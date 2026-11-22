"""
Deliberately thin inference and visualization CLI.
Loads checkpoint weights, executes single-image detection, and renders bounding boxes.
No registries or complex dispatchers: explicitly constructs the requested detector family.
Supports reading architecture configuration metadata from saved checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_detector
from transforms import letterbox_image, unletterbox_boxes_xyxy


def extract_checkpoint_payload(
    weights_path: str, device: torch.device
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Extract model state dict, model_config, data_config, and family from checkpoint."""
    ckpt = torch.load(weights_path, map_location=device)
    model_cfg = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    data_cfg = ckpt.get("data_config") if isinstance(ckpt, dict) else None
    family_hint = ckpt.get("family") if isinstance(ckpt, dict) else None

    # Fallback to CheckpointEnvelope metadata if present (e.g. from tools/convert_weights.py)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    if isinstance(meta, dict):
        if not family_hint:
            family_hint = meta.get("family")
        if not model_cfg and meta.get("arch"):
            model_cfg = {"arch": meta.get("arch")}
        if not data_cfg:
            data_cfg = {
                "num_classes": meta.get("num_classes", 80),
                "img_size": meta.get("input_size", (640, 640)),
            }

    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            ema_val = ckpt["ema"]
            # ModelEMA saves {"ema": model.state_dict(), "updates": ..., "decay": ...}
            sd = (
                ema_val["ema"] if "ema" in ema_val and isinstance(ema_val["ema"], dict) else ema_val
            )
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    clean_sd: Dict[str, Any] = {}
    for k, v in sd.items():
        clean_k = k[7:] if k.startswith("module.") else k
        clean_sd[clean_k] = v

    return clean_sd, model_cfg, data_cfg, family_hint


def load_weights(
    model: nn.Module,
    state_dict: Dict[str, Any],
    allow_partial: bool = False,
) -> nn.Module:
    """Load state dict into model with missing/unexpected key auditing."""
    if not allow_partial:
        model.load_state_dict(state_dict, strict=True)
    else:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Warning: {len(incompatible.missing_keys)} missing keys during weight loading.")
        if incompatible.unexpected_keys:
            print(f"Warning: {len(incompatible.unexpected_keys)} unexpected keys in checkpoint.")
    return model


def resolve_img_size(cli_size: Optional[int], data_cfg: Optional[Dict[str, Any]]) -> int:
    """Resolve input resolution with precedence: CLI flag -> data_cfg -> default 640."""
    if cli_size is not None:
        return cli_size
    if data_cfg and "img_size" in data_cfg:
        cfg_sz = data_cfg["img_size"]
        if isinstance(cfg_sz, (list, tuple)):
            return int(cfg_sz[0])
        elif isinstance(cfg_sz, (int, float)):
            return int(cfg_sz)
    return 640


def build_family_model(
    args: argparse.Namespace,
    model_cfg: Optional[Dict[str, Any]] = None,
    data_cfg: Optional[Dict[str, Any]] = None,
    family_hint: Optional[str] = None,
) -> nn.Module:
    """Instantiate detector module, defaulting to checkpoint metadata when available."""
    family = args.family
    if family is None:
        family = family_hint if family_hint is not None else "rtmdet"
    family = family.lower().strip()

    # Determine num_classes with precedence: CLI flag -> data_cfg -> model_cfg -> default 80
    num_classes = args.num_classes
    if num_classes is None:
        if data_cfg and "num_classes" in data_cfg:
            num_classes = int(data_cfg["num_classes"])
        elif model_cfg and "num_classes" in model_cfg:
            num_classes = int(model_cfg["num_classes"])
        else:
            num_classes = 80

    # Determine arch with precedence: CLI flag -> model_cfg -> default 'tiny'
    arch = args.arch
    if arch is None:
        if model_cfg and "arch" in model_cfg:
            arch = str(model_cfg["arch"])
        else:
            arch = "tiny"

    img_size = resolve_img_size(args.img_size, data_cfg)

    kwargs: Dict[str, Any] = {
        "num_classes": num_classes,
        "score_thresh": args.conf,
    }
    if family in ("rtmdet", "yolox"):
        kwargs["arch"] = arch
    elif family == "yolov1":
        split_size = (
            args.split_size
            if args.split_size is not None
            else (model_cfg.get("split_size", 7) if model_cfg else 7)
        )
        num_boxes = (
            args.num_boxes
            if args.num_boxes is not None
            else (model_cfg.get("num_boxes", 2) if model_cfg else 2)
        )
        kwargs.update(
            {
                "split_size": split_size,
                "num_boxes": num_boxes,
                "img_size": float(img_size),
            }
        )

    return build_detector(family, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet Lightweight Inference CLI")
    parser.add_argument("--img", type=str, required=True, help="Path to input image file")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=["rtmdet", "yolox", "yolov1"],
        help="Detector model family (default: auto-detected from checkpoint, or 'rtmdet')",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override architecture scale variant (default: from checkpoint, or 'tiny')",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Inference image resolution (default: from checkpoint, or 640)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override number of classes (default: from checkpoint, or 80)",
    )
    parser.add_argument(
        "--split-size",
        type=int,
        default=None,
        help="Override grid split size S for YOLOv1 (default: from checkpoint, or 7)",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=None,
        help="Override boxes per cell B for YOLOv1 (default: from checkpoint, or 2)",
    )
    parser.add_argument(
        "--conf",
        "--thr",
        type=float,
        default=0.3,
        help="Confidence score threshold (aliases: --conf, --thr)",
    )
    parser.add_argument(
        "--allow-partial-weights",
        action="store_true",
        help="Allow missing or unexpected keys when loading weights (default: strict check)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/vis.jpg",
        help="Output visualization file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 1. Read input image
    orig_bgr = cv2.imread(args.img)
    if orig_bgr is None:
        raise FileNotFoundError(f"Could not load image from: {args.img}")
    orig_h, orig_w = orig_bgr.shape[:2]

    # 2. Extract checkpoint and metadata on CPU to prevent redundant GPU memory allocation
    state_dict, model_cfg, data_cfg, family_hint = extract_checkpoint_payload(
        args.weights, torch.device("cpu")
    )

    # 3. Build model and load weights
    model = build_family_model(
        args, model_cfg=model_cfg, data_cfg=data_cfg, family_hint=family_hint
    )
    model = load_weights(model, state_dict, allow_partial=args.allow_partial_weights)
    model = model.to(device).eval()

    # 4. Determine image size and preprocess with letterbox
    img_size = resolve_img_size(args.img_size, data_cfg)

    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, scale, pad_left, pad_top, _, _ = letterbox_image(
        img_rgb, (img_size, img_size), color=(114, 114, 114)
    )
    # Normalize input matching training pipeline (ImageNet mean & std)
    inp = torch.from_numpy(letterboxed).permute(2, 0, 1).contiguous().float() / 255.0
    mean_tensor = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    inp = (inp - mean_tensor) / std_tensor
    inp = inp.unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        output = model(inp, training_override=False)
        det_output = (
            output if hasattr(output, "detections") else output.get("detections", [None])[0]
        )
        if hasattr(det_output, "detections"):
            detections = det_output.detections[0]
        elif isinstance(det_output, list):
            detections = det_output[0]
        else:
            detections = det_output

    boxes = (
        detections["boxes"].detach().cpu()
        if isinstance(detections, dict)
        else detections.boxes.detach().cpu()
    )
    scores = (
        detections["scores"].detach().cpu()
        if isinstance(detections, dict)
        else detections.scores.detach().cpu()
    )
    labels = (
        detections["labels"].detach().cpu()
        if isinstance(detections, dict)
        else detections.labels.detach().cpu()
    )

    # 6. Filter and rescale boxes to original image coordinates
    keep = scores >= args.conf
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    vis_img = orig_bgr.copy()
    if len(boxes) > 0:
        unscaled_boxes = unletterbox_boxes_xyxy(boxes, scale, pad_left, pad_top, orig_w, orig_h)
        boxes_np = unscaled_boxes.numpy().astype(np.int32)
        scores_np = scores.numpy()
        labels_np = labels.numpy().astype(int)

        for (x1, y1, x2, y2), sc, lb in zip(boxes_np, scores_np, labels_np):
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"cls {lb}: {sc:.2f}"
            cv2.putText(
                vis_img,
                caption,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, vis_img)
    print(f"Detected {len(boxes)} objects (conf >= {args.conf}). Saved to: {args.out}")


if __name__ == "__main__":
    main()
