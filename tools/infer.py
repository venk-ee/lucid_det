"""
Inference and visualization tool for LucidDet.
Runs single-image detector inference, performs NMS post-processing,
unletterboxes coordinates back to original image space, and renders detections.

Example usage:
    uv run python tools/infer.py --img data/dummy/images/dummy_0000.jpg \
        --weights work_dirs/exp/last.pth --model rtmdet_tiny --out outputs/vis.jpg
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transforms import letterbox_image, unletterbox_boxes_xyxy
from models import build_rtmdet, build_yolov1, build_yolox, RTMDet, YOLOX, YOLOv1


def create_model(model_name: str, num_classes: int) -> nn.Module:
    """Instantiate detector model architecture matching training configuration."""
    model_name = model_name.lower().replace("-", "_")
    if model_name in ("rtmdet_tiny", "rtmdet_t"):
        return build_rtmdet(arch="tiny", num_classes=num_classes)
    elif model_name in ("rtmdet_s", "rtmdet"):
        return build_rtmdet(arch="s", num_classes=num_classes)
    elif model_name in ("yolox_s", "yolox"):
        return build_yolox(arch="s", num_classes=num_classes)
    elif model_name in ("yolov1", "yolo_v1"):
        return build_yolov1(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model architecture: '{model_name}'. Choose from: rtmdet_tiny, rtmdet_s, yolox_s, yolov1"
        )


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Robust checkpoint loader.
    Handles CheckpointEnvelope metadata, ModelEMA shadow dictionaries,
    standard training checkpoints, and raw state_dicts.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    meta = None
    state_dict = None
    if isinstance(ckpt, dict):
        meta = ckpt.get("meta", None)
        if "ema" in ckpt:
            ema_data = ckpt["ema"]
            if isinstance(ema_data, dict) and "ema" in ema_data:
                state_dict = ema_data["ema"]
            elif isinstance(ema_data, dict):
                state_dict = ema_data
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Clean potential 'module.' prefixes from DataParallel / DDP
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Note: {len(incompatible.missing_keys)} missing keys during weight loading.")
    if incompatible.unexpected_keys:
        print(f"Note: {len(incompatible.unexpected_keys)} unexpected keys during weight loading.")

    model.to(device)
    model.eval()
    return model, meta


def get_color(idx: int) -> Tuple[int, int, int]:
    """Generate distinct pseudo-random color for visualization."""
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)
    ]
    return palette[idx % len(palette)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet Inference & Visualization")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--model", type=str, default="rtmdet_tiny", help="Model architecture")
    parser.add_argument("--out", type=str, default="outputs/vis.jpg", help="Path to save output visualization")
    parser.add_argument("--thr", type=float, default=0.3, help="Confidence score threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference resolution")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of foreground classes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class-names", type=str, default=None, help="Optional text file with class names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load image
    orig_img = cv2.imread(args.img)
    if orig_img is None:
        raise FileNotFoundError(f"Failed to read image at: {args.img}")
    orig_h, orig_w = orig_img.shape[:2]
    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # 2. Build model & load weights
    model = create_model(args.model, args.num_classes)
    if hasattr(model, "score_thresh"):
        model.score_thresh = min(model.score_thresh, args.thr)
    model, meta = load_weights(model, args.weights, device)

    # 3. Determine color space and normalization from metadata envelope if present
    color_space = "RGB"
    mean_vals = (0.485, 0.456, 0.406)
    std_vals = (0.229, 0.224, 0.225)
    img_size = args.img_size

    if meta is not None and isinstance(meta, dict):
        if "color_space" in meta and meta["color_space"]:
            color_space = str(meta["color_space"]).upper()
        if "mean" in meta and meta["mean"]:
            mean_vals = tuple(meta["mean"])
        if "std" in meta and meta["std"]:
            std_vals = tuple(meta["std"])
        if "input_size" in meta and meta["input_size"] and args.img_size == 640:
            img_size = meta["input_size"][0] if isinstance(meta["input_size"], (list, tuple)) else meta["input_size"]

    if color_space == "RGB":
        processed_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    else:
        processed_img = orig_img.copy()

    # Preprocess image
    letterboxed, scale, pad_left, pad_top, rw, rh = letterbox_image(
        processed_img, new_shape=img_size, color=(114, 114, 114)
    )

    mean_arr = np.array(mean_vals, dtype=np.float32)
    std_arr = np.array(std_vals, dtype=np.float32)
    if (meta is not None and meta.get("family") == "yolox") or args.model.startswith("yolox") or np.any(mean_arr > 1.0):
        inp = (letterboxed.astype(np.float32) - mean_arr) / std_arr
    else:
        inp = (letterboxed.astype(np.float32) / 255.0 - mean_arr) / std_arr

    inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. Model Forward Pass
    with torch.no_grad():
        outputs = model(inp_tensor, training_override=False)

    # Extract detections from SingleStageDetector output
    if isinstance(outputs, dict) and "detections" in outputs:
        detections = outputs["detections"][0]
    elif hasattr(outputs, "detections") and getattr(outputs, "detections") is not None:
        detections = getattr(outputs, "detections")[0]
    else:
        raise RuntimeError("Model output did not produce NMS detections.")

    pred_boxes = detections["boxes"]  # (K, 4) in letterbox space
    pred_scores = detections["scores"]  # (K,)
    pred_labels = detections["labels"]  # (K,)

    # 5. Coordinate transformation back to original image resolution
    orig_boxes = unletterbox_boxes_xyxy(
        boxes=pred_boxes,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        orig_w=orig_w,
        orig_h=orig_h,
    )

    # 6. Confidence Thresholding
    keep_mask = pred_scores >= args.thr
    filt_boxes = orig_boxes[keep_mask].cpu().numpy()
    filt_scores = pred_scores[keep_mask].cpu().numpy()
    filt_labels = pred_labels[keep_mask].cpu().numpy()

    # Load class names if provided
    class_names: List[str] = []
    if args.class_names and os.path.isfile(args.class_names):
        with open(args.class_names, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

    # 7. Render bounding boxes and labels
    vis = orig_img.copy()
    num_dets = len(filt_boxes)

    for (x1, y1, x2, y2), score, label_id in zip(filt_boxes, filt_scores, filt_labels):
        x1_i, y1_i, x2_i, y2_i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cls_idx = int(label_id)
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"cls_{cls_idx}"
        label_text = f"{cls_name} {score:.2f}"

        color = get_color(cls_idx)
        cv2.rectangle(vis, (x1_i, y1_i), (x2_i, y2_i), color, 2)

        # Draw filled text background
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_y1 = max(0, y1_i - th - baseline - 4)
        bg_y2 = y1_i
        cv2.rectangle(vis, (x1_i, bg_y1), (x1_i + tw + 4, bg_y2), color, -1)
        cv2.putText(
            vis,
            label_text,
            (x1_i + 2, bg_y2 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 8. Save visualization
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out, vis)

    print(f"Detected {num_dets} objects (threshold={args.thr:.2f})")
    print(f"Visualization saved to: {args.out}")


if __name__ == "__main__":
    main()
