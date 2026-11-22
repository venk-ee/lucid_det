"""
YOLOv1 training launcher.
Defines model, loss, parses arguments, and delegates to generic_train.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses import YOLOv1Loss, encode_yolov1_targets
from models import YOLOv1
from runners import generic_train, set_seed
from training_scripts.common import add_common_training_args, apply_common_training_args
from training_scripts.yolov1.config import TrainConfig


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    """Construct and place YOLOv1 on the target device."""
    return YOLOv1(
        num_classes=cfg.model.num_classes,
        score_thresh=cfg.model.score_thresh,
        nms_thresh=cfg.model.nms_thresh,
        split_size=cfg.model.split_size,
        num_boxes=cfg.model.num_boxes,
    ).to(device)


def build_loss_fn(cfg: TrainConfig, device: torch.device) -> Callable:
    """Build the YOLOv1 loss closure."""
    yolo_loss_fn = YOLOv1Loss(
        split_size=cfg.model.split_size,
        num_boxes=cfg.model.num_boxes,
        num_classes=cfg.model.num_classes,
    )
    img_sz = float(
        cfg.data.img_size[0] if isinstance(cfg.data.img_size, tuple) else cfg.data.img_size
    )

    def loss_fn(outputs, targets):
        if isinstance(targets, dict) and "gt_boxes" in targets:
            targets_matrix = encode_yolov1_targets(
                gt_boxes=targets["gt_boxes"],
                gt_labels=targets["gt_labels"],
                split_size=cfg.model.split_size,
                num_classes=cfg.model.num_classes,
                img_size=img_sz,
                device=device,
            )
        else:
            b_size = outputs.shape[0] if hasattr(outputs, "shape") else 1
            targets_matrix = torch.zeros(
                b_size,
                cfg.model.split_size,
                cfg.model.split_size,
                cfg.model.num_classes + 5,
                device=device,
            )
        return yolo_loss_fn(outputs, targets_matrix)

    return loss_fn


def run_training(cfg: TrainConfig) -> None:
    """Execute complete YOLOv1 training lifecycle via generic_train."""
    set_seed(cfg.seed)
    device = torch.device(cfg.optim.device)
    model = build_model(cfg, device)
    loss_fn = build_loss_fn(cfg, device)

    print("==================================================")
    print(f"Initializing YOLOv1 Training: {cfg.exp_name} (classes={cfg.model.num_classes})")
    print(f"Device: {device} | Epochs: {cfg.optim.epochs} | Base LR: {cfg.optim.lr:.2e}")
    print(f"Work dir: {cfg.work_dir}")
    print("==================================================")

    generic_train(cfg, model, loss_fn, family_tag="yolov1")
    print(f"\nYOLOv1 training completed successfully! Artifacts in: {cfg.work_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet YOLOv1 Training Launcher")
    add_common_training_args(parser)
    parser.add_argument(
        "--split-size",
        type=int,
        default=None,
        help="Override grid split size S (default: 7)",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        choices=[2],
        default=None,
        help="Override boxes per cell B (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig()
    apply_common_training_args(cfg, args)
    if args.split_size is not None:
        cfg.model.split_size = args.split_size
    if args.num_boxes is not None:
        if args.num_boxes != 2:
            raise ValueError(
                f"YOLOv1Loss currently supports only --num-boxes 2, got {args.num_boxes}"
            )
        cfg.model.num_boxes = args.num_boxes
    run_training(cfg)
