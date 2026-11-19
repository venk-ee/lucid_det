"""
RTMDet training launcher.
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

from losses import det_loss, resolve_box_loss, resolve_cls_loss
from models import RTMDet
from runners import generic_train
from training_scripts.common import add_common_training_args, apply_common_training_args
from training_scripts.rtmdet.config import TrainConfig


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    """Construct and place RTMDet on the target device."""
    return RTMDet(
        arch=cfg.model.arch,
        num_classes=cfg.model.num_classes,
        score_thresh=cfg.model.score_thresh,
        nms_thresh=cfg.model.nms_thresh,
    ).to(device)


def build_loss_fn(cfg: TrainConfig) -> Callable:
    """Build the RTMDet loss closure."""
    cls_loss_fn = resolve_cls_loss(cfg.model.cls_loss)
    box_loss_fn = resolve_box_loss(cfg.model.box_loss)

    def loss_fn(outputs, targets):
        return det_loss(
            model_out=outputs,
            gt_boxes_list=targets["gt_boxes"],
            gt_labels_list=targets["gt_labels"],
            cls_loss=cls_loss_fn,
            box_loss=box_loss_fn,
        )

    return loss_fn


def run_training(cfg: TrainConfig) -> None:
    """Execute complete RTMDet training lifecycle via generic_train."""
    device = torch.device(cfg.optim.device)
    model = build_model(cfg, device)
    loss_fn = build_loss_fn(cfg)

    print("==================================================")
    print(f"Initializing RTMDet Training: {cfg.exp_name} (arch={cfg.model.arch})")
    print(f"Device: {device} | Epochs: {cfg.optim.epochs} | Base LR: {cfg.optim.lr:.2e}")
    print(f"Work dir: {cfg.work_dir}")
    print("==================================================")

    generic_train(cfg, model, loss_fn, family_tag="rtmdet")
    print(f"\nRTMDet training completed successfully! Artifacts in: {cfg.work_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet RTMDet Training Launcher")
    add_common_training_args(parser)
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override model architecture scale (tiny, s, m, l, x)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig()
    apply_common_training_args(cfg, args)
    if args.arch is not None:
        cfg.model.arch = args.arch
    run_training(cfg)
