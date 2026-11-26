"""
YOLOv1 evaluation launcher.
Constructs YOLOv1, parses evaluation arguments, and delegates to generic_eval.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runners import generic_eval, set_seed
from training_scripts.common import add_common_eval_args, apply_common_eval_args
from training_scripts.yolov1.config import TrainConfig
from training_scripts.yolov1.train import build_model


def run_eval(cfg: TrainConfig, weights_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute YOLOv1 evaluation lifecycle via generic_eval."""
    set_seed(cfg.seed)
    device = torch.device(cfg.optim.device)
    model = build_model(cfg, device)

    print("==================================================")
    print(f"Initializing YOLOv1 Evaluation: {cfg.exp_name} (classes={cfg.model.num_classes})")
    print(f"Device: {device} | Weights: {weights_path or 'None (random init)'}")
    print(f"Work dir: {cfg.work_dir}")
    print("==================================================")

    return generic_eval(
        cfg=cfg,
        model=model,
        weights_path=weights_path,
        family_tag="yolov1",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for YOLOv1 evaluation."""
    parser = argparse.ArgumentParser(description="LucidDet YOLOv1 Evaluation Launcher")
    add_common_eval_args(parser)
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig()
    apply_common_eval_args(cfg, args)
    if args.split_size is not None:
        cfg.model.split_size = args.split_size
    if args.num_boxes is not None:
        cfg.model.num_boxes = args.num_boxes
    run_eval(cfg, weights_path=args.weights)
