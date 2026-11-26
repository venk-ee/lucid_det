"""
YOLOX evaluation launcher.
Constructs YOLOX, parses evaluation arguments, and delegates to generic_eval.
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
from training_scripts.yolox.config import TrainConfig
from training_scripts.yolox.train import build_model
from transforms import build_det_transform


def run_eval(cfg: TrainConfig, weights_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute YOLOX evaluation lifecycle via generic_eval with BGR [0, 255] preprocessing."""
    set_seed(cfg.seed)
    device = torch.device(cfg.optim.device)
    model = build_model(cfg, device)

    val_transform = build_det_transform(
        img_size=cfg.data.img_size,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        max_pixel_value=1.0,
        bgr=True,
        is_train=False,
    )

    print("==================================================")
    print(f"Initializing YOLOX Evaluation: {cfg.exp_name} (arch={cfg.model.arch})")
    print(f"Device: {device} | Weights: {weights_path or 'None (random init)'}")
    print(f"Work dir: {cfg.work_dir}")
    print("==================================================")

    return generic_eval(
        cfg=cfg,
        model=model,
        weights_path=weights_path,
        family_tag="yolox",
        val_transform=val_transform,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for YOLOX evaluation."""
    parser = argparse.ArgumentParser(description="LucidDet YOLOX Evaluation Launcher")
    add_common_eval_args(parser)
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override model architecture scale (tiny, s, m, l, x)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig()
    apply_common_eval_args(cfg, args)
    if args.arch is not None:
        cfg.model.arch = args.arch
    run_eval(cfg, weights_path=args.weights)
