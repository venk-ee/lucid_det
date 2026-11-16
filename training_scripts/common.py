"""
Common CLI argument parser and config override utilities for training launchers.
Provides shared dataset, optimizer, and experiment lifecycle arguments across model families.
"""

import argparse
from typing import Any


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register common training arguments shared by all detector families."""
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override base learning rate")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override number of foreground classes",
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="Override dataset root directory"
    )
    parser.add_argument(
        "--train-img",
        type=str,
        default=None,
        help="Override training images subdirectory/path",
    )
    parser.add_argument(
        "--train-ann",
        type=str,
        default=None,
        help="Override training COCO annotations JSON path",
    )
    parser.add_argument(
        "--val-img",
        type=str,
        default=None,
        help="Override validation images subdirectory/path",
    )
    parser.add_argument(
        "--val-ann",
        type=str,
        default=None,
        help="Override validation COCO annotations JSON path",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override image input resolution (pixels)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override DataLoader worker count",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=None,
        help="Number of final epochs to switch to clean evaluation transform cooldown (default: from config, or 0)",
    )
    return parser


def apply_common_training_args(config: Any, args: argparse.Namespace) -> Any:
    """Apply parsed common CLI arguments onto a model family's TrainConfig."""
    if args.epochs is not None:
        config.optim.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optim.lr = args.lr
    if args.num_classes is not None:
        if hasattr(config, "model") and hasattr(config.model, "num_classes"):
            config.model.num_classes = args.num_classes
        if hasattr(config, "data") and hasattr(config.data, "num_classes"):
            config.data.num_classes = args.num_classes
    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.train_img is not None:
        config.data.train_img = args.train_img
    if args.train_ann is not None:
        config.data.train_ann = args.train_ann
    if args.val_img is not None:
        config.data.val_img = args.val_img
    if args.val_ann is not None:
        config.data.val_ann = args.val_ann
    if args.img_size is not None:
        if isinstance(getattr(config.data, "img_size", None), tuple):
            config.data.img_size = (args.img_size, args.img_size)
        else:
            config.data.img_size = args.img_size
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.workers is not None:
        config.data.num_workers = args.workers
    if args.device is not None:
        config.optim.device = args.device
    if getattr(args, "resume", None) is not None:
        config.optim.resume = args.resume
    if getattr(args, "cooldown_epochs", None) is not None:
        if hasattr(config, "optim") and hasattr(config.optim, "cooldown_epochs"):
            config.optim.cooldown_epochs = args.cooldown_epochs
    return config
