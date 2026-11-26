"""
Common CLI argument parser and config override utilities for training and evaluation launchers.
Provides shared dataset, optimizer, and experiment lifecycle arguments across model families.
"""

import argparse
from typing import Any


def _add_shared_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register dataset, resolution, device, and output arguments shared by train and eval."""
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
        help="Override output directory",
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
    return parser


def _apply_shared_runtime_args(config: Any, args: argparse.Namespace) -> Any:
    """Apply shared dataset, resolution, device, and output arguments onto TrainConfig."""
    if args.num_classes is not None:
        if hasattr(config, "model") and hasattr(config.model, "num_classes"):
            config.model.num_classes = args.num_classes
        if hasattr(config, "data") and hasattr(config.data, "num_classes"):
            config.data.num_classes = args.num_classes
    if args.data_root is not None:
        config.data.data_root = args.data_root
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
    return config


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register common training arguments shared by all detector families."""
    _add_shared_runtime_args(parser)
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override base learning rate")
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
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from",
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="Path to pretrained .pth weights to initialize model (weights-only)",
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
    _apply_shared_runtime_args(config, args)
    if args.epochs is not None:
        config.optim.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optim.lr = args.lr
    if args.train_img is not None:
        config.data.train_img = args.train_img
    if args.train_ann is not None:
        config.data.train_ann = args.train_ann
    if getattr(args, "resume", None) is not None:
        config.optim.resume = args.resume
    if (
        getattr(args, "cooldown_epochs", None) is not None
        and hasattr(config, "optim")
        and hasattr(config.optim, "cooldown_epochs")
    ):
        config.optim.cooldown_epochs = args.cooldown_epochs
    if (
        getattr(args, "load_from", None) is not None
        and hasattr(config, "optim")
        and hasattr(config.optim, "load_from")
    ):
        config.optim.load_from = args.load_from
    return config


def add_common_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register common evaluation arguments shared by all detector families."""
    _add_shared_runtime_args(parser)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override validation batch size",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=None,
        help="Override detection confidence threshold",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=None,
        help="Override NMS IoU threshold",
    )
    return parser


def apply_common_eval_args(config: Any, args: argparse.Namespace) -> Any:
    """Apply parsed common evaluation CLI arguments onto a model family's TrainConfig."""
    _apply_shared_runtime_args(config, args)
    if args.batch_size is not None:
        config.data.val_batch_size = args.batch_size
        config.data.batch_size = args.batch_size
    if args.score_thresh is not None and hasattr(config, "model"):
        config.model.score_thresh = args.score_thresh
    if args.nms_thresh is not None and hasattr(config, "model"):
        config.model.nms_thresh = args.nms_thresh
    return config
