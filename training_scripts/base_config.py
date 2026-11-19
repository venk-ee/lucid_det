"""
Central base configurations shared across model training scripts.
Standard dataclasses for data, optimization, and evaluation settings.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""

    data_root: str = "data"
    train_ann: str = "annotations/instances_train2017.json"
    val_ann: str = "annotations/instances_val2017.json"
    train_img: str = "train2017"
    val_img: str = "val2017"
    num_classes: int = 80
    img_size: Tuple[int, int] = (640, 640)
    batch_size: int = 16
    val_batch_size: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class OptimConfig:
    """Optimization hyperparameters."""

    epochs: int = 300
    lr: float = 2e-3
    weight_decay: float = 0.05
    clip_max_norm: float = 35.0
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_epochs: int = 3
    cooldown_epochs: int = 0
    min_lr_ratio: float = 0.05
    ema_decay: float = 0.9998
    resume: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation cadence and metric parameters."""

    eval_interval: int = 1
    compute_map: bool = True
