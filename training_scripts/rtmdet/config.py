"""
Dataclass training configurations for RTMDet detector.
Structured following clean configuration patterns.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class ModelConfig:
    """RTMDet architecture configuration."""

    arch: str = "s"
    num_classes: int = 80
    score_thresh: float = 0.05
    nms_thresh: float = 0.65
    cls_loss: str = "qfl"
    box_loss: str = "ciou"


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
    min_lr_ratio: float = 0.05
    ema_decay: float = 0.9998
    resume: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation cadence and metric parameters."""

    eval_interval: int = 1
    compute_map: bool = True


@dataclass
class TrainConfig:
    """Unified RTMDet training configuration."""

    exp_name: str = "rtmdet_s"
    work_dir: str = "work_dirs/rtmdet_s"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
