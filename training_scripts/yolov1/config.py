"""
Dataclass training configurations for YOLOv1 detector.
Structured following clean configuration patterns.
"""

from dataclasses import dataclass, field
from typing import Tuple

from training_scripts.base_config import (
    DataConfig as BaseDataConfig,
)
from training_scripts.base_config import (
    EvalConfig,
)
from training_scripts.base_config import (
    OptimConfig as BaseOptimConfig,
)


@dataclass
class ModelConfig:
    """YOLOv1 architecture configuration."""

    arch: str = "darknet24"
    num_classes: int = 20
    score_thresh: float = 0.05
    nms_thresh: float = 0.50
    split_size: int = 7
    num_boxes: int = 2


@dataclass
class DataConfig(BaseDataConfig):
    """YOLOv1 VOC dataset configuration."""

    train_ann: str = "annotations/voc07_train.json"
    val_ann: str = "annotations/voc07_val.json"
    train_img: str = "images"
    val_img: str = "images"
    num_classes: int = 20
    img_size: Tuple[int, int] = (448, 448)


@dataclass
class OptimConfig(BaseOptimConfig):
    """YOLOv1-specific optimization hyperparameters."""

    epochs: int = 135
    lr: float = 2e-5
    weight_decay: float = 5e-4
    warmup_epochs: int = 5


@dataclass
class TrainConfig:
    """Unified YOLOv1 training configuration."""

    exp_name: str = "yolov1_voc"
    work_dir: str = "work_dirs/yolov1_voc"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
