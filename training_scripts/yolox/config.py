"""
Dataclass training configurations for YOLOX detector.
Structured following clean configuration patterns.
"""

from dataclasses import dataclass, field

from training_scripts.base_config import (
    BaseTrainConfig,
)
from training_scripts.base_config import (
    OptimConfig as BaseOptimConfig,
)


@dataclass
class ModelConfig:
    """YOLOX architecture configuration."""

    arch: str = "s"
    num_classes: int = 80
    score_thresh: float = 0.05
    nms_thresh: float = 0.65
    cls_loss: str = "qfl"
    box_loss: str = "ciou"


@dataclass
class OptimConfig(BaseOptimConfig):
    """YOLOX-specific optimization hyperparameters."""

    lr: float = 1e-2
    weight_decay: float = 5e-4
    warmup_epochs: int = 5


@dataclass
class TrainConfig(BaseTrainConfig):
    """Unified YOLOX training configuration."""

    exp_name: str = "yolox_s"
    work_dir: str = "work_dirs/yolox_s"
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
