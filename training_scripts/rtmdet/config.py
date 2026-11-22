"""
Dataclass training configurations for RTMDet detector.
Structured following clean configuration patterns.
"""

from dataclasses import dataclass, field

from training_scripts.base_config import (
    BaseTrainConfig,
)


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
class TrainConfig(BaseTrainConfig):
    """Unified RTMDet training configuration."""

    exp_name: str = "rtmdet_s"
    work_dir: str = "work_dirs/rtmdet_s"
    model: ModelConfig = field(default_factory=ModelConfig)
