"""
RTMDet training scripts package.
"""

from training_scripts.rtmdet.config import TrainConfig
from training_scripts.rtmdet.train import build_optimizer, run_training

__all__ = ["TrainConfig", "build_optimizer", "run_training"]
