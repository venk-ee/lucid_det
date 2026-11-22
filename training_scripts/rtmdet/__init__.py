"""
RTMDet training scripts package.
"""

from runners import build_optimizer
from training_scripts.rtmdet.config import TrainConfig
from training_scripts.rtmdet.train import run_training

__all__ = ["TrainConfig", "build_optimizer", "run_training"]
