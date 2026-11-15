"""
Training scripts and experiment configuration definitions detector families.
"""

from training_scripts import rtmdet, yolov1, yolox
from training_scripts.common import add_common_training_args, apply_common_training_args

__all__ = [
    "rtmdet",
    "yolox",
    "yolov1",
    "add_common_training_args",
    "apply_common_training_args",
]
