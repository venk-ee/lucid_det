"""
Datasets package for lucid_det.
Exposes task-specific dataset loaders, collators, and tensor utilities.
"""

from datasets.det_dataset import DetDataset, det_collate
from datasets.utils import scale_boxes
from transforms.det_transforms import build_det_transform

__all__ = [
    "DetDataset",
    "det_collate",
    "scale_boxes",
    "build_det_transform",
]
