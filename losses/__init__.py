"""
Loss functions and multi-task loss dispatchers for LucidDet.
Direct imports of pure PyTorch loss modules and utilities.
"""

from losses.det_loss import (
    BOX_LOSS_MAP,
    CLS_LOSS_MAP,
    BCEWithLogitsLoss,
    det_loss,
    resolve_assigner,
    resolve_box_loss,
    resolve_cls_loss,
)
from losses.iou_loss import CIoULoss, GIoULoss, IoULoss
from losses.qfl import QualityFocalLoss
from losses.yolov1_loss import YOLOv1Loss, encode_yolov1_targets

__all__ = [
    "QualityFocalLoss",
    "BCEWithLogitsLoss",
    "GIoULoss",
    "CIoULoss",
    "IoULoss",
    "YOLOv1Loss",
    "encode_yolov1_targets",
    "det_loss",
    "resolve_cls_loss",
    "resolve_box_loss",
    "resolve_assigner",
    "CLS_LOSS_MAP",
    "BOX_LOSS_MAP",
]
