"""
Utils package: Box helpers, EMA, environment utilities.
Standard utility exports: from utils.boxes import ..., from utils.ema import ModelEMA.
"""

from .boxes import (
    box_ciou,
    box_cxcywh_to_xyxy,
    box_giou,
    box_iou,
    box_xywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_xyxy_to_xywh,
)
from .ema import ModelEMA

__all__ = [
    "box_xywh_to_xyxy",
    "box_xyxy_to_xywh",
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_iou",
    "box_giou",
    "box_ciou",
    "ModelEMA",
]
