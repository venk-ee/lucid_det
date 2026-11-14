"""
Bounding box transformations and geometric operations (Pure PyTorch).
Standard bounding box utility functions.
"""

from typing import Optional, Tuple
import torch
from core.box_coder import (
    xywh_to_xyxy,
    xyxy_to_xywh,
    cxcywh_to_xyxy,
    xyxy_to_cxcywh,
    box_xywh_to_xyxy,
    box_xyxy_to_xywh,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    decode_bboxes,
    encode_bboxes,
)
from core.iou import box_iou, box_giou, box_ciou

__all__ = [
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "cxcywh_to_xyxy",
    "xyxy_to_cxcywh",
    "box_xywh_to_xyxy",
    "box_xyxy_to_xywh",
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "decode_bboxes",
    "encode_bboxes",
    "box_iou",
    "box_giou",
    "box_ciou",
]
