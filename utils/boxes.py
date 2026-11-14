"""
Bounding box transformations and geometric operations.
Standard bounding box utility functions.
"""

from core.box_coder import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_xyxy_to_xywh,
    cxcywh_to_xyxy,
    decode_bboxes,
    encode_bboxes,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_xywh,
)
from core.iou import box_ciou, box_giou, box_iou

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
