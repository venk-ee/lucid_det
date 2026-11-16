"""
Core geometry, bounding box coding, IoU calculations, and NMS primitives.
"""

from core.box_coder import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_xyxy_to_xywh,
    cxcywh_to_xyxy,
    decode_ltrb_to_xyxy,
    encode_xyxy_to_ltrb,
    generate_points_priors,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    xyxy_to_xywh,
)
from core.iou import (
    box_area,
    box_ciou,
    box_giou,
    box_iou,
)
from core.nms import (
    batched_nms,
    multiclass_nms,
    multiclass_nms_batch,
)
from core.outputs import DetectionOutput

__all__ = [
    "box_xywh_to_xyxy",
    "box_xyxy_to_xywh",
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "cxcywh_to_xyxy",
    "xyxy_to_cxcywh",
    "generate_points_priors",
    "decode_ltrb_to_xyxy",
    "encode_xyxy_to_ltrb",
    "box_area",
    "box_iou",
    "box_giou",
    "box_ciou",
    "batched_nms",
    "multiclass_nms",
    "multiclass_nms_batch",
    "DetectionOutput",
]
