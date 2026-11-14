"""
Pre-wired model builders and model zoo factory functions.
Provides factory builders and class constructors for RTMDet, YOLOX, and YOLOv1.
"""

from typing import Any, Optional, Sequence
import torch.nn as nn

from models.outputs import DetectionOutput
from models.rtmdet import RTMDet
from models.single_stage import SingleStageDetector
from models.yolov1 import YOLOv1
from models.yolox import YOLOX


def build_rtmdet(
    arch: str = "tiny",
    num_classes: int = 80,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.65,
    max_num: int = 100,
    frozen_stages: int = -1,
    **kwargs: Any,
) -> RTMDet:
    """Build a complete RTMDet detector."""
    return RTMDet(
        arch=arch,
        num_classes=num_classes,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        max_num=max_num,
        frozen_stages=frozen_stages,
        **kwargs,
    )


def build_yolox(
    arch: str = "s",
    num_classes: int = 80,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.65,
    max_num: int = 100,
    frozen_stages: int = -1,
    **kwargs: Any,
) -> YOLOX:
    """Build a complete YOLOX detector."""
    return YOLOX(
        arch=arch,
        num_classes=num_classes,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        max_num=max_num,
        frozen_stages=frozen_stages,
        **kwargs,
    )


def build_yolov1(
    num_classes: int = 20,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.65,
    max_num: int = 100,
    split_size: int = 7,
    num_boxes: int = 2,
    S: Optional[int] = None,
    B: Optional[int] = None,
    **kwargs: Any,
) -> YOLOv1:
    """Build a complete YOLOv1 detector."""
    if S is not None:
        split_size = S
    if B is not None:
        num_boxes = B
    return YOLOv1(
        num_classes=num_classes,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        max_num=max_num,
        split_size=split_size,
        num_boxes=num_boxes,
        **kwargs,
    )
