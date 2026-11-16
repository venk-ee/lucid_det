"""
YOLOv1 detector model class.
Assembles Darknet24 backbone and YOLOv1Head.
"""

from typing import Any, Optional

from backbones import build_backbone
from heads import YOLOv1Head
from models.single_stage import SingleStageDetector
from necks import build_neck


class YOLOv1(SingleStageDetector):
    """
    YOLOv1: You Only Look Once Unified Real-Time Object Detector (Redmon et al. 2016).
    Composes Darknet24 Backbone -> YOLOv1Head (FC Detection Head).
    """

    def __init__(
        self,
        num_classes: int = 20,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.65,
        max_num: int = 100,
        split_size: int = 7,
        num_boxes: int = 2,
        S: Optional[int] = None,
        B: Optional[int] = None,
        img_size: float = 640.0,
        **kwargs: Any,
    ) -> None:
        if S is not None:
            split_size = S
        if B is not None:
            num_boxes = B

        backbone = build_backbone("darknet24", in_channels=3)
        neck = build_neck("identity")
        head = YOLOv1Head(
            in_channels=1024,
            num_classes=num_classes,
            split_size=split_size,
            num_boxes=num_boxes,
            img_size=img_size,
        )

        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            max_num=max_num,
            **kwargs,
        )
        self.num_classes = num_classes
        self.split_size = split_size
        self.num_boxes = num_boxes
