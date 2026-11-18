"""
YOLOv1 detector model class.
Assembles Darknet24 backbone and YOLOv1Head.
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch.nn as nn

from backbones import Darknet24
from heads import YOLOv1Head
from models.single_stage import SingleStageDetector
from necks import IdentityNeck


class YOLOv1(SingleStageDetector):
    """
    YOLOv1: You Only Look Once Unified Real-Time Object Detector (Redmon et al. 2016).
    Composes Darknet24 Backbone -> YOLOv1Head (FC Detection Head).
    Supports direct dependency injection of backbone, neck, and head modules.
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
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        weights_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if S is not None:
            split_size = S
        if B is not None:
            num_boxes = B

        if backbone is None:
            backbone = Darknet24(in_channels=3)
        if neck is None:
            neck = IdentityNeck()
        if head is None:
            in_ch = 1024
            if hasattr(neck, "out_channels"):
                out_c = getattr(neck, "out_channels")
                in_ch = out_c[0] if isinstance(out_c, (list, tuple)) else out_c
            elif hasattr(backbone, "out_channels"):
                out_c = getattr(backbone, "out_channels")
                in_ch = out_c[0] if isinstance(out_c, (list, tuple)) else out_c

            head = YOLOv1Head(
                in_channels=in_ch,
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
            checkpoint=checkpoint,
            weights_only=weights_only,
            **kwargs,
        )
        self.num_classes = num_classes
        self.split_size = split_size
        self.num_boxes = num_boxes
