"""
YOLOX detector model class.
Assembles CSPDarknet backbone, YOLOXPAFPN neck, and Decoupled YOLOXHead.
"""

from typing import Any

from backbones import build_backbone
from heads import YOLOXHead
from models.single_stage import SingleStageDetector
from necks import build_neck


class YOLOX(SingleStageDetector):
    """
    YOLOX: Exceeding YOLO Series in 2021.
    Composes CSPDarknet Backbone -> YOLOXPAFPN Neck -> Decoupled YOLOXHead.
    """

    def __init__(
        self,
        arch: str = "s",
        num_classes: int = 80,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.65,
        max_num: int = 100,
        frozen_stages: int = -1,
        **kwargs: Any,
    ) -> None:
        backbone = build_backbone("cspdarknet", arch=arch, frozen_stages=frozen_stages)
        in_channels = list(backbone.out_channels)
        strides = tuple(backbone.strides)

        neck_channels = in_channels[0]
        depth_mult = {
            "tiny": 0.33,
            "s": 0.33,
            "m": 0.67,
            "l": 1.00,
            "x": 1.33,
        }.get(arch, 0.33)
        num_csp_blocks = max(round(3 * depth_mult), 1)

        neck = build_neck(
            "yolox_pafpn",
            in_channels=in_channels,
            num_csp_blocks=num_csp_blocks,
        )

        head = YOLOXHead(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=neck_channels,
            strides=strides,
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
        self.arch = arch
        self.num_classes = num_classes
