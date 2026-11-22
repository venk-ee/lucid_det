"""
YOLOX detector model class.
Assembles CSPDarknet backbone, YOLOXPAFPN neck, and Decoupled YOLOXHead.
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch.nn as nn

from backbones import CSPDarknet
from heads import YOLOXHead
from models.single_stage import SingleStageDetector
from necks import YOLOXPAFPN


class YOLOX(SingleStageDetector):
    """
    YOLOX: Exceeding YOLO Series in 2021.
    Composes CSPDarknet Backbone -> YOLOXPAFPN Neck -> Decoupled YOLOXHead.
    Supports direct dependency injection of backbone, neck, and head modules.
    """

    def __init__(
        self,
        arch: str = "s",
        num_classes: int = 80,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.65,
        max_num: int = 100,
        frozen_stages: int = -1,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        weights_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if backbone is None:
            backbone = CSPDarknet(arch=arch, frozen_stages=frozen_stages)

        in_channels = list(getattr(backbone, "out_channels", [128, 256, 512]))
        strides = tuple(getattr(backbone, "strides", (8, 16, 32)))
        depth_mult = {
            "tiny": 0.33,
            "s": 0.33,
            "m": 0.67,
            "l": 1.00,
            "x": 1.33,
        }.get(arch, 0.33)
        num_csp_blocks = max(round(3 * depth_mult), 1)

        if neck is None:
            neck = YOLOXPAFPN(
                in_channels=in_channels,
                num_csp_blocks=num_csp_blocks,
            )

        head_in_channels = in_channels
        if hasattr(neck, "out_channels"):
            neck_out = getattr(neck, "out_channels")
            if isinstance(neck_out, (list, tuple)):
                head_in_channels = list(neck_out)
            elif isinstance(neck_out, int):
                head_in_channels = [neck_out] * len(strides)

        if head is None:
            head = YOLOXHead(
                num_classes=num_classes,
                in_channels=head_in_channels,
                feat_channels=head_in_channels[0],
                strides=strides,
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
        self.arch = arch
        self.num_classes = num_classes
