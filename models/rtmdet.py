"""
RTMDet detector model class.
Assembles CSPNeXt backbone, CSPNeXtPAFPN neck, and SepBNHead head.
"""

from typing import Any

from backbones import CSPNeXt
from heads import SepBNHead
from models.single_stage import SingleStageDetector
from necks import CSPNeXtPAFPN


class RTMDet(SingleStageDetector):
    """
    RTMDet: Real-Time Models for High-Precision Object Detection.
    Composes CSPNeXt Backbone -> CSPNeXtPAFPN Neck -> SepBNHead Head.
    """

    def __init__(
        self,
        arch: str = "tiny",
        num_classes: int = 80,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.65,
        max_num: int = 100,
        frozen_stages: int = -1,
        **kwargs: Any,
    ) -> None:
        backbone = CSPNeXt(arch=arch, frozen_stages=frozen_stages)
        in_channels = list(backbone.out_channels)
        strides = tuple(backbone.strides)

        neck_channels = in_channels[0]
        use_depthwise = arch in ("tiny", "tiny-dw")
        exp_on_reg = arch not in ("tiny", "tiny-dw")

        neck = CSPNeXtPAFPN(
            in_channels=in_channels,
            out_channels=neck_channels,
            num_csp_blocks=1,
            use_depthwise=use_depthwise,
        )

        head = SepBNHead(
            num_classes=num_classes,
            in_channels=neck_channels,
            feat_channels=neck_channels,
            strides=strides,
            exp_on_reg=exp_on_reg,
            share_conv=True,
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
