"""
Standard YOLOX PAFPN Neck.
Bidirectional Feature Pyramid Network with standard CSPDarknet blocks.
"""

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.common import ConvModule, CSPLayer
from necks import register_neck


@register_neck("pafpn")
@register_neck("yolox_pafpn")
class YOLOXPAFPN(nn.Module):
    """
    Standard Path Aggregation FPN for YOLOX.

    Args:
        in_channels: Sequence of input channel dimensions [P3, P4, P5] (default: (128, 256, 512) for 's').
        out_channels: Output channel dimension across all levels (default: 128 for 's', or 256).
        num_csp_blocks: Number of Darknet bottlenecks in each CSP layer.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 256, 512),
        out_channels: Optional[Any] = None,
        num_csp_blocks: int = 1,
    ) -> None:
        super().__init__()
        assert len(in_channels) == 3, (
            f"YOLOX PAFPN expects 3 levels [P3, P4, P5], got {len(in_channels)}"
        )
        self.in_channels = list(in_channels)
        c3, c4, c5 = in_channels
        self.out_channels = tuple(in_channels)

        # Top-down pathway
        # P5 (c5) -> lateral 1x1 conv to match c4
        self.lateral_conv0 = ConvModule(c5, c4, kernel_size=1, stride=1)
        # CSP block after fusing upsampled P5 with P4
        self.top_down_csp0 = CSPLayer(
            c4 * 2,
            c4,
            num_blocks=num_csp_blocks,
            add_identity=False,
            use_cspnext_block=False,
            channel_attention=False,
        )

        # lateral 1x1 conv from fused P4 to match c3
        self.lateral_conv1 = ConvModule(c4, c3, kernel_size=1, stride=1)
        # CSP block after fusing upsampled P4 with P3 -> outputs final F3
        self.top_down_csp1 = CSPLayer(
            c3 * 2,
            c3,
            num_blocks=num_csp_blocks,
            add_identity=False,
            use_cspnext_block=False,
            channel_attention=False,
        )

        # Bottom-up pathway
        # Downsample F3 -> match F4
        self.downsample0 = ConvModule(c3, c3, kernel_size=3, stride=2, padding=1)
        self.bottom_up_csp0 = CSPLayer(
            c3 * 2,
            c4,
            num_blocks=num_csp_blocks,
            add_identity=False,
            use_cspnext_block=False,
            channel_attention=False,
        )

        # Downsample F4 -> match F5
        self.downsample1 = ConvModule(c4, c4, kernel_size=3, stride=2, padding=1)
        self.bottom_up_csp1 = CSPLayer(
            c4 * 2,
            c5,
            num_blocks=num_csp_blocks,
            add_identity=False,
            use_cspnext_block=False,
            channel_attention=False,
        )

    def forward(
        self, feats: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = feats

        # Top-down
        p5_lat = self.lateral_conv0(p5)
        p5_up = F.interpolate(p5_lat, size=p4.shape[-2:], mode="nearest")
        p4_cat = torch.cat([p5_up, p4], dim=1)
        p4_fused = self.top_down_csp0(p4_cat)

        p4_lat = self.lateral_conv1(p4_fused)
        p4_up = F.interpolate(p4_lat, size=p3.shape[-2:], mode="nearest")
        p3_cat = torch.cat([p4_up, p3], dim=1)
        f3 = self.top_down_csp1(p3_cat)

        # Bottom-up
        f3_down = self.downsample0(f3)
        f4_cat = torch.cat([f3_down, p4_lat], dim=1)
        f4 = self.bottom_up_csp0(f4_cat)

        f4_down = self.downsample1(f4)
        f5_cat = torch.cat([f4_down, p5_lat], dim=1)
        f5 = self.bottom_up_csp1(f5_cat)

        return (f3, f4, f5)
