"""
CSPNeXtPAFPN Neck
Multi-scale Path Aggregation Feature Pyramid Network with CSP blocks.
"""

from collections.abc import Sequence

import torch
from torch import nn

from backbones.common import ConvModule, CSPLayer, DepthwiseSeparableConv


class CSPNeXtPAFPN(nn.Module):
    """
    Path Aggregation FPN with CSPNeXt blocks.

    Args:
        in_channels: Input channels from backbone stages [P3, P4, P5].
        out_channels: Output feature channels for each level.
        num_csp_blocks: Number of blocks inside each CSPLayer.
        use_depthwise: Whether to use depthwise separable convolutions.
        expand_ratio: Channel expansion ratio inside CSPLayer.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384),
        out_channels: int = 96,
        num_csp_blocks: int = 1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        assert len(in_channels) >= 2, "PAFPN requires at least 2 input feature levels."
        self.in_channels = list(in_channels)
        self.out_channels = out_channels

        def make_conv(
            cin: int, cout: int, k: int = 3, s: int = 1, p: int | None = None
        ) -> nn.Module:
            if use_depthwise:
                if k == 1 or s != 1:
                    return ConvModule(cin, cout, k, s, p)
                return DepthwiseSeparableConv(cin, cout, k)
            return ConvModule(cin, cout, k, s, p)

        n = len(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down pathway
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(n - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    channel_attention=False,
                )
            )

        # Bottom-up pathway
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(n - 1):
            self.downsamples.append(make_conv(in_channels[idx], in_channels[idx], 3, 2))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    channel_attention=False,
                )
            )

        # Out projections
        self.out_convs = nn.ModuleList(
            [make_conv(in_channels[i], out_channels, 3) for i in range(n)]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels), (
            f"Expected {len(self.in_channels)} inputs, got {len(inputs)}"
        )
        n = len(inputs)
        inner = [inputs[-1]]

        # Top-down
        for idx in range(n - 1, 0, -1):
            j = n - 1 - idx
            feat_h = self.reduce_layers[j](inner[0])
            inner[0] = feat_h
            up = self.upsample(feat_h)
            inner.insert(0, self.top_down_blocks[j](torch.cat([up, inputs[idx - 1]], dim=1)))

        # Bottom-up
        outs = [inner[0]]
        for idx in range(n - 1):
            down = self.downsamples[idx](outs[-1])
            outs.append(self.bottom_up_blocks[idx](torch.cat([down, inner[idx + 1]], dim=1)))

        return tuple(self.out_convs[i](outs[i]) for i in range(n))
