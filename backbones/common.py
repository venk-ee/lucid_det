"""
Common neural network layers and building blocks backbones and necks.
Faithful to CSPNeXt / RTMDet architecture specifications.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """
    Standard Convolution + BatchNorm2d + SiLU activation.
    Exposes .conv and .bn for easy level weight sharing in detection heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise convolution followed by pointwise (1x1) convolution,
    each equipped with BatchNorm and SiLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.dw = ConvModule(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pw = ConvModule(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ChannelAttention(nn.Module):
    """
    Channel Attention Module: GAP -> 1x1 Conv -> Hardsigmoid.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class CSPNeXtBlock(nn.Module):
    """
    Official CSPNeXt Block:
        conv1: 3x3 (depthwise-separable if use_depthwise)
        conv2: 5x5 depthwise-separable
        residual connection if in_channels == out_channels and add_identity=True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(in_channels, hidden, 3)
        else:
            self.conv1 = ConvModule(in_channels, hidden, 3)
        self.conv2 = DepthwiseSeparableConv(hidden, out_channels, kernel_size)
        self.add_identity = add_identity and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class DarknetBottleneck(nn.Module):
    """Fallback Darknet bottleneck block (1x1 -> 3x3)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden, 1)
        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(hidden, out_channels, 3)
        else:
            self.conv2 = ConvModule(hidden, out_channels, 3)
        self.add_identity = add_identity and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """
    Cross Stage Partial layer:
        main_conv (1x1) -> blocks -> concat with short_conv (1x1) -> attention -> final (1x1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = True,
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
    ) -> None:
        super().__init__()
        block_cls = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid, 1)
        self.short_conv = ConvModule(in_channels, mid, 1)
        self.blocks = nn.Sequential(
            *[
                block_cls(mid, mid, 1.0, add_identity=add_identity, use_depthwise=use_depthwise)
                for _ in range(num_blocks)
            ]
        )
        self.final_conv = ConvModule(2 * mid, out_channels, 1)
        self.attention = ChannelAttention(2 * mid) if channel_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        x_final = torch.cat([x_main, x_short], dim=1)
        return self.final_conv(self.attention(x_final))


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = ConvModule(in_channels, hidden, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes])
        self.conv2 = ConvModule(hidden * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.pools], dim=1))


class SPPF(nn.Module):
    """Fast Spatial Pyramid Pooling (SPPF) layer (receptive field equivalent to SPP)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = ConvModule(in_channels, hidden, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.conv2 = ConvModule(hidden * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
