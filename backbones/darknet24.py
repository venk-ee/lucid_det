"""
Darknet-24 backbone.
Faithful to YOLOv1 architecture specification (24 conv layers with LeakyReLU).
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from backbones import register_backbone


class CNNBlock(nn.Module):
    """Conv2d + BatchNorm2d + LeakyReLU(0.1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@register_backbone("darknet24")
class Darknet24(nn.Module):
    """
    YOLOv1 Darknet-24 backbone.
    Outputs single-scale P5 feature map (1024 channels).
    """

    ARCHITECTURE = [
        (7, 64, 2, 3),
        "M",
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
    ]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels: Tuple[int, ...] = (1024,)
        self.strides: Tuple[int, ...] = (64,)

        layers: List[nn.Module] = []
        curr_in = in_channels

        for item in self.ARCHITECTURE:
            if isinstance(item, tuple):
                k, out_c, s, p = item
                layers.append(CNNBlock(curr_in, out_c, kernel_size=k, stride=s, padding=p))
                curr_in = out_c
            elif item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(item, list):
                conv1, conv2, repeats = item
                for _ in range(repeats):
                    layers.append(
                        CNNBlock(
                            curr_in,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    )
                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    )
                    curr_in = conv2[1]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feat = self.net(x)
        return [feat]
