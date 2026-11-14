"""
Identity Neck (Pure PyTorch) for LucidDet.
Passthrough module for single-scale architectures like YOLOv1.
"""

from typing import Any, Sequence, Tuple
import torch
import torch.nn as nn

from necks import register_neck


@register_neck("identity")
class IdentityNeck(nn.Module):
    """
    Passthrough identity neck for single-scale detectors.
    """

    def __init__(self, in_channels: Any = 1024, out_channels: Any = 1024) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        return tuple(feats)
