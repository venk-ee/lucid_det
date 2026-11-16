"""
Identity Neck.
Passthrough module for single-scale architectures like YOLOv1.
"""

from typing import Any, Sequence, Tuple, Union

import torch
import torch.nn as nn


class IdentityNeck(nn.Module):
    """
    Passthrough identity neck for single-scale detectors.
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]] = 1024,
        out_channels: Union[int, Sequence[int]] = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        return tuple(feats)
