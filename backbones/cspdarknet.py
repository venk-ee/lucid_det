"""
CSPDarknet backbone (Pure PyTorch) for LucidDet.
Faithful to YOLOX architecture specifications with multi-scale variants (tiny, s, m, l, x).
"""

from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from backbones import register_backbone
from backbones.common import ConvModule, CSPLayer, SPPF


@register_backbone("cspdarknet")
class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone with Cross Stage Partial connections.

    Supported scale presets:
        'tiny': depth=0.33, width=0.375
        's'   : depth=0.33, width=0.50
        'm'   : depth=0.67, width=0.75
        'l'   : depth=1.00, width=1.00
        'x'   : depth=1.33, width=1.25
    """

    PRESETS: Dict[str, Tuple[float, float]] = {
        "tiny": (0.33, 0.375),
        "s": (0.33, 0.50),
        "m": (0.67, 0.75),
        "l": (1.00, 1.00),
        "x": (1.33, 1.25),
    }

    def __init__(
        self,
        arch: str = "s",
        depth_mult: Optional[float] = None,
        width_mult: Optional[float] = None,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
    ) -> None:
        super().__init__()
        if arch in self.PRESETS:
            preset_depth, preset_width = self.PRESETS[arch]
        else:
            preset_depth, preset_width = 0.33, 0.50

        depth = depth_mult if depth_mult is not None else preset_depth
        width = width_mult if width_mult is not None else preset_width

        self.arch = arch
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages

        base_channels = [64, 128, 256, 512, 1024]
        channels = [max(round(c * width), 16) for c in base_channels]
        num_blocks = [max(round(n * depth), 1) for n in [3, 9, 9, 3]]

        # Stem: 6x6 conv stride 2 (640 -> 320)
        self.stem = ConvModule(3, channels[0], kernel_size=6, stride=2, padding=2)

        # Stage 1 (Dark2): Stride 4 (320 -> 160)
        self.dark2 = nn.Sequential(
            ConvModule(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
            CSPLayer(
                channels[1],
                channels[1],
                num_blocks=num_blocks[0],
                add_identity=True,
                use_cspnext_block=False,
                channel_attention=False,
            ),
        )

        # Stage 2 (Dark3): Stride 8 (160 -> 80) -> P3
        self.dark3 = nn.Sequential(
            ConvModule(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
            CSPLayer(
                channels[2],
                channels[2],
                num_blocks=num_blocks[1],
                add_identity=True,
                use_cspnext_block=False,
                channel_attention=False,
            ),
        )

        # Stage 3 (Dark4): Stride 16 (80 -> 40) -> P4
        self.dark4 = nn.Sequential(
            ConvModule(channels[2], channels[3], kernel_size=3, stride=2, padding=1),
            CSPLayer(
                channels[3],
                channels[3],
                num_blocks=num_blocks[2],
                add_identity=True,
                use_cspnext_block=False,
                channel_attention=False,
            ),
        )

        # Stage 4 (Dark5): Stride 32 (40 -> 20) -> P5
        self.dark5 = nn.Sequential(
            ConvModule(channels[3], channels[4], kernel_size=3, stride=2, padding=1),
            SPPF(channels[4], channels[4]),
            CSPLayer(
                channels[4],
                channels[4],
                num_blocks=num_blocks[3],
                add_identity=False,
                use_cspnext_block=False,
                channel_attention=False,
            ),
        )

        self.stages = (self.stem, self.dark2, self.dark3, self.dark4, self.dark5)

        all_channels = [channels[0], channels[1], channels[2], channels[3], channels[4]]
        self.out_channels: Tuple[int, ...] = tuple(all_channels[i] for i in self.out_indices)
        self.strides: Tuple[int, ...] = tuple(2 ** (i + 1) for i in self.out_indices)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(min(self.frozen_stages + 1, len(self.stages))):
                stage = self.stages[i]
                stage.eval()
                for p in stage.parameters():
                    p.requires_grad = False

    def unfreeze(self) -> None:
        self.frozen_stages = -1
        for stage in self.stages:
            for p in stage.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> "CSPDarknet":
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
