"""
CSPNeXt backbone.
Faithful to RTMDet architecture specification.
Contract: forward(x) -> List[Tensor] ([P3, P4, P5]), exposes .out_channels and .strides.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from backbones import register_backbone
from backbones.common import SPPF, ConvModule, CSPLayer, SPPBottleneck


@register_backbone("cspnext")
class CSPNeXt(nn.Module):
    """
    CSPNeXt: CNN backbone designed for real-time object detection (RTMDet).
    """

    ARCH_P5 = [
        (64, 128, 3, True, False),
        (128, 256, 6, True, False),
        (256, 512, 6, True, False),
        (512, 1024, 3, False, True),
    ]

    PRESETS = {
        "tiny": (0.167, 0.375, False),
        "s": (0.333, 0.5, False),
        "m": (0.667, 0.75, False),
        "l": (1.0, 1.0, False),
        "tiny-dw": (0.167, 0.375, True),
    }

    def __init__(
        self,
        arch: Optional[str] = "tiny",
        deepen_factor: Optional[float] = None,
        widen_factor: Optional[float] = None,
        use_depthwise: Optional[bool] = None,
        out_indices: Sequence[int] = (2, 3, 4),
        channel_attention: bool = True,
        use_sppf: bool = True,
        frozen_stages: int = -1,
    ) -> None:
        super().__init__()
        if arch is not None:
            if arch not in self.PRESETS:
                raise ValueError(
                    f"Unknown arch '{arch}', choose from {sorted(list(self.PRESETS.keys()))}"
                )
            preset_deepen, preset_widen, preset_dw = self.PRESETS[arch]
        else:
            preset_deepen, preset_widen, preset_dw = 0.333, 0.5, False

        deepen = deepen_factor if deepen_factor is not None else preset_deepen
        widen = widen_factor if widen_factor is not None else preset_widen
        use_dw = use_depthwise if use_depthwise is not None else preset_dw

        self.arch = arch
        self.deepen = deepen
        self.widen = widen
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.use_dw = use_dw

        c0 = int(self.ARCH_P5[0][0] * widen // 2)
        c1 = int(self.ARCH_P5[0][0] * widen)
        self.stem = nn.Sequential(
            ConvModule(3, c0, 3, stride=2),
            ConvModule(c0, c0, 3, stride=1),
            ConvModule(c0, c1, 3, stride=1),
        )

        stages = []
        for i, (cin0, cout0, n0, add_id, use_spp) in enumerate(self.ARCH_P5):
            cin = int(cin0 * widen)
            cout = int(cout0 * widen)
            nblocks = max(round(n0 * deepen), 1)
            stage: List[nn.Module] = [ConvModule(cin, cout, 3, stride=2)]
            if use_spp:
                stage.append(SPPF(cout, cout) if use_sppf else SPPBottleneck(cout, cout))
            stage.append(
                CSPLayer(
                    cout,
                    cout,
                    num_blocks=nblocks,
                    add_identity=add_id,
                    use_depthwise=use_dw,
                    use_cspnext_block=True,
                    expand_ratio=0.5,
                    channel_attention=channel_attention,
                )
            )
            stages.append(nn.Sequential(*stage))

        self.stage1, self.stage2, self.stage3, self.stage4 = stages
        self.layers = (self.stem, self.stage1, self.stage2, self.stage3, self.stage4)

        all_ch = [c1] + [int(c[1] * widen) for c in self.ARCH_P5]
        self.out_channels: Tuple[int, ...] = tuple(all_ch[i] for i in self.out_indices)
        self.strides: Tuple[int, ...] = tuple(2 ** (i + 1) for i in self.out_indices)
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Ensure stages up to frozen_stages stay in eval mode with requires_grad=False."""
        if self.frozen_stages >= 0:
            for i in range(min(self.frozen_stages + 1, len(self.layers))):
                layer = self.layers[i]
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone stages for full end-to-end training."""
        self.frozen_stages = -1
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True) -> "CSPNeXt":
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
