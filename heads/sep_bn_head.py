"""
SepBNHead: Decoupled anchor-free detection head with level-separated BatchNorm.
Faithful to RTMDet specification.
"""

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from backbones.common import ConvModule, DepthwiseSeparableConv
from heads import register_head
from models.outputs import DetectionOutput


def _share_submodule(src: nn.Module, dst: nn.Module) -> None:
    """Share convolution weights across levels while keeping independent BatchNorm statistics."""
    if isinstance(src, ConvModule) and isinstance(dst, ConvModule):
        dst.conv = src.conv
    elif isinstance(src, DepthwiseSeparableConv) and isinstance(dst, DepthwiseSeparableConv):
        dst.dw.conv = src.dw.conv
        dst.pw.conv = src.pw.conv
    else:
        raise TypeError(
            f"Unsupported module type for weight sharing: src={type(src)}, dst={type(dst)}"
        )


@register_head("sep_bn_head")
@register_head("sep_bn")
@register_head("rtmdet")
class SepBNHead(nn.Module):
    """
    Decoupled anchor-free detection head with level-separated BatchNorm.

    Args:
        num_classes: Number of object categories.
        in_channels: Input channels per level (single int or tuple/list matching strides).
        feat_channels: Intermediate convolutional feature channels.
        stacked_convs: Number of conv layers in each branch per level.
        strides: Feature level strides (default: (8, 16, 32)).
        share_conv: Whether to share convolution weights across levels.
        use_depthwise: Whether to use depthwise separable convolutions.
        with_objectness: Whether to predict an objectness branch.
        exp_on_reg: Whether to apply exp() to regression predictions (e.g. for scales s/m/l).
        pred_kernel_size: Kernel size of the final prediction 1x1 conv.
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: Union[int, Sequence[int]] = 96,
        feat_channels: int = 96,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        share_conv: bool = True,
        use_depthwise: bool = False,
        with_objectness: bool = False,
        exp_on_reg: bool = False,
        pred_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.strides = tuple(strides)
        self.share_conv = share_conv
        self.with_objectness = with_objectness
        self.exp_on_reg = exp_on_reg

        if isinstance(in_channels, int):
            self.in_channels: Tuple[int, ...] = (in_channels,) * len(self.strides)
        else:
            self.in_channels = tuple(in_channels)
            if len(self.in_channels) != len(self.strides):
                raise ValueError(
                    f"in_channels length ({len(self.in_channels)}) must match "
                    f"strides length ({len(self.strides)})"
                )

        if share_conv and len(set(self.in_channels)) > 1:
            raise ValueError(
                f"share_conv=True requires identical in_channels across all levels, "
                f"got {self.in_channels}."
            )

        def make_conv(cin: int, cout: int) -> nn.Module:
            if use_depthwise:
                return DepthwiseSeparableConv(cin, cout, 3)
            return ConvModule(cin, cout, 3)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_obj: Optional[nn.ModuleList] = nn.ModuleList() if with_objectness else None

        for lvl, stride in enumerate(self.strides):
            cls_branch, reg_branch = nn.ModuleList(), nn.ModuleList()
            for i in range(stacked_convs):
                cin = self.in_channels[lvl] if i == 0 else feat_channels
                cls_branch.append(make_conv(cin, feat_channels))
                reg_branch.append(make_conv(cin, feat_channels))
            self.cls_convs.append(cls_branch)
            self.reg_convs.append(reg_branch)

            self.rtm_cls.append(
                nn.Conv2d(
                    feat_channels,
                    num_classes,
                    pred_kernel_size,
                    padding=pred_kernel_size // 2,
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(
                    feat_channels,
                    4,
                    pred_kernel_size,
                    padding=pred_kernel_size // 2,
                )
            )
            if self.rtm_obj is not None:
                self.rtm_obj.append(
                    nn.Conv2d(
                        feat_channels,
                        1,
                        pred_kernel_size,
                        padding=pred_kernel_size // 2,
                    )
                )

        if share_conv:
            for lvl in range(1, len(self.strides)):
                for i in range(stacked_convs):
                    _share_submodule(self.cls_convs[0][i], self.cls_convs[lvl][i])
                    _share_submodule(self.reg_convs[0][i], self.reg_convs[lvl][i])

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: normal for convs, focal-loss prior bias for cls."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        bias_cls = -math.log((1 - 0.01) / 0.01)
        for c in self.rtm_cls:
            if isinstance(c, nn.Conv2d) and c.bias is not None:
                nn.init.constant_(c.bias, bias_cls)

    @staticmethod
    def _get_grid(
        h: int, w: int, stride: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor centers and stride tensor for one feature level."""
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        cx = (xs.reshape(-1) + 0.5) * stride
        cy = (ys.reshape(-1) + 0.5) * stride
        centers = torch.stack([cx, cy], dim=-1).to(dtype)
        strides = torch.full((h * w, 1), stride, device=device, dtype=dtype)
        return centers, strides

    def forward(self, feats: Sequence[torch.Tensor]) -> DetectionOutput:
        cls_list, box_list, ctr_list, str_list = [], [], [], []
        device = feats[0].device

        for lvl, (x, stride) in enumerate(zip(feats, self.strides)):
            cf, rf = x, x
            for layer in self.cls_convs[lvl]:
                cf = layer(cf)
            for layer in self.reg_convs[lvl]:
                rf = layer(rf)

            cls_score = self.rtm_cls[lvl](cf)  # B, num_classes, H, W
            reg = self.rtm_reg[lvl](rf)  # B, 4, H, W

            if self.with_objectness and self.rtm_obj is not None:
                obj = self.rtm_obj[lvl](rf)
                cls_score = cls_score + obj

            if self.exp_on_reg:
                reg_dist = reg.exp() * stride
            else:
                reg_dist = reg * stride

            b, _, h, w = cls_score.shape
            cls_flat = cls_score.permute(0, 2, 3, 1).reshape(b, h * w, self.num_classes)
            reg_flat = reg_dist.permute(0, 2, 3, 1).reshape(b, h * w, 4)

            centers, stride_tensor = self._get_grid(h, w, stride, device, reg_flat.dtype)

            left = reg_flat[..., 0]
            top = reg_flat[..., 1]
            right = reg_flat[..., 2]
            bottom = reg_flat[..., 3]

            x1 = centers[:, 0].unsqueeze(0) - left
            y1 = centers[:, 1].unsqueeze(0) - top
            x2 = centers[:, 0].unsqueeze(0) + right
            y2 = centers[:, 1].unsqueeze(0) + bottom
            decoded = torch.stack([x1, y1, x2, y2], dim=-1)

            cls_list.append(cls_flat)
            box_list.append(decoded)
            ctr_list.append(centers)
            str_list.append(stride_tensor)

        return DetectionOutput(
            pred_cls=torch.cat(cls_list, dim=1),
            pred_boxes=torch.cat(box_list, dim=1),
            anchor_centers=torch.cat(ctr_list, dim=0),
            anchor_strides=torch.cat(str_list, dim=0),
        )
