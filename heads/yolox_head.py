"""
YOLOX Decoupled Detection Head (Pure PyTorch) for LucidDet.
Decouples classification, bounding box regression, and objectness branches.
Returns standard DetectionOutput contract.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from backbones.common import ConvModule
from heads import register_head
from models.outputs import DetectionOutput


@register_head("yolox_head")
@register_head("yolox")
class YOLOXHead(nn.Module):
    """
    Decoupled Head for YOLOX.

    Args:
        num_classes: Number of foreground classes.
        in_channels: Input channels per feature level (single int or tuple matching strides).
        feat_channels: Intermediate channel dimensions (default: 128 for 's', or 256).
        strides: Stride per level (default: (8, 16, 32)).
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: Union[int, Sequence[int]] = 128,
        feat_channels: int = 128,
        strides: Sequence[int] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(strides)
        self.feat_channels = feat_channels

        if isinstance(in_channels, int):
            self.in_channels: Tuple[int, ...] = (in_channels,) * len(self.strides)
        else:
            self.in_channels = tuple(in_channels)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for cin in self.in_channels:
            # 1x1 stem
            self.stems.append(ConvModule(cin, feat_channels, kernel_size=1, stride=1))

            # 2x 3x3 classification convs
            self.cls_convs.append(
                nn.Sequential(
                    ConvModule(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
                    ConvModule(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
                )
            )

            # 2x 3x3 regression convs
            self.reg_convs.append(
                nn.Sequential(
                    ConvModule(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
                    ConvModule(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
                )
            )

            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, kernel_size=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, kernel_size=1))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, kernel_size=1))

        self._init_weights()

    def _init_weights(self) -> None:
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Prior probability for objectness and class outputs
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        for cls_pred in self.cls_preds:
            if cls_pred.bias is not None:
                nn.init.constant_(cls_pred.bias, bias_init)
        for obj_pred in self.obj_preds:
            if obj_pred.bias is not None:
                nn.init.constant_(obj_pred.bias, bias_init)

    def forward(self, feats: Sequence[torch.Tensor]) -> DetectionOutput:
        cls_list, box_list, ctr_list, str_list = [], [], [], []
        device = feats[0].device

        for lvl, (x, stride) in enumerate(zip(feats, self.strides)):
            stem_feat = self.stems[lvl](x)
            cls_feat = self.cls_convs[lvl](stem_feat)
            reg_feat = self.reg_convs[lvl](stem_feat)

            cls_out = self.cls_preds[lvl](cls_feat)  # (B, num_classes, H, W)
            reg_out = self.reg_preds[lvl](reg_feat)  # (B, 4, H, W)
            obj_out = self.obj_preds[lvl](reg_feat)  # (B, 1, H, W)

            b, _, h, w = cls_out.shape

            cls_flat = cls_out.permute(0, 2, 3, 1).reshape(b, h * w, self.num_classes)
            reg_flat = reg_out.permute(0, 2, 3, 1).reshape(b, h * w, 4)
            obj_flat = obj_out.permute(0, 2, 3, 1).reshape(b, h * w, 1)

            # Combined class probability: cls * obj
            cls_prob = cls_flat.sigmoid() * obj_flat.sigmoid()
            # Logit representation so sigmoid() in NMS recovers cls_prob
            cls_logits = torch.logit(cls_prob.clamp(min=1e-7, max=1.0 - 1e-7))

            # Generate grid coordinates
            ys, xs = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            grid_x = xs.reshape(-1)
            grid_y = ys.reshape(-1)
            centers = torch.stack([(grid_x + 0.5) * stride, (grid_y + 0.5) * stride], dim=-1).float()
            stride_tensor = torch.full((h * w, 1), stride, device=device, dtype=torch.float32)

            # YOLOX box decoding:
            # tx, ty offsets relative to grid origin; tw, th are log scales
            tx = reg_flat[..., 0]
            ty = reg_flat[..., 1]
            tw = reg_flat[..., 2]
            th = reg_flat[..., 3]

            cx_box = (grid_x.unsqueeze(0) + tx) * stride
            cy_box = (grid_y.unsqueeze(0) + ty) * stride
            w_box = tw.exp() * stride
            h_box = th.exp() * stride

            x1 = cx_box - 0.5 * w_box
            y1 = cy_box - 0.5 * h_box
            x2 = cx_box + 0.5 * w_box
            y2 = cy_box + 0.5 * h_box
            decoded = torch.stack([x1, y1, x2, y2], dim=-1)

            cls_list.append(cls_logits)
            box_list.append(decoded)
            ctr_list.append(centers)
            str_list.append(stride_tensor)

        return DetectionOutput(
            pred_cls=torch.cat(cls_list, dim=1),
            pred_boxes=torch.cat(box_list, dim=1),
            anchor_centers=torch.cat(ctr_list, dim=0),
            anchor_strides=torch.cat(str_list, dim=0),
        )
