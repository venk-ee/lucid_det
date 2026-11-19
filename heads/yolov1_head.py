"""
YOLOv1 Fully Connected Detection Head.
Decodes S x S grid cell predictions with B bounding boxes and C classes into DetectionOutput.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from core.outputs import DetectionOutput


class YOLOv1Head(nn.Module):
    """
    YOLOv1 Fully Connected Detection Head.

    Args:
        in_channels: Feature channels from backbone/neck (default: 1024).
        num_classes: Number of categories (default: 20 for VOC).
        split_size: Grid division S (default: 7).
        num_boxes: Number of bounding boxes predicted per grid cell (default: 2).
        hidden_dim: Intermediate linear layer units (default: 496).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        num_classes: int = 20,
        split_size: int = 7,
        num_boxes: int = 2,
        hidden_dim: int = 496,
        dropout: float = 0.5,
        img_size: float = 640.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.S = split_size
        self.B = num_boxes
        self.hidden_dim = hidden_dim
        self.img_size = float(img_size)
        self.out_dim = self.B * 5 + self.num_classes

        self.pool = nn.AdaptiveAvgPool2d((self.S, self.S))
        self.flatten = nn.Flatten(start_dim=1)
        self.fcs = nn.Sequential(
            nn.Linear(self.in_channels * self.S * self.S, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, self.S * self.S * self.out_dim),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> DetectionOutput:
        feat = feats[0] if isinstance(feats, (list, tuple)) else feats
        batch_size = feat.shape[0]
        device = feat.device

        pooled = self.pool(feat)
        flattened = self.flatten(pooled)
        raw = self.fcs(flattened)  # (B, S * S * out_dim)

        grid_preds = raw.view(batch_size, self.S, self.S, self.out_dim)

        # Decode for DetectionOutput compatibility (NMS and metrics)
        cell_size = self.img_size / self.S

        # Extract class probabilities and bounding box predictions: [classes (C), box1 (5), box2 (5)]
        cls_probs = grid_preds[..., : self.num_classes].softmax(dim=-1)

        boxes_list: List[torch.Tensor] = []
        cls_list: List[torch.Tensor] = []

        ys, xs = torch.meshgrid(
            torch.arange(self.S, device=device),
            torch.arange(self.S, device=device),
            indexing="ij",
        )
        grid_y = ys.unsqueeze(0).expand(batch_size, -1, -1)
        grid_x = xs.unsqueeze(0).expand(batch_size, -1, -1)

        for b in range(self.B):
            box_offset = self.num_classes + b * 5
            conf = grid_preds[..., box_offset : box_offset + 1].sigmoid()
            box_coords = grid_preds[..., box_offset + 1 : box_offset + 5]

            # In YOLOv1: x, y are offsets in cell [0, 1], w, h are cell-scaled units
            bx = (grid_x.float() + box_coords[..., 0].sigmoid()) * cell_size
            by = (grid_y.float() + box_coords[..., 1].sigmoid()) * cell_size
            bw = box_coords[..., 2].abs() * cell_size
            bh = box_coords[..., 3].abs() * cell_size

            x1 = bx - 0.5 * bw
            y1 = by - 0.5 * bh
            x2 = bx + 0.5 * bw
            y2 = by + 0.5 * bh
            xyxy = torch.stack([x1, y1, x2, y2], dim=-1).view(batch_size, self.S * self.S, 4)

            # Combined score: confidence * class probability
            combined_scores = (cls_probs * conf).view(batch_size, self.S * self.S, self.num_classes)
            # Logit representation so sigmoid() in NMS recovers score
            combined_logits = torch.logit(combined_scores.clamp(min=1e-7, max=1.0 - 1e-7))

            boxes_list.append(xyxy)
            cls_list.append(combined_logits)

        # Static anchor centers per prior box (independent of batch size)
        # Prior centers for each cell are at (col + 0.5) * cell_size, (row + 0.5) * cell_size
        cell_centers_x = (xs.float() + 0.5) * cell_size
        cell_centers_y = (ys.float() + 0.5) * cell_size
        single_box_centers = torch.stack(
            [cell_centers_x.reshape(-1), cell_centers_y.reshape(-1)], dim=-1
        )
        anchor_centers = single_box_centers.repeat(self.B, 1)

        pred_boxes = torch.cat(boxes_list, dim=1)
        pred_cls = torch.cat(cls_list, dim=1)
        anchor_strides = torch.full(
            (pred_boxes.shape[1], 1), cell_size, device=device, dtype=torch.float32
        )

        out = DetectionOutput(
            pred_cls=pred_cls,
            pred_boxes=pred_boxes,
            anchor_centers=anchor_centers,
            anchor_strides=anchor_strides,
        )
        setattr(out, "raw_preds", grid_preds)
        return out
