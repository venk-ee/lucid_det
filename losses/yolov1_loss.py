"""
YOLOv1 Multi-Task SSE Loss.
Computes coordinate regression, confidence, and classification loss over SxS grid cells.
"""

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn


def _box_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Intersection over union for boxes in [x, y, w, h] format."""
    b1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
    b1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
    b1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
    b1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2

    b2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
    b2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
    b2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
    b2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (b1_x2 - b1_x1).abs() * (b1_y2 - b1_y1).abs()
    box2_area = (b2_x2 - b2_x1).abs() * (b2_y2 - b2_y1).abs()

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YOLOv1Loss(nn.Module):
    """
    YOLOv1 Multi-Task Sum-Squared Error (SSE) Loss.

    Args:
        split_size: Grid division S (default: 7).
        num_boxes: Bounding boxes per grid cell B (default: 2).
        num_classes: Object categories C (default: 20).
        lambda_coord: Coordinate regression weight (default: 5.0).
        lambda_noobj: Background no-object confidence penalty weight (default: 0.5).
    """

    def __init__(
        self,
        split_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(
        self,
        predictions: Union[torch.Tensor, Any],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hasattr(predictions, "raw_preds"):
            preds = predictions.raw_preds
        elif isinstance(predictions, dict) and "raw_preds" in predictions:
            preds = predictions["raw_preds"]
        else:
            preds = predictions

        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)
        batch_size = preds.shape[0]

        # Calculate IoU for each predicted box with target box
        # target structure: [C classes, 1 objectness, 4 coords (x,y,w,h)]
        # predictions structure: [C classes, box1_conf (1), box1_coords (4), box2_conf (1), box2_coords (4)]
        iou_b1 = _box_iou_xywh(
            preds[..., self.C + 1 : self.C + 5], target[..., self.C + 1 : self.C + 5]
        )
        iou_b2 = _box_iou_xywh(
            preds[..., self.C + 6 : self.C + 10], target[..., self.C + 1 : self.C + 5]
        )

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, best_box = torch.max(ious, dim=0)
        exists_box = target[..., self.C : self.C + 1]

        # Coordinate Loss
        box_preds = exists_box * (
            best_box * preds[..., self.C + 6 : self.C + 10]
            + (1 - best_box) * preds[..., self.C + 1 : self.C + 5]
        )
        box_target = exists_box * target[..., self.C + 1 : self.C + 5]

        # Sqrt on width and height
        box_preds_scaled = box_preds.clone()
        box_preds_scaled[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(
            torch.abs(box_preds[..., 2:4]) + 1e-6
        )
        box_target_scaled = box_target.clone()
        box_target_scaled[..., 2:4] = torch.sqrt(torch.abs(box_target[..., 2:4]) + 1e-6)

        coord_loss = self.mse(
            torch.flatten(box_preds_scaled, end_dim=-2),
            torch.flatten(box_target_scaled, end_dim=-2),
        )

        # Object Confidence Loss
        pred_conf = (
            best_box * preds[..., self.C + 5 : self.C + 6]
            + (1 - best_box) * preds[..., self.C : self.C + 1]
        )
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_conf),
            torch.flatten(exists_box * target[..., self.C : self.C + 1]),
        )

        # No Object Confidence Loss
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * preds[..., self.C : self.C + 1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1], start_dim=1),
        ) + self.mse(
            torch.flatten((1 - exists_box) * preds[..., self.C + 5 : self.C + 6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1], start_dim=1),
        )

        # Class Probability Loss
        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., : self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., : self.C], end_dim=-2),
        )

        total_loss = (
            self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * no_obj_loss + class_loss
        ) / max(batch_size, 1)

        return total_loss, {
            "loss_coord": coord_loss / max(batch_size, 1),
            "loss_obj": obj_loss / max(batch_size, 1),
            "loss_noobj": no_obj_loss / max(batch_size, 1),
            "loss_class": class_loss / max(batch_size, 1),
        }
