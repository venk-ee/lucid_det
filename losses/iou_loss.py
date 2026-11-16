"""
Bounding box IoU regression losses.
Houses GIoULoss, CIoULoss, and standard IoULoss.
"""

import torch
import torch.nn as nn

from core.iou import box_ciou, box_giou


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss: 1.0 - GIoU.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) in (x1, y1, x2, y2).
            target_boxes: (N, 4) in (x1, y1, x2, y2).

        Returns:
            loss: (N,) elementwise loss.
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum(dim=-1) * 0.0
        return 1.0 - box_giou(pred_boxes, target_boxes, elementwise=True, eps=self.eps)


class CIoULoss(nn.Module):
    """
    Complete IoU Loss: 1.0 - CIoU.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) in (x1, y1, x2, y2).
            target_boxes: (N, 4) in (x1, y1, x2, y2).

        Returns:
            loss: (N,) elementwise loss.
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum(dim=-1) * 0.0
        return 1.0 - box_ciou(pred_boxes, target_boxes, eps=self.eps)


class IoULoss(nn.Module):
    """
    Standard IoU Loss: 1.0 - IoU.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) in (x1, y1, x2, y2).
            target_boxes: (N, 4) in (x1, y1, x2, y2).

        Returns:
            loss: (N,) elementwise loss.
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum(dim=-1) * 0.0

        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (
            pred_boxes[:, 3] - pred_boxes[:, 1]
        ).clamp(min=0)
        area2 = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        ).clamp(min=0)

        union = (area1 + area2 - inter).clamp(min=self.eps)
        iou = inter / union
        return 1.0 - iou
