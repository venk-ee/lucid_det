"""
Quality Focal Loss (Pure PyTorch) for LucidDet.
Computes elementwise QFL with soft IoU-weighted targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import register_cls_loss


@register_cls_loss("qfl")
@register_cls_loss("quality_focal_loss")
class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) from Generalized Focal Loss (GFL).

    Loss = |y - sigmoid(p)|^beta * BCE(p, y)

    Args:
        beta: Power scale factor for focal modulation (default: 2.0).
    """

    def __init__(self, beta: float = 2.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (..., num_classes) unnormalized logits.
            target_scores: (..., num_classes) continuous soft targets in [0, 1] (IoU for pos, 0 for neg).

        Returns:
            loss: (..., num_classes) elementwise loss tensor.
        """
        prob = pred_logits.sigmoid()
        scale = (target_scores - prob).abs().pow(self.beta)
        return scale * F.binary_cross_entropy_with_logits(
            pred_logits, target_scores, reduction="none"
        )
