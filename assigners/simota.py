"""
SimOTA / Dynamic Soft Label Assigner (Pure PyTorch) for LucidDet.
Matches ground-truth annotations to anchor priors per image.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from assigners import register_assigner
from core.iou import box_iou

EPS = 1e-7


def dynamic_soft_label_assign(
    pred_cls_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    centers: torch.Tensor,
    strides: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    soft_center_radius: float = 1.8,
    topk: int = 5,
    iou_weight: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SimOTA dynamic target assignment with soft label costs.

    Args:
        pred_cls_logits: (A, num_classes) raw class logits.
        pred_boxes: (A, 4) decoded boxes in (x1, y1, x2, y2).
        centers: (A, 2) anchor center points.
        strides: (A, 1) or (A,) anchor strides.
        gt_boxes: (G, 4) ground truth boxes in (x1, y1, x2, y2).
        gt_labels: (G,) ground truth integer class labels.
        num_classes: Number of foreground classes.
        soft_center_radius: Soft center radius multiplier.
        topk: Number of top IoU priors considered for dynamic k calculation.
        iou_weight: Regression cost weight factor.

    Returns:
        labels: (A,) matched class labels (background = num_classes).
        label_weights: (A,) loss weights.
        bbox_targets: (A, 4) matched target boxes in (x1, y1, x2, y2).
        assign_metrics: (A,) IoU alignment metric for positives, 0 for negatives.
    """
    A = pred_cls_logits.shape[0]
    G = gt_boxes.shape[0]
    device = pred_cls_logits.device
    dtype = pred_cls_logits.dtype

    gt_boxes = gt_boxes.to(device=device, dtype=dtype)
    gt_labels = gt_labels.to(device=device, dtype=torch.long)
    if strides.ndim == 1:
        strides = strides.unsqueeze(-1)

    bg = num_classes
    labels = torch.full((A,), bg, dtype=torch.long, device=device)
    label_weights = torch.zeros(A, dtype=dtype, device=device)
    bbox_targets = torch.zeros(A, 4, dtype=dtype, device=device)
    assign_metrics = torch.zeros(A, dtype=dtype, device=device)

    if G == 0 or A == 0:
        label_weights[:] = 1.0
        return labels, label_weights, bbox_targets, assign_metrics

    gt_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0  # (G, 2)
    dist = (centers[:, None, :] - gt_center[None, :, :]).norm(dim=-1) / strides

    lt = centers[:, None, :] - gt_boxes[None, :, :2]  # (A, G, 2)
    rb = gt_boxes[None, :, 2:] - centers[:, None, :]
    deltas = torch.cat([lt, rb], dim=-1)  # (A, G, 4)
    is_in = deltas.min(-1).values > 0  # (A, G)
    is_candidate = is_in | (dist < soft_center_radius)
    valid = is_candidate.sum(1) > 0

    if not valid.any():
        label_weights[:] = 1.0
        return labels, label_weights, bbox_targets, assign_metrics

    v_idx = torch.where(valid)[0]
    V = v_idx.numel()
    v_boxes = pred_boxes[valid]
    v_logits = pred_cls_logits[valid]
    v_centers = centers[valid]
    v_strides = strides[valid].squeeze(-1)

    pairwise_ious = box_iou(v_boxes, gt_boxes).clamp(min=EPS)  # (V, G)

    dist = (v_centers[:, None, :] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / v_strides[:, None]
    soft_prior = torch.pow(10.0, (dist - soft_center_radius).clamp(max=8.0))

    iou_cost = -torch.log(pairwise_ious + EPS) * iou_weight

    onehot = F.one_hot(gt_labels.long(), num_classes).float()  # (G, C)
    soft_label = onehot.unsqueeze(0) * pairwise_ious.unsqueeze(-1)  # (V, G, C)
    v_prob = v_logits.unsqueeze(1).repeat(1, G, 1)
    scale = (soft_label - v_prob.sigmoid()).abs().pow(2.0)
    bce = F.binary_cross_entropy_with_logits(v_prob, soft_label, reduction="none")
    cls_cost = (scale * bce).sum(-1)

    cost = cls_cost + iou_cost + soft_prior  # (V, G)

    cand_topk = min(topk, V)
    topk_ious, _ = torch.topk(pairwise_ious, cand_topk, dim=0)
    dyn_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    matching = torch.zeros_like(cost, dtype=torch.uint8)

    for g in range(G):
        _, pos = torch.topk(cost[:, g], k=int(dyn_ks[g]), largest=False)
        matching[pos, g] = 1

    multi = matching.sum(1) > 1
    if multi.sum() > 0:
        _, argmin = torch.min(cost[multi], dim=1)
        matching[multi] *= 0
        matching[multi, argmin] = 1

    fg = matching.sum(1) > 0  # V
    matched_gt = matching[fg].argmax(1)
    matched_iou = (matching.float() * pairwise_ious).sum(1)[fg]
    orig_pos = v_idx[torch.where(fg)[0]]

    labels[orig_pos] = gt_labels[matched_gt]
    label_weights[orig_pos] = 1.0
    neg = ~valid
    label_weights[neg] = 1.0
    bbox_targets[orig_pos] = gt_boxes[matched_gt]
    assign_metrics[orig_pos] = matched_iou

    return labels, label_weights, bbox_targets, assign_metrics


@register_assigner("simota")
@register_assigner("dynamic_soft_label_assigner")
class DynamicSoftLabelAssigner(nn.Module):
    """
    Module wrapper for SimOTA / Dynamic Soft Label Assignment.
    """

    def __init__(
        self,
        soft_center_radius: float = 1.8,
        topk: int = 5,
        iou_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    def forward(
        self,
        pred_cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        centers: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return dynamic_soft_label_assign(
            pred_cls_logits=pred_cls_logits,
            pred_boxes=pred_boxes,
            centers=centers,
            strides=strides,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=num_classes,
            soft_center_radius=self.soft_center_radius,
            topk=self.topk,
            iou_weight=self.iou_weight,
        )
