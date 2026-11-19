"""
Unified detection loss dispatcher.
Integrates target assignment (SimOTA), classification loss (QFL),
and bounding box regression loss (GIoU/CIoU).
Supports direct dependency injection of loss modules or callable functions.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from assigners.simota import dynamic_soft_label_assign
from losses.iou_loss import CIoULoss, GIoULoss, IoULoss
from losses.qfl import QualityFocalLoss

CLS_LOSS_MAP: Dict[str, Type[nn.Module]] = {
    "qfl": QualityFocalLoss,
    "quality_focal_loss": QualityFocalLoss,
}

BOX_LOSS_MAP: Dict[str, Type[nn.Module]] = {
    "giou": GIoULoss,
    "giou_loss": GIoULoss,
    "ciou": CIoULoss,
    "ciou_loss": CIoULoss,
    "iou": IoULoss,
    "iou_loss": IoULoss,
}


_DEFAULT_CLS_INSTANCES: Dict[str, Callable] = {}
_DEFAULT_BOX_INSTANCES: Dict[str, Callable] = {}


def resolve_cls_loss(
    cls_loss: Optional[Union[str, Type[nn.Module], nn.Module, Callable]] = None,
) -> Callable:
    """Resolve classification loss into a callable module or function."""
    if cls_loss is None:
        cls_loss = "qfl"
    if isinstance(cls_loss, str):
        key = cls_loss.lower()
        if key not in CLS_LOSS_MAP:
            raise ValueError(
                f"Unknown classification loss '{cls_loss}'. Available: {sorted(CLS_LOSS_MAP.keys())}"
            )
        if key not in _DEFAULT_CLS_INSTANCES:
            _DEFAULT_CLS_INSTANCES[key] = CLS_LOSS_MAP[key]()
        return _DEFAULT_CLS_INSTANCES[key]
    if isinstance(cls_loss, type) and issubclass(cls_loss, nn.Module):
        return cls_loss()
    if callable(cls_loss):
        return cls_loss
    raise TypeError(
        f"Invalid cls_loss: expected str, nn.Module class/instance, or Callable, got {type(cls_loss)}"
    )


def resolve_box_loss(
    box_loss: Optional[Union[str, Type[nn.Module], nn.Module, Callable]] = None,
) -> Callable:
    """Resolve box regression loss into a callable module or function."""
    if box_loss is None:
        box_loss = "giou"
    if isinstance(box_loss, str):
        key = box_loss.lower()
        if key not in BOX_LOSS_MAP:
            raise ValueError(
                f"Unknown box loss '{box_loss}'. Available: {sorted(BOX_LOSS_MAP.keys())}"
            )
        if key not in _DEFAULT_BOX_INSTANCES:
            _DEFAULT_BOX_INSTANCES[key] = BOX_LOSS_MAP[key]()
        return _DEFAULT_BOX_INSTANCES[key]
    if isinstance(box_loss, type) and issubclass(box_loss, nn.Module):
        return box_loss()
    if callable(box_loss):
        return box_loss
    raise TypeError(
        f"Invalid box_loss: expected str, nn.Module class/instance, or Callable, got {type(box_loss)}"
    )


def resolve_assigner(
    assigner: Optional[Union[str, Callable]] = None,
) -> Callable:
    """Resolve target assigner into a callable function or module."""
    if assigner is None:
        return dynamic_soft_label_assign
    if isinstance(assigner, str):
        from assigners import build_assigner

        return build_assigner(assigner)
    if callable(assigner):
        return assigner
    raise TypeError(f"Invalid assigner: expected str, Callable, or None, got {type(assigner)}")


def det_loss(
    model_out: Any,
    gt_boxes_list: Sequence[torch.Tensor],
    gt_labels_list: Sequence[torch.Tensor],
    cls_loss: Union[str, Type[nn.Module], nn.Module, Callable] = "qfl",
    box_loss: Union[str, Type[nn.Module], nn.Module, Callable] = "giou",
    box_weight: float = 3.5,
    l1_weight: float = 0.25,
    assigner: Optional[Callable] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-task detection loss for anchor-free detectors.

    Args:
        model_out: Model output (dict or DetectionOutput) containing:
            'pred_cls': (B, A, num_classes)
            'pred_boxes': (B, A, 4)
            'anchor_centers': (A, 2)
            'anchor_strides': (A, 1) or (A,)
        gt_boxes_list: List of length B with (G_i, 4) ground-truth boxes in xyxy.
        gt_labels_list: List of length B with (G_i,) ground-truth class labels.
        cls_loss: Classification loss instance, class, callable, or name (default: 'qfl').
        box_loss: Box regression loss instance, class, callable, or name (default: 'giou').
        box_weight: Multiplier weight for bounding box loss.
        l1_weight: Multiplier weight for auxiliary smooth L1 box loss.
        assigner: Optional callable assigner following the universal assigner protocol.
            If None, falls back to dynamic_soft_label_assign.

    Returns:
        total_loss: Scalar loss tensor with attached computational graph.
        loss_dict: Dictionary containing component losses ('loss_cls', 'loss_box', 'avg_factor').
    """
    cls_loss_fn = resolve_cls_loss(cls_loss)
    box_loss_fn = resolve_box_loss(box_loss)
    _assigner = resolve_assigner(assigner)

    pred_cls = model_out["pred_cls"]
    pred_boxes = model_out["pred_boxes"]
    centers = model_out["anchor_centers"]
    strides = model_out["anchor_strides"]

    if strides.ndim == 1:
        strides = strides.unsqueeze(-1)

    batch_size, num_anchors, num_classes = pred_cls.shape
    device = pred_cls.device

    cls_parts: List[torch.Tensor] = []
    box_parts: List[torch.Tensor] = []
    avg_terms: List[torch.Tensor] = []

    for b in range(batch_size):
        gb = gt_boxes_list[b].to(device)
        gl = gt_labels_list[b].to(device)

        labels, label_weights, bbox_targets, assign_metrics = _assigner(
            pred_cls_logits=pred_cls[b].detach().float(),
            pred_boxes=pred_boxes[b].detach().float(),
            centers=centers.float(),
            strides=strides.float(),
            gt_boxes=gb,
            gt_labels=gl,
            num_classes=num_classes,
        )

        pc_f = pred_cls[b].float()
        pb_f = pred_boxes[b].float()

        target_scores = torch.zeros_like(pc_f)
        pos = labels < num_classes
        if pos.any():
            target_scores[pos, labels[pos]] = assign_metrics[pos]

        cls_parts.append(cls_loss_fn(pc_f, target_scores).sum(-1) * label_weights)
        avg_terms.append(assign_metrics.sum())

        if pos.any():
            iou = box_loss_fn(pb_f[pos], bbox_targets[pos])
            l1 = F.smooth_l1_loss(
                pb_f[pos] / strides[pos], bbox_targets[pos] / strides[pos], reduction="none"
            ).sum(-1)
            box_parts.append((iou + l1_weight * l1) * assign_metrics[pos])

    avg_factor = torch.stack(avg_terms).sum().clamp(min=1.0)
    loss_cls = torch.cat(cls_parts).sum() / avg_factor

    if box_parts:
        loss_box = torch.cat(box_parts).sum() / avg_factor
    else:
        loss_box = pred_boxes.float().sum() * 0.0

    total_loss = loss_cls + box_weight * loss_box
    return total_loss, {
        "loss_cls": loss_cls,
        "loss_box": loss_box,
        "avg_factor": avg_factor,
    }
