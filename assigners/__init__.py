"""
Target assigners for LucidDet.
Exports dynamic_soft_label_assign and DynamicSoftLabelAssigner.

Universal Assigner Protocol
---------------------------
All assigners in this module follow a common callable protocol so they can be
interchanged via dependency injection (e.g. into losses.det_loss).

Inputs:
    pred_cls_logits: torch.Tensor (A, num_classes) — per-anchor classification logits.
    pred_boxes:      torch.Tensor (A, 4) — predicted boxes in absolute image-pixel xyxy format.
    centers:         torch.Tensor (A, 2) — anchor center coordinates in absolute pixel units.
    strides:         torch.Tensor (A, 1) — per-anchor stride values in pixel units.
    gt_boxes:        torch.Tensor (G, 4) — ground-truth boxes in absolute image-pixel xyxy format.
    gt_labels:       torch.Tensor (G,) — ground-truth class labels (0-indexed).
    num_classes:     int — total number of foreground classes.
    **kwargs:        additional implementation-specific arguments.

Returns:
    labels:        torch.Tensor (A,) — assigned label per anchor (background is num_classes).
    label_weights: torch.Tensor (A,) — weight per anchor for classification loss.
    bbox_targets:  torch.Tensor (A, 4) — target box per anchor in absolute image-pixel xyxy format (valid for positives).
    assign_metrics: torch.Tensor (A,) — assignment quality score (e.g. IoU, non-negative, 0 for background) per anchor.
"""

from typing import Any, Dict

from assigners.simota import DynamicSoftLabelAssigner, dynamic_soft_label_assign

ASSIGNER_MAP: Dict[str, Any] = {
    "simota": DynamicSoftLabelAssigner,
    "dynamic_soft_label_assigner": DynamicSoftLabelAssigner,
    "dynamic_soft_label_assign": dynamic_soft_label_assign,
}


def build_assigner(name: str = "simota", **kwargs: Any) -> Any:
    """
    Build an assigner by name from explicit ASSIGNER_MAP.

    Args:
        name: Name of the assigner ('simota', 'dynamic_soft_label_assigner').
        **kwargs: Arguments passed to the assigner constructor.

    Returns:
        Instantiated assigner module or function.
    """
    key = name.lower()
    if key not in ASSIGNER_MAP:
        raise ValueError(
            f"Unknown assigner '{name}'. Available assigners: {sorted(ASSIGNER_MAP.keys())}"
        )
    target = ASSIGNER_MAP[key]
    if isinstance(target, type):
        return target(**kwargs)
    return target


__all__ = [
    "DynamicSoftLabelAssigner",
    "dynamic_soft_label_assign",
    "ASSIGNER_MAP",
    "build_assigner",
]
