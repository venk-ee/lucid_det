"""
Unit tests for target assigners: SimOTA and DynamicSoftLabelAssigner.
Verifies assignment metrics, tensor shapes, and ASSIGNER_MAP lookup.
"""

import pytest
import torch

from assigners import (
    ASSIGNER_MAP,
    DynamicSoftLabelAssigner,
    build_assigner,
    dynamic_soft_label_assign,
)


def test_dynamic_soft_label_assign_forward() -> None:
    num_anchors = 100
    num_classes = 80

    pred_cls = torch.randn(num_anchors, num_classes)
    pred_boxes = torch.rand(num_anchors, 4) * 640.0
    centers = torch.rand(num_anchors, 2) * 640.0
    strides = torch.full((num_anchors, 1), 8.0)

    gt_boxes = torch.tensor(
        [
            [50.0, 50.0, 150.0, 150.0],
            [200.0, 200.0, 300.0, 300.0],
            [400.0, 400.0, 500.0, 500.0],
        ]
    )
    gt_labels = torch.tensor([0, 1, 2])

    labels, label_weights, bbox_targets, assign_metrics = dynamic_soft_label_assign(
        pred_cls_logits=pred_cls,
        pred_boxes=pred_boxes,
        centers=centers,
        strides=strides,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        num_classes=num_classes,
    )

    assert labels.shape == (num_anchors,)
    assert label_weights.shape == (num_anchors,)
    assert bbox_targets.shape == (num_anchors, 4)
    assert assign_metrics.shape == (num_anchors,)


def test_dynamic_soft_label_assigner_module() -> None:
    assigner = DynamicSoftLabelAssigner()
    num_anchors = 50
    num_classes = 20

    pred_cls = torch.randn(num_anchors, num_classes)
    pred_boxes = torch.rand(num_anchors, 4) * 320.0
    centers = torch.rand(num_anchors, 2) * 320.0
    strides = torch.full((num_anchors, 1), 8.0)

    gt_boxes = torch.tensor([[20.0, 20.0, 80.0, 80.0]])
    gt_labels = torch.tensor([5])

    labels, label_weights, bbox_targets, assign_metrics = assigner(
        pred_cls_logits=pred_cls,
        pred_boxes=pred_boxes,
        centers=centers,
        strides=strides,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        num_classes=num_classes,
    )
    assert labels.shape == (num_anchors,)


def test_build_assigner_map() -> None:
    assert "simota" in ASSIGNER_MAP
    assert ASSIGNER_MAP["simota"] is DynamicSoftLabelAssigner
    assert "dynamic_soft_label_assign" in ASSIGNER_MAP
    assert ASSIGNER_MAP["dynamic_soft_label_assign"] is dynamic_soft_label_assign

    assigner_fn = build_assigner("dynamic_soft_label_assign")
    assert assigner_fn is dynamic_soft_label_assign

    assigner_mod = build_assigner("simota")
    assert isinstance(assigner_mod, DynamicSoftLabelAssigner)

    with pytest.raises(ValueError, match="Unknown assigner 'invalid_assigner'"):
        build_assigner("invalid_assigner")
