"""
Unit tests for LucidDet loss functions and dependency injection dispatcher.
"""

import pytest
import torch

from losses import (
    BOX_LOSS_MAP,
    CLS_LOSS_MAP,
    CIoULoss,
    GIoULoss,
    IoULoss,
    QualityFocalLoss,
    YOLOv1Loss,
    det_loss,
    resolve_box_loss,
    resolve_cls_loss,
)


def test_quality_focal_loss_forward() -> None:
    loss_fn = QualityFocalLoss(beta=2.0)
    pred_logits = torch.randn(4, 10, requires_grad=True)
    target_scores = torch.rand(4, 10)  # Continuous soft IoU targets in [0, 1]

    loss = loss_fn(pred_logits, target_scores)
    assert loss.shape == pred_logits.shape
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert pred_logits.grad is not None


@pytest.mark.parametrize("loss_cls", [GIoULoss, CIoULoss, IoULoss])
def test_iou_losses_properties(loss_cls) -> None:
    loss_fn = loss_cls()

    # 1. Identical boxes: loss should be 0.0
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 30.0, 80.0, 90.0]])
    loss_same = loss_fn(boxes, boxes)
    assert torch.allclose(loss_same, torch.zeros(2), atol=1e-5)

    # 2. Empty boxes edge case
    empty_boxes = torch.empty(0, 4)
    loss_empty = loss_fn(empty_boxes, empty_boxes)
    assert loss_empty.numel() == 0

    # 3. Disjoint boxes: loss should be > 0.0
    box_a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    box_b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    loss_disjoint = loss_fn(box_a, box_b)
    assert loss_disjoint.item() > 0.0


def test_resolve_cls_loss() -> None:
    assert "qfl" in CLS_LOSS_MAP
    assert CLS_LOSS_MAP["qfl"] is QualityFocalLoss

    # String lookup
    resolved = resolve_cls_loss("qfl")
    assert isinstance(resolved, QualityFocalLoss)

    # Class injection
    resolved_cls = resolve_cls_loss(QualityFocalLoss)
    assert isinstance(resolved_cls, QualityFocalLoss)

    # Instance injection
    custom_inst = QualityFocalLoss(beta=1.5)
    assert resolve_cls_loss(custom_inst) is custom_inst

    # Invalid string / type
    with pytest.raises(ValueError):
        resolve_cls_loss("unknown_cls_loss")
    with pytest.raises(TypeError):
        resolve_cls_loss(12345)  # type: ignore


def test_resolve_box_loss() -> None:
    assert "giou" in BOX_LOSS_MAP
    assert BOX_LOSS_MAP["giou"] is GIoULoss
    assert "ciou" in BOX_LOSS_MAP
    assert BOX_LOSS_MAP["ciou"] is CIoULoss
    assert "iou" in BOX_LOSS_MAP
    assert BOX_LOSS_MAP["iou"] is IoULoss

    # String lookup
    resolved = resolve_box_loss("giou")
    assert isinstance(resolved, GIoULoss)

    # Class injection
    resolved_cls = resolve_box_loss(CIoULoss)
    assert isinstance(resolved_cls, CIoULoss)

    # Instance injection
    custom_inst = IoULoss(eps=1e-5)
    assert resolve_box_loss(custom_inst) is custom_inst

    # Invalid string / type
    with pytest.raises(ValueError):
        resolve_box_loss("unknown_box_loss")
    with pytest.raises(TypeError):
        resolve_box_loss(12345)  # type: ignore


def test_det_loss_direct_dependency_injection() -> None:
    batch_size = 2
    num_anchors = 100
    num_classes = 80

    pred_cls = torch.randn(batch_size, num_anchors, num_classes, requires_grad=True)
    pred_x1y1 = torch.rand(batch_size, num_anchors, 2) * 500.0
    pred_wh = torch.rand(batch_size, num_anchors, 2) * 100.0 + 1.0
    pred_boxes = torch.cat([pred_x1y1, pred_x1y1 + pred_wh], dim=-1).requires_grad_(True)
    assert (pred_boxes[..., 2:] >= pred_boxes[..., :2]).all()
    centers = torch.rand(num_anchors, 2) * 640.0
    strides = torch.full((num_anchors, 1), 8.0)

    model_out = {
        "pred_cls": pred_cls,
        "pred_boxes": pred_boxes,
        "anchor_centers": centers,
        "anchor_strides": strides,
    }

    gt_boxes = [
        torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
        torch.tensor([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]]),
    ]
    gt_labels = [
        torch.tensor([0]),
        torch.tensor([1, 2]),
    ]

    # Test 1: Direct instance injection
    custom_qfl = QualityFocalLoss(beta=2.5)
    custom_ciou = CIoULoss()
    total_loss, loss_dict = det_loss(
        model_out=model_out,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
        cls_loss=custom_qfl,
        box_loss=custom_ciou,
    )
    assert torch.isfinite(total_loss)
    assert "loss_cls" in loss_dict and "loss_box" in loss_dict
    total_loss.backward()
    assert pred_cls.grad is not None
    assert pred_boxes.grad is not None

    # Test 2: Direct class injection
    pred_cls.grad = None
    pred_boxes.grad = None
    loss_cls_inj, _ = det_loss(
        model_out=model_out,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
        cls_loss=QualityFocalLoss,
        box_loss=GIoULoss,
    )
    assert torch.isfinite(loss_cls_inj)


def test_yolov1_loss_forward() -> None:
    loss_fn = YOLOv1Loss()
    pred = torch.randn(2, 7, 7, 30, requires_grad=True)
    target = torch.zeros(2, 7, 7, 30)
    # Put an object in cell (3, 3)
    target[0, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
    target[0, 3, 3, 10] = 1.0  # Class 0

    total_loss, loss_dict = loss_fn(pred, target)
    assert torch.isfinite(total_loss)
    assert "loss_coord" in loss_dict and "loss_obj" in loss_dict and "loss_class" in loss_dict
    total_loss.backward()
    assert pred.grad is not None


def test_det_loss_zero_ground_truth() -> None:
    B, A, C = 2, 50, 5
    pred_cls = torch.randn(B, A, C, requires_grad=True)
    pred_boxes = torch.randn(B, A, 4, requires_grad=True)
    centers = torch.randn(A, 2)
    strides = torch.ones(A, 1)

    model_out = {
        "pred_cls": pred_cls,
        "pred_boxes": pred_boxes,
        "anchor_centers": centers,
        "anchor_strides": strides,
    }

    gt_boxes = [torch.empty(0, 4), torch.empty(0, 4)]
    gt_labels = [torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)]

    total_loss, loss_dict = det_loss(
        model_out=model_out,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
    )
    assert torch.isfinite(total_loss)
    assert loss_dict["loss_box"].item() == 0.0
    total_loss.backward()
    assert pred_cls.grad is not None
    assert pred_boxes.grad is not None
