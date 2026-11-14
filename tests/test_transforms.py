"""
Unit tests for transforms and pure PyTorch tensor utilities.
"""

import numpy as np
import pytest
import torch

from datasets.utils import scale_boxes
from transforms.det_transforms import (
    build_det_transform,
)


def test_scale_boxes_strict_tensor():
    # Must reject numpy array or list
    with pytest.raises(TypeError):
        scale_boxes(np.array([[10, 10, 20, 20]]), (640, 640), (480, 640))

    with pytest.raises(TypeError):
        scale_boxes([[10, 10, 20, 20]], (640, 640), (480, 640))

    # Reject non-positive dimensions
    valid_box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    with pytest.raises(ValueError):
        scale_boxes(valid_box, (640, 640), (0, 640))

    with pytest.raises(ValueError):
        scale_boxes(valid_box, (-640, 640), (480, 640))


def test_scale_boxes_empty():
    empty_boxes = torch.empty((0, 4), dtype=torch.float32)
    res = scale_boxes(empty_boxes, (640, 640), (480, 640))
    assert res.shape == (0, 4)
    assert isinstance(res, torch.Tensor)


def test_scale_boxes_roundtrip():
    # Original image: 720 (height) x 1280 (width)
    ori_h, ori_w = 720, 1280
    dst_h, dst_w = 640, 640

    # Test boxes in original coordinate space (xyxy)
    original_boxes = torch.tensor(
        [
            [100.0, 150.0, 450.0, 550.0],
            [50.0, 50.0, 200.0, 250.0],
        ],
        dtype=torch.float32,
    )

    # Calculate forward letterbox transformation matching PadIfNeeded
    gain = min(dst_w / float(ori_w), dst_h / float(ori_h))  # 640 / 1280 = 0.5
    pad_x = (dst_w - ori_w * gain) / 2.0  # (640 - 640) / 2 = 0
    pad_y = (dst_h - ori_h * gain) / 2.0  # (640 - 360) / 2 = 140

    letterboxed_boxes = original_boxes.clone()
    letterboxed_boxes[:, [0, 2]] = letterboxed_boxes[:, [0, 2]] * gain + pad_x
    letterboxed_boxes[:, [1, 3]] = letterboxed_boxes[:, [1, 3]] * gain + pad_y

    # Unletterbox using scale_boxes
    unscaled_boxes = scale_boxes(letterboxed_boxes, (dst_h, dst_w), (ori_h, ori_w))

    # Assert recovery within sub-pixel tolerance
    assert torch.allclose(unscaled_boxes, original_boxes, atol=1e-3)


def test_scale_boxes_clamping():
    ori_h, ori_w = 480, 640
    dst_h, dst_w = 640, 640

    # Box coordinates extending well beyond image boundaries
    out_of_bounds = torch.tensor([[-50.0, -100.0, 800.0, 900.0]], dtype=torch.float32)
    scaled = scale_boxes(out_of_bounds, (dst_h, dst_w), (ori_h, ori_w))

    assert scaled[0, 0] >= 0.0
    assert scaled[0, 1] >= 0.0
    assert scaled[0, 2] <= float(ori_w)
    assert scaled[0, 3] <= float(ori_h)


def test_scale_boxes_batched():
    # Shape: [Batch=2, Boxes=3, Coordinates=4]
    batched = torch.randn((2, 3, 4), dtype=torch.float32).abs() * 100
    # Make valid xyxy
    batched[..., 2] += batched[..., 0] + 10
    batched[..., 3] += batched[..., 1] + 10

    scaled = scale_boxes(batched, (640, 640), (480, 640))
    assert scaled.shape == (2, 3, 4)
    assert isinstance(scaled, torch.Tensor)


def test_detection_transform_execution():
    transform = build_det_transform(img_size=640, is_train=False)

    # Dummy image (H, W, 3)
    dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # COCO format [x, y, w, h]
    bboxes = [[50.0, 50.0, 100.0, 150.0], [200.0, 200.0, 80.0, 100.0]]
    labels = [0, 1]

    img_tensor, gt_boxes, gt_labels = transform(dummy_img, bboxes, labels)

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 640, 640)
    assert img_tensor.dtype == torch.float32

    assert isinstance(gt_boxes, torch.Tensor)
    assert gt_boxes.shape == (2, 4)
    assert gt_boxes.dtype == torch.float32

    assert isinstance(gt_labels, torch.Tensor)
    assert gt_labels.shape == (2,)
    assert gt_labels.dtype == torch.int64


def test_detection_transform_zero_boxes():
    transform = build_det_transform(img_size=640, is_train=False)
    dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8)

    img_tensor, gt_boxes, gt_labels = transform(dummy_img, [], [])

    assert img_tensor.shape == (3, 640, 640)
    assert gt_boxes.shape == (0, 4)
    assert gt_labels.shape == (0,)
    assert gt_boxes.dtype == torch.float32
    assert gt_labels.dtype == torch.int64
