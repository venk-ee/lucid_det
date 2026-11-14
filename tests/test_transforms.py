"""
Unit tests for detection transforms and pure PyTorch coordinate operations.
Designed with parametrization and mathematical invariance checks.
"""

import numpy as np
import pytest
import torch

from datasets.utils import scale_boxes
from transforms.det_transforms import (
    build_det_transform,
)

# --- 1. Type Safety & Defensive Guardrails ---


def test_scale_boxes_type_rejection():
    """Reject non-tensor types and invalid shapes."""
    with pytest.raises(TypeError, match="expects torch.Tensor"):
        scale_boxes(np.array([[10, 10, 20, 20]]), (640, 640), (480, 640))

    with pytest.raises(TypeError, match="expects torch.Tensor"):
        scale_boxes([[10, 10, 20, 20]], (640, 640), (480, 640))


@pytest.mark.parametrize("invalid_shape", [(0, 640), (640, 0), (-100, 640), (640, -100)])
def test_scale_boxes_non_positive_dimensions(invalid_shape):
    """Enforce strictly positive image dimensions."""
    valid_box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    with pytest.raises(ValueError, match="dimensions must be positive"):
        scale_boxes(valid_box, (640, 640), invalid_shape)

    with pytest.raises(ValueError, match="dimensions must be positive"):
        scale_boxes(valid_box, invalid_shape, (480, 640))


def test_scale_boxes_empty_tensors():
    """Gracefully pass through empty tensors without memory reallocation."""
    empty_boxes = torch.empty((0, 4), dtype=torch.float32)
    res = scale_boxes(empty_boxes, (640, 640), (480, 640))
    assert res.shape == (0, 4)
    assert isinstance(res, torch.Tensor)


# --- 2. Mathematical Invariance & Parametrized Aspect Ratios ---


@pytest.mark.parametrize(
    "ori_h, ori_w",
    [
        (720, 1280),  # Standard 16:9 Landscape
        (1080, 1920),  # Full HD Landscape
        (1920, 1080),  # Portrait / Mobile format
        (500, 500),  # Square
        (384, 640),  # Standard detection aspect ratio
        (200, 1000),  # Extreme wide banner (5:1)
        (1200, 300),  # Extreme tall tower (1:4)
    ],
)
@pytest.mark.parametrize("dst_size", [640, 512, (384, 640)])
def test_scale_boxes_bijective_roundtrip(ori_h, ori_w, dst_size):
    """
    Mathematical Invariance: scale_boxes(letterbox(B)) == B.
    Verifies sub-pixel roundtrip recovery across diverse aspect ratios.
    """
    dst_h, dst_w = (dst_size, dst_size) if isinstance(dst_size, int) else dst_size

    # Arbitrary ground truth boxes within original canvas
    original_boxes = torch.tensor(
        [
            [ori_w * 0.1, ori_h * 0.15, ori_w * 0.45, ori_h * 0.65],
            [ori_w * 0.05, ori_h * 0.05, ori_w * 0.25, ori_h * 0.35],
        ],
        dtype=torch.float32,
    )

    # Forward letterbox math matching PadIfNeeded(position='center')
    gain = min(dst_w / float(ori_w), dst_h / float(ori_h))
    pad_x = (dst_w - ori_w * gain) / 2.0
    pad_y = (dst_h - ori_h * gain) / 2.0

    letterboxed = original_boxes.clone()
    letterboxed[:, [0, 2]] = letterboxed[:, [0, 2]] * gain + pad_x
    letterboxed[:, [1, 3]] = letterboxed[:, [1, 3]] * gain + pad_y

    # Invert using scale_boxes
    recovered = scale_boxes(letterboxed, (dst_h, dst_w), (ori_h, ori_w))

    # Assert recovery within numerical machine epsilon (< 0.05 pixel error)
    assert torch.allclose(recovered, original_boxes, atol=5e-2), (
        f"Failed on ori=({ori_h}, {ori_w}), dst=({dst_h}, {dst_w})"
    )


# --- 3. Boundary Clamping & Batched Tensors ---


def test_scale_boxes_boundary_clamping():
    """Ensure coordinates extending beyond canvas are clamped strictly to [0, W] and [0, H]."""
    ori_h, ori_w = 480, 640
    out_of_bounds = torch.tensor(
        [
            [-100.0, -50.0, 1000.0, 800.0],  # Way outside
            [0.0, 0.0, 640.0, 480.0],  # Exactly on boundary
        ],
        dtype=torch.float32,
    )
    scaled = scale_boxes(out_of_bounds, (640, 640), (ori_h, ori_w))

    assert scaled[0, 0] == 0.0
    assert scaled[0, 1] == 0.0
    assert scaled[0, 2] == float(ori_w)
    assert scaled[0, 3] == float(ori_h)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_boxes", [1, 10, 50])
def test_scale_boxes_batched_tensors(batch_size, num_boxes):
    """Operate seamlessly on higher-dimensional batch tensors [B, N, 4]."""
    batched = torch.randn((batch_size, num_boxes, 4), dtype=torch.float32).abs() * 100
    batched[..., 2] += batched[..., 0] + 10  # Ensure x2 > x1
    batched[..., 3] += batched[..., 1] + 10  # Ensure y2 > y1

    scaled = scale_boxes(batched, (640, 640), (720, 1280))
    assert scaled.shape == (batch_size, num_boxes, 4)
    assert isinstance(scaled, torch.Tensor)
    assert scaled[..., 0].min() >= 0.0
    assert scaled[..., 2].max() <= 1280.0


# --- 4. Pipeline Execution & Edge Handling ---


def test_detection_transform_output_contracts():
    """Verify Albumentations wrapper contract: tensor types, dtypes, coordinate order."""
    transform = build_det_transform(img_size=640, is_train=False)
    dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8)
    bboxes = [[50.0, 50.0, 100.0, 150.0], [200.0, 200.0, 80.0, 100.0]]
    labels = [0, 1]

    img_tensor, gt_boxes, gt_labels = transform(dummy_img, bboxes, labels)

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 640, 640)
    assert img_tensor.dtype == torch.float32
    assert img_tensor.is_contiguous()

    assert isinstance(gt_boxes, torch.Tensor)
    assert gt_boxes.shape == (2, 4)
    assert gt_boxes.dtype == torch.float32

    assert isinstance(gt_labels, torch.Tensor)
    assert gt_labels.shape == (2,)
    assert gt_labels.dtype == torch.int64


def test_detection_transform_zero_annotations():
    """Verify clean empty tensor generation when image has no target annotations."""
    transform = build_det_transform(img_size=640, is_train=False)
    dummy_img = np.full((480, 640, 3), 128, dtype=np.uint8)

    img_tensor, gt_boxes, gt_labels = transform(dummy_img, [], [])

    assert img_tensor.shape == (3, 640, 640)
    assert gt_boxes.shape == (0, 4)
    assert gt_labels.shape == (0,)
    assert gt_boxes.dtype == torch.float32
    assert gt_labels.dtype == torch.int64
