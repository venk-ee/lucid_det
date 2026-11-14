"""
Pure PyTorch tensor utilities for dataset coordinate transformations.
Zero-NumPy operations ensuring GPU device continuity and eliminating host-device synchronization.
"""

from typing import Tuple

import torch


def scale_boxes(
    boxes: torch.Tensor,
    img_shape: Tuple[int, int],
    ori_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Rescale bounding boxes from letterbox (dst_h, dst_w) back to original (ori_h, ori_w).

    Operates strictly on PyTorch tensors without converting to NumPy or moving across devices.
    Supports both unbatched (N, 4) and batched (..., 4) bounding box tensors.

    Args:
        boxes: Tensor of shape (..., 4) with coordinates (x1, y1, x2, y2).
        img_shape: Transformed image shape (dst_h, dst_w).
        ori_shape: Original image shape (ori_h, ori_w).

    Returns:
        torch.Tensor: Rescaled boxes clipped to original boundaries on the same device.
    """
    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f"scale_boxes expects torch.Tensor, got {type(boxes)}")

    if boxes.numel() == 0:
        return boxes

    dst_h, dst_w = img_shape
    ori_h, ori_w = ori_shape

    if ori_h <= 0 or ori_w <= 0:
        raise ValueError(f"ori_shape dimensions must be positive, got {ori_shape}")
    if dst_h <= 0 or dst_w <= 0:
        raise ValueError(f"img_shape dimensions must be positive, got {img_shape}")

    # Exact scaling gain
    gain = min(dst_w / float(ori_w), dst_h / float(ori_h))

    # Center padding offsets matching symmetric letterbox / PadIfNeeded
    pad_x = (dst_w - ori_w * gain) / 2.0
    pad_y = (dst_h - ori_h * gain) / 2.0

    scaled = boxes.clone()
    scaled[..., 0] = (scaled[..., 0] - pad_x) / gain
    scaled[..., 2] = (scaled[..., 2] - pad_x) / gain
    scaled[..., 1] = (scaled[..., 1] - pad_y) / gain
    scaled[..., 3] = (scaled[..., 3] - pad_y) / gain

    # Vectorized boundary clamping on device
    scaled[..., 0].clamp_(0.0, float(ori_w))
    scaled[..., 2].clamp_(0.0, float(ori_w))
    scaled[..., 1].clamp_(0.0, float(ori_h))
    scaled[..., 3].clamp_(0.0, float(ori_h))

    return scaled
