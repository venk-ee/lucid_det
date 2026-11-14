"""
Box coordinate transformations and anchor-point prior generation.
"""

from typing import Optional, Sequence, Tuple, Union

import torch


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def box_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) to (x, y, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def generate_points_priors(
    featmap_sizes: Sequence[Tuple[int, int]],
    strides: Sequence[int],
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor center points for multiple feature map levels.

    Args:
        featmap_sizes: List or sequence of (H, W) shapes for each level.
        strides: Stride for each level (e.g. [8, 16, 32]).
        device: Device on which tensors will be allocated.
        dtype: Data type for coordinates.

    Returns:
        priors: Tensor of shape (total_anchors, 2) containing center (x, y) coordinates.
        strides_tensor: Tensor of shape (total_anchors,) containing the stride of each anchor.
    """
    all_priors = []
    all_strides = []
    for (h, w), stride in zip(featmap_sizes, strides):
        x_center = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_center = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride

        grid_y, grid_x = torch.meshgrid(y_center, x_center, indexing="ij")
        priors = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        all_priors.append(priors)
        all_strides.append(torch.full((priors.shape[0],), stride, device=device, dtype=dtype))

    return torch.cat(all_priors, dim=0), torch.cat(all_strides, dim=0)


def decode_ltrb_to_xyxy(
    anchor_points: torch.Tensor,
    ltrb_offsets: torch.Tensor,
    max_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Decode predicted [left, top, right, bottom] offsets from anchor points
    into absolute [x1, y1, x2, y2] bounding boxes on the image.

    Args:
        anchor_points: (..., 2) center coordinates (x, y).
        ltrb_offsets: (..., 4) distance offsets [l, t, r, b].
        max_shape: Optional (H, W) image boundary for clamping.

    Returns:
        boxes_xyxy: (..., 4) decoded boxes.
    """
    x1 = anchor_points[..., 0] - ltrb_offsets[..., 0]
    y1 = anchor_points[..., 1] - ltrb_offsets[..., 1]
    x2 = anchor_points[..., 0] + ltrb_offsets[..., 2]
    y2 = anchor_points[..., 1] + ltrb_offsets[..., 3]

    if max_shape is not None:
        max_h, max_w = max_shape
        x1 = x1.clamp(min=0, max=max_w)
        y1 = y1.clamp(min=0, max=max_h)
        x2 = x2.clamp(min=0, max=max_w)
        y2 = y2.clamp(min=0, max=max_h)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def encode_xyxy_to_ltrb(
    anchor_points: torch.Tensor,
    bbox: torch.Tensor,
    max_dis: Optional[float] = None,
    eps: float = 0.1,
) -> torch.Tensor:
    """
    Encode ground-truth [x1, y1, x2, y2] boxes into target [l, t, r, b]
    offsets relative to anchor points.

    Args:
        anchor_points: (..., 2) center coordinates (x, y).
        bbox: (..., 4) ground-truth boxes in xyxy format.
        max_dis: Optional upper bound for distance offsets.
        eps: Margin subtracted from max_dis when clamping.

    Returns:
        ltrb: (..., 4) offsets [l, t, r, b].
    """
    left = anchor_points[..., 0] - bbox[..., 0]
    top = anchor_points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - anchor_points[..., 0]
    bottom = bbox[..., 3] - anchor_points[..., 1]

    distance = torch.stack([left, top, right, bottom], dim=-1)
    if max_dis is not None:
        distance = distance.clamp(min=0, max=max_dis - eps)
    return distance


# Shorthand aliases
cxcywh_to_xyxy = box_cxcywh_to_xyxy
xyxy_to_cxcywh = box_xyxy_to_cxcywh
xywh_to_xyxy = box_xywh_to_xyxy
xyxy_to_xywh = box_xyxy_to_xywh
decode_bboxes = decode_ltrb_to_xyxy
encode_bboxes = encode_xyxy_to_ltrb
