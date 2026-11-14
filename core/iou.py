"""
Bounding box IoU, GIoU, and CIoU geometric computations (Pure PyTorch).
"""

import math
from typing import Optional
import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes: (..., 4) tensor.

    Returns:
        area: (...,) tensor.
    """
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (
        boxes[..., 3] - boxes[..., 1]
    ).clamp(min=0)


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
        eps: Small constant for numerical stability.

    Returns:
        iou: (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=eps)


def box_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    elementwise: Optional[bool] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes in (x1, y1, x2, y2) format.

    Supports both:
        - Elementwise mode: (N, 4) and (N, 4) -> (N,)
        - Pairwise mode: (N, 4) and (M, 4) -> (N, M)

    Args:
        boxes1: (N, 4)
        boxes2: (N, 4) or (M, 4)
        elementwise: If True, forces elementwise calculation. If False, forces pairwise calculation.
            If None (default), infers elementwise only if shapes match and boxes1 is 2D.
        eps: Small constant for numerical stability.

    Returns:
        giou: (N,) if elementwise else (N, M)
    """
    if elementwise is None:
        do_elementwise = (
            boxes1.shape == boxes2.shape and boxes1.dim() == 2 and boxes1.shape[1] == 4
        )
    else:
        do_elementwise = elementwise

    if do_elementwise:
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
        iou = inter / union.clamp(min=eps)

        c_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        c_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        c_wh = (c_rb - c_lt).clamp(min=0)
        c_area = c_wh[:, 0] * c_wh[:, 1]

        return iou - (c_area - union) / c_area.clamp(min=eps)

    # Pairwise calculation
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=eps)

    c_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    c_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[:, :, 0] * c_wh[:, :, 1]

    return iou - (c_area - union) / c_area.clamp(min=eps)


def box_ciou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU), elementwise: (N, 4) and (N, 4) -> (N,).
    CIoU = IoU - (rho^2 / c^2) - alpha * v, with v aspect-ratio consistency penalty.

    Args:
        boxes1: (N, 4) in (x1, y1, x2, y2)
        boxes2: (N, 4) in (x1, y1, x2, y2)
        eps: Small constant for numerical stability.

    Returns:
        ciou: (N,) tensor.
    """
    x1, y1, x2, y2 = boxes1.unbind(-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(-1)

    inter_x1 = torch.max(x1, x1g)
    inter_y1 = torch.max(y1, y1g)
    inter_x2 = torch.min(x2, x2g)
    inter_y2 = torch.min(y2, y2g)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    a1 = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a2 = (x2g - x1g).clamp(min=0) * (y2g - y1g).clamp(min=0)
    union = (a1 + a2 - inter).clamp(min=eps)
    iou = inter / union

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    cxg = (x1g + x2g) * 0.5
    cyg = (y1g + y2g) * 0.5
    rho2 = (cx - cxg).pow(2) + (cy - cyg).pow(2)

    cw = torch.max(x2, x2g) - torch.min(x1, x1g)
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)
    c2 = (cw.pow(2) + ch.pow(2)).clamp(min=eps)

    w1 = (x2 - x1).clamp(min=eps)
    h1 = (y2 - y1).clamp(min=eps)
    w2 = (x2g - x1g).clamp(min=eps)
    h2 = (y2g - y1g).clamp(min=eps)

    v = (4.0 / (math.pi**2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return iou - rho2 / c2 - alpha * v
