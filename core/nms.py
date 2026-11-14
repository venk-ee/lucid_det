"""
Non-Maximum Suppression (NMS) utilities.
"""

from typing import Dict, List, Tuple

import torch
from torchvision.ops import batched_nms


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.65,
    max_num: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform multi-class Non-Maximum Suppression on a single image's predictions.

    Args:
        boxes: Tensor of shape (N, 4) in (x1, y1, x2, y2) format.
        scores: Tensor of shape (N, num_classes) with class probabilities or logits (sigmoid will be applied if max > 1).
        score_thresh: Minimum score threshold to retain a detection.
        nms_thresh: IoU threshold for NMS.
        max_num: Maximum number of detections to keep per image.

    Returns:
        det_boxes: Tensor of shape (K, 4)
        det_scores: Tensor of shape (K,)
        det_labels: Tensor of shape (K,) with class indices (int64)
    """
    if scores.numel() == 0 or boxes.numel() == 0:
        empty_boxes = torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype)
        empty_scores = torch.empty((0,), device=scores.device, dtype=scores.dtype)
        empty_labels = torch.empty((0,), device=scores.device, dtype=torch.long)
        return empty_boxes, empty_scores, empty_labels

    # If scores are raw logits, apply sigmoid
    if scores.max() > 1.0 or scores.min() < 0.0:
        scores = scores.sigmoid()

    # Best class score and label per candidate box
    max_scores, max_labels = torch.max(scores, dim=-1)
    keep_mask = max_scores > score_thresh

    if not keep_mask.any():
        empty_boxes = torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype)
        empty_scores = torch.empty((0,), device=scores.device, dtype=scores.dtype)
        empty_labels = torch.empty((0,), device=scores.device, dtype=torch.long)
        return empty_boxes, empty_scores, empty_labels

    cand_boxes = boxes[keep_mask]
    cand_scores = max_scores[keep_mask]
    cand_labels = max_labels[keep_mask]

    keep_idx = batched_nms(cand_boxes, cand_scores, cand_labels, nms_thresh)

    if max_num > 0 and keep_idx.numel() > max_num:
        keep_idx = keep_idx[:max_num]

    return cand_boxes[keep_idx], cand_scores[keep_idx], cand_labels[keep_idx]


def multiclass_nms_batch(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.65,
    max_num: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """
    Perform multi-class NMS across a batch of images.

    Args:
        pred_boxes: Tensor of shape (B, N, 4) in (x1, y1, x2, y2).
        pred_cls: Tensor of shape (B, N, num_classes) raw logits or probabilities.
        score_thresh: Score threshold.
        nms_thresh: IoU suppression threshold.
        max_num: Max boxes to retain per image.

    Returns:
        List of length B, where each element is a dict with keys:
            'boxes': (K, 4)
            'scores': (K,)
            'labels': (K,)
    """
    batch_size = pred_boxes.shape[0]
    results = []

    for b in range(batch_size):
        b_boxes, b_scores, b_labels = multiclass_nms(
            pred_boxes[b],
            pred_cls[b],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            max_num=max_num,
        )
        results.append(
            {
                "boxes": b_boxes,
                "scores": b_scores,
                "labels": b_labels,
            }
        )

    return results
