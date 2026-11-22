"""
Shared pytest fixtures for unit tests.
"""

from typing import List, Tuple

import pytest
import torch

from core.outputs import DetectionOutput


@pytest.fixture
def dummy_input_320() -> torch.Tensor:
    """Return a dummy batch of shape (2, 3, 320, 320)."""
    return torch.randn(2, 3, 320, 320)


@pytest.fixture
def dummy_input_448() -> torch.Tensor:
    """Return a dummy batch of shape (2, 3, 448, 448)."""
    return torch.randn(2, 3, 448, 448)


@pytest.fixture
def dummy_gt() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return dummy ground-truth boxes and labels for a batch of size 2.

    Returns:
        A tuple (gt_boxes_list, gt_labels_list) where each element corresponds
        to one image in the batch.
    """
    gt_boxes_list: List[torch.Tensor] = [
        torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
        torch.tensor([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]]),
    ]
    gt_labels_list: List[torch.Tensor] = [
        torch.tensor([0]),
        torch.tensor([1, 2]),
    ]
    return (gt_boxes_list, gt_labels_list)


@pytest.fixture
def dummy_detection_output() -> DetectionOutput:
    """Return a dummy DetectionOutput with batch size 2 and 100 anchors."""
    return DetectionOutput(
        pred_cls=torch.randn(2, 100, 80),
        pred_boxes=torch.rand(2, 100, 4) * 100,
        anchor_centers=torch.rand(100, 2) * 640,
        anchor_strides=torch.full((100, 1), 8.0),
    )
