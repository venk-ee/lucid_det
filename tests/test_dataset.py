"""
Unit tests for DetDataset and det_collate using local COCO-8 dataset.
"""

from pathlib import Path

import torch

from datasets.det_dataset import DetDataset, det_collate
from transforms.det_transforms import build_det_transform

COCO8_ROOT = Path("data/coco8/images/train")
COCO8_ANN = Path("data/coco8/annotations/instances_train.json")


def test_coco8_dataset_loading() -> None:
    transform = build_det_transform(img_size=640, is_train=False)
    dataset = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=transform,
        num_classes=80,
    )
    assert len(dataset) == 4

    sample = dataset[0]
    assert "img" in sample
    assert "gt_boxes" in sample
    assert "gt_labels" in sample
    assert "img_id" in sample
    assert "ori_shape" in sample

    assert isinstance(sample["img"], torch.Tensor)
    assert sample["img"].shape == (3, 640, 640)
    assert sample["gt_boxes"].ndim == 2
    assert sample["gt_boxes"].shape[1] == 4
    assert sample["gt_labels"].ndim == 1


def test_det_collate_batch() -> None:
    transform = build_det_transform(img_size=640, is_train=False)
    dataset = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=transform,
        num_classes=80,
    )

    batch = det_collate([dataset[0], dataset[1]])
    assert isinstance(batch["img"], torch.Tensor)
    assert batch["img"].shape == (2, 3, 640, 640)
    assert len(batch["gt_boxes"]) == 2
    assert len(batch["gt_labels"]) == 2
    assert len(batch["img_ids"]) == 2
    assert len(batch["ori_shapes"]) == 2


def test_dataset_without_transform_returns_tensors() -> None:
    dataset = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=None,
        num_classes=80,
    )
    sample = dataset[0]
    assert isinstance(sample["img"], torch.Tensor)
    assert isinstance(sample["gt_boxes"], torch.Tensor)
    assert isinstance(sample["gt_labels"], torch.Tensor)
    assert sample["gt_boxes"].shape[1] == 4
    assert sample["gt_boxes"].dtype == torch.float32
    assert sample["gt_labels"].dtype == torch.int64
