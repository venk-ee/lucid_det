"""
Unit tests for DetDataset, det_collate, and cooldown DataLoader swap using local COCO-8 dataset.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def test_dataset_clamps_out_of_bounds_boxes() -> None:
    dataset = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=None,
        num_classes=80,
    )
    img_id = dataset.ids[0]
    dataset.anns[img_id] = [([-20.0, -15.0, 2000.0, 2000.0], 0)]
    sample = dataset[0]
    orig_h, orig_w = sample["ori_shape"]
    boxes = sample["gt_boxes"]
    assert boxes[0, 0].item() == 0.0
    assert boxes[0, 1].item() == 0.0
    assert boxes[0, 2].item() <= float(orig_w)
    assert boxes[0, 3].item() <= float(orig_h)


def test_det_dataset_get_raw_sample() -> None:
    dataset = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=None,
        num_classes=80,
    )
    img, bboxes, labels, img_id, ori_shape = dataset.get_raw_sample(0)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    assert isinstance(bboxes, list)
    assert isinstance(labels, list)
    assert isinstance(img_id, int)
    assert isinstance(ori_shape, tuple) and len(ori_shape) == 2
    if len(bboxes) > 0:
        for b in bboxes:
            assert len(b) == 4
            x, y, w, h = b
            assert x >= 0.0 and y >= 0.0
            assert x + w <= float(ori_shape[1]) + 2.0
            assert y + h <= float(ori_shape[0]) + 2.0


def test_cooldown_dataloader_swap() -> None:
    """Test two-stage cooldown DataLoader swap pattern with copy.copy and val_transform."""
    train_transform = build_det_transform(img_size=640, is_train=True)
    val_transform = build_det_transform(img_size=640, is_train=False)

    train_ds = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=train_transform,
        num_classes=80,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=det_collate,
    )

    cooldown_ds = copy.copy(train_ds)
    cooldown_ds.transform = val_transform
    cooldown_loader = DataLoader(
        cooldown_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=det_collate,
    )

    # Verify independent transforms
    assert train_ds.transform is train_transform
    assert cooldown_ds.transform is val_transform

    # Verify both loaders yield valid batches
    train_batch = next(iter(train_loader))
    cooldown_batch = next(iter(cooldown_loader))

    assert train_batch["img"].shape == (2, 3, 640, 640)
    assert cooldown_batch["img"].shape == (2, 3, 640, 640)
    assert len(train_batch["gt_boxes"]) == 2
    assert len(cooldown_batch["gt_boxes"]) == 2


def test_cooldown_schedule_logic() -> None:
    """Verify cooldown epoch calculation and resume transition logic."""
    total_epochs = 300
    cooldown_epochs = 15

    cooldown_epoch = (
        (total_epochs - cooldown_epochs + 1) if cooldown_epochs > 0 else total_epochs + 1
    )
    assert cooldown_epoch == 286

    # Normal run from epoch 1
    start_epoch = 1
    current_loader = "cooldown" if (start_epoch >= cooldown_epoch) else "train"
    assert current_loader == "train"

    # Before transition (epoch 285)
    epoch = 285
    if epoch == cooldown_epoch:
        current_loader = "cooldown"
    assert current_loader == "train"

    # At transition (epoch 286)
    epoch = 286
    if epoch == cooldown_epoch:
        current_loader = "cooldown"
    assert current_loader == "cooldown"

    # Resumed run starting at epoch 290 (past transition)
    resume_epoch = 290
    resumed_loader = "cooldown" if (resume_epoch >= cooldown_epoch) else "train"
    assert resumed_loader == "cooldown"

    # When cooldown is disabled (cooldown_epochs == 0)
    zero_cooldown_epoch = (total_epochs - 0 + 1) if 0 > 0 else total_epochs + 1
    assert zero_cooldown_epoch == 301
    assert ("cooldown" if (1 >= zero_cooldown_epoch) else "train") == "train"
