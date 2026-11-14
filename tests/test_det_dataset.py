"""
Unit tests for DetDataset and det_collate in datasets/det_dataset.py.
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from datasets.det_dataset import DetDataset, det_collate


@pytest.fixture
def dummy_coco_env():
    """Create a temporary directory with a dummy COCO dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Create two test images
        # Image 1: 480x640
        im1 = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "im1.jpg"), im1)

        # Image 2: 600x800
        im2 = np.full((600, 800, 3), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "im2.jpg"), im2)

        # Image 3: 300x300 (zero annotations)
        im3 = np.full((300, 300, 3), 50, dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "im3.jpg"), im3)

        coco_data = {
            "images": [
                {"id": 1, "file_name": "im1.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "im2.jpg"},  # Missing height/width deliberately
                {"id": 3, "file_name": "im3.jpg", "height": 300, "width": 300},
            ],
            "categories": [
                {"id": 10, "name": "vehicle"},
                {"id": 25, "name": "pedestrian"},
            ],
            "annotations": [
                # Valid box on img 1
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 10,
                    "bbox": [50, 60, 100, 150],
                    "iscrowd": 0,
                },
                # Second valid box on img 1
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 25,
                    "bbox": [200, 250, 80, 120],
                    "iscrowd": 0,
                },
                # Crowd box (should be filtered)
                {"id": 3, "image_id": 1, "category_id": 10, "bbox": [10, 10, 50, 50], "iscrowd": 1},
                # Degenerate box with width=0 (should be filtered)
                {
                    "id": 4,
                    "image_id": 1,
                    "category_id": 10,
                    "bbox": [100, 100, 0, 50],
                    "iscrowd": 0,
                },
                # Valid box on img 2
                {
                    "id": 5,
                    "image_id": 2,
                    "category_id": 25,
                    "bbox": [300, 300, 150, 200],
                    "iscrowd": 0,
                },
                # Category not in cat_ids (should be filtered)
                {
                    "id": 6,
                    "image_id": 2,
                    "category_id": 999,
                    "bbox": [50, 50, 50, 50],
                    "iscrowd": 0,
                },
            ],
        }

        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(coco_data, f)

        yield img_dir, ann_path


def test_det_dataset_parsing(dummy_coco_env):
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    assert len(ds) == 3
    # Check category mappings
    assert ds.cat2label == {10: 0, 25: 1}
    assert ds.label2cat == {0: 10, 1: 25}

    # Verify item 0 (im1.jpg) has 2 valid boxes
    item1 = ds[0]
    assert item1["img_id"] == 1
    assert item1["ori_shape"] == (480, 640)
    assert item1["gt_boxes"].shape == (2, 4)
    assert item1["gt_labels"].shape == (2,)
    # Check conversion from COCO [x, y, w, h] to xyxy: [50, 60, 100, 150] -> [50, 60, 150, 210]
    expected_box_0 = torch.tensor([50.0, 60.0, 150.0, 210.0])
    assert torch.allclose(item1["gt_boxes"][0], expected_box_0)
    assert item1["gt_labels"][0].item() == 0  # cat 10 mapped to 0


def test_det_dataset_missing_dimensions_fallback(dummy_coco_env):
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    # im2.jpg has missing height/width in JSON
    item2 = ds[1]
    assert item2["img_id"] == 2
    # Should correctly infer (600, 800) from image file itself
    assert item2["ori_shape"] == (600, 800)
    assert item2["gt_boxes"].shape == (1, 4)
    assert item2["gt_labels"].shape == (1,)
    assert item2["gt_labels"][0].item() == 1  # cat 25 mapped to 1


def test_det_dataset_empty_annotations_image(dummy_coco_env):
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    # im3.jpg has zero annotations
    item3 = ds[2]
    assert item3["img_id"] == 3
    assert item3["ori_shape"] == (300, 300)
    assert item3["gt_boxes"].shape == (0, 4)
    assert item3["gt_labels"].shape == (0,)
    assert item3["gt_boxes"].dtype == torch.float32
    assert item3["gt_labels"].dtype == torch.int64


def test_det_collate(dummy_coco_env):
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    # Collate all 3 items (batch size 3) with varying box counts (2, 1, 0)
    batch_raw = [ds[0], ds[1], ds[2]]
    # Adjust im2 and im3 to same shape as im1 for stacking test
    # (in practice, transform standardizes shape to 640x640)
    target_shape = batch_raw[0]["img"].shape
    for b in batch_raw[1:]:
        b["img"] = torch.nn.functional.interpolate(
            b["img"].unsqueeze(0), size=target_shape[1:], mode="bilinear"
        ).squeeze(0)

    collated = det_collate(batch_raw)
    assert "img" in collated
    assert "gt_boxes" in collated
    assert "gt_labels" in collated
    assert "img_ids" in collated
    assert "ori_shapes" in collated

    assert collated["img"].shape == (3, target_shape[0], target_shape[1], target_shape[2])
    assert len(collated["gt_boxes"]) == 3
    assert collated["gt_boxes"][0].shape == (2, 4)
    assert collated["gt_boxes"][1].shape == (1, 4)
    assert collated["gt_boxes"][2].shape == (0, 4)
    assert collated["img_ids"] == [1, 2, 3]
