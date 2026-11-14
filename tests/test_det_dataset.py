"""
Unit tests for DetDataset and det_collate in datasets/det_dataset.py.
Dataset robustness, adversarial annotation filtering, and stress testing.
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
    """Create a temporary directory with an adversarial COCO dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Image 1: 480x640 (standard)
        cv2.imwrite(os.path.join(img_dir, "im1.jpg"), np.full((480, 640, 3), 128, dtype=np.uint8))
        # Image 2: 600x800 (missing height/width in JSON)
        cv2.imwrite(os.path.join(img_dir, "im2.jpg"), np.full((600, 800, 3), 200, dtype=np.uint8))
        # Image 3: 300x300 (zero annotations / pure background)
        cv2.imwrite(os.path.join(img_dir, "im3.jpg"), np.full((300, 300, 3), 50, dtype=np.uint8))
        # Image 4: 720x1280 (high density stress test: 200 boxes)
        cv2.imwrite(os.path.join(img_dir, "im4.jpg"), np.full((720, 1280, 3), 80, dtype=np.uint8))

        annotations = [
            # Valid boxes on im1
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [50, 60, 100, 150], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 25, "bbox": [200, 250, 80, 120], "iscrowd": 0},
            # Adversarial corruptions on im1 (must be filtered)
            {"id": 3, "image_id": 1, "category_id": 10, "bbox": [10, 10, 50, 50], "iscrowd": 1},
            {"id": 4, "image_id": 1, "category_id": 10, "bbox": [100, 100, 0, 50], "iscrowd": 0},
            {"id": 5, "image_id": 1, "category_id": 10, "bbox": [100, 100, 50, 0], "iscrowd": 0},
            {"id": 6, "image_id": 1, "category_id": 10, "bbox": [100, 100, -10, 50], "iscrowd": 0},
            # Valid box on im2
            {"id": 7, "image_id": 2, "category_id": 25, "bbox": [300, 300, 150, 200], "iscrowd": 0},
            # Unmapped category (must be filtered)
            {"id": 8, "image_id": 2, "category_id": 999, "bbox": [50, 50, 50, 50], "iscrowd": 0},
        ]

        # Add 200 dense synthetic boxes to im4 for stress testing
        for box_id in range(100, 300):
            annotations.append(
                {
                    "id": box_id,
                    "image_id": 4,
                    "category_id": 10,
                    "bbox": [float(box_id % 500), float(box_id % 300), 40.0, 40.0],
                    "iscrowd": 0,
                }
            )

        coco_data = {
            "images": [
                {"id": 1, "file_name": "im1.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "im2.jpg"},  # Deliberately missing dimensions
                {"id": 3, "file_name": "im3.jpg", "height": 300, "width": 300},
                {"id": 4, "file_name": "im4.jpg", "height": 720, "width": 1280},
            ],
            "categories": [
                {"id": 10, "name": "vehicle"},
                {"id": 25, "name": "pedestrian"},
            ],
            "annotations": annotations,
        }

        ann_path = os.path.join(tmpdir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(coco_data, f)

        yield img_dir, ann_path


def test_det_dataset_parsing_and_adversarial_filtering(dummy_coco_env):
    """Verify parsing, category mapping, and filtering of crowd/zero-area boxes."""
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    assert len(ds) == 4
    # Category synchronization
    assert ds.num_classes == 2
    assert ds.cat2label == {10: 0, 25: 1}
    assert ds.label2cat == {0: 10, 1: 25}

    # im1: only the 2 valid boxes survive out of 6 annotations
    item1 = ds[0]
    assert item1["img_id"] == 1
    assert item1["ori_shape"] == (480, 640)
    assert item1["gt_boxes"].shape == (2, 4)
    assert item1["gt_labels"].shape == (2,)

    expected_box_0 = torch.tensor([50.0, 60.0, 150.0, 210.0])
    assert torch.allclose(item1["gt_boxes"][0], expected_box_0)
    assert item1["gt_labels"][0].item() == 0


def test_det_dataset_missing_dimensions_fallback(dummy_coco_env):
    """Gracefully infer image dimensions directly from file when JSON metadata lacks them."""
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    item2 = ds[1]
    assert item2["img_id"] == 2
    assert item2["ori_shape"] == (600, 800)
    assert item2["gt_boxes"].shape == (1, 4)
    assert item2["gt_labels"][0].item() == 1


def test_det_dataset_empty_annotations_image(dummy_coco_env):
    """Images with zero annotations must return well-formed empty tensors."""
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    item3 = ds[2]
    assert item3["img_id"] == 3
    assert item3["ori_shape"] == (300, 300)
    assert item3["gt_boxes"].shape == (0, 4)
    assert item3["gt_labels"].shape == (0,)
    assert item3["gt_boxes"].dtype == torch.float32
    assert item3["gt_labels"].dtype == torch.int64


def test_det_dataset_high_density_stress(dummy_coco_env):
    """Stress test: verify 200 boxes parsed, stacked, and bounded correctly."""
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    item4 = ds[3]
    assert item4["img_id"] == 4
    assert item4["ori_shape"] == (720, 1280)
    assert item4["gt_boxes"].shape == (200, 4)
    assert item4["gt_labels"].shape == (200,)
    # Assert coordinates strictly clamped to image boundaries
    assert item4["gt_boxes"][:, 0].min() >= 0.0
    assert item4["gt_boxes"][:, 2].max() <= 1280.0
    assert item4["gt_boxes"][:, 1].min() >= 0.0
    assert item4["gt_boxes"][:, 3].max() <= 720.0


def test_det_collate_variable_box_counts(dummy_coco_env):
    """Collate mixed batch: images with 2, 1, 0, and 200 boxes."""
    img_dir, ann_path = dummy_coco_env
    ds = DetDataset(root=img_dir, ann_file=ann_path, num_classes=2, cat_ids=[10, 25])

    batch_raw = [ds[0], ds[1], ds[2], ds[3]]
    # Resize image tensors to uniform [3, 640, 640] for stacking
    for b in batch_raw:
        b["img"] = torch.nn.functional.interpolate(
            b["img"].unsqueeze(0), size=(640, 640), mode="bilinear"
        ).squeeze(0)

    collated = det_collate(batch_raw)
    assert collated["img"].shape == (4, 3, 640, 640)
    assert len(collated["gt_boxes"]) == 4
    assert collated["gt_boxes"][0].shape == (2, 4)
    assert collated["gt_boxes"][1].shape == (1, 4)
    assert collated["gt_boxes"][2].shape == (0, 4)
    assert collated["gt_boxes"][3].shape == (200, 4)
    assert collated["img_ids"] == [1, 2, 3, 4]
    assert collated["ori_shapes"] == [(480, 640), (600, 800), (300, 300), (720, 1280)]
