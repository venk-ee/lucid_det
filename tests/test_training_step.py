"""
End-to-end integration test for training data pipeline and detector training step.
Verifies DetDataset -> det_collate -> model forward pass -> loss -> autograd backward pass.
"""

import json
import os
import tempfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.det_dataset import DetDataset, det_collate
from losses.det_loss import det_loss
from models.zoo import build_rtmdet
from transforms.det_transforms import build_det_transform


def test_full_training_step_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Create two sample images
        cv2.imwrite(
            os.path.join(img_dir, "train_0.jpg"), np.full((480, 640, 3), 100, dtype=np.uint8)
        )
        cv2.imwrite(
            os.path.join(img_dir, "train_1.jpg"), np.full((500, 500, 3), 150, dtype=np.uint8)
        )

        coco_data = {
            "images": [
                {"id": 1, "file_name": "train_0.jpg", "height": 480, "width": 640},
                {"id": 2, "file_name": "train_1.jpg", "height": 500, "width": 500},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50, 50, 100, 100],
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [200, 200, 80, 80],
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [100, 120, 150, 150],
                    "iscrowd": 0,
                },
            ],
        }

        ann_path = os.path.join(tmpdir, "train_ann.json")
        with open(ann_path, "w") as f:
            json.dump(coco_data, f)

        # Build transform and dataset
        transform = build_det_transform(img_size=640, is_train=True)
        dataset = DetDataset(root=img_dir, ann_file=ann_path, transform=transform, num_classes=2)

        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=det_collate)
        batch = next(iter(loader))

        # Build detector
        model = build_rtmdet(arch="tiny", num_classes=2)
        model.train()

        # Forward pass
        outputs = model(batch["img"], training_override=True)

        # Loss calculation
        total_loss, loss_dict = det_loss(
            model_out=outputs,
            gt_boxes_list=batch["gt_boxes"],
            gt_labels_list=batch["gt_labels"],
            cls_loss="qfl",
            box_loss="giou",
        )

        assert torch.isfinite(total_loss), f"Loss must be finite, got {total_loss.item()}"
        assert total_loss.item() > 0.0

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Check gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients were computed!"
        for g in grads:
            assert torch.isfinite(g).all(), "Detected NaN or Inf gradient!"


def test_model_ema_unwrapping():
    from engine.ema import ModelEMA

    model = build_rtmdet(arch="tiny", num_classes=2)
    ema = ModelEMA(model)

    # Simulate torch.compile wrapper
    class DummyCompiledWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m

    wrapped = DummyCompiledWrapper(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
            break

    # Update should succeed and update weights without error
    ema.update(wrapped)
    assert ema.updates == 1


def test_val_one_epoch_coordinate_scaling():
    from runners.det_runner import val_one_epoch

    model = build_rtmdet(arch="tiny", num_classes=2)
    model.eval()

    # Create dummy batch with ori_shape 720x1280 and 640x640 images
    batch = {
        "img": torch.randn(1, 3, 640, 640),
        "img_ids": [42],
        "ori_shapes": [(720, 1280)],
    }
    loader = [batch]

    captured_preds = []

    class DummyEvaluator:
        label2cat = {0: 10, 1: 25}

        def evaluate(self, preds):
            captured_preds.extend(preds)
            return {"mAP_50": 0.5}

    evaluator = DummyEvaluator()
    metrics = val_one_epoch(model, loader, device="cpu", evaluator=evaluator)
    assert "mAP_50" in metrics

    # If any detections were made, ensure coordinates are rescaled and category_id is mapped
    for p in captured_preds:
        assert p["image_id"] == 42
        assert p["category_id"] in (10, 25)
        x, y, w, h = p["bbox"]
        assert x >= 0 and y >= 0
        assert x + w <= 1280.1 and y + h <= 720.1
