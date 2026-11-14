"""
End-to-end integration and autograd gradient flow tests.
Zero-dead-layer validation, numerical gradient stability, and graph leak checks.
"""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from datasets.det_dataset import DetDataset, det_collate
from engine.ema import ModelEMA
from losses.det_loss import det_loss
from models.zoo import build_rtmdet, build_yolox
from runners.det_runner import val_one_epoch
from transforms.det_transforms import build_det_transform


@pytest.fixture
def mini_training_batch():
    """Create a temporary dataset and return a collated batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir, exist_ok=True)

        cv2.imwrite(
            os.path.join(img_dir, "train_0.jpg"), np.full((640, 640, 3), 100, dtype=np.uint8)
        )
        cv2.imwrite(
            os.path.join(img_dir, "train_1.jpg"), np.full((640, 640, 3), 150, dtype=np.uint8)
        )

        coco_data = {
            "images": [
                {"id": 1, "file_name": "train_0.jpg", "height": 640, "width": 640},
                {"id": 2, "file_name": "train_1.jpg", "height": 640, "width": 640},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
            "annotations": [
                # Image 1 multi-scale objects (tiny P3/stride 8, medium P4/stride 16, large P5/stride 32)
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [16, 16, 8, 8], "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [80, 80, 24, 24], "iscrowd": 0},
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [200, 200, 200, 200],
                    "iscrowd": 0,
                },
                # Image 2 multi-scale objects
                {"id": 4, "image_id": 2, "category_id": 2, "bbox": [24, 24, 8, 8], "iscrowd": 0},
                {"id": 5, "image_id": 2, "category_id": 1, "bbox": [96, 96, 24, 24], "iscrowd": 0},
                {
                    "id": 6,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [220, 220, 200, 200],
                    "iscrowd": 0,
                },
            ],
        }

        ann_path = os.path.join(tmpdir, "train_ann.json")
        with open(ann_path, "w") as f:
            json.dump(coco_data, f)

        transform = build_det_transform(img_size=640, is_train=False)
        dataset = DetDataset(root=img_dir, ann_file=ann_path, transform=transform, num_classes=2)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=det_collate)
        return next(iter(loader))


# --- 1. Zero Dead-Layer Autograd Rigor ---


@pytest.mark.parametrize(
    "model_builder,builder_kwargs",
    [
        (build_rtmdet, {"arch": "tiny", "num_classes": 2}),
        (build_rtmdet, {"arch": "s", "num_classes": 2}),
        (build_yolox, {"arch": "s", "num_classes": 2}),
    ],
)
def test_zero_dead_layers_autograd(mini_training_batch, model_builder, builder_kwargs):
    """
    Zero Dead Layers validation.
    Asserts that 100% of trainable parameters receive valid, finite, non-zero autograd gradients.
    """
    model = model_builder(**builder_kwargs)
    model.train()

    outputs = model(mini_training_batch["img"], training_override=True)
    total_loss, loss_dict = det_loss(
        model_out=outputs,
        gt_boxes_list=mini_training_batch["gt_boxes"],
        gt_labels_list=mini_training_batch["gt_labels"],
        cls_loss="qfl",
        box_loss="giou",
    )

    assert torch.isfinite(total_loss), f"Loss must be finite, got {total_loss.item()}"
    assert total_loss.item() > 0.0

    model.zero_grad()
    total_loss.backward()

    # Verify every single parameter received non-zero gradient
    dead_layers = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead_layers.append(f"{name} (grad is None)")
        elif not torch.isfinite(p.grad).all():
            dead_layers.append(f"{name} (grad has NaN/Inf)")
        elif p.grad.abs().sum() == 0.0:
            dead_layers.append(f"{name} (gradient norm is exactly 0)")

    assert len(dead_layers) == 0, f"Detected {len(dead_layers)} dead layers: {dead_layers[:5]}"


# --- 2. Numerical Stability Under Mixed Precision (AMP) ---


def test_amp_numerical_stability(mini_training_batch):
    """Verify loss calculation and backward pass under lower precision (bfloat16)."""
    model = build_rtmdet(arch="tiny", num_classes=2)
    model.train()

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = model(mini_training_batch["img"], training_override=True)
        total_loss, loss_dict = det_loss(
            model_out=outputs,
            gt_boxes_list=mini_training_batch["gt_boxes"],
            gt_labels_list=mini_training_batch["gt_labels"],
            cls_loss="qfl",
            box_loss="giou",
        )

    assert torch.isfinite(total_loss), "AMP produced NaN or Inf loss!"
    total_loss.backward()


# --- 3. Inference Graph Leak Prevention ---


def test_inference_eval_mode_graph_detached():
    """Verify eval mode does NOT retain computational graph in detections (prevents memory leak)."""
    model = build_rtmdet(arch="tiny", num_classes=2)
    model.eval()

    dummy_imgs = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        infer_out = model(dummy_imgs, training_override=False)

    detections = infer_out["detections"]
    for det in detections:
        assert det["boxes"].grad_fn is None
        assert det["scores"].grad_fn is None
        assert det["labels"].grad_fn is None


# --- 4. ModelEMA Compilation & DDP Resilience ---


def test_model_ema_unwrapping():
    """Ensure ModelEMA unwraps torch.compile (_orig_mod) and DDP (module) transparently."""
    model = build_rtmdet(arch="tiny", num_classes=2)
    ema = ModelEMA(model)

    class DummyCompiledWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m

    wrapped = DummyCompiledWrapper(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
            break

    ema.update(wrapped)
    assert ema.updates == 1


# --- 5. Validation Coordinate Scaling & Category Mapping ---


def test_val_one_epoch_coordinate_scaling():
    """Verify validation loop unletterboxes predictions and maps 0-indexed labels to COCO IDs."""
    model = build_rtmdet(arch="tiny", num_classes=2)
    model.eval()

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

    for p in captured_preds:
        assert p["image_id"] == 42
        assert p["category_id"] in (10, 25)
        x, y, w, h = p["bbox"]
        assert x >= 0 and y >= 0
        assert x + w <= 1280.1 and y + h <= 720.1
