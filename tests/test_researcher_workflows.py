"""
Backtesting suite for AI researcher workflows.
Validates that LucidDet provides zero-friction experimentation for researchers:
- Workflow A: Injecting a custom backbone into an existing detector in 3 lines.
- Workflow B: Injecting a novel target assigner callable into det_loss.
- Workflow C: Injecting a custom classification and box loss function.
- Workflow D: Implementing a custom detector from scratch and training with generic_train.
- Production Tracing: Deterministic torch.jit.trace export with zero dynamic control flow errors.
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from losses import det_loss
from models import RTMDet
from runners import generic_train
from training_scripts.base_config import BaseTrainConfig, DataConfig, OptimConfig


class ToyResearchBackbone(nn.Module):
    """Example novel backbone an AI researcher might invent."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=8, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.out_channels = (64, 128, 256)
        self.strides = (8, 16, 32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = torch.relu(self.conv1(x))
        p4 = torch.relu(self.conv2(p3))
        p5 = torch.relu(self.conv3(p4))
        return (p3, p4, p5)


def test_workflow_a_custom_backbone_injection(dummy_input_320: torch.Tensor) -> None:
    """Workflow A: Testing a novel backbone on an existing detector."""
    # 1. Researcher instantiates their custom backbone
    custom_backbone = ToyResearchBackbone()

    # 2. Researcher injects it directly into RTMDet (Constructor Injection)
    # The detector accepts the backbone and automatically adapts neck channels
    model = RTMDet(backbone=custom_backbone, arch="tiny", num_classes=80)
    assert model.backbone is custom_backbone

    # 3. Verify forward and backward pass
    model.train()
    outputs = model(dummy_input_320, training_override=True)
    assert "pred_cls" in outputs and "pred_boxes" in outputs

    loss = outputs["pred_cls"].sum() + outputs["pred_boxes"].sum()
    loss.backward()

    # Confirm gradients flow all the way to the researcher's custom backbone conv1
    assert custom_backbone.conv1.weight.grad is not None
    assert torch.isfinite(custom_backbone.conv1.weight.grad).all()


def test_workflow_b_custom_assigner_injection(
    dummy_input_320: torch.Tensor, dummy_gt: Tuple[Any, Any]
) -> None:
    """Workflow B: Benchmarking a novel target assigner callable."""
    model = RTMDet(arch="tiny", num_classes=80)
    model.train()
    outputs = model(dummy_input_320, training_override=True)
    gt_boxes, gt_labels = dummy_gt

    assigner_invoked = 0

    # Researcher defines their custom assigner function conforming to 4-tensor contract
    def my_novel_assigner(
        pred_cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        centers: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        num_classes: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal assigner_invoked
        assigner_invoked += 1

        A = pred_cls_logits.shape[0]
        device = pred_cls_logits.device
        # Assign background by default (num_classes)
        labels = torch.full((A,), num_classes, dtype=torch.long, device=device)
        label_weights = torch.ones(A, device=device)
        bbox_targets = torch.zeros((A, 4), device=device)
        assign_metrics = torch.zeros(A, device=device)

        # Match first anchor to first ground truth box
        if gt_boxes.numel() > 0:
            labels[0] = gt_labels[0]
            bbox_targets[0] = gt_boxes[0]
            assign_metrics[0] = 0.95

        return labels, label_weights, bbox_targets, assign_metrics

    # Inject into det_loss
    total_loss, loss_dict = det_loss(
        model_out=outputs,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
        assigner=my_novel_assigner,
    )

    assert assigner_invoked == dummy_input_320.shape[0]
    assert torch.isfinite(total_loss)
    total_loss.backward()


def test_workflow_c_custom_loss_injection(
    dummy_input_320: torch.Tensor, dummy_gt: Tuple[Any, Any]
) -> None:
    """Workflow C: Benchmarking a custom loss function."""
    model = RTMDet(arch="tiny", num_classes=80)
    model.train()
    outputs = model(dummy_input_320, training_override=True)
    gt_boxes, gt_labels = dummy_gt

    # Researcher writes custom functional loss
    def custom_cls_loss(pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        # Simple focal-weighted BCE loss
        prob = pred_logits.sigmoid()
        bce = -(
            target_scores * torch.log(prob + 1e-7)
            + (1.0 - target_scores) * torch.log(1.0 - prob + 1e-7)
        )
        weight = torch.abs(target_scores - prob)
        return weight * bce

    total_loss, loss_dict = det_loss(
        model_out=outputs,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
        cls_loss=custom_cls_loss,
        box_loss="ciou",
    )

    assert torch.isfinite(total_loss)
    total_loss.backward()


class InMemorySyntheticDataset(Dataset):
    """In-memory dataset for lightning-fast training backtesting without disk I/O."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        return {
            "img": torch.randn(3, 320, 320),
            "gt_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "gt_labels": torch.tensor([0]),
            "ori_shapes": (320, 320),
            "img_ids": idx,
        }


def synthetic_collate_fn(batch: list) -> dict:
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "ori_shapes": [b["ori_shapes"] for b in batch],
        "img_ids": [b["img_ids"] for b in batch],
    }


def test_workflow_d_custom_model_paper_training(tmp_path) -> None:
    """Workflow D: Building a custom model from scratch and training with generic_train."""
    # 1. Researcher defines their custom detector
    custom_model = RTMDet(arch="tiny", num_classes=80)

    # 2. Researcher creates lightweight config
    cfg = BaseTrainConfig(
        exp_name="custom_paper_detector",
        work_dir=str(tmp_path / "work_dirs"),
        optim=OptimConfig(epochs=2, lr=1e-3, device="cpu", use_amp=False),
        data=DataConfig(batch_size=2, num_workers=0),
    )

    # 3. Researcher creates in-memory DataLoader (Dependency Injection)
    train_dataset = InMemorySyntheticDataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=synthetic_collate_fn,
    )

    # 4. Custom loss closure
    def custom_loss_fn(outputs: Any, targets: Any) -> Tuple[torch.Tensor, dict]:
        return det_loss(
            model_out=outputs,
            gt_boxes_list=targets["gt_boxes"],
            gt_labels_list=targets["gt_labels"],
        )

    # 5. Execute training with injected loader - Zero disk annotation files needed!
    generic_train(
        cfg=cfg,
        model=custom_model,
        loss_fn=custom_loss_fn,
        family_tag="custom_paper",
        train_loader=train_loader,
    )

    # 6. Verify training produced artifacts atomically
    last_ckpt = tmp_path / "work_dirs" / "last.pth"
    assert last_ckpt.is_file()

    # 7. Verify checkpoint can be loaded cleanly
    state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    assert state["family"] == "custom_paper"
    assert state["epoch"] == 2


def test_production_torch_jit_and_export_tracing(dummy_input_320: torch.Tensor) -> None:
    """Verify that export_mode=True allows seamless graph tracing and torch.export without NMS crashes."""
    model = RTMDet(arch="tiny", num_classes=80)
    model.eval()
    model.export_mode = True

    # In export mode, model emits static tensor tuple: (pred_boxes, pred_cls.sigmoid())
    with torch.no_grad():
        boxes, scores = model(dummy_input_320)
        assert boxes.ndim == 3 and scores.ndim == 3

        # 1. Deterministic torch.export (PyTorch 2.x native export standard)
        ep = torch.export.export(model, (dummy_input_320,))
        assert ep is not None
        assert len(ep.graph.nodes) > 100

        # 2. Legacy TorchScript tracing
        traced_model = torch.jit.trace(model, dummy_input_320)
        traced_boxes, traced_scores = traced_model(dummy_input_320)

        assert torch.allclose(boxes, traced_boxes, atol=1e-5)
        assert torch.allclose(scores, traced_scores, atol=1e-5)
