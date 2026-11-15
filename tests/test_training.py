"""
Integration test suite for detector training loops, gradient flow, and checkpointing.
Verifies real-batch forward, loss computation, backward passes, EMA weight adaptation,
full checkpoint contract validation, and checkpoint resume behavior across all three
detector families (RTMDet, YOLOX, YOLOv1) on COCO-8.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from datasets.det_dataset import DetDataset, det_collate
from engine.ema import ModelEMA
from engine.lr_scheduler import build_lr_scheduler
from losses.det_loss import det_loss
from losses.yolov1_loss import YOLOv1Loss, encode_yolov1_targets
from models import YOLOX, RTMDet, YOLOv1
from runners.det_runner import load_checkpoint, save_checkpoint
from tools.infer import build_family_model, extract_checkpoint_payload, load_weights
from transforms.det_transforms import build_det_transform

COCO8_ROOT = Path("data/coco8/images/train")
COCO8_ANN = Path("data/coco8/annotations/instances_train.json")


def get_coco8_batch(img_size: int = 640, batch_size: int = 2) -> Dict[str, Any]:
    transform = build_det_transform(img_size=img_size, is_train=False)
    dataset = DetDataset(
        root=str(COCO8_ROOT),
        ann_file=str(COCO8_ANN),
        transform=transform,
        num_classes=80,
    )
    items = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    return det_collate(items)


def test_rtmdet_training_step() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RTMDet(arch="tiny", num_classes=80).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, decay=0.99)

    batch = get_coco8_batch(img_size=640, batch_size=2)
    imgs = batch["img"].to(device)
    gt_boxes = [b.to(device) for b in batch["gt_boxes"]]
    gt_labels = [lbl.to(device) for lbl in batch["gt_labels"]]

    outputs = model(imgs, training_override=True)
    loss, parts = det_loss(
        model_out=outputs,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Snapshot parameter values before optimizer step and EMA update
    p_model_before = next(model.parameters()).clone()
    p_ema_before = next(ema.module.parameters()).clone()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Model parameters must have updated after optimizer step
    p_model_after = next(model.parameters())
    assert not torch.allclose(p_model_after, p_model_before), (
        "Optimizer step did not alter model parameters."
    )

    ema.update(model)

    # EMA shadow parameters must have adapted towards the updated model parameters
    p_ema_after = next(ema.module.parameters())
    assert not torch.allclose(p_ema_after, p_ema_before), (
        "EMA update did not alter shadow parameters."
    )

    # Verify EMA shadow module tracks model parameter shapes
    for p_model, p_ema in zip(model.parameters(), ema.module.parameters()):
        assert p_ema.shape == p_model.shape


def test_yolox_training_step() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOX(arch="s", num_classes=80).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = get_coco8_batch(img_size=640, batch_size=2)
    imgs = batch["img"].to(device)
    gt_boxes = [b.to(device) for b in batch["gt_boxes"]]
    gt_labels = [lbl.to(device) for lbl in batch["gt_labels"]]

    outputs = model(imgs, training_override=True)
    loss, parts = det_loss(
        model_out=outputs,
        gt_boxes_list=gt_boxes,
        gt_labels_list=gt_labels,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    p_before = next(model.parameters()).clone()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    p_after = next(model.parameters())
    assert not torch.allclose(p_after, p_before)


def test_yolov1_training_step() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 640.0
    num_classes = 80
    S = 7
    model = YOLOv1(num_classes=num_classes, img_size=img_size, split_size=S).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    batch = get_coco8_batch(img_size=int(img_size), batch_size=2)
    imgs = batch["img"].to(device)
    gt_boxes = [b.to(device) for b in batch["gt_boxes"]]
    gt_labels = [lbl.to(device) for lbl in batch["gt_labels"]]

    outputs = model(imgs, training_override=True)

    # Build cell-scaled YOLOv1 targets matrix
    targets_matrix = encode_yolov1_targets(
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        split_size=S,
        num_classes=num_classes,
        img_size=img_size,
        device=device,
    )

    loss_fn = YOLOv1Loss(split_size=S, num_boxes=2, num_classes=num_classes)
    loss, parts = loss_fn(outputs, targets_matrix)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    p_before = next(model.parameters()).clone()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    p_after = next(model.parameters())
    assert not torch.allclose(p_after, p_before)


@pytest.mark.parametrize(
    "family,model_cls,model_cfg,data_cfg",
    [
        (
            "rtmdet",
            RTMDet,
            {"arch": "tiny", "num_classes": 80, "score_thresh": 0.05, "nms_thresh": 0.65},
            {"data_root": "data/coco8", "num_classes": 80, "img_size": (640, 640), "batch_size": 4},
        ),
        (
            "yolox",
            YOLOX,
            {"arch": "s", "num_classes": 80, "score_thresh": 0.05, "nms_thresh": 0.65},
            {"data_root": "data/coco8", "num_classes": 80, "img_size": (640, 640), "batch_size": 4},
        ),
        (
            "yolov1",
            YOLOv1,
            {"arch": "darknet24", "num_classes": 20, "split_size": 7, "num_boxes": 2},
            {"data_root": "data/coco8", "num_classes": 20, "img_size": (448, 448), "batch_size": 4},
        ),
    ],
)
def test_checkpoint_contract_and_inference_reconstruction(
    tmp_path: Path,
    family: str,
    model_cls: type,
    model_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
) -> None:
    """Verifies that checkpoints adhere to the self-describing contract and that

    tools/infer.py can rebuild and strictly load the detector from checkpoint metadata alone.
    """
    if family == "rtmdet":
        model = RTMDet(arch=model_cfg["arch"], num_classes=model_cfg["num_classes"])
    elif family == "yolox":
        model = YOLOX(arch=model_cfg["arch"], num_classes=model_cfg["num_classes"])
    else:
        model = YOLOv1(
            num_classes=model_cfg["num_classes"],
            split_size=model_cfg["split_size"],
            num_boxes=model_cfg["num_boxes"],
            img_size=float(data_cfg["img_size"][0]),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, decay=0.99)
    ckpt_path = tmp_path / f"test_{family}_contract.pth"

    contract_payload = {
        "epoch": 5,
        "step": 120,
        "family": family,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": 2.15,
        "best_metric": 0.48,
        "model_config": model_cfg,
        "data_config": data_cfg,
    }
    save_checkpoint(contract_payload, str(ckpt_path))
    assert ckpt_path.exists()

    # 1. Verify contract dictionary keys and values
    loaded_ckpt = load_checkpoint(str(ckpt_path), device="cpu")
    assert loaded_ckpt is not None
    required_keys = [
        "epoch",
        "step",
        "family",
        "model",
        "ema",
        "optimizer",
        "best_loss",
        "best_metric",
        "model_config",
        "data_config",
    ]
    for key in required_keys:
        assert key in loaded_ckpt, f"Missing promised contract key: '{key}'"

    assert loaded_ckpt["family"] == family
    assert loaded_ckpt["epoch"] == 5
    assert loaded_ckpt["step"] == 120
    assert loaded_ckpt["best_loss"] == 2.15
    assert loaded_ckpt["best_metric"] == 0.48
    assert isinstance(loaded_ckpt["ema"], dict)
    assert "ema" in loaded_ckpt["ema"]

    # 2. Verify tools/infer.py reconstruction without manual CLI flags
    sd, extracted_m_cfg, extracted_d_cfg, family_hint = extract_checkpoint_payload(
        str(ckpt_path), device=torch.device("cpu")
    )
    assert family_hint == family
    assert extracted_m_cfg == model_cfg
    assert extracted_d_cfg == data_cfg

    infer_args = argparse.Namespace(
        family=None,
        arch=None,
        num_classes=None,
        img_size=None,
        conf=0.3,
        split_size=None,
        num_boxes=None,
    )
    reconstructed_model = build_family_model(
        infer_args,
        model_cfg=extracted_m_cfg,
        data_cfg=extracted_d_cfg,
        family_hint=family_hint,
    )
    assert isinstance(reconstructed_model, model_cls)
    assert reconstructed_model.num_classes == model_cfg["num_classes"]

    # 3. Enforce strict weight loading (no missing or unexpected keys)
    loaded_detector = load_weights(reconstructed_model, sd, allow_partial=False)
    assert loaded_detector is not None


def test_checkpoint_resume_behavior(tmp_path: Path) -> None:
    """Verifies that resuming from a saved checkpoint restores exact model weights,
    EMA parameters, optimizer state, LR scheduler state, epoch offset, and best metrics.
    """
    model = RTMDet(arch="tiny", num_classes=80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, decay=0.99)
    lr_scheduler = build_lr_scheduler(optimizer, total_steps=100, warmup_steps=10)

    # Simulate 3 training updates
    for _ in range(3):
        x = torch.randn(2, 3, 320, 320)
        out = model(x, training_override=True)
        loss = out["pred_cls"].sum() + out["pred_boxes"].sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.update(model)

    ckpt_path = tmp_path / "resume_test.pth"
    payload = {
        "epoch": 3,
        "step": 15,
        "family": "rtmdet",
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "best_loss": 1.45,
        "best_metric": 0.35,
        "model_config": {"arch": "tiny", "num_classes": 80},
        "data_config": {"num_classes": 80},
    }
    save_checkpoint(payload, str(ckpt_path))

    # Reconstitute in a fresh instance
    resumed_ckpt = load_checkpoint(str(ckpt_path), device="cpu")
    assert resumed_ckpt is not None

    fresh_model = RTMDet(arch="tiny", num_classes=80)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_ema = ModelEMA(fresh_model, decay=0.99)
    fresh_scheduler = build_lr_scheduler(fresh_optimizer, total_steps=100, warmup_steps=10)

    fresh_model.load_state_dict(resumed_ckpt["model"])
    fresh_ema.load_state_dict(resumed_ckpt["ema"])
    fresh_optimizer.load_state_dict(resumed_ckpt["optimizer"])
    fresh_scheduler.load_state_dict(resumed_ckpt["lr_scheduler"])

    # Assert exact parameter equality across restored model and EMA
    for p_orig, p_restored in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p_orig, p_restored), "Restored model weights diverged."

    for p_orig, p_restored in zip(ema.module.parameters(), fresh_ema.module.parameters()):
        assert torch.equal(p_orig, p_restored), "Restored EMA weights diverged."

    # Assert exact LR scheduler state restoration
    assert fresh_scheduler.last_epoch == lr_scheduler.last_epoch
    assert fresh_scheduler.get_last_lr() == lr_scheduler.get_last_lr()

    assert resumed_ckpt["epoch"] + 1 == 4
    assert resumed_ckpt["step"] == 15
    assert resumed_ckpt["best_loss"] == 1.45
    assert resumed_ckpt["best_metric"] == 0.35


def test_model_ema_unwraps_wrapped_model() -> None:
    class DummyWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.module = inner

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    raw_model = torch.nn.Linear(10, 5)
    wrapped = DummyWrapper(raw_model)
    ema = ModelEMA(wrapped, decay=0.9)

    # Shadow should not have 'module.' prefix in parameter names
    ema_param_names = list(dict(ema.module.named_parameters()).keys())
    assert "weight" in ema_param_names
    assert "module.weight" not in ema_param_names

    # Update should update weights without skipping
    with torch.no_grad():
        raw_model.weight.fill_(10.0)
    ema.update(wrapped)
    assert not torch.allclose(ema.module.weight, torch.zeros_like(ema.module.weight))


def test_infer_payload_reads_converted_envelope(tmp_path: Path) -> None:
    from tools.infer import extract_checkpoint_payload

    dummy_sd = {"backbone.stem.conv.weight": torch.randn(8, 3, 3, 3)}
    envelope_meta = {
        "family": "rtmdet",
        "arch": "s",
        "num_classes": 80,
        "input_size": (640, 640),
    }
    ckpt_path = tmp_path / "converted_ckpt.pth"
    torch.save({"state_dict": dummy_sd, "meta": envelope_meta}, ckpt_path)

    sd, model_cfg, data_cfg, family = extract_checkpoint_payload(
        str(ckpt_path), torch.device("cpu")
    )
    assert family == "rtmdet"
    assert model_cfg == {"arch": "s"}
    assert data_cfg["num_classes"] == 80
    assert "backbone.stem.conv.weight" in sd


def test_all_detector_configs_support_cooldown() -> None:
    from training_scripts.common import add_common_training_args, apply_common_training_args
    from training_scripts.rtmdet.config import TrainConfig as RTMDetConfig
    from training_scripts.yolov1.config import TrainConfig as YOLOv1Config
    from training_scripts.yolox.config import TrainConfig as YOLOXConfig

    for config_cls in (RTMDetConfig, YOLOXConfig, YOLOv1Config):
        cfg = config_cls()
        assert hasattr(cfg.optim, "cooldown_epochs")
        assert cfg.optim.cooldown_epochs == 0

        parser = argparse.ArgumentParser()
        add_common_training_args(parser)
        args = parser.parse_args(["--cooldown-epochs", "10"])
        apply_common_training_args(cfg, args)
        assert cfg.optim.cooldown_epochs == 10
