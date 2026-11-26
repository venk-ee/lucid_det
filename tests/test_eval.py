"""
Unit and integration tests for generic_eval, per-model eval launchers,
metrics.jsonl logging, and load_from transfer-learning initialization.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.det_dataset import DetDataset, det_collate
from losses.det_loss import det_loss
from models import RTMDet
from runners import generic_eval, generic_train
from runners.det_runner import save_checkpoint
from training_scripts.base_config import BaseTrainConfig, DataConfig, EvalConfig, OptimConfig
from training_scripts.common import apply_common_eval_args
from training_scripts.rtmdet.config import TrainConfig as RTMDetTrainConfig
from training_scripts.rtmdet.eval import parse_args as parse_rtmdet_eval_args
from training_scripts.yolov1.config import TrainConfig as YOLOv1TrainConfig
from training_scripts.yolov1.eval import parse_args as parse_yolov1_eval_args
from training_scripts.yolox.config import TrainConfig as YOLOXTrainConfig
from training_scripts.yolox.eval import parse_args as parse_yolox_eval_args
from transforms.det_transforms import build_det_transform

COCO8_VAL_IMG = Path("data/coco8/images/val")
COCO8_VAL_ANN = Path("data/coco8/annotations/instances_val.json")


def test_eval_cli_parsers() -> None:
    """Verify per-model eval CLI parsers apply common and family-specific flags."""
    rtm_args = parse_rtmdet_eval_args(
        ["--weights", "w.pth", "--arch", "tiny", "--batch-size", "4", "--img-size", "320"]
    )
    rtm_cfg = apply_common_eval_args(RTMDetTrainConfig(), rtm_args)
    assert rtm_args.weights == "w.pth"
    assert rtm_args.arch == "tiny"
    assert rtm_cfg.data.val_batch_size == 4
    assert rtm_cfg.data.img_size == (320, 320)

    yolox_args = parse_yolox_eval_args(
        ["--arch", "s", "--num-classes", "20", "--score-thresh", "0.1"]
    )
    yolox_cfg = apply_common_eval_args(YOLOXTrainConfig(), yolox_args)
    assert yolox_args.arch == "s"
    assert yolox_cfg.model.num_classes == 20
    assert yolox_cfg.model.score_thresh == 0.1

    v1_args = parse_yolov1_eval_args(
        ["--split-size", "7", "--num-boxes", "2", "--nms-thresh", "0.5"]
    )
    v1_cfg = apply_common_eval_args(YOLOv1TrainConfig(), v1_args)
    assert v1_args.split_size == 7
    assert v1_args.num_boxes == 2
    assert v1_cfg.model.nms_thresh == 0.5


def test_generic_eval_and_checkpoint_loading(tmp_path: Path) -> None:
    """Verify generic_eval loads EMA/model checkpoints and writes eval_metrics.json."""
    model = RTMDet(arch="tiny", num_classes=80)
    ckpt_path = tmp_path / "ema_ckpt.pth"
    save_checkpoint(
        {"ema": {"ema": model.state_dict(), "updates": 1, "decay": 0.99}}, str(ckpt_path)
    )

    val_ds = DetDataset(
        root=str(COCO8_VAL_IMG),
        ann_file=str(COCO8_VAL_ANN),
        transform=build_det_transform(img_size=320, is_train=False),
        num_classes=80,
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=det_collate)

    work_dir = tmp_path / "eval_out"
    cfg = BaseTrainConfig(
        work_dir=str(work_dir),
        data=DataConfig(
            data_root="data/coco8",
            val_img=str(COCO8_VAL_IMG),
            val_ann=str(COCO8_VAL_ANN),
            num_classes=80,
            img_size=(320, 320),
            batch_size=2,
            num_workers=0,
        ),
        optim=OptimConfig(device="cpu", use_amp=False),
        eval=EvalConfig(compute_map=True),
    )

    fresh_model = RTMDet(arch="tiny", num_classes=80)
    results = generic_eval(
        cfg=cfg,
        model=fresh_model,
        weights_path=str(ckpt_path),
        family_tag="rtmdet",
        val_loader=val_loader,
    )

    assert results["family"] == "rtmdet"
    assert results["num_images"] == len(val_ds)
    assert "ms_per_img" in results
    assert "mAP_50_95" in results
    saved = json.loads((work_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert saved["num_images"] == len(val_ds)


def test_generic_train_load_from_and_metrics_jsonl(tmp_path: Path) -> None:
    """Verify load_from handles class-count shape mismatch and generic_train writes metrics.jsonl."""
    # Save an 80-class checkpoint
    src_model = RTMDet(arch="tiny", num_classes=80)
    src_ckpt = tmp_path / "coco80.pth"
    save_checkpoint({"model": src_model.state_dict()}, str(src_ckpt))

    # Transfer into a 20-class model on a tiny 2-sample DataLoader
    dst_model = RTMDet(arch="tiny", num_classes=20)
    ds = DetDataset(
        root=str(COCO8_VAL_IMG),
        ann_file=str(COCO8_VAL_ANN),
        transform=build_det_transform(img_size=320, is_train=False),
        num_classes=20,
    )
    sub_ds = torch.utils.data.Subset(ds, [0, 1])
    loader = DataLoader(sub_ds, batch_size=2, shuffle=False, collate_fn=det_collate)

    work_dir = tmp_path / "train_out"
    cfg = BaseTrainConfig(
        work_dir=str(work_dir),
        data=DataConfig(num_classes=20, img_size=(320, 320), batch_size=2, num_workers=0),
        optim=OptimConfig(
            epochs=1,
            lr=1e-4,
            device="cpu",
            use_amp=False,
            warmup_epochs=0,
            load_from=str(src_ckpt),
        ),
        eval=EvalConfig(compute_map=False),
    )

    def loss_fn(outputs, targets):
        return det_loss(outputs, targets["gt_boxes"], targets["gt_labels"])

    generic_train(cfg, dst_model, loss_fn, family_tag="rtmdet", train_loader=loader)

    metrics_file = work_dir / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["epoch"] == 1
    assert "train_loss" in entry
    assert "epoch_time_s" in entry
