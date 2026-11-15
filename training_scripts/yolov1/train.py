"""
Production training launcher for YOLOv1.
Handles optimizer grouping, learning rate scheduling, runner invocation, and experiment lifecycle.
Direct imports matching modular standards.
"""

import argparse
import copy
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DetDataset, det_collate
from engine import COCOEvaluator, ModelEMA, build_lr_scheduler
from losses import YOLOv1Loss, encode_yolov1_targets
from models import YOLOv1
from runners import load_checkpoint, save_checkpoint, train_one_epoch, val_one_epoch
from training_scripts.common import add_common_training_args, apply_common_training_args
from training_scripts.yolov1.config import TrainConfig
from transforms import build_det_transform


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
) -> torch.optim.Optimizer:
    """Build optimizer with bias/norm parameter group separation."""
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr)


def run_training(cfg: TrainConfig) -> None:
    """Execute complete YOLOv1 training lifecycle."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    os.makedirs(cfg.work_dir, exist_ok=True)
    device = torch.device(cfg.optim.device)

    print("==================================================")
    print(f"Initializing YOLOv1 Training: {cfg.exp_name} (classes={cfg.model.num_classes})")
    print(f"Device: {device} | Epochs: {cfg.optim.epochs} | Base LR: {cfg.optim.lr:.2e}")
    print(f"Work dir: {cfg.work_dir}")
    print("==================================================")

    # 1. Dataset & DataLoader
    train_root = (
        os.path.join(cfg.data.data_root, cfg.data.train_img)
        if not os.path.isabs(cfg.data.train_img) and not os.path.exists(cfg.data.train_img)
        else cfg.data.train_img
    )
    train_ann = (
        os.path.join(cfg.data.data_root, cfg.data.train_ann)
        if not os.path.isabs(cfg.data.train_ann) and not os.path.exists(cfg.data.train_ann)
        else cfg.data.train_ann
    )

    train_transform = build_det_transform(img_size=cfg.data.img_size, is_train=True)
    val_transform = build_det_transform(img_size=cfg.data.img_size, is_train=False)

    train_dataset = DetDataset(
        root=train_root,
        ann_file=train_ann,
        transform=train_transform,
        num_classes=cfg.data.num_classes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=det_collate,
        pin_memory=(device.type == "cuda" and cfg.data.pin_memory),
        drop_last=False,
    )

    cooldown_loader = None
    cooldown_epochs = getattr(cfg.optim, "cooldown_epochs", 0)
    if cooldown_epochs > 0:
        cooldown_ds = copy.copy(train_dataset)
        cooldown_ds.transform = val_transform
        cooldown_loader = DataLoader(
            cooldown_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=det_collate,
            pin_memory=(device.type == "cuda" and cfg.data.pin_memory),
            drop_last=False,
        )

    val_loader = None
    evaluator = None
    val_root = (
        os.path.join(cfg.data.data_root, cfg.data.val_img)
        if not os.path.isabs(cfg.data.val_img) and not os.path.exists(cfg.data.val_img)
        else cfg.data.val_img
    )
    val_ann = (
        os.path.join(cfg.data.data_root, cfg.data.val_ann)
        if not os.path.isabs(cfg.data.val_ann) and not os.path.exists(cfg.data.val_ann)
        else cfg.data.val_ann
    )
    if os.path.exists(val_root) and os.path.exists(val_ann):
        val_dataset = DetDataset(
            root=val_root,
            ann_file=val_ann,
            transform=val_transform,
            num_classes=cfg.data.num_classes,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.val_batch_size or cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=det_collate,
            pin_memory=(device.type == "cuda" and cfg.data.pin_memory),
        )
        if cfg.eval.compute_map:
            evaluator = COCOEvaluator(ann_file=val_ann)

    # 2. Model & Optimizer
    model = YOLOv1(
        num_classes=cfg.model.num_classes,
        score_thresh=cfg.model.score_thresh,
        nms_thresh=cfg.model.nms_thresh,
        split_size=cfg.model.split_size,
        num_boxes=cfg.model.num_boxes,
    ).to(device)

    optimizer = build_optimizer(model, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    # 3. Scheduler, EMA, Scaler
    total_steps = cfg.optim.epochs * len(train_loader)
    warmup_steps = cfg.optim.warmup_epochs * len(train_loader)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=cfg.optim.min_lr_ratio,
    )
    ema = ModelEMA(model, decay=cfg.optim.ema_decay)
    use_amp = cfg.optim.use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    yolo_loss_fn = YOLOv1Loss(
        split_size=cfg.model.split_size,
        num_boxes=cfg.model.num_boxes,
        num_classes=cfg.model.num_classes,
    )
    img_sz = float(
        cfg.data.img_size[0] if isinstance(cfg.data.img_size, tuple) else cfg.data.img_size
    )

    def loss_fn(outputs, targets):
        if isinstance(targets, dict) and "gt_boxes" in targets:
            targets_matrix = encode_yolov1_targets(
                gt_boxes=targets["gt_boxes"],
                gt_labels=targets["gt_labels"],
                split_size=cfg.model.split_size,
                num_classes=cfg.model.num_classes,
                img_size=img_sz,
                device=device,
            )
        else:
            b_size = outputs.shape[0] if hasattr(outputs, "shape") else 1
            targets_matrix = torch.zeros(
                b_size,
                cfg.model.split_size,
                cfg.model.split_size,
                cfg.model.num_classes + 5,
                device=device,
            )
        return yolo_loss_fn(outputs, targets_matrix)

    # 4. Training Loop & Resume
    start_epoch = 1
    current_step = 0
    best_loss = float("inf")
    best_metric = 0.0

    resume_path = getattr(cfg.optim, "resume", None)
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_checkpoint(resume_path, device=device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
                ema.load_state_dict(ckpt["ema"])
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if (
                lr_scheduler is not None
                and "lr_scheduler" in ckpt
                and ckpt["lr_scheduler"] is not None
            ):
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            current_step = ckpt.get("step", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            best_metric = ckpt.get("best_metric", 0.0)
            print(
                f"--> Resumed successfully from {cfg.optim.resume} "
                f"at epoch {start_epoch}, step {current_step}"
            )
            del ckpt

    cooldown_epoch = (
        (cfg.optim.epochs - cooldown_epochs + 1) if cooldown_epochs > 0 else cfg.optim.epochs + 1
    )
    current_train_loader = (
        cooldown_loader
        if (cooldown_loader is not None and start_epoch >= cooldown_epoch)
        else train_loader
    )

    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        if epoch == cooldown_epoch and cooldown_loader is not None:
            print(f"\n[Epoch {epoch}] Augmentation Cool-Down: Switched to clean DataLoader.")
            current_train_loader = cooldown_loader

        mean_loss, current_step = train_one_epoch(
            model=model,
            dataloader=current_train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            epochs=cfg.optim.epochs,
            current_step=current_step,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            ema=ema,
            clip_max_norm=cfg.optim.clip_max_norm,
            use_amp=use_amp,
        )
        print(f"--> Epoch [{epoch}/{cfg.optim.epochs}] Training Loss: {mean_loss:.4f}")

        # Validation
        if val_loader is not None and (
            epoch % cfg.eval.eval_interval == 0 or epoch == cfg.optim.epochs
        ):
            eval_model = ema.module if ema is not None else model
            val_metrics = val_one_epoch(eval_model, val_loader, device=device, evaluator=evaluator)
            print(f"--> Epoch [{epoch}/{cfg.optim.epochs}] Validation Metrics: {val_metrics}")
            mAP = val_metrics.get("mAP_50_95", val_metrics.get("mAP_50", 0.0))
            is_best = mAP > best_metric
            if is_best:
                best_metric = mAP
        else:
            is_best = mean_loss < best_loss
            if is_best:
                best_loss = mean_loss

        # Checkpoint saving
        ckpt = {
            "epoch": epoch,
            "step": current_step,
            "family": "yolov1",
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "best_loss": best_loss,
            "best_metric": best_metric,
            "model_config": asdict(cfg.model),
            "data_config": asdict(cfg.data),
        }
        last_ckpt_path = os.path.join(cfg.work_dir, "last.pth")
        save_checkpoint(ckpt, last_ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(cfg.work_dir, "best.pth")
            save_checkpoint(ckpt, best_ckpt_path)

    print(f"\nYOLOv1 training completed successfully! Artifacts in: {cfg.work_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet YOLOv1 Training Launcher")
    add_common_training_args(parser)
    parser.add_argument(
        "--split-size",
        type=int,
        default=None,
        help="Override grid split size S (default: 7)",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        choices=[2],
        default=None,
        help="Override boxes per cell B (default: 2, currently only B=2 supported by YOLOv1Loss)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig()
    apply_common_training_args(config, args)
    if args.split_size is not None:
        config.model.split_size = args.split_size
    if args.num_boxes is not None:
        if args.num_boxes != 2:
            raise ValueError(
                f"YOLOv1Loss currently supports only --num-boxes 2, got {args.num_boxes}"
            )
        config.model.num_boxes = args.num_boxes
    run_training(config)
