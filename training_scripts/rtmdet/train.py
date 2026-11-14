"""
Production training launcher for RTMDet.
Handles optimizer grouping, learning rate scheduling, runner invocation, and experiment lifecycle.
Direct imports matching modular standards.
"""

import argparse
import os
import random
import sys
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
from losses import det_loss
from models import RTMDet
from runners import save_checkpoint, train_one_epoch, val_one_epoch
from training_scripts.rtmdet.config import TrainConfig
from transforms import build_det_transform


def build_optimizer(
    model: nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 0.05,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with bias/norm parameter group separation."""
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
    """Execute complete RTMDet training lifecycle."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    os.makedirs(cfg.work_dir, exist_ok=True)
    device = torch.device(cfg.optim.device)

    print("==================================================")
    print(f"Initializing RTMDet Training: {cfg.exp_name} (arch={cfg.model.arch})")
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
        val_transform = build_det_transform(img_size=cfg.data.img_size, is_train=False)
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
    model = RTMDet(
        arch=cfg.model.arch,
        num_classes=cfg.model.num_classes,
        score_thresh=cfg.model.score_thresh,
        nms_thresh=cfg.model.nms_thresh,
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

    def loss_fn(outputs, targets):
        return det_loss(
            model_out=outputs,
            gt_boxes_list=targets["gt_boxes"],
            gt_labels_list=targets["gt_labels"],
            cls_loss=cfg.model.cls_loss,
            box_loss=cfg.model.box_loss,
        )

    # 4. Training Loop
    current_step = 0
    best_loss = float("inf")
    best_metric = 0.0

    for epoch in range(1, cfg.optim.epochs + 1):
        mean_loss, current_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
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
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        last_ckpt_path = os.path.join(cfg.work_dir, "last.pth")
        save_checkpoint(ckpt, last_ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(cfg.work_dir, "best.pth")
            save_checkpoint(ckpt, best_ckpt_path)

    print(f"\nRTMDet training completed successfully! Artifacts in: {cfg.work_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LucidDet RTMDet Training Launcher")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig()
    if args.epochs is not None:
        config.optim.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optim.lr = args.lr
    run_training(config)
