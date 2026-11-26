"""
Generic training runner for object detection models.
Provides build_optimizer and generic_train to coordinate the full training lifecycle,
eliminating duplication across model-family launchers.
"""

import copy
import json
import os
import random
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import DetDataset, det_collate
from engine import COCOEvaluator, ModelEMA, build_lr_scheduler
from runners.det_runner import load_checkpoint, save_checkpoint, train_one_epoch, val_one_epoch
from transforms import build_det_transform


def build_optimizer(
    model: nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 0.05,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with bias/norm parameter group separation.

    Parameters with ndim <= 1 or whose names end with '.bias' receive zero
    weight decay; all remaining parameters receive the specified weight_decay.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

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


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_path(data_root: str, sub_or_abs: str) -> str:
    """Resolve a dataset path relative to data_root unless already absolute or existing."""
    if os.path.isabs(sub_or_abs) or os.path.exists(sub_or_abs):
        return sub_or_abs
    return os.path.join(data_root, sub_or_abs)


def extract_state_dict(ckpt: Any) -> Dict[str, Any]:
    """Extract a clean model state_dict from a checkpoint payload."""
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            ema_val = ckpt["ema"]
            sd = (
                ema_val["ema"] if "ema" in ema_val and isinstance(ema_val["ema"], dict) else ema_val
            )
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _load_pretrained_weights(
    model: nn.Module,
    weights_path: str,
    device: torch.device,
    strict: bool = True,
) -> None:
    """Load model weights from a checkpoint file with optional shape filtering."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights checkpoint not found: {weights_path}")
    ckpt = load_checkpoint(weights_path, device=device, weights_only=False)
    sd = extract_state_dict(ckpt)
    if strict:
        model.load_state_dict(sd, strict=True)
        print(f"--> Loaded weights (strict=True) from {weights_path}")
        return

    model_sd = model.state_dict()
    filtered_sd: Dict[str, Any] = {}
    skipped_shape = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        filtered_sd[k] = v

    incompatible = model.load_state_dict(filtered_sd, strict=False)
    print(
        f"--> Loaded pretrained weights from {weights_path} "
        f"(missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}, "
        f"skipped_shape={len(skipped_shape)})"
    )


def _build_val_loader_and_evaluator(
    cfg: Any,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    evaluator: Optional[Any] = None,
    val_transform: Optional[Callable] = None,
    require_exists: bool = False,
) -> Tuple[Optional[DataLoader], Optional[Any]]:
    """Construct validation DataLoader and COCOEvaluator when not injected."""
    if val_loader is None and hasattr(cfg, "data") and getattr(cfg.data, "val_ann", None):
        if val_transform is None:
            val_transform = build_det_transform(img_size=cfg.data.img_size, is_train=False)
        val_root = _resolve_data_path(cfg.data.data_root, cfg.data.val_img)
        val_ann = _resolve_data_path(cfg.data.data_root, cfg.data.val_ann)
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
                pin_memory=device.type == "cuda" and cfg.data.pin_memory,
            )
            if evaluator is None and getattr(cfg.eval, "compute_map", True):
                evaluator = COCOEvaluator(ann_file=val_ann)
        elif require_exists:
            raise FileNotFoundError(
                f"Validation dataset not found: val_root={val_root!r}, val_ann={val_ann!r}"
            )
    elif val_loader is not None and evaluator is None and getattr(cfg.eval, "compute_map", True):
        if hasattr(cfg, "data") and getattr(cfg.data, "val_ann", None):
            val_ann = _resolve_data_path(cfg.data.data_root, cfg.data.val_ann)
            if os.path.exists(val_ann):
                evaluator = COCOEvaluator(ann_file=val_ann)

    return val_loader, evaluator


def generic_train(
    cfg: Any,
    model: nn.Module,
    loss_fn: Callable,
    family_tag: str = "detector",
    optimizer: Optional[torch.optim.Optimizer] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    evaluator: Optional[Any] = None,
    cooldown_loader: Optional[DataLoader] = None,
) -> None:
    """Coordinate the full training lifecycle for a detection model.

    Handles seeding, dataset and DataLoader construction, optimizer fallback,
    LR scheduler, EMA, GradScaler, checkpoint resume, cooldown transition,
    epoch loop with training and validation, and checkpoint saving.
    Supports direct dependency injection of train_loader, val_loader, and evaluator.

    Args:
        cfg: Training configuration object with nested data, optim, eval, and model attrs.
        model: Detection model already placed on the target device.
        loss_fn: Callable(outputs, targets) -> (total_loss, loss_dict).
        family_tag: Short identifier stored in checkpoints (e.g. 'rtmdet').
        optimizer: Pre-built optimizer; if None, built via build_optimizer using cfg.optim.
        train_loader: Optional pre-built DataLoader for training.
        val_loader: Optional pre-built DataLoader for validation.
        evaluator: Optional metric evaluator (e.g. COCOEvaluator).
        cooldown_loader: Optional pre-built DataLoader for augmentation cooldown epochs.
    """
    # -- Seeding --
    seed = getattr(cfg, "seed", 42)
    set_seed(seed)

    work_dir = getattr(cfg, "work_dir", "work_dirs")
    os.makedirs(work_dir, exist_ok=True)
    device = torch.device(cfg.optim.device)

    # -- Dataset & DataLoaders fallback --
    cooldown_epochs = getattr(cfg.optim, "cooldown_epochs", 0)
    if train_loader is None:
        train_root = _resolve_data_path(cfg.data.data_root, cfg.data.train_img)
        train_ann = _resolve_data_path(cfg.data.data_root, cfg.data.train_ann)

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
            pin_memory=device.type == "cuda" and cfg.data.pin_memory,
            drop_last=False,
        )

        if cooldown_epochs > 0 and cooldown_loader is None:
            cooldown_ds = copy.copy(train_dataset)
            cooldown_ds.transform = val_transform
            cooldown_loader = DataLoader(
                cooldown_ds,
                batch_size=cfg.data.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                collate_fn=det_collate,
                pin_memory=device.type == "cuda" and cfg.data.pin_memory,
                drop_last=False,
            )

    val_loader, evaluator = _build_val_loader_and_evaluator(
        cfg=cfg,
        device=device,
        val_loader=val_loader,
        evaluator=evaluator,
        require_exists=False,
    )

    # -- Optional weights-only transfer-learning initialization --
    load_from_path = getattr(cfg.optim, "load_from", None)
    resume_path = getattr(cfg.optim, "resume", None)
    if load_from_path and not resume_path:
        _load_pretrained_weights(model, load_from_path, device=device, strict=False)

    # -- Optimizer fallback --
    if optimizer is None:
        optimizer = build_optimizer(model, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    # -- Scheduler, EMA, Scaler --
    normal_epochs = max(0, cfg.optim.epochs - cooldown_epochs)
    cd_epochs = min(cfg.optim.epochs, cooldown_epochs)
    cd_len = len(cooldown_loader) if cooldown_loader is not None else len(train_loader)
    total_steps = normal_epochs * len(train_loader) + cd_epochs * cd_len
    warmup_steps = cfg.optim.warmup_epochs * len(train_loader)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=getattr(cfg.optim, "min_lr_ratio", 0.05),
    )
    ema = ModelEMA(model, decay=getattr(cfg.optim, "ema_decay", 0.9998))
    use_amp = getattr(cfg.optim, "use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -- Resume checkpoint --
    start_epoch = 1
    current_step = 0
    best_loss = float("inf")
    best_metric = 0.0

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_checkpoint(resume_path, device=device, weights_only=False)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            if "ema" in ckpt and ckpt["ema"] is not None:
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
            print(f"--> Resumed from {resume_path} at epoch {start_epoch}, step {current_step}")
            del ckpt

    # -- Cooldown transition --
    cooldown_epoch = (
        (cfg.optim.epochs - cooldown_epochs + 1) if cooldown_epochs > 0 else cfg.optim.epochs + 1
    )
    current_train_loader = (
        cooldown_loader
        if cooldown_loader is not None and start_epoch >= cooldown_epoch
        else train_loader
    )

    # -- Epoch loop --
    metrics_path = os.path.join(work_dir, "metrics.jsonl")
    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        epoch_start_t = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

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
            clip_max_norm=getattr(cfg.optim, "clip_max_norm", 35.0),
            use_amp=use_amp,
        )
        print(f"--> Epoch [{epoch}/{cfg.optim.epochs}] Training Loss: {mean_loss:.4f}")

        # -- Validation --
        is_best = False
        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            if epoch % getattr(cfg.eval, "eval_interval", 1) == 0 or epoch == cfg.optim.epochs:
                eval_model = ema.module if ema is not None else model
                val_metrics = val_one_epoch(
                    eval_model, val_loader, device=device, evaluator=evaluator
                )
                print(f"--> Epoch [{epoch}/{cfg.optim.epochs}] Validation Metrics: {val_metrics}")
                if "mAP_50_95" in val_metrics or "mAP_50" in val_metrics:
                    mAP = val_metrics.get("mAP_50_95", val_metrics.get("mAP_50", 0.0))
                    is_best = mAP > best_metric
                    if is_best:
                        best_metric = mAP
                elif "val_loss" in val_metrics:
                    val_loss = val_metrics["val_loss"]
                    is_best = val_loss < best_loss
                    if is_best:
                        best_loss = val_loss
                else:
                    is_best = mean_loss < best_loss
                    if is_best:
                        best_loss = mean_loss
            else:
                is_best = False
        else:
            is_best = mean_loss < best_loss
            if is_best:
                best_loss = mean_loss

        epoch_time_s = time.perf_counter() - epoch_start_t
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "step": current_step,
            "train_loss": round(float(mean_loss), 6),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_s": round(epoch_time_s, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
        }
        for k, v in val_metrics.items():
            record[k] = round(float(v), 6)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # -- Checkpoint saving --
        ckpt = {
            "epoch": epoch,
            "step": current_step,
            "family": family_tag,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_loss": best_loss,
            "best_metric": best_metric,
            "model_config": asdict(cfg.model) if hasattr(cfg, "model") else None,
            "data_config": asdict(cfg.data) if hasattr(cfg, "data") else None,
        }
        save_checkpoint(ckpt, os.path.join(work_dir, "last.pth"))
        if is_best:
            save_checkpoint(ckpt, os.path.join(work_dir, "best.pth"))


def generic_eval(
    cfg: Any,
    model: nn.Module,
    weights_path: Optional[str] = None,
    family_tag: str = "detector",
    val_loader: Optional[DataLoader] = None,
    evaluator: Optional[Any] = None,
    val_transform: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Evaluate a detection model checkpoint and record COCO metrics, latency, and VRAM.

    Args:
        cfg: Evaluation/training configuration with nested data, optim, eval, and model attrs.
        model: Detection model already placed on the target device.
        weights_path: Optional path to a .pth checkpoint to load before evaluation.
        family_tag: Short identifier stored in eval_metrics.json (e.g. 'rtmdet').
        val_loader: Optional pre-built DataLoader for validation.
        evaluator: Optional metric evaluator (e.g. COCOEvaluator).
        val_transform: Optional validation transform override (e.g. BGR [0, 255] for YOLOX).

    Returns:
        Dictionary containing COCO metrics, image count, elapsed time, ms/img, and peak VRAM.
    """
    set_seed(getattr(cfg, "seed", 42))
    work_dir = getattr(cfg, "work_dir", "work_dirs")
    os.makedirs(work_dir, exist_ok=True)
    device = torch.device(cfg.optim.device)

    if weights_path:
        _load_pretrained_weights(model, weights_path, device=device, strict=True)

    val_loader, evaluator = _build_val_loader_and_evaluator(
        cfg=cfg,
        device=device,
        val_loader=val_loader,
        evaluator=evaluator,
        val_transform=val_transform,
        require_exists=True,
    )
    if val_loader is None:
        raise ValueError("Validation DataLoader could not be constructed; check cfg.data settings.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_t = time.perf_counter()
    val_metrics = val_one_epoch(model, val_loader, device=device, evaluator=evaluator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start_t

    num_images = len(val_loader.dataset)
    ms_per_img = (elapsed_s / max(num_images, 1)) * 1000.0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    results: Dict[str, Any] = {
        "family": family_tag,
        "weights": weights_path,
        "num_images": num_images,
        "elapsed_s": round(elapsed_s, 3),
        "ms_per_img": round(ms_per_img, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    for k, v in val_metrics.items():
        results[k] = round(float(v), 6)

    out_path = os.path.join(work_dir, "eval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"--> Evaluation Metrics ({family_tag}): {val_metrics}")
    print(
        f"--> Benchmark: {num_images} images in {elapsed_s:.2f}s "
        f"({ms_per_img:.2f} ms/img, peak_vram={peak_vram_mb:.1f} MB)"
    )
    print(f"--> Saved evaluation summary to {out_path}")
    return results
