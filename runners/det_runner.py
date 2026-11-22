"""
Training and evaluation execution runner.
Clean detection runner design with AMP, ModelEMA, and validation cadence.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.utils import scale_boxes


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save training checkpoint atomically."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    temp_path = f"{path}.tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(
    path: str,
    device: Union[str, torch.device] = "cpu",
    weights_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint dictionary from disk.
    Defaults to weights_only=True for secure unpickling. Set weights_only=False
    when loading legacy checkpoints with complex pickled objects.
    """
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=weights_only)


def _move_to_device(item: Any, device: torch.device) -> Any:
    """Recursively move tensors (including those inside lists/tuples/dicts) to device."""
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, list):
        return [_move_to_device(x, device) for x in item]
    elif isinstance(item, tuple):
        return tuple(_move_to_device(x, device) for x in item)
    elif isinstance(item, dict):
        return {k: _move_to_device(v, device) for k, v in item.items()}
    return item


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., Tuple[torch.Tensor, Dict[str, float]]],
    device: Union[str, torch.device],
    epoch: int = 1,
    epochs: int = 50,
    current_step: int = 0,
    scaler: Optional[torch.amp.GradScaler] = None,
    lr_scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    clip_max_norm: float = 35.0,
    use_amp: bool = True,
) -> Tuple[float, int]:
    """
    Execute one training epoch with optional AMP, GradScaler, gradient clipping, and EMA.

    Args:
        model: Detection model nn.Module.
        dataloader: Training dataloader.
        optimizer: PyTorch optimizer.
        loss_fn: Multi-task loss computation function (returns total_loss, loss_dict).
        device: Execution device (e.g. 'cuda', 'cpu', or torch.device).
        epoch: Current epoch index.
        epochs: Total epochs.
        current_step: Global step counter.
        scaler: GradScaler for mixed precision training.
        lr_scheduler: Step-wise learning rate scheduler.
        ema: ModelEMA tracker instance.
        clip_max_norm: Max gradient norm for clipping.
        use_amp: Whether to use mixed precision.

    Returns:
        mean_loss: Average loss over epoch.
        current_step: Updated global step counter.
    """
    device = torch.device(device)
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    effective_amp = use_amp and (device.type == "cuda")
    amp_dtype = (
        torch.bfloat16 if (effective_amp and torch.cuda.is_bf16_supported()) else torch.float16
    )
    effective_scaler = scaler if (effective_amp and amp_dtype == torch.float16) else None

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", leave=False)
    for batch in pbar:
        current_step += 1
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, dict):
            images = batch["img"].to(device, non_blocking=True)
            batch_targets = {k: _move_to_device(v, device) for k, v in batch.items() if k != "img"}
        elif isinstance(batch, (tuple, list)):
            images = batch[0].to(device, non_blocking=True)
            batch_targets = _move_to_device(batch[1], device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=effective_amp):
            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, batch_targets)

        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected at step {current_step}: {loss_dict}")
            continue

        if effective_scaler is not None:
            effective_scaler.scale(loss).backward()
            if clip_max_norm > 0:
                effective_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            scale_before = effective_scaler.get_scale()
            effective_scaler.step(optimizer)
            effective_scaler.update()
            scale_after = effective_scaler.get_scale()
            stepped = scale_after >= scale_before
        else:
            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()
            stepped = True

        if lr_scheduler is not None and stepped:
            lr_scheduler.step()

        if ema is not None and hasattr(ema, "update") and stepped:
            ema.update(model)

        loss_val = loss.item()
        running_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    mean_loss = running_loss / max(num_batches, 1)
    return mean_loss, current_step


def val_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device],
    evaluator: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Run evaluation loop over validation dataloader.

    Args:
        model: Model in evaluation mode.
        dataloader: Validation dataloader.
        device: Device to run evaluation on (str or torch.device).
        evaluator: Optional evaluator instance (e.g. COCO evaluator).

    Returns:
        metrics: Dictionary of metric scores (e.g. mAP_50_95, mAP_50).
    """
    device = torch.device(device)
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            if isinstance(batch, dict):
                images = batch["img"].to(device, non_blocking=True)
                img_ids = batch.get(
                    "img_ids", [batch_idx * len(images) + i for i in range(len(images))]
                )
            elif isinstance(batch, (tuple, list)):
                images = batch[0].to(device, non_blocking=True)
                img_ids = [batch_idx * len(images) + i for i in range(len(images))]
            else:
                continue

            outputs = model(images)
            batch_dets = None
            if isinstance(outputs, dict) and "detections" in outputs:
                batch_dets = outputs["detections"]
            elif hasattr(outputs, "detections") and getattr(outputs, "detections") is not None:
                batch_dets = getattr(outputs, "detections")

            if batch_dets is not None:
                img_h, img_w = images.shape[-2:]
                ori_shapes = (
                    batch.get("ori_shapes", [(img_h, img_w)] * len(images))
                    if isinstance(batch, dict)
                    else [(img_h, img_w)] * len(images)
                )
                dataset = getattr(dataloader, "dataset", None)
                label2cat = getattr(dataset, "label2cat", None)
                if not label2cat and evaluator is not None:
                    label2cat = getattr(evaluator, "label2cat", {})
                if not label2cat:
                    label2cat = {}

                for img_id, det, ori_shape in zip(img_ids, batch_dets, ori_shapes):
                    boxes = det["boxes"]
                    scores = det["scores"]
                    labels = det["labels"]
                    if boxes.numel() > 0:
                        boxes = scale_boxes(boxes, (img_h, img_w), ori_shape)

                    for b, s, lbl in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = b.tolist()
                        img_id_val = (
                            int(img_id) if not hasattr(img_id, "item") else int(img_id.item())
                        )
                        lbl_val = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)
                        cat_id = label2cat.get(lbl_val, lbl_val + 1)
                        all_predictions.append(
                            {
                                "image_id": img_id_val,
                                "category_id": cat_id,
                                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                "score": float(s.item()) if hasattr(s, "item") else float(s),
                            }
                        )

    if evaluator is not None and hasattr(evaluator, "evaluate"):
        return evaluator.evaluate(all_predictions)

    return {"val_samples": float(len(all_predictions))}
