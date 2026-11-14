"""
Lightweight command-line trainer.
Training loop with ModelEMA, parameter-group weight decay separation,
AMP mixed-precision, and COCO checkpointing.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from datasets import DetDataset, det_collate
from engine import ModelEMA, build_lr_scheduler
from engine.evaluator import COCOEvaluator
from losses import YOLOv1Loss, det_loss
from models import build_rtmdet, build_yolov1, build_yolox
from runners.det_runner import val_one_epoch
from transforms import build_det_transform


def create_model(model_name: str, num_classes: int, img_size: int = 640) -> torch.nn.Module:
    """Instantiate chosen detector model architecture."""
    model_name = model_name.lower()
    if model_name in ("rtmdet_tiny", "rtmdet-tiny"):
        return build_rtmdet(arch="tiny", num_classes=num_classes)
    elif model_name in ("rtmdet_s", "rtmdet-s", "rtmdet"):
        return build_rtmdet(arch="s", num_classes=num_classes)
    elif model_name in ("yolox_s", "yolox-s", "yolox"):
        return build_yolox(arch="s", num_classes=num_classes)
    elif model_name in ("yolov1", "yolo_v1"):
        return build_yolov1(num_classes=num_classes, img_size=img_size)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: rtmdet_tiny, rtmdet_s, yolox_s, yolov1"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LucidDet Training CLI")
    parser.add_argument("--model", type=str, default="rtmdet_tiny", help="Detector model name")
    parser.add_argument("--root", type=str, required=True, help="Path to image directory")
    parser.add_argument("--ann", type=str, required=True, help="Path to COCO JSON annotations")
    parser.add_argument(
        "--val-root", type=str, default=None, help="Path to validation image directory"
    )
    parser.add_argument(
        "--val-ann", type=str, default=None, help="Path to validation COCO JSON annotations"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=1, help="Validation evaluation epoch cadence"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Base learning rate (default: 2e-5 for yolov1, 2e-3 for rtmdet/yolox)",
    )
    parser.add_argument("--img-size", type=int, default=640, help="Image input resolution")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of foreground classes")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--out", type=str, default="work_dirs/exp", help="Checkpoint output directory"
    )
    parser.add_argument("--log-interval", type=int, default=5, help="Step logging cadence")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    print("==================================================")
    print(f"LucidDet Training: {args.model} | Device: {device} | Epochs: {args.epochs}")
    print(f"Data: {args.root} ({args.ann})")
    print("==================================================")

    # 1. Dataset & DataLoader
    transform = build_det_transform(img_size=args.img_size, is_train=True)
    dataset = DetDataset(
        root=args.root, ann_file=args.ann, transform=transform, num_classes=args.num_classes
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=det_collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_dataloader = None
    evaluator = None
    if args.val_ann and args.val_root:
        val_transform = build_det_transform(img_size=args.img_size, is_train=False)
        val_dataset = DetDataset(
            root=args.val_root,
            ann_file=args.val_ann,
            transform=val_transform,
            num_classes=args.num_classes,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=det_collate,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        evaluator = COCOEvaluator(args.val_ann)

    # 2. Model & Optimizer
    model = create_model(args.model, args.num_classes, img_size=args.img_size).to(device)

    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    base_lr = args.lr if args.lr is not None else (2e-5 if "yolov1" in args.model else 2e-3)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.05},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=base_lr,
    )

    total_steps = args.epochs * len(dataloader)
    warmup_steps = min(500, max(1, total_steps // 5))
    scheduler = build_lr_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    ema = ModelEMA(model, decay=0.9998)
    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=use_amp)
        if (use_amp and amp_dtype == torch.float16)
        else None
    )

    yolov1_loss_fn = (
        YOLOv1Loss(split_size=7, num_boxes=2, num_classes=args.num_classes)
        if "yolov1" in args.model
        else None
    )

    # 3. Training Loop
    step = 0
    best_loss = float("inf")
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            step += 1
            images = batch["img"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images, training_override=True)

                if yolov1_loss_fn is not None:
                    # Construct cell target matrix from boxes
                    S = 7
                    targets_matrix = torch.zeros(
                        len(images), S, S, args.num_classes + 5, device=device
                    )
                    for b_i in range(len(images)):
                        gt_b = batch["gt_boxes"][b_i]
                        gt_l = batch["gt_labels"][b_i]
                        for box, lbl in zip(gt_b, gt_l):
                            x1, y1, x2, y2 = box.tolist()
                            cx, cy = (
                                (x1 + x2) / (2.0 * args.img_size),
                                (y1 + y2) / (2.0 * args.img_size),
                            )
                            w, h = (x2 - x1) / args.img_size, (y2 - y1) / args.img_size
                            ci, cj = min(int(cy * S), S - 1), min(int(cx * S), S - 1)
                            targets_matrix[b_i, ci, cj, int(lbl.item())] = 1.0
                            targets_matrix[b_i, ci, cj, args.num_classes] = 1.0
                            targets_matrix[
                                b_i, ci, cj, args.num_classes + 1 : args.num_classes + 5
                            ] = torch.tensor(
                                [cx * S - cj, cy * S - ci, w * S, h * S], device=device
                            )
                    loss, loss_dict = yolov1_loss_fn(outputs, targets_matrix)
                else:
                    gt_boxes = [b.to(device, non_blocking=True) for b in batch["gt_boxes"]]
                    gt_labels = [lbl.to(device, non_blocking=True) for lbl in batch["gt_labels"]]
                    loss, loss_dict = det_loss(
                        model_out=outputs,
                        gt_boxes_list=gt_boxes,
                        gt_labels_list=gt_labels,
                        cls_loss="qfl",
                        box_loss="giou",
                    )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                stepped = scale_after >= scale_before
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
                optimizer.step()
                stepped = True

            if stepped:
                scheduler.step()
                ema.update(model)

            loss_val = loss.item()
            epoch_loss += loss_val

            if step % args.log_interval == 0 or batch_idx == num_batches - 1:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx + 1}/{num_batches}] "
                    f"Step {step}/{total_steps} | Loss: {loss_val:.4f} | LR: {cur_lr:.2e}",
                    flush=True,
                )

        mean_loss = epoch_loss / max(num_batches, 1)
        print(f"--> Epoch {epoch} Complete. Average Loss: {mean_loss:.4f}")

        # Checkpointing
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "args": vars(args),
        }
        last_path = os.path.join(args.out, "last.pth")
        torch.save(ckpt, last_path)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = os.path.join(args.out, "best.pth")
            torch.save(ckpt, best_path)

        # Validation evaluation cadence
        if val_dataloader is not None and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            eval_model = ema.module if hasattr(ema, "module") else model
            metrics = val_one_epoch(eval_model, val_dataloader, device=device, evaluator=evaluator)
            map_val = metrics.get("mAP_50_95", 0.0)
            map_50 = metrics.get("mAP_50", 0.0)
            print(
                f"--> Epoch {epoch} Validation: mAP@0.50:0.95 = {map_val:.4f} | mAP@0.50 = {map_50:.4f}"
            )
            if map_val > best_map:
                best_map = map_val
                best_map_path = os.path.join(args.out, "best_map.pth")
                torch.save(ckpt, best_map_path)

    print(f"\nTraining Complete! Checkpoints saved to {args.out}/ (best.pth, last.pth)")


if __name__ == "__main__":
    main()
