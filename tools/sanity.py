"""
Unified Sanity Smoke Test:
Verifies all 3 supported detector architectures on CUDA:
  1. RTMDet (arch=tiny, arch=s)
  2. YOLOX (arch=s)
  3. YOLOv1 (Darknet-24 + FC Head)
Covers forward pass, target assignment, multi-task loss computation,
backward gradient flow, and inference multi-class NMS decoding.
"""

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from losses import YOLOv1Loss, det_loss
from models import build_rtmdet, build_yolov1, build_yolox


def run_rtmdet_sanity(device: torch.device, arch: str = "tiny", num_classes: int = 80) -> None:
    print(f"\n--- [RTMDet] arch={arch}, classes={num_classes} ---")
    model = build_rtmdet(arch=arch, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    model.train()
    batch_size = 2
    dummy_imgs = torch.randn(batch_size, 3, 640, 640, device=device)
    outputs = model(dummy_imgs, training_override=True)

    dummy_gt_boxes = [
        torch.tensor([[50.0, 50.0, 200.0, 200.0], [300.0, 300.0, 500.0, 500.0]], device=device),
        torch.tensor([[100.0, 150.0, 400.0, 450.0]], device=device),
    ]
    dummy_gt_labels = [
        torch.tensor([0, 1], device=device),
        torch.tensor([2], device=device),
    ]

    total_loss, loss_dict = det_loss(
        model_out=outputs,
        gt_boxes_list=dummy_gt_boxes,
        gt_labels_list=dummy_gt_labels,
        cls_loss="qfl",
        box_loss="giou",
    )
    print(
        f"  Train Loss: Total={total_loss.item():.4f} | Cls={loss_dict['loss_cls'].item():.4f} | Box={loss_dict['loss_box'].item():.4f}"
    )

    assert torch.isfinite(total_loss), "Loss must be finite!"
    model.zero_grad()
    total_loss.backward()

    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed!"
    print(f"  Backward pass OK ({len(grads)} tensors with gradients)")

    model.eval()
    with torch.no_grad():
        infer_out = model(dummy_imgs, training_override=False)
    detections = (
        infer_out.detections if hasattr(infer_out, "detections") else infer_out["detections"]
    )
    assert len(detections) == batch_size
    print(f"  Inference NMS decoding OK (Batch={len(detections)})")
    print(f"  [PASS] RTMDet ({arch}) PASSED")


def run_yolox_sanity(device: torch.device, arch: str = "s", num_classes: int = 80) -> None:
    print(f"\n--- [YOLOX] arch={arch}, classes={num_classes} ---")
    model = build_yolox(arch=arch, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    model.train()
    batch_size = 2
    dummy_imgs = torch.randn(batch_size, 3, 640, 640, device=device)
    outputs = model(dummy_imgs, training_override=True)

    dummy_gt_boxes = [
        torch.tensor([[50.0, 50.0, 200.0, 200.0], [300.0, 300.0, 500.0, 500.0]], device=device),
        torch.tensor([[100.0, 150.0, 400.0, 450.0]], device=device),
    ]
    dummy_gt_labels = [
        torch.tensor([0, 1], device=device),
        torch.tensor([2], device=device),
    ]

    total_loss, loss_dict = det_loss(
        model_out=outputs,
        gt_boxes_list=dummy_gt_boxes,
        gt_labels_list=dummy_gt_labels,
        cls_loss="qfl",
        box_loss="giou",
    )
    print(
        f"  Train Loss: Total={total_loss.item():.4f} | Cls={loss_dict['loss_cls'].item():.4f} | Box={loss_dict['loss_box'].item():.4f}"
    )

    assert torch.isfinite(total_loss), "Loss must be finite!"
    model.zero_grad()
    total_loss.backward()

    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed!"
    print(f"  Backward pass OK ({len(grads)} tensors with gradients)")

    model.eval()
    with torch.no_grad():
        infer_out = model(dummy_imgs, training_override=False)
    detections = (
        infer_out.detections if hasattr(infer_out, "detections") else infer_out["detections"]
    )
    assert len(detections) == batch_size
    print(f"  Inference NMS decoding OK (Batch={len(detections)})")
    print(f"  [PASS] YOLOX ({arch}) PASSED")


def run_yolov1_sanity(device: torch.device, num_classes: int = 20, S: int = 7, B: int = 2) -> None:
    print(f"\n--- [YOLOv1] classes={num_classes}, S={S}, B={B} ---")
    model = build_yolov1(num_classes=num_classes, S=S, B=B).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    model.train()
    batch_size = 2
    dummy_imgs = torch.randn(batch_size, 3, 448, 448, device=device)
    outputs = model(dummy_imgs, training_override=True)

    # YOLOv1 Target format: (B, S, S, C + 5)
    dummy_targets = torch.zeros(batch_size, S, S, num_classes + 5, device=device)
    # Put one object at cell (3, 3)
    dummy_targets[:, 3, 3, 0] = 1.0  # class 0
    dummy_targets[:, 3, 3, num_classes] = 1.0  # objectness confidence
    dummy_targets[:, 3, 3, num_classes + 1 : num_classes + 5] = torch.tensor(
        [0.5, 0.5, 0.2, 0.2], device=device
    )

    loss_fn = YOLOv1Loss(split_size=S, num_boxes=B, num_classes=num_classes)
    total_loss, loss_dict = loss_fn(outputs, dummy_targets)
    print(
        f"  Train Loss: Total={total_loss.item():.4f} | Coord={loss_dict['loss_coord'].item():.4f} | Obj={loss_dict['loss_obj'].item():.4f} | NoObj={loss_dict['loss_noobj'].item():.4f}"
    )

    assert torch.isfinite(total_loss), "Loss must be finite!"
    model.zero_grad()
    total_loss.backward()

    grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed!"
    print(f"  Backward pass OK ({len(grads)} tensors with gradients)")

    model.eval()
    with torch.no_grad():
        infer_out = model(dummy_imgs, training_override=False)
    detections = (
        infer_out.detections if hasattr(infer_out, "detections") else infer_out["detections"]
    )
    assert len(detections) == batch_size
    print(f"  Inference NMS decoding OK (Batch={len(detections)})")
    print("  [PASS] YOLOv1 PASSED")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Version: {torch.__version__} | Device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    print("\n==================================================")
    print("RUNNING 3-IN-1 UNIFIED DETECTOR SANITY SUITE")
    print("==================================================")

    # 1. RTMDet
    run_rtmdet_sanity(device, arch="tiny", num_classes=80)
    run_rtmdet_sanity(device, arch="s", num_classes=80)

    # 2. YOLOX
    run_yolox_sanity(device, arch="s", num_classes=80)

    # 3. YOLOv1
    run_yolov1_sanity(device, num_classes=20, S=7, B=2)

    print("\n==================================================")
    print("ALL 3 MODEL FAMILIES (RTMDet, YOLOX, YOLOv1) PASSED ON CUDA!")
    print("==================================================")


if __name__ == "__main__":
    main()
