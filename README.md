# LucidDet

A clean, modular, pure PyTorch object detection library unifying **RTMDet**, **YOLOX**, and **YOLOv1** with **zero OpenMMLab / MMCV runtime dependencies**.

Designed with 1:1 modular architectural parity, flat package hierarchy, typed dataclass configs, and pure PyTorch components.

---

## Supported Architectures

| Family | Backbone | Neck | Head | Assigner | Primary Losses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTMDet** | `CSPNeXt` | `CSPNeXtPAFPN` | `SepBNHead` | `SimOTA` | `QualityFocalLoss` + `GIoULoss` |
| **YOLOX** | `CSPDarknet` | `PAFPN` | `DecoupledHead` | `SimOTA` | `BCEWithLogits` + `IoULoss` + `L1Loss` |
| **YOLOv1** | `Darknet19` / `ResNet` | Direct pass | `YOLOv1Head` | Dense grid | Detection MSE Loss |

---

## Repository Structure

```
lucid_det/
├── backbones/          # CSPNeXt, CSPDarknet, Darknet19
├── necks/              # CSPNeXtPAFPN, PAFPN
├── heads/              # SepBNHead, DecoupledHead, YOLOv1Head
├── models/             # Detectors: RTMDet, YOLOX, YOLOv1, SingleStageDetector
├── assigners/          # SimOTA dynamic soft-label assigner
├── losses/             # QFL, GIoU, IoU, YOLOv1 MSE loss
├── datasets/           # COCODetection, VOCDetection, dataloader builders
├── transforms/         # Albumentations & PyTorch vision transforms
├── engine/             # ModelEMA, AMP, Optimizer & LR schedulers
├── runners/            # Training epoch runner & COCO evaluation
├── training_scripts/   # Per-family configs and training entrypoints
│   ├── rtmdet/         # config.py, train.py
│   ├── yolox/          # config.py, train.py
│   └── yolov1/         # config.py, train.py
├── tools/              # CLI utilities (sanity, convert_weights, infer, train_det)
├── weights/            # Decoupled official checkpoint converters & loaders
└── train.sh            # Universal execution launcher with CUDA/cuDNN handling
```

---

## Quick Start

### 1. Environment Setup

Using [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`:

```bash
# Clone and enter repo
cd lucid_det

# Install dependencies
uv sync
```

### 2. Sanity Check

Verify all backbones, necks, heads, and full model forward passes:

```bash
./train.sh python tools/sanity.py
```

### 3. Pretrained Weights

Convert official checkpoints (OpenMMLab RTMDet, Megvii YOLOX) into LucidDet pure PyTorch format:

```bash
# List all available presets
./train.sh python tools/convert_weights.py --list

# Download & convert a specific preset (e.g. RTMDet-s or YOLOX-s)
./train.sh python tools/convert_weights.py --preset rtmdet_s
./train.sh python tools/convert_weights.py --preset yolox_s
```

### 4. Inference & Visualization

Run inference on an image and save visualized detections:

```bash
./train.sh python tools/infer.py \
    --img data/coco/val2017/000000000036.jpg \
    --weights checkpoints/rtmdet_s.pth \
    --out outputs/pred_rtmdet.jpg \
    --thr 0.35
```

### 5. Training

Launch training for any model family via `train.sh` or direct script execution:

```bash
# Train RTMDet
./train.sh rtmdet

# Train YOLOX
./train.sh yolox

# Train YOLOv1
./train.sh yolov1
```

Or run with custom arguments:

```bash
./train.sh python training_scripts/rtmdet/train.py --batch-size 16 --epochs 100 --device cuda:0
```

---

## Design Principles

- **Pure PyTorch**: Zero reliance on `mmcv`, `mmengine`, or complex C++ compilation extensions.
- **Flat Modular Hierarchy**: Clear separation of `backbones`, `necks`, `heads`, and `models` for seamless mixing and matching.
- **Clean Execution**: Environment and CUDA runtime paths are encapsulated in `train.sh` without polluting library code.
