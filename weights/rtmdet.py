"""
RTMDet weight converter plugin and model presets.
Handles OpenMMLab / MMYOLO checkpoints with safe unpickling,
data_preprocessor removal, and bbox_head/depthwise/pointwise key remapping.
"""

import pickle
from typing import Any, Dict

import torch

from weights.base import ModelWeightPreset
from weights.registry import register_converter, register_preset


def safe_unpickle_load(f: Any) -> Any:
    """Safe unpickler mock that ignores missing mmcv/mmengine modules in pickles."""

    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if any(module.startswith(pkg) for pkg in ("mmengine", "mmcv", "mmdet", "mmyolo")):
                return type(name, (), {})
            return super().find_class(module, name)

    return SafeUnpickler(f).load()


@register_converter("rtmdet")
def convert_rtmdet(raw_sd: Any, preset: ModelWeightPreset) -> Dict[str, torch.Tensor]:
    """
    Convert OpenMMLab RTMDet checkpoint to clean LucidDet state_dict.

    Args:
        raw_sd: Raw checkpoint dictionary loaded from disk.
        preset: ModelWeightPreset specifying target architecture.

    Returns:
        clean_sd: Clean state dict matching build_rtmdet(arch=preset.arch, num_classes=preset.num_classes).
    """
    # 1. Extract state dict from container dictionary if wrapped
    if isinstance(raw_sd, dict):
        if "state_dict" in raw_sd:
            sd = raw_sd["state_dict"]
        elif "model" in raw_sd:
            sd = raw_sd["model"]
        elif "ema" in raw_sd:
            ema_data = raw_sd["ema"]
            sd = ema_data.get("ema", ema_data) if isinstance(ema_data, dict) else ema_data
        else:
            sd = raw_sd
    else:
        sd = raw_sd

    clean_sd: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue

        # Strip module. wrapper
        if k.startswith("module."):
            k = k[7:]

        # Filter out OpenMMLab input preprocessor stats
        if k.startswith("data_preprocessor."):
            continue

        # Remap OpenMMLab MMDetection / MMYOLO layer names
        new_k = k
        new_k = new_k.replace("bbox_head.", "head.")
        new_k = new_k.replace(".depthwise_conv.", ".dw.")
        new_k = new_k.replace(".pointwise_conv.", ".pw.")

        clean_sd[new_k] = v

    return clean_sd


# Register Official Presets
register_preset(
    ModelWeightPreset(
        name="rtmdet_s",
        family="rtmdet",
        arch="s",
        builder_fn="build_rtmdet",
        num_classes=80,
        url="https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-888a1e51.pth",
        cache_hints=(),
        color_space="RGB",
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        input_size=(640, 640),
        description="RTMDet-s pretrained on COCO (80 classes, 40.3 mAP)",
    )
)

register_preset(
    ModelWeightPreset(
        name="rtmdet_tiny",
        family="rtmdet",
        arch="tiny",
        builder_fn="build_rtmdet",
        num_classes=80,
        url="https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-db32cd39.pth",
        cache_hints=(),
        color_space="RGB",
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        input_size=(640, 640),
        description="RTMDet-tiny pretrained on COCO (80 classes, 33.2 mAP)",
    )
)
