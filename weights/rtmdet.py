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
_RTMDET_BASE_URL = "https://download.openmmlab.com/mmyolo/v0/rtmdet"
_RTMDET_PRESET_SPECS = (
    (
        "tiny",
        f"{_RTMDET_BASE_URL}/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-db32cd39.pth",
        "RTMDet-tiny pretrained on COCO (80 classes, 33.2 mAP)",
    ),
    (
        "s",
        f"{_RTMDET_BASE_URL}/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-888a1e51.pth",
        "RTMDet-s pretrained on COCO (80 classes, 40.3 mAP)",
    ),
    (
        "m",
        f"{_RTMDET_BASE_URL}/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth",
        "RTMDet-m pretrained on COCO (80 classes, 49.4 mAP)",
    ),
    (
        "l",
        f"{_RTMDET_BASE_URL}/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth",
        "RTMDet-l pretrained on COCO (80 classes, 51.4 mAP)",
    ),
)

for _arch, _url, _description in _RTMDET_PRESET_SPECS:
    register_preset(
        ModelWeightPreset(
            name=f"rtmdet_{_arch}",
            family="rtmdet",
            arch=_arch,
            builder_fn="build_rtmdet",
            num_classes=80,
            url=_url,
            cache_hints=(),
            color_space="RGB",
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            input_size=(640, 640),
            description=_description,
        )
    )
