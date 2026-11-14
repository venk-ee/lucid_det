"""
YOLOv1 weight converter plugin and model presets.
Handles legacy YOLOv1 checkpoints with darknet -> backbone.net and fcs -> head.fcs remapping.
"""

from typing import Any, Dict

import torch

from weights.base import ModelWeightPreset
from weights.registry import register_converter, register_preset


@register_converter("yolov1")
def convert_yolov1(raw_sd: Any, preset: ModelWeightPreset) -> Dict[str, torch.Tensor]:
    """
    Convert legacy YOLOv1 checkpoint to clean LucidDet state_dict.

    Args:
        raw_sd: Raw checkpoint dictionary.
        preset: ModelWeightPreset specifying target architecture.

    Returns:
        clean_sd: Clean state dict matching build_yolov1(num_classes=preset.num_classes).
    """
    if isinstance(raw_sd, dict):
        if "model" in raw_sd:
            sd = raw_sd["model"]
        elif "state_dict" in raw_sd:
            sd = raw_sd["state_dict"]
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

        if k.startswith("module."):
            k = k[7:]

        new_k = k
        # Remap legacy darknet backbone
        if new_k.startswith("darknet."):
            new_k = "backbone.net." + new_k[len("darknet.") :]
        # Remap batchnorm name
        new_k = new_k.replace(".batchnorm.", ".bn.")
        # Remap fcs head
        if new_k.startswith("fcs."):
            new_k = "head.fcs." + new_k[len("fcs.") :]

        clean_sd[new_k] = v

    return clean_sd


# Register Official Presets
register_preset(
    ModelWeightPreset(
        name="yolov1_voc",
        family="yolov1",
        arch="darknet24",
        builder_fn="build_yolov1",
        num_classes=20,
        url=None,
        cache_hints=(
            "/mnt/ken/proj/vision_lib/YoloV1/yolov1_voc.pth",
            "/mnt/ken/proj/vision_lib/YoloV1/best.pth",
            "checkpoints/yolov1_voc.pth",
        ),
        color_space="RGB",
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        input_size=(448, 448),
        description="YOLOv1 Darknet24 trained on Pascal VOC (20 classes)",
    )
)
