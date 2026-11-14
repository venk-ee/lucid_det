"""
YOLOX weight converter plugin and model presets for LucidDet.
Handles official Megvii YOLOX checkpoints with prefix stripping,
PAFPN neck key remapping, and Focus -> Conv 6x6 spatial weight reshaping:
    w = focus_weight.view(c_out, 3, 2, 2, 3, 3).permute(0, 1, 4, 2, 5, 3).reshape(c_out, 3, 6, 6)
"""

from typing import Any, Dict
import torch

from weights.base import ModelWeightPreset
from weights.registry import register_converter, register_preset


def translate_yolox_key(k: str) -> str:
    """Translate Megvii YOLOX key to LucidDet naming convention."""
    new_k = k
    if new_k.startswith("module."):
        new_k = new_k[7:]

    # Strip double backbone prefix
    if new_k.startswith("backbone.backbone."):
        new_k = "backbone." + new_k[len("backbone.backbone."):]

    # Remap PAFPN neck blocks
    neck_map = {
        "backbone.lateral_conv0.": "neck.lateral_conv0.",
        "backbone.C3_p4.": "neck.top_down_csp0.",
        "backbone.reduce_conv1.": "neck.lateral_conv1.",
        "backbone.C3_p3.": "neck.top_down_csp1.",
        "backbone.bu_conv2.": "neck.downsample0.",
        "backbone.C3_n3.": "neck.bottom_up_csp0.",
        "backbone.bu_conv1.": "neck.downsample1.",
        "backbone.C3_n4.": "neck.bottom_up_csp1.",
    }
    for old_pfx, new_pfx in neck_map.items():
        if new_k.startswith(old_pfx):
            new_k = new_pfx + new_k[len(old_pfx):]
            break

    # CSPLayer sub-modules: conv1 -> main_conv, conv2 -> short_conv, conv3 -> final_conv, m -> blocks
    new_k = new_k.replace(".conv1.", ".main_conv.")
    new_k = new_k.replace(".conv2.", ".short_conv.")
    new_k = new_k.replace(".conv3.", ".final_conv.")
    new_k = new_k.replace(".m.", ".blocks.")

    # In SPPF (dark5.1), conv1 and conv2 are standard convs, not CSPLayer
    new_k = new_k.replace("dark5.1.main_conv.", "dark5.1.conv1.")
    new_k = new_k.replace("dark5.1.short_conv.", "dark5.1.conv2.")

    # Inside DarknetBottleneck blocks (blocks.X), conv1 and conv2 are standard convs
    for b_idx in range(10):
        new_k = new_k.replace(f".blocks.{b_idx}.main_conv.", f".blocks.{b_idx}.conv1.")
        new_k = new_k.replace(f".blocks.{b_idx}.short_conv.", f".blocks.{b_idx}.conv2.")

    return new_k


@register_converter("yolox")
def convert_yolox(raw_sd: Any, preset: ModelWeightPreset) -> Dict[str, torch.Tensor]:
    """
    Convert official Megvii YOLOX or native YOLOX checkpoint to clean LucidDet state_dict.

    Args:
        raw_sd: Raw checkpoint dictionary.
        preset: ModelWeightPreset specifying target architecture.

    Returns:
        clean_sd: Clean state dict matching build_yolox(arch=preset.arch, num_classes=preset.num_classes).
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

    # Check for Focus layer weights to reshape into Conv 6x6
    focus_weight_key = None
    for k in sd.keys():
        if "stem.conv.conv.weight" in k or "stem.focus.conv.conv.weight" in k or "stem.focus.conv.weight" in k:
            focus_weight_key = k
            break

    if focus_weight_key is not None:
        focus_w = sd[focus_weight_key]
        if focus_w.ndim == 4 and focus_w.shape[1] == 12:
            c_out = focus_w.shape[0]
            # Reshape Focus 3x3 over 4 spatial patches x 3 color channels into equivalent 6x6 conv over 3 channels
            # Megvii patch order is (0,0), (1,0), (0,1), (1,1) -> permute to (dy, dx) grid: [0, 2, 1, 3]
            w_4patches = focus_w.view(c_out, 4, 3, 3, 3)
            w_grid = w_4patches[:, [0, 2, 1, 3], :, :, :].view(c_out, 2, 2, 3, 3, 3)
            reshaped_stem_w = w_grid.permute(0, 3, 4, 1, 5, 2).reshape(c_out, 3, 6, 6)
            clean_sd["backbone.stem.conv.weight"] = reshaped_stem_w

            # Map stem batchnorm params
            base_prefix = focus_weight_key.rsplit(".conv.", 1)[0]
            if f"{base_prefix}.bn.weight" in sd:
                clean_sd["backbone.stem.bn.weight"] = sd[f"{base_prefix}.bn.weight"]
                clean_sd["backbone.stem.bn.bias"] = sd[f"{base_prefix}.bn.bias"]
                clean_sd["backbone.stem.bn.running_mean"] = sd[f"{base_prefix}.bn.running_mean"]
                clean_sd["backbone.stem.bn.running_var"] = sd[f"{base_prefix}.bn.running_var"]
                if f"{base_prefix}.bn.num_batches_tracked" in sd:
                    clean_sd["backbone.stem.bn.num_batches_tracked"] = sd[f"{base_prefix}.bn.num_batches_tracked"]

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue

        # Skip focus layer since it was reshaped and loaded above
        if "stem.conv.conv." in k or "stem.conv.bn." in k or "stem.focus." in k:
            continue

        new_k = translate_yolox_key(k)
        clean_sd[new_k] = v

    return clean_sd


# Register Official Presets
register_preset(
    ModelWeightPreset(
        name="yolox_s",
        family="yolox",
        arch="s",
        builder_fn="build_yolox",
        num_classes=80,
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
        cache_hints=(
            "checkpoints/downloads/yolox_s.pth",
            "/mnt/ken/proj/vision_lib/yolo_x/yolox_s.pth",
        ),
        color_space="BGR",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        input_size=(640, 640),
        description="YOLOX-s official Megvii pretrained on COCO (80 classes, 40.5 mAP)",
    )
)

register_preset(
    ModelWeightPreset(
        name="yolox_tiny",
        family="yolox",
        arch="tiny",
        builder_fn="build_yolox",
        num_classes=80,
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
        cache_hints=(
            "checkpoints/downloads/yolox_tiny.pth",
            "/mnt/ken/proj/vision_lib/yolo_x/yolox_tiny.pth",
        ),
        color_space="BGR",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        input_size=(416, 416),
        description="YOLOX-tiny official Megvii pretrained on COCO (80 classes, 32.8 mAP)",
    )
)
