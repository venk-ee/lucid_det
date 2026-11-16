"""
Detection Heads for LucidDet.
Exports SepBNHead (RTMDet), YOLOXHead (YOLOX), and YOLOv1Head (YOLOv1).
"""

from typing import Any, Dict, Type

import torch.nn as nn

from heads.sep_bn_head import SepBNHead
from heads.yolov1_head import YOLOv1Head
from heads.yolox_head import YOLOXHead

HEAD_MAP: Dict[str, Type[nn.Module]] = {
    "sep_bn_head": SepBNHead,
    "sep_bn": SepBNHead,
    "rtmdet": SepBNHead,
    "yolox_head": YOLOXHead,
    "yolox": YOLOXHead,
    "yolov1_head": YOLOv1Head,
    "yolov1": YOLOv1Head,
}


def build_head(name: str, **kwargs: Any) -> nn.Module:
    """
    Build a detection head by name from explicit HEAD_MAP.

    Args:
        name: Name of the head ('sep_bn_head', 'yolox_head', 'yolov1_head').
        **kwargs: Arguments passed to the head constructor.

    Returns:
        Instantiated head nn.Module.
    """
    key = name.lower()
    if key not in HEAD_MAP:
        raise ValueError(f"Unknown head '{name}'. Available heads: {sorted(HEAD_MAP.keys())}")
    return HEAD_MAP[key](**kwargs)


__all__ = [
    "SepBNHead",
    "YOLOXHead",
    "YOLOv1Head",
    "HEAD_MAP",
    "build_head",
]
