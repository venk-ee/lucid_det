"""
Models package: Assembled neural network detector models.
Top-level model exports: from models import RTMDet, YOLOX, YOLOv1, build_detector.
"""

from typing import Any, Dict, Type, Union

from core.outputs import DetectionOutput
from models.rtmdet import RTMDet
from models.single_stage import SingleStageDetector
from models.yolov1 import YOLOv1
from models.yolox import YOLOX

MODEL_MAP: Dict[str, Type[SingleStageDetector]] = {
    "rtmdet": RTMDet,
    "yolox": YOLOX,
    "yolov1": YOLOv1,
}


def build_detector(
    name_or_cls: Union[str, Type[SingleStageDetector]],
    **kwargs: Any,
) -> SingleStageDetector:
    """
    Build a detector instance by name from MODEL_MAP or by class type.

    Args:
        name_or_cls: Name of the detector ('rtmdet', 'yolox', 'yolov1') or class.
        **kwargs: Arguments forwarded to the detector constructor.

    Returns:
        Instantiated SingleStageDetector subclass.
    """
    if isinstance(name_or_cls, str):
        key = name_or_cls.lower()
        if key not in MODEL_MAP:
            raise ValueError(
                f"Unknown detector model '{name_or_cls}'. Available: {sorted(MODEL_MAP.keys())}"
            )
        target_cls = MODEL_MAP[key]
    elif isinstance(name_or_cls, type) and issubclass(name_or_cls, SingleStageDetector):
        target_cls = name_or_cls
    else:
        raise TypeError(
            f"Expected model name (str) or SingleStageDetector subclass, got {type(name_or_cls)}"
        )
    return target_cls(**kwargs)


def build_rtmdet(**kwargs: Any) -> RTMDet:
    """Convenience builder forwarding directly to RTMDet."""
    return RTMDet(**kwargs)


def build_yolox(**kwargs: Any) -> YOLOX:
    """Convenience builder forwarding directly to YOLOX."""
    return YOLOX(**kwargs)


def build_yolov1(**kwargs: Any) -> YOLOv1:
    """Convenience builder forwarding directly to YOLOv1."""
    return YOLOv1(**kwargs)


__all__ = [
    "RTMDet",
    "YOLOX",
    "YOLOv1",
    "SingleStageDetector",
    "DetectionOutput",
    "MODEL_MAP",
    "build_detector",
    "build_rtmdet",
    "build_yolox",
    "build_yolov1",
]
