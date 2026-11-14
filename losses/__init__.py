"""
Loss registries and builders.

Houses classification losses (CLS) and bounding box regression losses (BOX).
"""

from typing import Any, Callable, Dict, Optional, Type, Union

import torch.nn as nn

CLS: Dict[str, Type[nn.Module]] = {}
BOX: Dict[str, Type[nn.Module]] = {}


def register_cls_loss(
    name_or_cls: Optional[Union[str, Type[nn.Module]]] = None,
) -> Any:
    """
    Decorator to register a classification loss.
    Supports both @register_cls_loss and @register_cls_loss("custom_name").
    """
    if callable(name_or_cls):
        cls = name_or_cls
        key = cls.__name__.lower()
        if key in CLS:
            raise KeyError(f"Classification loss '{key}' is already registered.")
        CLS[key] = cls
        return cls

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name_or_cls if name_or_cls is not None else cls.__name__.lower()
        if key in CLS:
            raise KeyError(f"Classification loss '{key}' is already registered.")
        CLS[key] = cls
        return cls

    return decorator


def register_box_loss(
    name_or_cls: Optional[Union[str, Type[nn.Module]]] = None,
) -> Any:
    """
    Decorator to register a bounding box regression loss.
    Supports both @register_box_loss and @register_box_loss("custom_name").
    """
    if callable(name_or_cls):
        cls = name_or_cls
        key = cls.__name__.lower()
        if key in BOX:
            raise KeyError(f"Box loss '{key}' is already registered.")
        BOX[key] = cls
        return cls

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name_or_cls if name_or_cls is not None else cls.__name__.lower()
        if key in BOX:
            raise KeyError(f"Box loss '{key}' is already registered.")
        BOX[key] = cls
        return cls

    return decorator


def build_cls_loss(name: str = "qfl", **kwargs: Any) -> nn.Module:
    """
    Build a classification loss by name.

    Args:
        name: Name of registered classification loss.
        **kwargs: Arguments passed to loss constructor.
    """
    if name not in CLS:
        raise ValueError(
            f"Unknown classification loss '{name}'. Available: {sorted(list(CLS.keys()))}"
        )
    return CLS[name](**kwargs)


def build_box_loss(name: str = "ciou", **kwargs: Any) -> nn.Module:
    """
    Build a bounding box regression loss by name.

    Args:
        name: Name of registered box loss.
        **kwargs: Arguments passed to loss constructor.
    """
    if name not in BOX:
        raise ValueError(f"Unknown box loss '{name}'. Available: {sorted(list(BOX.keys()))}")
    return BOX[name](**kwargs)


# Import concrete implementations to trigger registration
from losses.det_loss import det_loss
from losses.iou_loss import CIoULoss, GIoULoss, IoULoss
from losses.qfl import QualityFocalLoss
from losses.yolov1_loss import YOLOv1Loss

__all__ = [
    "CLS",
    "BOX",
    "QualityFocalLoss",
    "GIoULoss",
    "CIoULoss",
    "IoULoss",
    "YOLOv1Loss",
    "det_loss",
    "register_cls_loss",
    "register_box_loss",
    "build_cls_loss",
    "build_box_loss",
]
