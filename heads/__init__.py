"""
Head registry and builder for LucidDet.

Contract:
    forward(feats: Sequence[Tensor]) -> Dict[str, Tensor] or Tuple[Tensor, ...]
"""

from typing import Any, Callable, Dict, Optional, Type, Union
import torch.nn as nn

HEADS: Dict[str, Type[nn.Module]] = {}


def register_head(
    name_or_cls: Optional[Union[str, Type[nn.Module]]] = None,
) -> Any:
    """
    Decorator to register a head class in the HEADS registry.
    Supports both @register_head and @register_head("custom_name").
    """
    if callable(name_or_cls):
        cls = name_or_cls
        key = cls.__name__.lower()
        if key in HEADS:
            raise KeyError(f"Head '{key}' is already registered.")
        HEADS[key] = cls
        return cls

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name_or_cls if name_or_cls is not None else cls.__name__.lower()
        if key in HEADS:
            raise KeyError(f"Head '{key}' is already registered.")
        HEADS[key] = cls
        return cls

    return decorator


def build_head(name: str, **kwargs: Any) -> nn.Module:
    """
    Build a detection head by name from the HEADS registry.

    Args:
        name: Name of the registered head.
        **kwargs: Arguments passed to the head constructor.

    Returns:
        Instantiated head nn.Module.
    """
    if name not in HEADS:
        raise ValueError(
            f"Unknown head '{name}'. Available heads: {sorted(list(HEADS.keys()))}"
        )
    return HEADS[name](**kwargs)


# Import concrete implementations to trigger registration
from heads.sep_bn_head import SepBNHead
from heads.yolox_head import YOLOXHead
from heads.yolov1_head import YOLOv1Head

__all__ = ["HEADS", "SepBNHead", "YOLOXHead", "YOLOv1Head", "register_head", "build_head"]



