"""
Backbone registry and builder.

Contract:
    forward(x) -> List[Tensor] or Tuple[Tensor, ...] representing [P3, P4, P5] feature maps.
    Module must expose:
        .out_channels: Sequence[int] (e.g. [256, 512, 1024])
        .strides: Sequence[int] (e.g. [8, 16, 32])
"""

from typing import Any, Callable, Dict, Optional, Type, Union

import torch.nn as nn

BACKBONES: Dict[str, Type[nn.Module]] = {}


def register_backbone(
    name_or_cls: Optional[Union[str, Type[nn.Module]]] = None,
) -> Any:
    """
    Decorator to register a backbone class in the BACKBONES registry.
    Supports both @register_backbone and @register_backbone("custom_name").
    """
    if callable(name_or_cls):
        cls = name_or_cls
        key = cls.__name__.lower()
        if key in BACKBONES:
            raise KeyError(f"Backbone '{key}' is already registered.")
        BACKBONES[key] = cls
        return cls

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name_or_cls if name_or_cls is not None else cls.__name__.lower()
        if key in BACKBONES:
            raise KeyError(f"Backbone '{key}' is already registered.")
        BACKBONES[key] = cls
        return cls

    return decorator


def build_backbone(name: str = "cspnext", **kwargs: Any) -> nn.Module:
    """
    Build a backbone by name from the BACKBONES registry.

    Args:
        name: Name of the registered backbone.
        **kwargs: Arguments passed to the backbone constructor.

    Returns:
        Instantiated backbone nn.Module.
    """
    if name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{name}'. Available backbones: {sorted(BACKBONES.keys())}"
        )
    return BACKBONES[name](**kwargs)


# Import concrete implementations to trigger registration
from backbones.cspdarknet import CSPDarknet
from backbones.cspnext import CSPNeXt
from backbones.darknet24 import Darknet24

__all__ = ["BACKBONES", "CSPNeXt", "CSPDarknet", "Darknet24", "register_backbone", "build_backbone"]
