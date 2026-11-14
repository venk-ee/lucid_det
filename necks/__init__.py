"""
Neck registry and builder for LucidDet.

Contract:
    forward(feats: Sequence[Tensor]) -> Tuple[Tensor, ...]
    Module must expose:
        .out_channels: int or Sequence[int]
"""

from typing import Any, Callable, Dict, Optional, Type, Union
import torch.nn as nn

NECKS: Dict[str, Type[nn.Module]] = {}


def register_neck(
    name_or_cls: Optional[Union[str, Type[nn.Module]]] = None,
) -> Any:
    """
    Decorator to register a neck class in the NECKS registry.
    Supports both @register_neck and @register_neck("custom_name").
    """
    if callable(name_or_cls):
        cls = name_or_cls
        key = cls.__name__.lower()
        if key in NECKS:
            raise KeyError(f"Neck '{key}' is already registered.")
        NECKS[key] = cls
        return cls

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name_or_cls if name_or_cls is not None else cls.__name__.lower()
        if key in NECKS:
            raise KeyError(f"Neck '{key}' is already registered.")
        NECKS[key] = cls
        return cls

    return decorator


def build_neck(name: str = "pafpn", **kwargs: Any) -> nn.Module:
    """
    Build a neck by name from the NECKS registry.

    Args:
        name: Name of the registered neck.
        **kwargs: Arguments passed to the neck constructor.

    Returns:
        Instantiated neck nn.Module.
    """
    if name not in NECKS:
        raise ValueError(
            f"Unknown neck '{name}'. Available necks: {sorted(list(NECKS.keys()))}"
        )
    return NECKS[name](**kwargs)


# Import concrete implementations to trigger registration
from necks.cspnext_pafpn import CSPNeXtPAFPN
from necks.pafpn import YOLOXPAFPN
from necks.identity import IdentityNeck

__all__ = ["NECKS", "CSPNeXtPAFPN", "YOLOXPAFPN", "IdentityNeck", "register_neck", "build_neck"]



