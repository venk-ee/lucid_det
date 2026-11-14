"""
Target assigners registry and builder.

Assigners match ground-truth boxes to priors/anchors to produce training targets.
"""

from typing import Any, Callable, Dict, Optional, Union

ASSIGNERS: Dict[str, Any] = {}


def register_assigner(
    name_or_cls: Optional[Union[str, Any]] = None,
) -> Any:
    """
    Decorator to register an assigner in the ASSIGNERS registry.
    Supports both @register_assigner and @register_assigner("custom_name").
    """
    if callable(name_or_cls) and not isinstance(name_or_cls, str):
        target = name_or_cls
        key = getattr(target, "__name__", str(target)).lower()
        if key in ASSIGNERS:
            raise KeyError(f"Assigner '{key}' is already registered.")
        ASSIGNERS[key] = target
        return target

    def decorator(cls_or_fn: Any) -> Any:
        key = (
            name_or_cls
            if name_or_cls is not None
            else getattr(cls_or_fn, "__name__", str(cls_or_fn)).lower()
        )
        if key in ASSIGNERS:
            raise KeyError(f"Assigner '{key}' is already registered.")
        ASSIGNERS[key] = cls_or_fn
        return cls_or_fn

    return decorator


def build_assigner(name: str = "simota", **kwargs: Any) -> Any:
    """
    Build an assigner by name from the ASSIGNERS registry.

    Args:
        name: Name of the registered assigner.
        **kwargs: Arguments passed to the assigner constructor or callable.

    Returns:
        Instantiated assigner or callable.
    """
    if name not in ASSIGNERS:
        raise ValueError(
            f"Unknown assigner '{name}'. Available assigners: {sorted(list(ASSIGNERS.keys()))}"
        )
    target = ASSIGNERS[name]
    if isinstance(target, type):
        return target(**kwargs)
    return target


# Import concrete implementations to trigger registration
from assigners.simota import DynamicSoftLabelAssigner, dynamic_soft_label_assign

__all__ = [
    "ASSIGNERS",
    "DynamicSoftLabelAssigner",
    "dynamic_soft_label_assign",
    "register_assigner",
    "build_assigner",
]
