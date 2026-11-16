"""
Target assigners for LucidDet.
Exports dynamic_soft_label_assign and DynamicSoftLabelAssigner.
"""

from typing import Any, Callable, Dict

from assigners.simota import DynamicSoftLabelAssigner, dynamic_soft_label_assign

ASSIGNER_MAP: Dict[str, Any] = {
    "simota": DynamicSoftLabelAssigner,
    "dynamic_soft_label_assigner": DynamicSoftLabelAssigner,
    "dynamic_soft_label_assign": dynamic_soft_label_assign,
}


def build_assigner(name: str = "simota", **kwargs: Any) -> Any:
    """
    Build an assigner by name from explicit ASSIGNER_MAP.

    Args:
        name: Name of the assigner ('simota', 'dynamic_soft_label_assigner').
        **kwargs: Arguments passed to the assigner constructor.

    Returns:
        Instantiated assigner module or function.
    """
    key = name.lower()
    if key not in ASSIGNER_MAP:
        raise ValueError(
            f"Unknown assigner '{name}'. Available assigners: {sorted(ASSIGNER_MAP.keys())}"
        )
    target = ASSIGNER_MAP[key]
    if isinstance(target, type):
        return target(**kwargs)
    return target


__all__ = [
    "DynamicSoftLabelAssigner",
    "dynamic_soft_label_assign",
    "ASSIGNER_MAP",
    "build_assigner",
]
