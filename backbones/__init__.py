"""
Backbones for LucidDet.
Exports CSPNeXt, CSPDarknet, and Darknet24.
"""

from typing import Any, Dict, Type

import torch.nn as nn

from backbones.cspdarknet import CSPDarknet
from backbones.cspnext import CSPNeXt
from backbones.darknet24 import Darknet24

BACKBONE_MAP: Dict[str, Type[nn.Module]] = {
    "cspnext": CSPNeXt,
    "cspdarknet": CSPDarknet,
    "darknet24": Darknet24,
    "darknet": Darknet24,
}


def build_backbone(name: str = "cspnext", **kwargs: Any) -> nn.Module:
    """
    Build a backbone by name from the explicit BACKBONE_MAP.

    Args:
        name: Name of the backbone ('cspnext', 'cspdarknet', 'darknet24').
        **kwargs: Arguments passed to the backbone constructor.

    Returns:
        Instantiated backbone nn.Module.
    """
    key = name.lower()
    if key not in BACKBONE_MAP:
        raise ValueError(
            f"Unknown backbone '{name}'. Available backbones: {sorted(BACKBONE_MAP.keys())}"
        )
    return BACKBONE_MAP[key](**kwargs)


__all__ = [
    "CSPNeXt",
    "CSPDarknet",
    "Darknet24",
    "BACKBONE_MAP",
    "build_backbone",
]
