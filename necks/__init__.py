"""
Multi-scale feature fusion necks for LucidDet.
Exports CSPNeXtPAFPN (RTMDet), YOLOXPAFPN (YOLOX), and IdentityNeck (YOLOv1).
"""

from typing import Any, Dict, Type

import torch.nn as nn

from necks.cspnext_pafpn import CSPNeXtPAFPN
from necks.identity import IdentityNeck
from necks.pafpn import YOLOXPAFPN

NECK_MAP: Dict[str, Type[nn.Module]] = {
    "cspnext_pafpn": CSPNeXtPAFPN,
    "pafpn": YOLOXPAFPN,
    "yolox_pafpn": YOLOXPAFPN,
    "identity": IdentityNeck,
    "identity_neck": IdentityNeck,
}


def build_neck(name: str = "pafpn", **kwargs: Any) -> nn.Module:
    """
    Build a neck by name from the explicit NECK_MAP.

    Args:
        name: Name of the neck ('cspnext_pafpn', 'yolox_pafpn', 'identity').
        **kwargs: Arguments passed to the neck constructor.

    Returns:
        Instantiated neck nn.Module.
    """
    key = name.lower()
    if key not in NECK_MAP:
        raise ValueError(f"Unknown neck '{name}'. Available necks: {sorted(NECK_MAP.keys())}")
    return NECK_MAP[key](**kwargs)


__all__ = [
    "CSPNeXtPAFPN",
    "YOLOXPAFPN",
    "IdentityNeck",
    "NECK_MAP",
    "build_neck",
]
