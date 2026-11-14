"""
LucidDet Model Weights and Checkpoint Converter System.
Provides modular weight converters, official presets, and auto-discovery.
"""

import importlib
import pkgutil
from typing import List

from weights.base import (
    CheckpointEnvelope,
    ConversionReport,
    ModelWeightPreset,
)
from weights.registry import (
    WEIGHT_CONVERTERS,
    WEIGHT_PRESETS,
    get_converter,
    get_preset,
    list_presets,
    register_converter,
    register_preset,
)

# Auto-discover all converter modules in this package so their registers fire automatically
__all_modules: List[str] = []
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg and module_name not in ("base", "registry"):
        importlib.import_module(f".{module_name}", package=__name__)
        __all_modules.append(module_name)

__all__ = [
    "ModelWeightPreset",
    "CheckpointEnvelope",
    "ConversionReport",
    "WEIGHT_CONVERTERS",
    "WEIGHT_PRESETS",
    "register_converter",
    "register_preset",
    "get_converter",
    "get_preset",
    "list_presets",
]
