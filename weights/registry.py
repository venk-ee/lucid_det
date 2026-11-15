"""
Registry weight conversion functions and model weight presets.
"""

from typing import Any, Callable, Dict, List, Union

import torch

from weights.base import ModelWeightPreset

# Converter function type: takes (raw_state_dict, preset) -> clean_models_state_dict
ConverterFn = Callable[[Dict[str, Any], ModelWeightPreset], Dict[str, torch.Tensor]]

WEIGHT_CONVERTERS: Dict[str, ConverterFn] = {}
WEIGHT_PRESETS: Dict[str, ModelWeightPreset] = {}


def register_converter(family_or_fn: Union[str, ConverterFn]) -> Callable:
    """
    Decorator to register a weight converter for a specific model family.

    Usage:
        @register_converter("rtmdet")
        def convert_rtmdet(raw_sd: Dict[str, Any], preset: ModelWeightPreset) -> Dict[str, torch.Tensor]:
            ...
    """
    if callable(family_or_fn):
        # Used without parens: family derived from function name (e.g. convert_rtmdet -> rtmdet)
        fn = family_or_fn
        family = fn.__name__.replace("convert_", "")
        WEIGHT_CONVERTERS[family.lower()] = fn
        return fn

    family = str(family_or_fn).lower()

    def decorator(fn: ConverterFn) -> ConverterFn:
        WEIGHT_CONVERTERS[family] = fn
        return fn

    return decorator


def register_preset(preset: ModelWeightPreset) -> None:
    """
    Register a ModelWeightPreset instance.
    Raises ValueError on duplicate preset names.
    """
    name = preset.name.lower()
    WEIGHT_PRESETS[name] = preset


def get_converter(family: str) -> ConverterFn:
    """
    Retrieve converter function for a model family.
    Raises KeyError if family is not registered.
    """
    family = family.lower()
    if family not in WEIGHT_CONVERTERS:
        raise KeyError(
            f"Weight converter for family '{family}' is not registered. "
            f"Available converters: {list(WEIGHT_CONVERTERS.keys())}"
        )
    return WEIGHT_CONVERTERS[family]


def get_preset(name: str) -> ModelWeightPreset:
    """
    Retrieve ModelWeightPreset by name.
    Raises KeyError if preset is not registered.
    """
    name = name.lower()
    if name not in WEIGHT_PRESETS:
        raise KeyError(
            f"Model preset '{name}' is not registered. "
            f"Available presets: {list(WEIGHT_PRESETS.keys())}"
        )
    return WEIGHT_PRESETS[name]


def list_presets() -> List[ModelWeightPreset]:
    """Return all registered ModelWeightPreset instances sorted by name."""
    return sorted(WEIGHT_PRESETS.values(), key=lambda p: p.name)
