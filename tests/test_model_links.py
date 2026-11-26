"""Tests for per-family weight preset plugins, registry, and model scale variants."""

import pytest
import torch

from models import build_detector
from weights import get_converter, get_preset, list_presets


def test_model_links_presets_loaded():
    """All expected per-family presets are registered and valid."""
    names = {preset.name for preset in list_presets()}
    expected = {
        "rtmdet_tiny",
        "rtmdet_s",
        "rtmdet_m",
        "rtmdet_l",
        "yolox_tiny",
        "yolox_s",
        "yolox_m",
        "yolox_l",
        "yolox_x",
        "yolov1_voc",
    }
    assert expected.issubset(names)

    for name in expected:
        preset = get_preset(name)
        assert callable(get_converter(preset.family))
        assert isinstance(preset.mean, tuple)
        assert isinstance(preset.std, tuple)
        assert isinstance(preset.input_size, tuple) and len(preset.input_size) == 2
        assert isinstance(preset.cache_hints, tuple)


@pytest.mark.parametrize(
    ("family", "arch"),
    [
        ("rtmdet", "tiny"),
        ("rtmdet", "s"),
        ("rtmdet", "m"),
        ("rtmdet", "l"),
        ("yolox", "tiny"),
        ("yolox", "s"),
        ("yolox", "m"),
        ("yolox", "l"),
        ("yolox", "x"),
    ],
)
def test_rtmdet_and_yolox_preset_forward_passes(family: str, arch: str):
    """Every registered RTMDet and YOLOX preset scale builds and runs inference."""
    model = build_detector(family, arch=arch, num_classes=80)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        outputs = model(x, training_override=False)
    assert outputs.detections is not None
    assert len(outputs.detections) == 1
