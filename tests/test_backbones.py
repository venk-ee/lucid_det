"""
Unit tests for backbones: CSPNeXt, CSPDarknet, and Darknet24.
Verifies direct instantiation, output shapes, strides, and BACKBONE_MAP lookup.
"""

import pytest
import torch

from backbones import BACKBONE_MAP, CSPDarknet, CSPNeXt, Darknet24, build_backbone


def test_cspnext_forward() -> None:
    model = CSPNeXt(arch="tiny")
    assert model.strides == (8, 16, 32)
    assert len(model.out_channels) == 3

    x = torch.randn(2, 3, 256, 256)
    feats = model(x)
    assert len(feats) == 3
    # P3: 256/8 = 32, P4: 256/16 = 16, P5: 256/32 = 8
    assert feats[0].shape == (2, model.out_channels[0], 32, 32)
    assert feats[1].shape == (2, model.out_channels[1], 16, 16)
    assert feats[2].shape == (2, model.out_channels[2], 8, 8)


def test_cspdarknet_forward() -> None:
    model = CSPDarknet(arch="s")
    assert model.strides == (8, 16, 32)
    assert len(model.out_channels) == 3

    x = torch.randn(2, 3, 256, 256)
    feats = model(x)
    assert len(feats) == 3
    assert feats[0].shape == (2, model.out_channels[0], 32, 32)
    assert feats[1].shape == (2, model.out_channels[1], 16, 16)
    assert feats[2].shape == (2, model.out_channels[2], 8, 8)


def test_darknet24_forward() -> None:
    model = Darknet24(in_channels=3)
    assert model.strides == (64,)
    assert model.out_channels == (1024,)

    x = torch.randn(2, 3, 448, 448)
    feats = model(x)
    assert len(feats) == 1
    # 448 / 64 = 7 (standard YOLOv1 7x7 grid)
    assert feats[0].shape == (2, 1024, 7, 7)


def test_build_backbone_map() -> None:
    b_rtm = build_backbone("cspnext", arch="tiny")
    assert isinstance(b_rtm, CSPNeXt)

    b_yolox = build_backbone("cspdarknet", arch="s")
    assert isinstance(b_yolox, CSPDarknet)

    b_v1 = build_backbone("darknet24")
    assert isinstance(b_v1, Darknet24)

    with pytest.raises(ValueError, match="Unknown backbone 'invalid_backbone'"):
        build_backbone("invalid_backbone")
