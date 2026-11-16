"""
Unit tests for necks: CSPNeXtPAFPN, YOLOXPAFPN, and IdentityNeck.
Verifies direct instantiation, multi-scale feature aggregation, and NECK_MAP lookup.
"""

import pytest
import torch

from necks import CSPNeXtPAFPN, IdentityNeck, NECK_MAP, YOLOXPAFPN, build_neck


def test_cspnext_pafpn_forward() -> None:
    neck = CSPNeXtPAFPN(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
    )
    feats = [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]

    out = neck(feats)
    assert len(out) == 3
    assert out[0].shape == (2, 64, 32, 32)
    assert out[1].shape == (2, 64, 16, 16)
    assert out[2].shape == (2, 64, 8, 8)


def test_yolox_pafpn_forward() -> None:
    neck = YOLOXPAFPN(
        in_channels=[128, 256, 512],
        num_csp_blocks=1,
    )
    feats = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]

    out = neck(feats)
    assert len(out) == 3
    assert out[0].shape == (2, 128, 32, 32)
    assert out[1].shape == (2, 256, 16, 16)
    assert out[2].shape == (2, 512, 8, 8)


def test_identity_neck_forward() -> None:
    neck = IdentityNeck(in_channels=1024, out_channels=1024)
    feat = [torch.randn(2, 1024, 7, 7)]
    out = neck(feat)
    assert len(out) == 1
    assert out[0].shape == (2, 1024, 7, 7)


def test_build_neck_map() -> None:
    n_rtm = build_neck("cspnext_pafpn", in_channels=[64, 128, 256], out_channels=64)
    assert isinstance(n_rtm, CSPNeXtPAFPN)

    n_yolox = build_neck("yolox_pafpn", in_channels=[128, 256, 512])
    assert isinstance(n_yolox, YOLOXPAFPN)

    n_id = build_neck("identity")
    assert isinstance(n_id, IdentityNeck)

    with pytest.raises(ValueError, match="Unknown neck 'invalid_neck'"):
        build_neck("invalid_neck")
