"""
Unit tests for detection heads: SepBNHead, YOLOXHead, and YOLOv1Head.
Verifies direct instantiation, forward pass contracts, and explicit HEAD_MAP lookup.
"""

import pytest
import torch

from core.outputs import DetectionOutput
from heads import HEAD_MAP, SepBNHead, YOLOv1Head, YOLOXHead, build_head


def test_sep_bn_head_forward() -> None:
    head = SepBNHead(
        num_classes=80,
        in_channels=128,
        feat_channels=128,
        strides=(8, 16, 32),
        share_conv=True,
    )
    # 3 pyramid levels with downsampled spatial resolutions
    feats = [
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 128, 20, 20),
        torch.randn(2, 128, 10, 10),
    ]

    out = head(feats)
    assert isinstance(out, DetectionOutput)
    # Total anchors = 40*40 + 20*20 + 10*10 = 1600 + 400 + 100 = 2100
    assert out.pred_cls.shape == (2, 2100, 80)
    assert out.pred_boxes.shape == (2, 2100, 4)
    assert out.anchor_centers.shape == (2100, 2)
    assert out.anchor_strides.shape in ((2100,), (2100, 1))


def test_yolox_head_forward() -> None:
    head = YOLOXHead(
        num_classes=80,
        in_channels=[128, 256, 512],
        feat_channels=128,
        strides=(8, 16, 32),
    )
    feats = [
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 256, 20, 20),
        torch.randn(2, 512, 10, 10),
    ]

    out = head(feats)
    assert isinstance(out, DetectionOutput)
    assert out.pred_cls.shape == (2, 2100, 80)
    assert out.pred_boxes.shape == (2, 2100, 4)


def test_yolov1_head_forward() -> None:
    head = YOLOv1Head(
        in_channels=1024,
        num_classes=20,
        split_size=7,
        num_boxes=2,
        img_size=448.0,
    )
    feat = torch.randn(2, 1024, 7, 7)

    out = head(feat)
    assert isinstance(out, DetectionOutput)
    # Total anchors = 7 * 7 * 2 = 98
    assert out.pred_boxes.shape == (2, 98, 4)
    assert out.pred_cls.shape == (2, 98, 20)


def test_build_head_via_head_map() -> None:
    assert "sep_bn_head" in HEAD_MAP
    assert HEAD_MAP["sep_bn_head"] is SepBNHead
    assert "yolox_head" in HEAD_MAP
    assert HEAD_MAP["yolox_head"] is YOLOXHead
    assert "yolov1_head" in HEAD_MAP
    assert HEAD_MAP["yolov1_head"] is YOLOv1Head

    # Explicit builder verification
    head_rtm = build_head(
        "sep_bn_head", num_classes=10, in_channels=64, feat_channels=64, strides=(8, 16, 32)
    )
    assert isinstance(head_rtm, SepBNHead)

    head_yolox = build_head(
        "yolox_head",
        num_classes=10,
        in_channels=[64, 128, 256],
        feat_channels=64,
        strides=(8, 16, 32),
    )
    assert isinstance(head_yolox, YOLOXHead)

    head_v1 = build_head("yolov1_head", in_channels=1024, num_classes=20)
    assert isinstance(head_v1, YOLOv1Head)

    with pytest.raises(ValueError, match="Unknown head 'non_existent_head'"):
        build_head("non_existent_head")


def test_detection_output_mapping_interface() -> None:
    pred_cls = torch.randn(2, 10, 80)
    pred_boxes = torch.randn(2, 10, 4)
    centers = torch.randn(10, 2)
    strides = torch.randn(10, 1)

    out = DetectionOutput(
        pred_cls=pred_cls,
        pred_boxes=pred_boxes,
        anchor_centers=centers,
        anchor_strides=strides,
    )

    # 1. Mapping protocol: len, iter, in, dict conversion
    assert len(out) == 4
    assert list(out) == ["pred_cls", "pred_boxes", "anchor_centers", "anchor_strides"]
    assert "pred_cls" in out
    assert "non_existent" not in out
    d = dict(out)
    assert d["pred_cls"] is pred_cls
    assert d["pred_boxes"] is pred_boxes

    # 2. Key access & setitem
    assert out["pred_cls"] is pred_cls
    assert out.get("pred_cls") is pred_cls
    assert out.get("missing", 42) == 42

    dummy_dets = [{"boxes": torch.zeros(1, 4)}]
    out["detections"] = dummy_dets
    assert len(out) == 5
    assert out.detections is dummy_dets
