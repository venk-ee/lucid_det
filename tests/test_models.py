"""
Unit tests for detector models direct instantiation and builders.
"""

import pytest
import torch

from models import YOLOX, RTMDet, YOLOv1, build_rtmdet, build_yolov1, build_yolox


@pytest.mark.parametrize("arch", ["tiny", "s"])
def test_rtmdet_instantiation(arch: str) -> None:
    model = RTMDet(arch=arch, num_classes=10)
    assert isinstance(model, RTMDet)
    assert model.num_classes == 10
    assert model.arch == arch


@pytest.mark.parametrize("arch", ["tiny", "s"])
def test_yolox_instantiation(arch: str) -> None:
    model = YOLOX(arch=arch, num_classes=10)
    assert isinstance(model, YOLOX)
    assert model.num_classes == 10
    assert model.arch == arch


def test_yolov1_instantiation() -> None:
    model = YOLOv1(num_classes=20, img_size=640.0)
    assert isinstance(model, YOLOv1)
    assert model.num_classes == 20


def test_zoo_builders() -> None:
    m_rtm = build_rtmdet(arch="tiny", num_classes=5)
    assert isinstance(m_rtm, RTMDet)
    assert m_rtm.num_classes == 5

    m_yolox = build_yolox(arch="s", num_classes=5)
    assert isinstance(m_yolox, YOLOX)
    assert m_yolox.num_classes == 5

    m_yolo1 = build_yolov1(num_classes=5, img_size=448.0)
    assert isinstance(m_yolo1, YOLOv1)
    assert m_yolo1.num_classes == 5


def test_forward_pass_training_and_eval() -> None:
    model = RTMDet(arch="tiny", num_classes=4)
    model.train()
    x = torch.randn(2, 3, 320, 320)
    out = model(x, training_override=True)
    assert "pred_cls" in out
    assert "pred_boxes" in out

    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert hasattr(eval_out, "detections")


def test_yolov1_positional_arguments_compatibility() -> None:
    # Positional signature: num_classes, score_thresh, nms_thresh, max_num, split_size, num_boxes, S, B
    model = YOLOv1(20, 0.05, 0.65, 100, 7, 2, 7, 2)
    assert model.split_size == 7
    assert model.num_boxes == 2


def test_detector_direct_dependency_injection() -> None:
    from backbones import CSPDarknet, CSPNeXt, Darknet24
    from heads import SepBNHead, YOLOv1Head, YOLOXHead
    from necks import YOLOXPAFPN, CSPNeXtPAFPN, IdentityNeck

    # RTMDet DI
    bb_rtm = CSPNeXt(arch="tiny")
    neck_rtm = CSPNeXtPAFPN(in_channels=list(bb_rtm.out_channels), out_channels=96)
    head_rtm = SepBNHead(num_classes=5, in_channels=96, feat_channels=96, strides=bb_rtm.strides)
    model_rtm = RTMDet(backbone=bb_rtm, neck=neck_rtm, head=head_rtm, num_classes=5)
    assert model_rtm.backbone is bb_rtm
    assert model_rtm.neck is neck_rtm
    assert model_rtm.head is head_rtm

    # YOLOX DI
    bb_yolox = CSPDarknet(arch="s")
    neck_yolox = YOLOXPAFPN(in_channels=list(bb_yolox.out_channels))
    head_yolox = YOLOXHead(
        num_classes=5,
        in_channels=list(bb_yolox.out_channels),
        feat_channels=128,
        strides=bb_yolox.strides,
    )
    model_yolox = YOLOX(backbone=bb_yolox, neck=neck_yolox, head=head_yolox, num_classes=5)
    assert model_yolox.backbone is bb_yolox
    assert model_yolox.neck is neck_yolox
    assert model_yolox.head is head_yolox

    # YOLOv1 DI
    bb_v1 = Darknet24(in_channels=3)
    neck_v1 = IdentityNeck(in_channels=1024, out_channels=1024)
    head_v1 = YOLOv1Head(in_channels=1024, num_classes=10)
    model_v1 = YOLOv1(num_classes=10, backbone=bb_v1, neck=neck_v1, head=head_v1)
    assert model_v1.backbone is bb_v1
    assert model_v1.neck is neck_v1
    assert model_v1.head is head_v1
