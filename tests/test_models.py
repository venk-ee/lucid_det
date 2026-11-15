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
