"""
Unit tests for detector models direct instantiation and builders.
"""

import pytest
import torch

from models import (
    MODEL_MAP,
    YOLOX,
    RTMDet,
    YOLOv1,
    build_detector,
    build_rtmdet,
    build_yolov1,
    build_yolox,
)


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


def test_model_builders_and_model_map() -> None:
    assert "rtmdet" in MODEL_MAP
    assert MODEL_MAP["rtmdet"] is RTMDet
    assert "yolox" in MODEL_MAP
    assert MODEL_MAP["yolox"] is YOLOX
    assert "yolov1" in MODEL_MAP
    assert MODEL_MAP["yolov1"] is YOLOv1

    m_rtm = build_detector("rtmdet", arch="tiny", num_classes=5)
    assert isinstance(m_rtm, RTMDet)
    assert m_rtm.num_classes == 5

    m_yolox = build_detector("yolox", arch="s", num_classes=5)
    assert isinstance(m_yolox, YOLOX)
    assert m_yolox.num_classes == 5

    m_yolo1 = build_detector("yolov1", num_classes=5, img_size=448.0)
    assert isinstance(m_yolo1, YOLOv1)
    assert m_yolo1.num_classes == 5

    with pytest.raises(ValueError, match="Unknown detector model"):
        build_detector("unknown_detector")

    # Convenience aliases
    assert isinstance(build_rtmdet(arch="tiny"), RTMDet)
    assert isinstance(build_yolox(arch="s"), YOLOX)
    assert isinstance(build_yolov1(num_classes=20), YOLOv1)


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


def test_detector_export_mode_and_postprocess() -> None:
    model = RTMDet(arch="tiny", num_classes=4)

    # Verify strides attribute exists
    assert hasattr(model, "strides")
    assert isinstance(model.strides, (tuple, list))

    # Verify export_mode defaults to False
    assert model.export_mode is False

    x = torch.randn(1, 3, 320, 320)

    # Test export_mode=True returns a 2-tuple of tensors
    model.eval()
    model.export_mode = True
    with torch.no_grad():
        result = model(x)
    assert isinstance(result, tuple)
    assert len(result) == 2
    pred_boxes, pred_cls = result
    assert isinstance(pred_boxes, torch.Tensor)
    assert isinstance(pred_cls, torch.Tensor)
    # Sigmoid output should be in [0, 1]
    assert (pred_cls >= 0).all() and (pred_cls <= 1).all()

    # Test postprocess() works independently
    model.export_mode = False
    with torch.no_grad():
        feats = model.extract_feat(x)
        raw_outputs = model.head(feats)
        processed = model.postprocess(raw_outputs, x)
    assert hasattr(processed, "detections") or "detections" in processed


def test_checkpoint_and_from_pretrained_methods(tmp_path) -> None:
    model = RTMDet(arch="tiny", num_classes=5)
    ckpt_path = tmp_path / "model.pth"
    torch.save(
        {"model": model.state_dict(), "meta": {"arch": "tiny", "num_classes": 5}},
        ckpt_path,
    )

    # 1. Constructor checkpoint loading (restricted unpickling by default)
    model_loaded = RTMDet(arch="tiny", num_classes=5, checkpoint=ckpt_path)
    assert isinstance(model_loaded, RTMDet)

    # 2. from_pretrained classmethod (restricted unpickling by default)
    model_pretrained = RTMDet.from_pretrained(ckpt_path)
    assert isinstance(model_pretrained, RTMDet)
    assert model_pretrained.num_classes == 5

    # 3. Explicit weights_only opt-in
    meta = model.load_checkpoint(ckpt_path, weights_only=True)
    assert meta["arch"] == "tiny"
    meta_legacy = model.load_checkpoint(ckpt_path, weights_only=False)
    assert meta_legacy["arch"] == "tiny"

    # 3. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        RTMDet.from_pretrained(tmp_path / "non_existent.pth")


def test_multiclass_nms_from_logits() -> None:
    from core.nms import multiclass_nms, multiclass_nms_batch

    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]])
    # Logits where values fall inside [0, 1] - heuristic would fail, but from_logits=True works deterministically
    logits = torch.tensor([[0.5, 0.2], [0.1, 0.4]])

    det_boxes, det_scores, det_labels = multiclass_nms(
        boxes, logits, score_thresh=0.01, from_logits=True
    )
    expected_scores = logits.sigmoid().max(dim=-1)[0]
    assert torch.allclose(det_scores[0], expected_scores[0], atol=1e-5)

    # With from_logits=False, values are preserved without sigmoid
    probs = torch.tensor([[0.7, 0.2], [0.1, 0.8]])
    det_boxes_p, det_scores_p, det_labels_p = multiclass_nms(
        boxes, probs, score_thresh=0.01, from_logits=False
    )
    assert torch.allclose(det_scores_p[0], torch.tensor(0.8), atol=1e-5)

    # Batch version with from_logits=True
    batch_boxes = boxes.unsqueeze(0)
    batch_logits = logits.unsqueeze(0)
    batch_res = multiclass_nms_batch(batch_boxes, batch_logits, score_thresh=0.01, from_logits=True)
    assert len(batch_res) == 1
    assert "boxes" in batch_res[0] and "scores" in batch_res[0]
