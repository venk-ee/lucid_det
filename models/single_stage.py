"""
Single-Stage Detector base architecture (Pure PyTorch).
Composes swappable Backbone -> Neck -> Head with support for both training outputs
and inference multi-class NMS decoding.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from core.nms import multiclass_nms_batch


class SingleStageDetector(nn.Module):
    """
    Standard Single-Stage Detector base class.

    Pipeline:
        Images -> Backbone -> Neck (optional) -> Head -> (Training loss inputs or NMS detections)

    Args:
        backbone: Feature extractor module (e.g. CSPNeXt, CSPDarknet, Darknet24).
        neck: Optional multi-scale feature aggregator (e.g. PAFPN, Identity).
        head: Detection head module predicting class logits and bounding boxes.
        score_thresh: Confidence threshold for inference filtering.
        nms_thresh: IoU threshold for inference Non-Maximum Suppression.
        max_num: Maximum number of detections retained per image during inference.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.65,
        max_num: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_num = max_num

    def extract_feat(self, x: torch.Tensor) -> Any:
        """Extract multi-scale features through backbone and neck."""
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(
        self,
        x: torch.Tensor,
        training_override: Optional[bool] = None,
    ) -> Any:
        """
        Forward pass of the detector.

        Args:
            x: Input batch tensor of shape (B, C, H, W).
            training_override: Explicitly force training or inference mode when provided.

        Returns:
            In training mode:
                Raw head outputs (e.g. dict, tuple, or dataclass of cls scores, bbox preds).
            In inference mode:
                If head outputs expose 'pred_cls' and 'pred_boxes' (dict or dataclass),
                attaches or includes 'detections' (list of dicts per image) and returns results.
                Otherwise returns raw head outputs.
        """
        feats = self.extract_feat(x)
        if self.head is None:
            return feats

        outputs = self.head(feats)

        is_train = self.training if training_override is None else training_override
        if is_train:
            return outputs

        # Inference post-processing: extract pred_cls and pred_boxes from dict or dataclass
        pred_cls = None
        pred_boxes = None

        if isinstance(outputs, Mapping):
            pred_cls = outputs.get("pred_cls")
            pred_boxes = outputs.get("pred_boxes")
        elif hasattr(outputs, "pred_cls") and hasattr(outputs, "pred_boxes"):
            pred_cls = getattr(outputs, "pred_cls")
            pred_boxes = getattr(outputs, "pred_boxes")

        if pred_cls is not None and pred_boxes is not None:
            detections = multiclass_nms_batch(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh,
                max_num=self.max_num,
            )

            if isinstance(outputs, Mapping):
                result = dict(outputs)
                result["detections"] = detections
                return result
            elif hasattr(outputs, "__setitem__"):
                try:
                    outputs["detections"] = detections
                    return outputs
                except Exception:
                    pass

            if hasattr(outputs, "detections"):
                try:
                    outputs.detections = detections
                    return outputs
                except Exception:
                    pass

            return {"detections": detections, "outputs": outputs}

        return outputs
