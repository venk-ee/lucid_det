"""
Single-Stage Detector base architecture.
Composes swappable Backbone -> Neck -> Head with support for both training outputs
and inference multi-class NMS decoding.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn

from core.nms import multiclass_nms_batch

T = TypeVar("T", bound="SingleStageDetector")


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
        checkpoint: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_num = max_num

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load weights into the detector from a checkpoint file.
        Supports raw state_dicts and CheckpointEnvelope dictionaries with 'model' key.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        state = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
            meta = state.get("meta", {})
        elif isinstance(state, dict):
            state_dict = state
            meta = {}
        else:
            raise TypeError(f"Expected dict or state_dict in checkpoint, got {type(state)}")

        self.load_state_dict(state_dict, strict=strict)
        return meta

    @classmethod
    def from_pretrained(
        cls: Type[T],
        checkpoint_or_preset: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
        **kwargs: Any,
    ) -> T:
        """
        Instantiate detector model and load pretrained weights from path or local preset name.
        """
        path = Path(checkpoint_or_preset)
        if not path.is_file():
            possible_paths = [
                Path("checkpoints") / f"{checkpoint_or_preset}.pth",
                Path("checkpoints") / str(checkpoint_or_preset),
            ]
            for p in possible_paths:
                if p.is_file():
                    path = p
                    break

        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find checkpoint '{checkpoint_or_preset}'. "
                f"Please provide an existing checkpoint path."
            )

        state = torch.load(path, map_location=map_location, weights_only=False)
        meta = state.get("meta", {}) if isinstance(state, dict) else {}

        model_kwargs = dict(kwargs)
        if "arch" not in model_kwargs and "arch" in meta:
            model_kwargs["arch"] = meta["arch"]
        if "num_classes" not in model_kwargs and "num_classes" in meta:
            model_kwargs["num_classes"] = meta["num_classes"]

        model = cls(**model_kwargs)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=strict)
        return model

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
