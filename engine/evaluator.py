"""
Fast COCO Evaluation Engine.
Computes standard COCO bbox mAP (AP50:95, AP50, AP75, AP_s, AP_m, AP_l).
"""

import os
from typing import Any, Dict, List, Optional, Union

try:
    import faster_coco_eval

    faster_coco_eval.init_as_pycocotools()
    _BACKEND = "faster-coco-eval"
except Exception:
    _BACKEND = "pycocotools"

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_eval_backend() -> str:
    """Return active evaluation backend engine name."""
    return _BACKEND


def load_coco(gt: Union[str, COCO, None]) -> Optional[COCO]:
    """Load COCO instance from path or return existing COCO instance."""
    if gt is None or isinstance(gt, COCO):
        return gt
    if isinstance(gt, str) and os.path.exists(gt):
        return COCO(gt)
    return None


def eval_bbox(
    gt: Union[str, COCO],
    preds: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate bounding box detection predictions against ground truth.

    Args:
        gt: File path to COCO JSON annotations or an instantiated COCO object.
        preds: List of prediction dictionaries with keys:
            'image_id': int
            'category_id': int
            'bbox': [x, y, w, h]
            'score': float

    Returns:
        metrics: Dictionary with COCO mAP metrics:
            'mAP_50_95', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    """
    default_res = {
        "mAP_50_95": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP_s": 0.0,
        "mAP_m": 0.0,
        "mAP_l": 0.0,
    }

    if not preds:
        return default_res

    coco_gt = load_coco(gt)
    if coco_gt is None:
        raise ValueError(f"Invalid COCO ground truth instance or path: {gt}")

    coco_dt = coco_gt.loadRes(preds)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()

    return {
        "mAP_50_95": float(e.stats[0]),
        "mAP_50": float(e.stats[1]),
        "mAP_75": float(e.stats[2]),
        "mAP_s": float(e.stats[3]),
        "mAP_m": float(e.stats[4]),
        "mAP_l": float(e.stats[5]),
    }


class COCOEvaluator:
    """
    Evaluator wrapper holding preloaded COCO ground truth.
    """

    def __init__(self, ann_file: str) -> None:
        self.ann_file = ann_file
        self.coco_gt = COCO(ann_file)

    def evaluate(self, preds: List[Dict[str, Any]]) -> Dict[str, float]:
        return eval_bbox(self.coco_gt, preds)
