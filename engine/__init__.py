"""
Engine modules for LucidDet: evaluation, EMA, learning rate scheduling, and training helpers.
"""

from engine.ema import ModelEMA
from engine.evaluator import (
    COCOEvaluator,
    eval_bbox,
    get_eval_backend,
    load_coco,
)
from engine.lr_scheduler import (
    build_lr_scheduler,
    get_warmup_cosine_lambda,
)

__all__ = [
    "ModelEMA",
    "build_lr_scheduler",
    "get_warmup_cosine_lambda",
    "COCOEvaluator",
    "eval_bbox",
    "get_eval_backend",
    "load_coco",
]
