"""
Learning rate scheduling.
Implements Linear Warmup followed by Cosine Annealing.
"""

import math
from typing import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
) -> Callable[[int], float]:
    """Returns a lambda multiplier for Linear Warmup + Cosine Annealing."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """
    Build a step-wise Linear Warmup + Cosine Annealing LR scheduler.

    Args:
        optimizer: PyTorch optimizer.
        total_steps: Total training steps across all epochs.
        warmup_steps: Number of initial steps for linear warmup.
        min_lr_ratio: Minimum learning rate multiplier at end of cosine decay.

    Returns:
        LambdaLR scheduler instance.
    """
    lr_lambda = get_warmup_cosine_lambda(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
