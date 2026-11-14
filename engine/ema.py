"""
Model Exponential Moving Average (EMA).
Maintains a smoothed shadow copy of model weights for robust evaluation and checkpointing.
"""

import copy
import math
from typing import Any, Dict

import torch
import torch.nn as nn


class ModelEMA:
    """
    Model Exponential Moving Average.

    Keeps an exponential moving average of all model parameters:
        ema_param = decay * ema_param + (1 - decay) * model_param

    Includes exponential warmup ramp-up:
        d = decay * (1 - exp(-updates / tau))
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9998,
        tau: float = 2000.0,
    ) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @property
    def module(self) -> nn.Module:
        """Shadowed EMA model instance."""
        return self.ema

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters given current model weights."""
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))

        # Safely unwrap DDP / DataParallel and torch.compile wrappers
        m = model.module if hasattr(model, "module") else model
        m = getattr(m, "_orig_mod", m)

        model_params = dict(m.named_parameters())
        for name, ema_p in self.ema.named_parameters():
            if name in model_params:
                model_p = model_params[name]
                ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)

        model_buffers = dict(m.named_buffers())
        for name, ema_b in self.ema.named_buffers():
            if name in model_buffers:
                ema_b.copy_(model_buffers[name])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema": self.ema.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema.load_state_dict(state_dict["ema"])
        self.updates = state_dict.get("updates", 0)
        self.decay = state_dict.get("decay", 0.9998)
