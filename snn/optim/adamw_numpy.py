from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


class AdamWOptimizer:
    """Minimal NumPy-based AdamW optimizer supporting parameter groups."""

    def __init__(
        self,
        param_groups: Sequence[dict],
        *,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not param_groups:
            raise ValueError("param_groups must be non-empty")
        beta1, beta2 = betas
        if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
            raise ValueError("betas must lie in (0, 1)")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.step_count = 0
        self.param_groups: List[dict] = []
        for group in param_groups:
            params = list(group.get("params", []))
            if not params:
                continue
            states = [{"m": np.zeros_like(param), "v": np.zeros_like(param)} for param in params]
            self.param_groups.append(
                {
                    "params": params,
                    "states": states,
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "lr": float(group.get("lr", 1e-3)),
                }
            )
        if not self.param_groups:
            raise ValueError("param_groups must contain parameters")

    def step(
        self,
        grads_by_group: Sequence[Sequence[Optional[np.ndarray]]],
        lr_by_group: Sequence[float],
    ) -> float:
        if len(grads_by_group) != len(self.param_groups) or len(lr_by_group) != len(self.param_groups):
            raise ValueError("grads and lr specifications must match parameter groups")
        self.step_count += 1
        total_update_sq = 0.0
        for group, grads, lr in zip(self.param_groups, grads_by_group, lr_by_group):
            weight_decay = group["weight_decay"]
            lr_value = float(lr)
            for param, state, grad in zip(group["params"], group["states"], grads):
                if grad is None:
                    continue
                if grad.shape != param.shape:
                    raise ValueError("gradient shape must match parameter shape")
                state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * grad
                state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * (grad ** 2)
                bias_correction1 = 1.0 - self.beta1 ** self.step_count
                bias_correction2 = 1.0 - self.beta2 ** self.step_count
                m_hat = state["m"] / max(bias_correction1, 1e-12)
                v_hat = state["v"] / max(bias_correction2, 1e-12)
                denom = np.sqrt(v_hat) + self.eps
                update = m_hat / denom
                if weight_decay > 0.0:
                    update = update + weight_decay * param
                delta = lr_value * update
                param -= delta
                total_update_sq += float(np.sum(delta.astype(np.float64) ** 2))
        return math.sqrt(total_update_sq)
