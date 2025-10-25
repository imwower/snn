from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class _HeadCache:
    inputs: np.ndarray
    hidden: np.ndarray


class ReadoutMLP:
    """Two-layer MLP readout implemented purely with NumPy."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        rng: np.random.Generator,
        *,
        momentum: float = 0.9,
    ) -> None:
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("ReadoutMLP dimensions must be positive")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        scale1 = np.sqrt(2.0 / max(1, self.in_dim))
        scale2 = np.sqrt(2.0 / max(1, self.hidden_dim))
        self.W1 = rng.normal(0.0, scale1, size=(self.in_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0.0, scale2, size=(self.hidden_dim, self.out_dim)).astype(np.float32)
        self.b2 = np.zeros(self.out_dim, dtype=np.float32)
        self.momentum = float(np.clip(momentum, 0.0, 0.999))
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, inputs: np.ndarray, *, return_cache: bool = False) -> Tuple[np.ndarray, Optional[_HeadCache]]:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_dim:
            raise ValueError(f"ReadoutMLP expected input shape (batch,{self.in_dim}), got {inputs.shape}")
        z1 = inputs @ self.W1 + self.b1
        hidden = np.tanh(z1).astype(np.float32, copy=False)
        logits = hidden @ self.W2 + self.b2
        cache: Optional[_HeadCache] = None
        if return_cache:
            cache = _HeadCache(inputs=inputs, hidden=hidden)
        return logits.astype(np.float32, copy=False), cache

    def backward(
        self,
        grad_logits: np.ndarray,
        cache: _HeadCache,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if grad_logits.ndim != 2 or grad_logits.shape[1] != self.out_dim:
            raise ValueError(f"ReadoutMLP grad logits shape mismatch, got {grad_logits.shape}")
        inputs = cache.inputs
        hidden = cache.hidden
        batch = max(1, grad_logits.shape[0])
        grad_W2 = hidden.T @ grad_logits / batch
        grad_b2 = np.sum(grad_logits, axis=0) / batch
        grad_hidden = grad_logits @ self.W2.T
        grad_z1 = grad_hidden * (1.0 - hidden ** 2)
        grad_W1 = inputs.T @ grad_z1 / batch
        grad_b1 = np.sum(grad_z1, axis=0) / batch
        grad_inputs = grad_z1 @ self.W1.T
        grads = {
            "W1": grad_W1.astype(np.float32, copy=False),
            "b1": grad_b1.astype(np.float32, copy=False),
            "W2": grad_W2.astype(np.float32, copy=False),
            "b2": grad_b2.astype(np.float32, copy=False),
        }
        return grad_inputs.astype(np.float32, copy=False), grads

    def apply_gradients(self, grads: Dict[str, np.ndarray], lr: float, *, weight_decay: float = 0.0) -> None:
        wd = float(max(0.0, weight_decay))
        mom = self.momentum

        def _update(param: np.ndarray, grad: np.ndarray, velocity: np.ndarray) -> np.ndarray:
            if wd > 0.0:
                param *= (1.0 - lr * wd)
            velocity[:] = mom * velocity + grad
            param -= lr * velocity
            return velocity

        self.v_W1 = _update(self.W1, grads["W1"], self.v_W1)
        self.v_b1 = _update(self.b1, grads["b1"], self.v_b1)
        self.v_W2 = _update(self.W2, grads["W2"], self.v_W2)
        self.v_b2 = _update(self.b2, grads["b2"], self.v_b2)
