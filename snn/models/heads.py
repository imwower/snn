from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class _HeadCache:
    inputs: np.ndarray
    hidden: np.ndarray
    tanh_out: np.ndarray
    ln_norm: Optional[np.ndarray]
    ln_inv_std: Optional[np.ndarray]
    dropout_mask: Optional[np.ndarray]
    raw_logits: np.ndarray


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
        use_layer_norm: bool = True,
        dropout_prob: float = 0.0,
        ln_eps: float = 1e-5,
        learn_logit_scale: bool = False,
        logit_scale_init: float = 1.0,
        logit_scale_bounds: Tuple[float, float] = (0.5, 3.0),
    ) -> None:
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("ReadoutMLP dimensions must be positive")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.rng = rng
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
        self.use_layer_norm = bool(use_layer_norm)
        self.layer_norm_eps = float(ln_eps)
        if self.use_layer_norm:
            self.ln_gamma = np.ones(self.hidden_dim, dtype=np.float32)
            self.ln_beta = np.zeros(self.hidden_dim, dtype=np.float32)
            self.grad_ln_gamma = np.zeros_like(self.ln_gamma)
            self.grad_ln_beta = np.zeros_like(self.ln_beta)
            self.v_ln_gamma = np.zeros_like(self.ln_gamma)
            self.v_ln_beta = np.zeros_like(self.ln_beta)
        else:
            self.ln_gamma = None
            self.ln_beta = None
            self.grad_ln_gamma = None
            self.grad_ln_beta = None
            self.v_ln_gamma = None
            self.v_ln_beta = None
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 0.99))
        self.keep_prob = 1.0 - self.dropout_prob
        self.learn_logit_scale = bool(learn_logit_scale)
        self.logit_scale = np.array([float(logit_scale_init)], dtype=np.float32)
        lo, hi = logit_scale_bounds
        self.logit_bounds = (float(min(lo, hi)), float(max(lo, hi)))
        self.grad_logit_scale = np.zeros_like(self.logit_scale)
        self.v_logit_scale = np.zeros_like(self.logit_scale)

    def _layer_norm(self, pre_act: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(pre_act, axis=1, keepdims=True)
        var = np.var(pre_act, axis=1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.layer_norm_eps)
        norm = (pre_act - mean) * inv_std
        return norm, inv_std, norm * self.ln_gamma + self.ln_beta

    def forward(
        self,
        inputs: np.ndarray,
        *,
        training: bool = False,
        return_cache: bool = False,
    ) -> Tuple[np.ndarray, Optional[_HeadCache]]:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_dim:
            raise ValueError(f"ReadoutMLP expected input shape (batch,{self.in_dim}), got {inputs.shape}")
        z1 = inputs @ self.W1 + self.b1
        ln_norm: Optional[np.ndarray] = None
        ln_inv_std: Optional[np.ndarray] = None
        if self.use_layer_norm:
            norm, inv_std, z1 = self._layer_norm(z1)
            ln_norm = norm.astype(np.float32, copy=False)
            ln_inv_std = inv_std.astype(np.float32, copy=False)
        tanh_out = np.tanh(z1).astype(np.float32, copy=False)
        hidden = tanh_out
        dropout_mask: Optional[np.ndarray] = None
        if training and self.dropout_prob > 0.0:
            mask = (self.rng.random(size=hidden.shape) >= self.dropout_prob).astype(np.float32)
            mask /= max(self.keep_prob, 1e-6)
            hidden = hidden * mask
            dropout_mask = mask
        raw_logits = hidden @ self.W2 + self.b2
        logits = raw_logits * float(self.logit_scale[0])
        cache: Optional[_HeadCache] = None
        if return_cache:
            cache = _HeadCache(
                inputs=inputs,
                hidden=hidden,
                tanh_out=tanh_out,
                ln_norm=ln_norm,
                ln_inv_std=ln_inv_std,
                dropout_mask=dropout_mask,
                raw_logits=raw_logits,
            )
        return logits.astype(np.float32, copy=False), cache

    def backward(
        self,
        grad_logits: np.ndarray,
        cache: _HeadCache,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if grad_logits.ndim != 2 or grad_logits.shape[1] != self.out_dim:
            raise ValueError(f"ReadoutMLP grad logits shape mismatch, got {grad_logits.shape}")
        batch = max(1, grad_logits.shape[0])
        inputs = cache.inputs
        hidden = cache.hidden
        grad_W2 = hidden.T @ grad_logits / batch
        grad_b2 = np.sum(grad_logits, axis=0) / batch
        grad_hidden = grad_logits @ self.W2.T
        if cache.dropout_mask is not None:
            grad_hidden = grad_hidden * cache.dropout_mask
        grad_pre = grad_hidden * (1.0 - cache.tanh_out ** 2)
        if self.use_layer_norm and cache.ln_norm is not None and cache.ln_inv_std is not None:
            norm = cache.ln_norm
            inv_std = cache.ln_inv_std
            self.grad_ln_gamma = np.sum(grad_pre * norm, axis=0) / batch
            self.grad_ln_beta = np.sum(grad_pre, axis=0) / batch
            grad_scaled = grad_pre * self.ln_gamma
            sum_scaled = np.sum(grad_scaled, axis=1, keepdims=True)
            sum_scaled_norm = np.sum(grad_scaled * norm, axis=1, keepdims=True)
            hidden_dim = grad_scaled.shape[1]
            grad_pre = (inv_std / hidden_dim) * (
                hidden_dim * grad_scaled - sum_scaled - norm * sum_scaled_norm
            )
        grad_W1 = inputs.T @ grad_pre / batch
        grad_b1 = np.sum(grad_pre, axis=0) / batch
        grad_inputs = grad_pre @ self.W1.T
        if self.learn_logit_scale:
            self.grad_logit_scale = np.sum(grad_logits * cache.raw_logits, axis=0, keepdims=True) / batch
        grads = {
            "W1": grad_W1.astype(np.float32, copy=False),
            "b1": grad_b1.astype(np.float32, copy=False),
            "W2": grad_W2.astype(np.float32, copy=False),
            "b2": grad_b2.astype(np.float32, copy=False),
        }
        if self.use_layer_norm and self.grad_ln_gamma is not None and self.grad_ln_beta is not None:
            grads["ln_gamma"] = self.grad_ln_gamma.astype(np.float32, copy=False)
            grads["ln_beta"] = self.grad_ln_beta.astype(np.float32, copy=False)
        if self.learn_logit_scale:
            grads["logit_scale"] = self.grad_logit_scale.astype(np.float32, copy=False)
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
        if self.use_layer_norm and "ln_gamma" in grads and "ln_beta" in grads:
            self.v_ln_gamma = _update(self.ln_gamma, grads["ln_gamma"], self.v_ln_gamma)
            self.v_ln_beta = _update(self.ln_beta, grads["ln_beta"], self.v_ln_beta)
        if self.learn_logit_scale and "logit_scale" in grads:
            self.v_logit_scale = _update(self.logit_scale, grads["logit_scale"], self.v_logit_scale)
            self.logit_scale[...] = np.clip(self.logit_scale, *self.logit_bounds)
