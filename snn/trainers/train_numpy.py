# -*- coding: utf-8 -*-
"""
NumPy trainer for SNN:
- T-step (three-compartment) + MLP readout head
- FPT (fixed-point) with optional Anderson acceleration
- Stable softmax/NLL, AdamW + Warmup-Cosine, grad-clip
- NATS JetStream events (metrics_batch / metrics_epoch / train_iter / spike / log) via stdlib json
- Dataset normalization (mean/std) + light augmentation (optional)

Usage:
  python -m snn.trainers.train_numpy --config config.yaml

Config keys (subset):
  nats.url
  subjects.metrics / subjects.spikes / subjects.logs
  train: dataset, mode (tstep|fpt), batch_size, epochs, steps_per_epoch (optional),
         lr/logit_scale, optimizer{...}, scheduler{...},
         T, tau_m/tau_a/tau_b, v_th, beta, g_apical,
         K, tol, solver(anderson|plain), anderson_m, anderson_beta,
         head_only(true|false), unfreeze_at_conf, augment(true|false)
  data.root
"""

import os, math, time, json, gzip, struct, argparse, logging, random, asyncio
from typing import Tuple, Dict, Any, Optional
import urllib.request
import numpy as np
import yaml
try:
    import nats  # type: ignore
except Exception:  # pragma: no cover - optional dependency for offline mode
    nats = None

from snn.fpt import fpt_solve

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
log_srv = logging.getLogger("snn.server")
log_fpt = logging.getLogger("snn.fpt")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def now_ms() -> int:
    return int(time.time()*1000)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_npy(path: str, arr: np.ndarray):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)

def load_npy(path: str) -> Optional[np.ndarray]:
    return np.load(path) if os.path.exists(path) else None

# ---------------------------------------------------------------------
# NATS helpers (stdlib json only)
# ---------------------------------------------------------------------
async def js_publish(js, subject: str, payload: Dict[str, Any]):
    headers = {"Content-Type": "application/json", "Nats-Msg-Id": f"{subject}:{now_ms()}"}
    await js.publish(subject, json.dumps(payload).encode("utf-8"), headers=headers)

async def send_log(js, subject_log: str, msg: str):
    await js_publish(js, subject_log, {"ts": time.time(), "msg": msg})

# ---------------------------------------------------------------------
# Dataset: MNIST / FashionMNIST (IDX) with stdlib only + normalization + light aug
# ---------------------------------------------------------------------
MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}
FASHION_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

def _dl(url: str, path: str):
    if not os.path.exists(path):
        log_srv.info(f"[data] downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)

def _read_idx_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        assert magic == 0x00000803, f"bad magic (images): {magic}"
        n,     = struct.unpack(">I", f.read(4))
        rows,  = struct.unpack(">I", f.read(4))
        cols,  = struct.unpack(">I", f.read(4))
        raw = f.read(n*rows*cols)
        X = np.frombuffer(raw, dtype=np.uint8).reshape(n, rows*cols).astype(np.float32)/255.0
        return X

def _read_idx_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        assert magic == 0x00000801, f"bad magic (labels): {magic}"
        n,     = struct.unpack(">I", f.read(4))
        y = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)
        return y

def load_dataset(name: str, data_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    name = name.upper()
    urls = MNIST_URLS if name == "MNIST" else FASHION_URLS if name in ("FASHIONMNIST","FASHION-MNIST") else None
    if urls is None:
        raise ValueError(f"Unsupported dataset: {name}")
    # Prefer pre-downloaded Keras-style mnist.npz if available under common paths
    if name == "MNIST":
        candidates = [
            os.path.join(data_root, "mnist", "mnist.npz"),
            os.path.join(data_root, "MNIST", "mnist.npz"),
            os.path.join(data_root, "mnist.npz"),
        ]
        for npz_path in candidates:
            if os.path.exists(npz_path):
                with np.load(npz_path) as data:
                    Xtr = data["x_train"].reshape(data["x_train"].shape[0], -1).astype(np.float32) / 255.0
                    ytr = data["y_train"].astype(np.int64)
                    Xte = data["x_test"].reshape(data["x_test"].shape[0], -1).astype(np.float32) / 255.0
                    yte = data["y_test"].astype(np.int64)
                in_dim = Xtr.shape[1]
                num_classes = 10
                return Xtr, ytr, Xte, yte, in_dim, num_classes

    base = os.path.join(data_root, name)
    ensure_dir(base)
    paths = {k: os.path.join(base, os.path.basename(v)) for k, v in urls.items()}
    for k, u in urls.items():
        _dl(u, paths[k])
    Xtr = _read_idx_images(paths["train_images"])
    ytr = _read_idx_labels(paths["train_labels"])
    Xte = _read_idx_images(paths["test_images"])
    yte = _read_idx_labels(paths["test_labels"])
    in_dim = Xtr.shape[1]
    num_classes = 10
    return Xtr, ytr, Xte, yte, in_dim, num_classes


def spectral_norm(W: np.ndarray, power_iters: int = 20, eps: float = 1e-8) -> float:
    if W.size == 0:
        return 0.0
    mat = np.asarray(W, dtype=np.float64)
    rng = np.random.default_rng()
    v = rng.standard_normal((mat.shape[1], 1))
    v /= np.linalg.norm(v) + eps
    u = mat @ v
    for _ in range(power_iters):
        norm_u = np.linalg.norm(u)
        if norm_u < eps:
            break
        u /= max(norm_u, eps)
        v = mat.T @ u
        norm_v = np.linalg.norm(v)
        if norm_v < eps:
            break
        v /= max(norm_v, eps)
        u = mat @ v
    sigma = float((u.T @ mat @ v))
    return abs(sigma)


def rescale_to_spectral_radius(W: np.ndarray, rho: float, power_iters: int = 20) -> Tuple[float, float]:
    sigma = spectral_norm(W, power_iters=power_iters)
    if rho <= 0.0 or sigma <= 0.0:
        return sigma, sigma
    if sigma <= rho:
        return sigma, sigma
    scale = rho / (sigma + 1e-12)
    W *= scale
    return sigma, sigma * scale


def as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "off"}
    return bool(value)

def compute_or_load_norm(name: str, data_root: str, Xtr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    stat_dir = os.path.join(data_root, name.upper())
    m_path = os.path.join(stat_dir, "mean.npy")
    s_path = os.path.join(stat_dir, "std.npy")
    mean = load_npy(m_path)
    std  = load_npy(s_path)
    if mean is None or std is None:
        mean = Xtr.mean(axis=0, dtype=np.float32)
        var  = Xtr.var(axis=0, dtype=np.float32)
        std  = np.sqrt(np.clip(var, 1e-6, None))
        save_npy(m_path, mean); save_npy(s_path, std)
        log_srv.info(f"[data] computed mean/std -> {stat_dir}")
    return mean.astype(np.float32), std.astype(np.float32)

def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / np.clip(std, 1e-6, None)

def _img_reshape(x: np.ndarray, side: int) -> np.ndarray:
    # reshape [B, side*side] -> [B, side, side]
    return x.reshape(-1, side, side)

def _img_unshape(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)

def random_augment(flat: np.ndarray, side: int = 28) -> np.ndarray:
    """ Minimal augmentation: random shift (±2px), random horizontal flip. """
    B = flat.shape[0]
    img = _img_reshape(flat, side)
    # random flip
    mask = np.random.rand(B) < 0.5
    img[mask] = img[mask, :, ::-1]
    # random shift
    tx = np.random.randint(-2, 3)
    ty = np.random.randint(-2, 3)
    pad = ((0,0), (2,2), (2,2))
    pad_img = np.pad(img, pad, mode='constant')
    sx = 2 + tx; sy = 2 + ty
    img = pad_img[:, sx:sx+side, sy:sy+side]
    return _img_unshape(img)

def iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, steps_per_epoch: Optional[int]=None, augment=False) -> Tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    idx = np.arange(N)
    if steps_per_epoch is None:
        np.random.shuffle(idx)
        for i in range(0, N, batch_size):
            sel = idx[i:i+batch_size]
            xb = X[sel]
            if augment:
                xb = random_augment(xb)
            yield xb, y[sel]
    else:
        # sample with replacement to reach steps_per_epoch
        for _ in range(steps_per_epoch):
            sel = np.random.choice(N, size=batch_size, replace=True)
            xb = X[sel]
            if augment:
                xb = random_augment(xb)
            yield xb, y[sel]

# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------
def softmax_nll(logits: np.ndarray, y_idx: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Stable softmax + NLL using log-sum-exp trick."""
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    nll = -log_probs[np.arange(y_idx.shape[0]), y_idx].mean()
    probs = np.exp(log_probs)
    return float(nll), probs, log_probs

def topk_acc(probs: np.ndarray, y_idx: np.ndarray, k: int=5) -> float:
    idx = np.argpartition(-probs, k-1, axis=1)[:, :k]
    return float(np.mean((idx == y_idx[:, None]).any(axis=1)))

def l2_norm(arrs) -> float:
    if isinstance(arrs, (list, tuple)):
        s = 0.0
        for a in arrs:
            s += float(np.sum(a*a))
        return math.sqrt(s)
    return math.sqrt(float(np.sum(arrs*arrs)))

# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------
def save_np_checkpoint(path: str, mode: str, core, head) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    if mode == "fpt":
        payload.update({
            "Wxh": np.asarray(core.Wxh, dtype=np.float32),
            "Whh": np.asarray(core.Whh, dtype=np.float32),
            "bh":  np.asarray(core.bh,  dtype=np.float32),
        })
    else:
        payload.update({
            "Wxb": np.asarray(core.Wxb, dtype=np.float32),
            "Wxa": np.asarray(core.Wxa, dtype=np.float32),
            "Whb": np.asarray(core.Whb, dtype=np.float32),
            "Wha": np.asarray(core.Wha, dtype=np.float32),
            "g":   np.array([float(getattr(core, "g", 0.0))], dtype=np.float32),
            "tau_m": np.array([float(getattr(core, "tau_m", 0.0))], dtype=np.float32),
            "tau_a": np.array([float(getattr(core, "tau_a", 0.0))], dtype=np.float32),
            "tau_b": np.array([float(getattr(core, "tau_b", 0.0))], dtype=np.float32),
            "v_th": np.array([float(getattr(core, "v_th", 0.0))], dtype=np.float32),
            "beta": np.array([float(getattr(core, "beta", 0.0))], dtype=np.float32),
            "T":    np.array([int(getattr(core, "T", 0))], dtype=np.int32),
        })
    payload.update({
        "head_W1": np.asarray(head.W1, dtype=np.float32),
        "head_b1": np.asarray(head.b1, dtype=np.float32),
        "head_W2": np.asarray(head.W2, dtype=np.float32),
        "head_b2": np.asarray(head.b2, dtype=np.float32),
        "logit_scale": np.asarray(head.logit_scale, dtype=np.float32),
        "mode": np.array([1 if mode == "fpt" else 0], dtype=np.int32),
    })
    if getattr(head, "layer_norm", False):
        payload["ln_gamma"] = np.asarray(head.ln_gamma, dtype=np.float32)
        payload["ln_beta"] = np.asarray(head.ln_beta, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path

# ---------------------------------------------------------------------
# Readout Head (MLP)
# ---------------------------------------------------------------------
class ReadoutMLP:
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        seed: int = 42,
        init_scale: float = 0.02,
        *,
        layer_norm: bool = True,
        dropout_prob: float = 0.0,
        learn_logit_scale: bool = False,
        logit_scale_init: float = 1.25,
        scale_bounds: Tuple[float, float] = (0.5, 3.0),
        ln_eps: float = 1e-5,
    ):
        self.rng = np.random.RandomState(seed)
        self.W1 = (self.rng.randn(in_dim, hidden).astype(np.float32) * init_scale)
        self.b1 = np.zeros((hidden,), np.float32)
        self.W2 = (self.rng.randn(hidden, out_dim).astype(np.float32) * init_scale)
        self.b2 = np.zeros((out_dim,), np.float32)
        self.layer_norm = bool(layer_norm)
        self.layer_norm_eps = float(ln_eps)
        if self.layer_norm:
            self.ln_gamma = np.ones((hidden,), np.float32)
            self.ln_beta = np.zeros((hidden,), np.float32)
            self.g_ln_gamma = np.zeros_like(self.ln_gamma)
            self.g_ln_beta = np.zeros_like(self.ln_beta)
        else:
            self.ln_gamma = None
            self.ln_beta = None
            self.g_ln_gamma = None
            self.g_ln_beta = None
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 0.99))
        self.keep_prob = 1.0 - self.dropout_prob
        self.learn_logit_scale = bool(learn_logit_scale)
        self.logit_scale = np.array([float(logit_scale_init)], dtype=np.float32)
        lo, hi = scale_bounds
        self.logit_bounds = (float(min(lo, hi)), float(max(lo, hi)))
        # grads
        self.gW1 = np.zeros_like(self.W1)
        self.gb1 = np.zeros_like(self.b1)
        self.gW2 = np.zeros_like(self.W2)
        self.gb2 = np.zeros_like(self.b2)
        self.g_scale = np.zeros_like(self.logit_scale)

    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        pre = X @ self.W1 + self.b1
        ln_norm = None
        ln_inv_std = None
        if self.layer_norm:
            mean = np.mean(pre, axis=1, keepdims=True)
            var = np.var(pre, axis=1, keepdims=True)
            ln_inv_std = 1.0 / np.sqrt(var + self.layer_norm_eps)
            ln_norm = (pre - mean) * ln_inv_std
            pre = ln_norm * self.ln_gamma + self.ln_beta
            ln_norm = ln_norm.astype(np.float32, copy=False)
            ln_inv_std = ln_inv_std.astype(np.float32, copy=False)
        tanh_out = np.tanh(pre).astype(np.float32, copy=False)
        hidden = tanh_out
        dropout_mask = None
        if training and self.dropout_prob > 0.0:
            mask = (self.rng.rand(*hidden.shape) >= self.dropout_prob).astype(np.float32)
            mask /= max(self.keep_prob, 1e-6)
            hidden = hidden * mask
            dropout_mask = mask
        raw_logits = hidden @ self.W2 + self.b2
        logits = raw_logits * float(self.logit_scale[0])
        cache = {
            "X": X,
            "hidden": hidden,
            "tanh": tanh_out,
            "ln_norm": ln_norm,
            "ln_inv_std": ln_inv_std,
            "dropout": dropout_mask,
            "raw_logits": raw_logits,
        }
        return logits, cache

    def backward(self, cache: Dict[str, np.ndarray], dlogits: np.ndarray):
        scale = float(self.logit_scale[0])
        dlogits_scaled = dlogits * scale

        hidden = cache["hidden"]
        X = cache["X"]
        self.gW2[...] = hidden.T @ dlogits_scaled
        self.gb2[...] = dlogits_scaled.sum(axis=0)
        d_hidden = dlogits_scaled @ self.W2.T
        if cache["dropout"] is not None:
            d_hidden *= cache["dropout"]
        d_pre = d_hidden * (1.0 - cache["tanh"] ** 2)
        if self.layer_norm and cache["ln_norm"] is not None:
            norm = cache["ln_norm"]
            inv_std = cache["ln_inv_std"]
            self.g_ln_gamma[...] = np.sum(d_pre * norm, axis=0)
            self.g_ln_beta[...] = np.sum(d_pre, axis=0)
            scaled = d_pre * self.ln_gamma
            sum_scaled = np.sum(scaled, axis=1, keepdims=True)
            sum_scaled_norm = np.sum(scaled * norm, axis=1, keepdims=True)
            H = scaled.shape[1]
            d_pre = (inv_std / H) * (H * scaled - sum_scaled - norm * sum_scaled_norm)
        self.gW1[...] = X.T @ d_pre
        self.gb1[...] = d_pre.sum(axis=0)
        if self.learn_logit_scale:
            raw_logits = cache["raw_logits"]
            self.g_scale[...] = np.sum(dlogits * raw_logits, dtype=np.float32, keepdims=True)

    def params(self):
        params = [self.W1, self.b1, self.W2, self.b2]
        if self.layer_norm:
            params.extend([self.ln_gamma, self.ln_beta])
        if self.learn_logit_scale:
            params.append(self.logit_scale)
        return params

    def grads(self):
        grads = [self.gW1, self.gb1, self.gW2, self.gb2]
        if self.layer_norm:
            grads.extend([self.g_ln_gamma, self.g_ln_beta])
        if self.learn_logit_scale:
            grads.append(self.g_scale)
        return grads

    def clamp_scale(self):
        lo, hi = self.logit_bounds
        self.logit_scale[...] = np.clip(self.logit_scale, lo, hi)
# ---------------------------------------------------------------------
# Three-compartment T-step (hidden fixed, train head only)
# ---------------------------------------------------------------------
class ThreeCompTStep:
    def __init__(self, in_dim: int, hidden: int, cfg: Dict[str, Any], seed: int = 42):
        rs = np.random.RandomState(seed)
        self.hidden = hidden
        # fixed random hidden weights (not trained)
        self.Wxb = rs.randn(in_dim, hidden).astype(np.float32) * math.sqrt(2/(in_dim+hidden))
        self.Wxa = rs.randn(in_dim, hidden).astype(np.float32) * math.sqrt(2/(in_dim+hidden))
        self.Whb = rs.randn(hidden, hidden).astype(np.float32) * math.sqrt(2/hidden)
        self.Wha = rs.randn(hidden, hidden).astype(np.float32) * math.sqrt(2/hidden)
        self.g = float(cfg.get("g_apical", 0.5))
        # dynamics
        self.T = int(cfg.get("T", 20))
        self.tau_m = float(cfg.get("tau_m", 10.0))
        self.tau_b = float(cfg.get("tau_b", 8.0))
        self.tau_a = float(cfg.get("tau_a", 12.0))
        self.dt    = float(cfg.get("dt", 1.0))
        self.v_th  = float(cfg.get("v_th", 0.5))
        self.beta  = float(cfg.get("beta", 5.0))

    def forward_states(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return S=[B,T,H] (soft spikes) and s_mean=[B,H]"""
        B = X.shape[0]; H = self.hidden; T = self.T
        a = np.zeros((B,H), np.float32); b = np.zeros((B,H), np.float32); v = np.zeros((B,H), np.float32)
        s_prev = np.zeros((B,H), np.float32)
        alpha_m = 1.0 - self.dt/self.tau_m
        alpha_b = 1.0 - self.dt/self.tau_b
        alpha_a = 1.0 - self.dt/self.tau_a
        S = []
        for _ in range(T):
            b = alpha_b*b + X @ self.Wxb + s_prev @ self.Whb
            a = alpha_a*a + X @ self.Wxa + s_prev @ self.Wha
            v = alpha_m*v + b + self.g*a - self.v_th*s_prev
            s = 1.0/(1.0 + np.exp(-self.beta*(v - self.v_th)))  # surrogate spike
            S.append(s)
            s_prev = s
        S = np.stack(S, axis=1)   # [B,T,H]
        s_mean = S.mean(axis=1)   # [B,H]
        return S, s_mean

# ---------------------------------------------------------------------
# FPT block
# ---------------------------------------------------------------------
class FPTBlock:
    def __init__(self, in_dim: int, hidden: int, seed: int = 42):
        rs = np.random.RandomState(seed)
        self.Wxh = rs.randn(in_dim, hidden).astype(np.float32) * math.sqrt(2/(in_dim+hidden))
        self.Whh = rs.randn(hidden, hidden).astype(np.float32) * math.sqrt(2/hidden)
        self.bh  = np.zeros((hidden,), np.float32)

    @staticmethod
    def phi(x, h, Wxh, Whh, bh):
        pre = x @ Wxh + h @ Whh + bh
        pre = np.clip(pre, -8.0, 8.0)
        return np.tanh(pre)

    @staticmethod
    def anderson_solve(
        x: np.ndarray,
        h0: np.ndarray,
        Wxh: np.ndarray,
        Whh: np.ndarray,
        bh: np.ndarray,
        K: int,
        tol: float,
        m: int = 4,
        beta: float = 0.5,
        *,
        solver: str = "anderson",
        line_search: bool = True,
        ridge: float = 1e-4,
        fp_err_guard: float = 5.0,
        damping: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[np.ndarray, int, float, float]:
        x_term = np.asarray(x @ Wxh, dtype=np.float64)
        Whh64 = np.asarray(Whh, dtype=np.float64)
        bh64 = np.asarray(bh, dtype=np.float64)

        def _phi(state: np.ndarray) -> np.ndarray:
            state64 = np.asarray(state, dtype=np.float64)
            pre = np.clip(x_term + state64 @ Whh64 + bh64, -8.0, 8.0)
            return np.tanh(pre)

        trace: List[Tuple[int, float, float]] = []
        h_init = np.asarray(h0, dtype=np.float64)
        h_next, k_eff, fp_err, iter_err, solver_used = fpt_solve(
            _phi,
            h_init,
            K,
            tol,
            solver=solver,
            m=m,
            beta=beta,
            ridge=ridge,
            line_search=line_search,
            fp_err_guard=fp_err_guard,
            damping=damping,
            logger=logger,
            trace=trace,
        )
        return h_next.astype(np.float32), k_eff, fp_err, iter_err, solver_used
# ---------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------
class AdamW:
    def __init__(self, param_groups: list, betas=(0.9, 0.999), eps=1e-8):
        if not param_groups:
            raise ValueError("param_groups must be non-empty")
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.groups = []
        self.group_map = {}
        for idx, group in enumerate(param_groups):
            name = str(group.get("name", f"group{idx}"))
            if name in self.group_map:
                raise ValueError(f"duplicate optimizer group name: {name}")
            params = list(group.get("params", []))
            entry = {
                "name": name,
                "params": params,
                "m": [np.zeros_like(p) for p in params],
                "v": [np.zeros_like(p) for p in params],
                "lr": float(group.get("lr", 1e-3)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "grads": None,
            }
            self.groups.append(entry)
            self.group_map[name] = entry

    @property
    def group_names(self) -> list:
        return [g["name"] for g in self.groups]

    def has_group(self, name: str) -> bool:
        return name in self.group_map

    def get_group_lr(self, name: str) -> Optional[float]:
        group = self.group_map.get(name)
        return float(group["lr"]) if group is not None else None

    def set_group_lrs(self, lrs: list):
        if len(lrs) != len(self.groups):
            raise ValueError("lr schedule length must match optimizer groups")
        for lr, group in zip(lrs, self.groups):
            group["lr"] = float(lr)

    def add_params(self, name: str, new_params: list):
        if not new_params:
            return
        group = self._require_group(name)
        group["params"].extend(new_params)
        group["m"].extend([np.zeros_like(p) for p in new_params])
        group["v"].extend([np.zeros_like(p) for p in new_params])
        group["grads"] = None

    def set_grads(self, grads_by_group: Dict[str, Optional[list]]):
        for group in self.groups:
            grads = grads_by_group.get(group["name"])
            if grads is None:
                group["grads"] = None
                continue
            if len(grads) != len(group["params"]):
                raise ValueError(f"gradient list for {group['name']} does not match parameters")
            group["grads"] = grads

    def step(self, grad_clip: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        self.t += 1
        group_norm_sq = {}
        total_gnorm_sq = 0.0
        for group in self.groups:
            grads = group["grads"]
            if not grads:
                group_norm_sq[group["name"]] = 0.0
                continue
            g_sq = 0.0
            for grad in grads:
                g_sq += float(np.sum(grad * grad))
            group_norm_sq[group["name"]] = g_sq
            total_gnorm_sq += g_sq
        total_gnorm = math.sqrt(total_gnorm_sq)
        scale = 1.0
        if grad_clip is not None and grad_clip > 0.0 and total_gnorm > grad_clip:
            scale = grad_clip / (total_gnorm + 1e-12)

        stats: Dict[str, Dict[str, float]] = {}
        for group in self.groups:
            grads = group["grads"]
            delta_sq = 0.0
            if grads:
                for i, (param, grad) in enumerate(zip(group["params"], grads)):
                    gi = grad * scale
                    param *= (1.0 - group["weight_decay"])
                    group["m"][i] = self.b1 * group["m"][i] + (1.0 - self.b1) * gi
                    group["v"][i] = self.b2 * group["v"][i] + (1.0 - self.b2) * (gi * gi)
                    m_hat = group["m"][i] / (1.0 - self.b1 ** self.t)
                    v_hat = group["v"][i] / (1.0 - self.b2 ** self.t)
                    step = group["lr"] * m_hat / (np.sqrt(v_hat) + self.eps)
                    param[...] -= step
                    delta_sq += float(np.sum(step * step))
            stats[group["name"]] = {
                "grad_norm": math.sqrt(group_norm_sq.get(group["name"], 0.0)),
                "delta_norm": math.sqrt(delta_sq),
            }
        return stats

    def _require_group(self, name: str) -> dict:
        if name not in self.group_map:
            raise KeyError(f"optimizer group '{name}' not found")
        return self.group_map[name]

class WarmupCosine:
    def __init__(self, base_lrs: list, total_steps: int, warmup_ratio: float=0.05, min_lr_ratio: float=0.1):
        self.base = base_lrs
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(self.total_steps * warmup_ratio))
        self.min_lr_ratio = min_lr_ratio
        self.step_idx = 0

    def get_lrs(self) -> list:
        t = self.step_idx
        if t < self.warmup_steps:
            w = (t+1) / max(1, self.warmup_steps)
            return [b * w for b in self.base]
        # cosine
        tt = min(1.0, (t - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        ratio = self.min_lr_ratio + 0.5*(1.0 - self.min_lr_ratio)*(1.0 + math.cos(math.pi*tt))
        return [b * ratio for b in self.base]

    def step(self):
        self.step_idx += 1
# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class NPTrainer:
    def __init__(self, cfg: Dict[str, Any], js, subjects: Dict[str, str]):
        self.cfg = cfg
        self.js = js
        self.sb = subjects
        self.tr = cfg["train"]
        self.data = cfg["data"]
        self.mode = self.tr.get("mode", "tstep").lower()
        self.seed = int(self.tr.get("seed", 42))
        np.random.seed(self.seed); random.seed(self.seed)

    async def train(self):
        # Load data
        Xtr, ytr, Xte, yte, in_dim, num_classes = load_dataset(self.tr.get("dataset","MNIST"), self.data.get("root","./data"))
        mean, std = compute_or_load_norm(self.tr.get("dataset","MNIST"), self.data.get("root","./data"), Xtr)
        Xtr = normalize(Xtr, mean, std); Xte = normalize(Xte, mean, std)
        await send_log(self.js, self.sb["logs"], f"dataset={self.tr.get('dataset')} in_dim={in_dim}, train={Xtr.shape[0]} val={Xte.shape[0]}")

        # Build model
        hidden = int(self.tr.get("network_size", 256))
        head_hidden = int(self.tr.get("head_hidden", 256))
        head_ln = as_bool(self.tr.get("head_ln", True), True)
        head_dropout = float(self.tr.get("head_dropout", 0.0))
        head_dropout = float(np.clip(head_dropout, 0.0, 0.99))
        learn_scale = bool(self.tr.get("learn_logit_scale", True))
        logit_scale_init = float(self.tr.get("logit_scale_init", 1.60))
        scale_lo = float(self.tr.get("logit_scale_min", 0.5))
        scale_hi = float(self.tr.get("logit_scale_max", 3.0))
        scale_bounds = (min(scale_lo, scale_hi), max(scale_lo, scale_hi))

        if self.mode == "tstep":
            core = ThreeCompTStep(in_dim, hidden, self.tr, seed=self.seed)
            head_in = hidden
        else:
            core = FPTBlock(in_dim, hidden, seed=self.seed)
            head_in = hidden

        head = ReadoutMLP(
            head_in,
            head_hidden,
            num_classes,
            seed=self.seed,
            layer_norm=head_ln,
            dropout_prob=head_dropout,
            learn_logit_scale=learn_scale,
            logit_scale_init=logit_scale_init,
            scale_bounds=scale_bounds,
        )
        solver_cfg = self.tr.get("solver", "anderson").lower()
        ema_decay_cfg = float(self.tr.get("ema_decay", 0.0))
        ema_in_use = 0.0 < ema_decay_cfg < 1.0

        # Optimizer / Scheduler
        head_lr = float(self.tr.get("head_lr", self.tr.get("lr", 1e-3)))
        rec_lr  = float(self.tr.get("rec_lr",  self.tr.get("lr", 1e-3)))
        head_only = bool(self.tr.get("head_only", True))
        unfreeze_at_conf = float(self.tr.get("unfreeze_at_conf", 0.15))
        weight_decay_head = float(self.tr.get("weight_decay_head", self.tr.get("weight_decay", 1e-4)))
        weight_decay_rec = float(self.tr.get("weight_decay_rec", self.tr.get("weight_decay", 1e-4)))
        grad_clip = float(self.tr.get("grad_clip", 1.0))
        total_steps = int(self.tr.get("total_steps", 0))
        warmup_ratio = float(self.tr.get("warmup_ratio", 0.05))
        min_lr_ratio = float(self.tr.get("min_lr_ratio", 0.1))
        fp_damping = float(self.tr.get("fp_damping", 1.0))
        line_search_cfg = as_bool(self.tr.get("line_search", True), True)
        fp_guard = float(self.tr.get("fp_err_guard", 1e-3))
        val_max_steps = int(self.tr.get("val_max_steps", 0))

        param_groups = [
            {
                "name": "head",
                "params": head.params(),
                "lr": head_lr,
                "weight_decay": weight_decay_head,
            }
        ]
        has_rec_group = self.mode == "fpt"
        if has_rec_group:
            rec_params = [] if head_only else [core.Wxh, core.Whh, core.bh]
            param_groups.append(
                {
                    "name": "rec",
                    "params": rec_params,
                    "lr": rec_lr,
                    "weight_decay": weight_decay_rec,
                }
            )

        opt = AdamW(param_groups, betas=(0.9,0.999), eps=1e-8)

        # steps_per_epoch
        batch = int(self.tr.get("batch_size", 64))
        steps_per_epoch = self.tr.get("steps_per_epoch", None)
        if steps_per_epoch is None or int(steps_per_epoch) <= 0:
            steps_per_epoch = math.ceil(Xtr.shape[0] / batch)
        total_steps = total_steps if total_steps > 0 else steps_per_epoch * int(self.tr.get("epochs", 1))
        base_lrs = [head_lr]
        if has_rec_group:
            base_lrs.append(rec_lr)
        sch = WarmupCosine(base_lrs, total_steps, warmup_ratio=warmup_ratio, min_lr_ratio=min_lr_ratio)

        augment = bool(self.tr.get("augment", True))

        await send_log(
            self.js,
            self.sb["logs"],
            f"trainer start mode={self.mode} hidden={hidden} head_hidden={head_hidden} head_only={head_only} "
            f"ln={head_ln} dropout={head_dropout:.2f} lr_head={head_lr} lr_rec={rec_lr} "
            f"logit_scale={float(head.logit_scale[0])} solver={solver_cfg} steps/ep={steps_per_epoch}",
        )

        best_acc = 0.0
        best_loss = 1e9
        global_step = 0

        epochs = int(self.tr.get("epochs", 1))
        rate_target = float(self.tr.get("rate_target", 0.20))
        rate_lambda = float(self.tr.get("rate_reg_lambda", 0.0))

        for ep in range(1, epochs+1):
            t0 = time.perf_counter()
            seen = 0; corr = 0; loss_sum = 0.0; steps = 0
            top5_acc_sum = 0.0
            train_meter = {
                "count": 0,
                "loss": 0.0,
                "acc": 0.0,
                "top5": 0.0,
                "conf": 0.0,
                "entropy": 0.0,
                "logit_mean": 0.0,
                "logit_std": 0.0,
                "logit_scale": 0.0,
                "grad_norm": 0.0,
                "delta_norm": 0.0,
                "gnorm_rec": 0.0,
                "gnorm_rec_count": 0,
                "lr_head": 0.0,
                "lr_rec": 0.0,
                "lr_rec_count": 0,
                "throughput": 0.0,
                "s_rate": 0.0,
                "s_rate_count": 0,
                "fp_err": 0.0,
                "fp_err_count": 0,
                "iter_err": 0.0,
                "iter_err_count": 0,
            }
            # optional anneal g_apical/beta/v_th for T-step
            if self.mode == "tstep":
                g0 = float(self.tr.get("g_apical_start", core.g))
                g1 = float(self.tr.get("g_apical_end",   core.g))
                core.g = g0 + (g1 - g0) * (ep-1)/max(1, (epochs-1))
                # optional beta/v_th anneal
                b0 = float(self.tr.get("beta_start", core.beta)); b1 = float(self.tr.get("beta_end", core.beta))
                v0 = float(self.tr.get("vth_start", core.v_th));  v1 = float(self.tr.get("vth_end", core.v_th))
                core.beta = b0 + (b1-b0)*(ep-1)/max(1,(epochs-1))
                core.v_th = v0 + (v1-v0)*(ep-1)/max(1,(epochs-1))

            # train loop
            last_conf_b = None
            low_conf_streak = 0
            for step, (xb, yb) in enumerate(iter_minibatches(Xtr, ytr, batch, steps_per_epoch, augment=augment), start=1):
                fp_err = None
                iter_err = None
                global_step += 1
                # forward (core)
                skip_updates_due_fp = False
                if self.mode == "tstep":
                    S, s_mean = core.forward_states(xb)
                    Z = s_mean
                else:
                    K_max = int(self.tr.get("K", 4))
                    tol = float(self.tr.get("tol", 1e-5))
                    h0 = np.zeros((xb.shape[0], hidden), np.float32)
                    beta_use = float(self.tr.get("anderson_beta", 0.5))
                    hK, k_eff, fp_err, iter_err, solver_used = FPTBlock.anderson_solve(
                        xb,
                        h0,
                        core.Wxh,
                        core.Whh,
                        core.bh,
                        K=K_max,
                        tol=tol,
                        m=int(self.tr.get("anderson_m", 4)),
                        beta=beta_use,
                        solver=solver_cfg,
                        line_search=line_search_cfg,
                        ridge=float(self.tr.get("anderson_ridge", 1e-4)),
                        fp_err_guard=fp_guard,
                        damping=fp_damping,
                        logger=log_fpt,
                    )
                    auto_fallback = solver_used != solver_cfg
                    bad_fp = fp_err is None or not np.isfinite(fp_err) or (fp_err is not None and fp_err > fp_guard)
                    if auto_fallback:
                        fp_err_msg = f"{fp_err:.3e}" if fp_err is not None else "nan"
                        await send_log(
                            self.js,
                            self.sb["logs"],
                            f"[solver] epoch={ep} step={step} fallback {solver_cfg}->{solver_used} "
                            f"k={int(k_eff)} fp_err={fp_err_msg}",
                        )
                    if bad_fp:
                        skip_updates_due_fp = True
                        fp_err_msg = f"{fp_err:.3e}" if fp_err is not None else "nan"
                        await send_log(
                            self.js,
                            self.sb["logs"],
                            f"[solver] epoch={ep} step={step} fp_err={fp_err_msg} trigger plain fallback",
                        )
                        hK, k_eff, fp_err, iter_err, solver_used = FPTBlock.anderson_solve(
                            xb,
                            h0,
                            core.Wxh,
                            core.Whh,
                            core.bh,
                            K=max(K_max, 6),
                            tol=tol,
                            m=1,
                            beta=0.2,
                            solver="plain",
                            line_search=False,
                            ridge=float(self.tr.get("anderson_ridge", 1e-4)),
                            fp_err_guard=fp_guard,
                            damping=fp_damping,
                            logger=log_fpt,
                        )
                    Z = hK
                    await js_publish(
                        self.js,
                        self.sb["train_iter"] if "train_iter" in self.sb else self.sb["logs"],
                        {
                            "epoch": ep,
                            "step": step,
                            "k": int(k_eff),
                            "solver": solver_used,
                            "residual": float(fp_err) if fp_err is not None else None,
                            "iter_err": float(iter_err) if iter_err is not None else None,
                            "ts": time.time(),
                        },
                    )

                if not np.all(np.isfinite(Z)):
                    await send_log(
                        self.js,
                        self.sb["logs"],
                        f"[nan-guard] epoch={ep} step={step} 检测到隐藏状态非法值，跳过该 batch",
                    )
                    continue

                # forward (head)
                logits, cache = head.forward(Z, training=True)
                if not np.all(np.isfinite(logits)):
                    await send_log(
                        self.js,
                        self.sb["logs"],
                        f"[nan-guard] epoch={ep} step={step} 检测到 logits 非法值，跳过该 batch",
                    )
                    continue
                nll, probs, log_probs = softmax_nll(logits, yb)
                pred = np.argmax(probs, axis=1)
                acc_b = float(np.mean(pred == yb))
                top5_b = topk_acc(probs, yb, k=5)
                conf_b = float(np.mean(probs[np.arange(yb.shape[0]), yb]))
                entropy_b = float(-np.mean(np.sum(probs * np.log(np.clip(probs,1e-9,None)), axis=1)))
                logit_mean = float(np.mean(logits))
                logit_std  = float(np.std(logits))

                conf_drop = last_conf_b is not None and (last_conf_b - conf_b) > 0.1
                low_conf_streak = low_conf_streak + 1 if conf_b < 0.18 else 0
                if low_conf_streak >= 200:
                    prev_scale = float(head.logit_scale[0])
                    head.logit_scale[...] = np.clip(head.logit_scale * 1.1, scale_bounds[0], scale_bounds[1])
                    low_conf_streak = 0
                    await send_log(
                        self.js,
                        self.sb["logs"],
                        f"[auto-calibrate] epoch={ep} step={step} logit_scale {prev_scale:.4f}->{float(head.logit_scale[0]):.4f}",
                    )
                skip_reason_conf = logit_std > 10.0 or conf_drop
                skip_update = skip_reason_conf or skip_updates_due_fp
                if skip_reason_conf:
                    head.logit_scale[...] = np.clip(head.logit_scale * 0.8, scale_bounds[0], scale_bounds[1])
                    await send_log(
                        self.js,
                        self.sb["logs"],
                        f"[fuse] epoch={ep} step={step} reason={'logit_std' if logit_std > 10.0 else 'confidence_drop'}",
                    )

                stats: Dict[str, Dict[str, float]] = {}
                grads_map: Optional[Dict[str, list]] = None
                if not skip_update:
                    dlogits = probs.copy()
                    dlogits[np.arange(yb.shape[0]), yb] -= 1.0
                    dlogits /= max(1, yb.shape[0])

                    # backward head
                    head.backward(cache, dlogits)
                    grads_map = {"head": head.grads()}

                cur_lrs = sch.get_lrs()
                opt.set_group_lrs(cur_lrs)
                if grads_map is not None:
                    opt.set_grads(grads_map)
                    stats = opt.step(grad_clip=grad_clip)
                sch.step()

                # rate regularization (T-step only; monitor; no gradient to head)
                rate = None
                reg_loss = 0.0
                if self.mode == "tstep":
                    rate = float(np.mean(Z))
                    if rate_lambda > 0.0:
                        reg_loss = rate_lambda * (rate - rate_target) ** 2
                        nll += reg_loss  # add to scalar loss, no head gradient

                # optional: unfreeze recurrent after conf high enough (FPT)
                if self.mode == "fpt" and head_only and conf_b >= unfreeze_at_conf:
                    head_only = False
                    opt.add_params("rec", [core.Wxh, core.Whh, core.bh])
                    await send_log(self.js, self.sb["logs"], f"unfreeze recurrent at conf={conf_b:.3f}")

                # meters
                seen += xb.shape[0]
                corr += int((pred == yb).sum())
                loss_sum += float(nll)
                top5_acc_sum += top5_b
                steps += 1
                tps = seen / max(1e-6, time.time()-t0)
                # publish metrics batch
                lr_head_val = opt.get_group_lr("head") or head_lr
                lr_rec_val = opt.get_group_lr("rec") if has_rec_group else None
                head_stats = stats.get("head", {"grad_norm": 0.0, "delta_norm": 0.0})
                rec_stats = None
                if has_rec_group:
                    rec_stats = stats.get("rec", {"grad_norm": 0.0, "delta_norm": 0.0})
                batch_loss = float(nll)
                scale_value = float(head.logit_scale[0])
                payload = {
                    "epoch": ep, "step": step,
                    "loss": batch_loss, "acc": acc_b, "top5": top5_b,
                    "throughput": tps, "lr": lr_head_val,
                    "nll": batch_loss, "conf": conf_b, "entropy": entropy_b,
                    "logit_mean": logit_mean, "logit_std": logit_std,
                    "logit_scale": scale_value,
                    "low_conf_streak": int(low_conf_streak),
                    "s_rate": rate if self.mode == "tstep" else None,
                    "grad_norm": head_stats["grad_norm"], "delta_norm": head_stats["delta_norm"],
                    "gnorm_head": head_stats["grad_norm"],
                    "gnorm_rec": rec_stats["grad_norm"] if rec_stats else None,
                    "fp_err": float(fp_err) if fp_err is not None else None,
                    "iter_err": float(iter_err) if iter_err is not None else None,
                    "lr_head": lr_head_val,
                    "lr_rec": lr_rec_val,
                    "ts": time.time()
                }
                await js_publish(self.js, self.sb["metrics"], payload)

                train_meter["count"] += 1
                train_meter["loss"] += batch_loss
                train_meter["acc"] += acc_b
                train_meter["top5"] += top5_b
                train_meter["conf"] += conf_b
                train_meter["entropy"] += entropy_b
                train_meter["logit_mean"] += logit_mean
                train_meter["logit_std"] += logit_std
                train_meter["logit_scale"] += scale_value
                train_meter["grad_norm"] += head_stats["grad_norm"]
                train_meter["delta_norm"] += head_stats["delta_norm"]
                train_meter["lr_head"] += lr_head_val
                train_meter["throughput"] += tps
                if rate is not None:
                    train_meter["s_rate"] += rate
                    train_meter["s_rate_count"] += 1
                if (not head_only) and rec_stats and rec_stats["grad_norm"] is not None:
                    train_meter["gnorm_rec"] += rec_stats["grad_norm"]
                    train_meter["gnorm_rec_count"] += 1
                if (not head_only) and lr_rec_val is not None:
                    train_meter["lr_rec"] += lr_rec_val
                    train_meter["lr_rec_count"] += 1
                if fp_err is not None:
                    train_meter["fp_err"] += float(fp_err)
                    train_meter["fp_err_count"] += 1
                if iter_err is not None:
                    train_meter["iter_err"] += float(iter_err)
                    train_meter["iter_err_count"] += 1

                # optional spikes (visual)
                if step % 10 == 0:
                    await self._emit_spikes(Z)

                head.clamp_scale()
                last_conf_b = conf_b

            steps_recorded = train_meter["count"]
            if steps_recorded > 0:
                avg = lambda key: train_meter[key] / max(1, steps_recorded)
                lr_rec_avg = (
                    train_meter["lr_rec"] / max(1, train_meter["lr_rec_count"])
                    if train_meter["lr_rec_count"] > 0
                    else None
                )
                gnorm_rec_avg = (
                    train_meter["gnorm_rec"] / max(1, train_meter["gnorm_rec_count"])
                    if train_meter["gnorm_rec_count"] > 0
                    else None
                )
                s_rate_avg = (
                    train_meter["s_rate"] / max(1, train_meter["s_rate_count"])
                    if train_meter["s_rate_count"] > 0
                    else None
                )
                fp_err_avg = (
                    train_meter["fp_err"] / max(1, train_meter["fp_err_count"])
                    if train_meter["fp_err_count"] > 0
                    else None
                )
                iter_err_avg = (
                    train_meter["iter_err"] / max(1, train_meter["iter_err_count"])
                    if train_meter["iter_err_count"] > 0
                    else None
                )
                train_epoch_payload = {
                    "epoch": ep,
                    "step": 0,
                    "phase": "train",
                    "loss": avg("loss"),
                    "nll": avg("loss"),
                    "acc": avg("acc"),
                    "top5": avg("top5"),
                    "conf": avg("conf"),
                    "entropy": avg("entropy"),
                    "logit_mean": avg("logit_mean"),
                    "logit_std": avg("logit_std"),
                    "logit_scale": avg("logit_scale"),
                    "grad_norm": avg("grad_norm"),
                    "delta_norm": avg("delta_norm"),
                    "gnorm_head": avg("grad_norm"),
                    "gnorm_rec": gnorm_rec_avg,
                    "throughput": avg("throughput"),
                    "s_rate": s_rate_avg,
                    "fp_err": fp_err_avg,
                    "iter_err": iter_err_avg,
                    "lr": avg("lr_head"),
                    "lr_head": avg("lr_head"),
                    "lr_rec": lr_rec_avg,
                    "ts": time.time(),
                    "best_acc": float(best_acc),
                    "best_loss": float(best_loss),
                    "ema_in_use": ema_in_use,
                }
                await js_publish(self.js, self.sb["metrics"], train_epoch_payload)

            if self.mode == "fpt" and not head_only:
                rho_target = float(self.tr.get("spectral_rho", 0.9))
                if rho_target > 0.0:
                    sigma_before, sigma_after = rescale_to_spectral_radius(core.Whh, rho_target)
                    if sigma_after < sigma_before - 1e-6:
                        await send_log(
                            self.js,
                            self.sb["logs"],
                            f"[spectral] clamp Whh sigma={sigma_before:.4f}->{sigma_after:.4f}",
                        )

            # validation
            v_seen=v_cor=v_steps=0; v_loss=0.0; v_top5=0.0; v_conf=0.0
            val_steps_use = math.ceil(Xte.shape[0] / batch)
            if val_max_steps > 0:
                val_steps_use = min(val_steps_use, val_max_steps)
            v_entropy_sum = 0.0
            v_logit_mean_sum = 0.0
            v_logit_std_sum = 0.0
            v_fp_err_sum = 0.0
            v_fp_err_count = 0
            v_iter_err_sum = 0.0
            v_iter_err_count = 0
            v_s_rate_sum = 0.0
            v_s_rate_count = 0
            val_loop_start = time.perf_counter()
            for xb, yb in iter_minibatches(Xte, yte, batch, steps_per_epoch=val_steps_use, augment=False):
                if self.mode == "tstep":
                    _, s_mean = core.forward_states(xb)
                    Z = s_mean
                    batch_s_rate = float(np.mean(s_mean))
                    v_s_rate_sum += batch_s_rate
                    v_s_rate_count += 1
                else:
                    h0 = np.zeros((xb.shape[0], hidden), np.float32)
                    hK, _, fp_err_val, iter_err_val, solver_used_val = FPTBlock.anderson_solve(
                        xb,
                        h0,
                        core.Wxh,
                        core.Whh,
                        core.bh,
                        K=int(self.tr.get("K", 4)),
                        tol=float(self.tr.get("tol", 1e-5)),
                        m=int(self.tr.get("anderson_m", 4)),
                        beta=float(self.tr.get("anderson_beta", 0.5)),
                        solver=solver_cfg,
                        line_search=line_search_cfg,
                        ridge=float(self.tr.get("anderson_ridge", 1e-4)),
                        fp_err_guard=fp_guard,
                        damping=fp_damping,
                    )
                    Z = hK
                    if solver_used_val != solver_cfg:
                        fp_val_msg = f"{fp_err_val:.3e}" if fp_err_val is not None else "nan"
                        await send_log(
                            self.js,
                            self.sb["logs"],
                            f"[solver] (val) epoch={ep} fallback {solver_cfg}->{solver_used_val} fp_err={fp_val_msg}",
                        )
                    if fp_err_val is not None:
                        v_fp_err_sum += float(fp_err_val)
                        v_fp_err_count += 1
                    if iter_err_val is not None:
                        v_iter_err_sum += float(iter_err_val)
                        v_iter_err_count += 1
                logits, _ = head.forward(Z, training=False)
                nll, probs, _ = softmax_nll(logits, yb)
                batch_size_eval = xb.shape[0]
                v_loss += float(nll) * batch_size_eval
                v_steps += 1
                v_seen += batch_size_eval
                pred = np.argmax(probs, axis=1)
                batch_correct = int((pred == yb).sum())
                v_cor += batch_correct
                v_top5 += topk_acc(probs, yb, k=5) * batch_size_eval
                v_conf += float(np.mean(probs[np.arange(yb.shape[0]), yb])) * batch_size_eval
                entropy_val = float(-np.mean(np.sum(probs * np.log(np.clip(probs, 1e-9, None)), axis=1)))
                v_entropy_sum += entropy_val * batch_size_eval
                logit_mean_val = float(np.mean(logits))
                logit_std_val = float(np.std(logits))
                v_logit_mean_sum += logit_mean_val * batch_size_eval
                v_logit_std_sum += logit_std_val * batch_size_eval

            train_loss = loss_sum/max(1,steps)
            train_acc  = corr/max(1,seen)
            val_loss = v_loss/max(1,v_seen)
            val_acc  = v_cor/max(1,v_seen)
            top5_epoch = v_top5/max(1,v_seen)
            v_conf_epoch = v_conf/max(1,v_seen)
            val_entropy_avg = v_entropy_sum / max(1, v_seen)
            val_logit_mean_avg = v_logit_mean_sum / max(1, v_seen)
            val_logit_std_avg = v_logit_std_sum / max(1, v_seen)
            val_s_rate_avg = (
                v_s_rate_sum / max(1, v_s_rate_count) if v_s_rate_count > 0 else None
            )
            val_fp_err_avg = (
                v_fp_err_sum / max(1, v_fp_err_count) if v_fp_err_count > 0 else None
            )
            val_iter_err_avg = (
                v_iter_err_sum / max(1, v_iter_err_count) if v_iter_err_count > 0 else None
            )
            val_elapsed = max(1e-6, time.perf_counter() - val_loop_start)
            val_throughput = v_seen / val_elapsed
            val_lr_head = opt.get_group_lr("head") or head_lr
            val_lr_rec = opt.get_group_lr("rec") if (has_rec_group and not head_only) else None
            logit_scale_val = float(head.logit_scale[0])

            prev_best = best_acc
            best_acc = max(best_acc, val_acc)
            best_loss = min(best_loss, val_loss)

            # save best checkpoint
            if val_acc > prev_best:
                ckpt_path = save_np_checkpoint(os.path.join("checkpoints", "np_trainer_best.npz"), self.mode, core, head)
                await send_log(self.js, self.sb["logs"], f"[checkpoint] saved best np trainer epoch={ep} acc={val_acc:.4f} path={ckpt_path}")

            # epoch metrics
            await js_publish(self.js, self.sb["metrics"], {
                "epoch": ep, "step": 0,
                "phase": "val",
                "loss": float(val_loss), "nll": float(val_loss),
                "acc": float(val_acc), "top5": float(top5_epoch),
                "conf": float(v_conf_epoch), "entropy": val_entropy_avg,
                "logit_mean": val_logit_mean_avg, "logit_std": val_logit_std_avg,
                "logit_scale": logit_scale_val,
                "grad_norm": None, "delta_norm": None,
                "gnorm_head": None, "gnorm_rec": None,
                "throughput": val_throughput,
                "s_rate": val_s_rate_avg,
                "fp_err": val_fp_err_avg,
                "iter_err": val_iter_err_avg,
                "lr": val_lr_head,
                "lr_head": val_lr_head,
                "lr_rec": val_lr_rec,
                "best_acc": float(best_acc), "best_loss": float(best_loss),
                "ema_in_use": ema_in_use,
                "ts": time.time()
            })

            # log line
            sec = time.perf_counter()-t0
            await send_log(self.js, self.sb["logs"],
                           f"[epoch {ep}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                           f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} top5={top5_epoch:.4f} "
                           f"avg_throughput={seen/max(1e-6,sec):.2f}")

        await send_log(self.js, self.sb["logs"], "训练完成")

    async def _emit_spikes(self, Z: np.ndarray):
        # Visualize top-active neurons; create light edges to class ids
        act = np.mean(np.abs(Z), axis=0)  # [H]
        thr = float(act.mean() + act.std())
        neurons = np.where(act > thr)[0].tolist()
        edges = []
        for _ in range(min(300, len(neurons))):
            a = int(np.random.choice(neurons))
            b = int(np.random.randint(0, 10))
            edges.append([a, b])
        await js_publish(self.js, self.sb["spikes"], {
            "layer": 1, "t": time.time(), "neurons": neurons[:2000], "edges": edges[:4000]
        })
# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def main_async(args):
    cfg = load_cfg(args.config)
    # setup NATS or offline stub with local capture
    class _LocalCaptureJS:
        def __init__(self) -> None:
            self.events = []  # type: ignore[var-annotated]
        async def publish(self, subject: str, data: bytes, *, headers=None):
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception:
                payload = {"raw": data.decode("utf-8", errors="ignore")}
            self.events.append((subject, payload))
    js = _LocalCaptureJS()
    nc = None
    try:
        url = os.getenv("NATS_URL") or (cfg.get("nats") or {}).get("url")
        if nats is not None and url:
            nc = await nats.connect(url)
            js = nc.jetstream()
    except Exception:
        js = _LocalCaptureJS()

    # subjects map with robust defaults
    subj_cfg = cfg.get("subjects") or {}
    subjects = {
        "metrics": subj_cfg.get("metrics", "snn.metrics.training"),
        "spikes":  subj_cfg.get("spikes",  "snn.spikes.layer.1"),
        "logs":    subj_cfg.get("logs",    "snn.ui.log.training"),
        "train_iter": subj_cfg.get("train_iter", subj_cfg.get("logs", "snn.ui.log.training")),
    }

    trainer = NPTrainer(cfg, js, subjects)
    await trainer.train()
    if nc is not None:
        await nc.drain()
    # Print final validation summary when running offline
    if isinstance(js, _LocalCaptureJS):
        last = None
        for (subj, payload) in js.events:
            if subj == subjects["metrics"] and isinstance(payload, dict) and payload.get("phase") == "val":
                last = payload
        if last is not None:
            print(
                "FINAL:",
                {
                    "epoch": int(last.get("epoch", 0)),
                    "val_loss": float(last.get("loss", 0.0)),
                    "val_acc": float(last.get("acc", 0.0)),
                    "top5": float(last.get("top5", 0.0)),
                },
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
