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
import nats

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

    base = os.path.join(data_root, name)
    ensure_dir(base)
    paths = {k: os.path.join(base, os.path.basename(v)) for k, v in urls.items()}
    for k,u in urls.items():
        _dl(u, paths[k])
    Xtr = _read_idx_images(paths["train_images"])
    ytr = _read_idx_labels(paths["train_labels"])
    Xte = _read_idx_images(paths["test_images"])
    yte = _read_idx_labels(paths["test_labels"])
    in_dim = Xtr.shape[1]
    num_classes = 10
    return Xtr, ytr, Xte, yte, in_dim, num_classes

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
# Readout Head (MLP)
# ---------------------------------------------------------------------
class ReadoutMLP:
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 42, init_scale: float = 0.02, learn_logit_scale: bool = False, logit_scale_init: float = 1.25, scale_bounds=(0.5, 3.0)):
        rs = np.random.RandomState(seed)
        self.W1 = (rs.randn(in_dim, hidden).astype(np.float32) * init_scale)
        self.b1 = np.zeros((hidden,), np.float32)
        self.W2 = (rs.randn(hidden, out_dim).astype(np.float32) * init_scale)
        self.b2 = np.zeros((out_dim,), np.float32)
        self.learn_logit_scale = learn_logit_scale
        self.logit_scale = np.array([logit_scale_init], dtype=np.float32)
        self.logit_bounds = scale_bounds
        # grads
        self.gW1 = np.zeros_like(self.W1); self.gb1 = np.zeros_like(self.b1)
        self.gW2 = np.zeros_like(self.W2); self.gb2 = np.zeros_like(self.b2)
        self.g_scale = np.zeros_like(self.logit_scale)

    @staticmethod
    def act(x):  # tanh
        return np.tanh(x)

    @staticmethod
    def d_act(y):  # derivative wrt output y = tanh(x)
        return 1.0 - y*y

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        H1 = self.act(X @ self.W1 + self.b1)
        logits = H1 @ self.W2 + self.b2
        logits = logits * float(self.logit_scale[0])
        cache = {"X": X, "H1": H1, "logits": logits}
        return logits, cache

    def backward(self, cache: Dict[str, np.ndarray], dlogits: np.ndarray):
        # d logits (with scale)
        scale = float(self.logit_scale[0])
        dlogits_scaled = dlogits * scale

        H1 = cache["H1"]; X = cache["X"]
        # W2, b2
        self.gW2[...] = H1.T @ dlogits_scaled
        self.gb2[...] = dlogits_scaled.sum(axis=0)
        # back to H1
        dH1 = dlogits_scaled @ self.W2.T
        # through tanh
        dpre = dH1 * self.d_act(H1)
        # W1, b1
        self.gW1[...] = X.T @ dpre
        self.gb1[...] = dpre.sum(axis=0)
        # grad to logit_scale if learnable: d(loss)/d(scale) = sum(dlogits * logits_raw)
        if self.learn_logit_scale:
            raw_logits = cache["logits"] / scale  # logits before scaling
            self.g_scale[...] = np.sum(dlogits * raw_logits, dtype=np.float32, keepdims=True)

    def params(self):
        if self.learn_logit_scale:
            return [self.W1, self.b1, self.W2, self.b2, self.logit_scale]
        return [self.W1, self.b1, self.W2, self.b2]

    def grads(self):
        if self.learn_logit_scale:
            return [self.gW1, self.gb1, self.gW2, self.gb2, self.g_scale]
        return [self.gW1, self.gb1, self.gW2, self.gb2]

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
        return np.tanh(x @ Wxh + h @ Whh + bh)

    @staticmethod
    def anderson_solve(x, h0, Wxh, Whh, bh, K, tol, m=4, beta=0.5, logger=None):
        h_prev = h0
        deltas = []
        for k in range(1, K+1):
            h_next = FPTBlock.phi(x, h_prev, Wxh, Whh, bh)
            d = (h_next - h_prev).reshape(h_next.shape[0], -1)
            res = float(np.sqrt((d*d).mean()))
            if logger:
                if k == 1:
                    logger.info(f"固定点迭代开始：步数={x.shape[0]}, 迭代次数={K}, 阈值={tol:.2e}, 阻尼={beta:.2f}")
                logger.info(f"迭代 {k}/{K}，残差={res:.3e}")
            if res < tol:
                if logger:
                    logger.info(f"残差 {res:.3e} 已低于阈值 {tol:.3e}，提前停止迭代")
                return h_next, k, res
            deltas.append((h_next - h_prev).copy())
            if len(deltas) >= m:
                # Simple batch-averaged Anderson mixing
                G = np.stack(deltas[-m:], axis=1).mean(axis=0)  # [H] averaged vector per slot
                # reduce to vector
                g = G.reshape(-1)
                denom = float(np.dot(g, g)) + 1e-9
                alpha = float(np.dot(g, (h_next - h_prev).reshape(-1))) / denom
                h_mix = h_prev + alpha * (h_next - h_prev)
                h_prev = (1.0 - beta) * h_prev + beta * h_mix
            else:
                h_prev = h_next
        return h_prev, K, res
# ---------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------
class AdamW:
    def __init__(self, params: list, lrs: list, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4):
        self.params = params      # list of arrays
        self.grads  = None        # to be set each step (same length as params)
        self.m      = [np.zeros_like(p) for p in params]
        self.v      = [np.zeros_like(p) for p in params]
        self.t      = 0
        self.lrs    = lrs         # per-parameter lr
        self.b1, self.b2 = betas
        self.eps    = eps
        self.weight_decay = weight_decay

    def set_grads(self, grads: list):
        assert len(grads) == len(self.params)
        self.grads = grads

    def step(self, grad_clip: Optional[float]=None) -> Dict[str, float]:
        self.t += 1
        # concat grad norm
        gnorm_sq = 0.0
        for g in self.grads:
            gnorm_sq += float(np.sum(g*g))
        gnorm = math.sqrt(gnorm_sq)

        # clip
        scale = 1.0
        if grad_clip is not None and gnorm > grad_clip:
            scale = grad_clip / (gnorm + 1e-12)

        delta_sq = 0.0
        for i, (p, g) in enumerate(zip(self.params, self.grads)):
            gi = g * scale
            # decoupled weight decay
            p *= (1.0 - self.weight_decay)
            # AdamW
            self.m[i] = self.b1*self.m[i] + (1.0-self.b1)*gi
            self.v[i] = self.b2*self.v[i] + (1.0-self.b2)*(gi*gi)
            m_hat = self.m[i] / (1.0 - self.b1**self.t)
            v_hat = self.v[i] / (1.0 - self.b2**self.t)
            step = self.lrs[i] * m_hat / (np.sqrt(v_hat) + self.eps)
            p[...] -= step
            delta_sq += float(np.sum(step*step))
        return {"grad_norm": gnorm, "delta_norm": math.sqrt(delta_sq)}

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
        learn_scale = bool(self.tr.get("learn_logit_scale", False))
        logit_scale_init = float(self.tr.get("logit_scale_init", 1.25))
        scale_bounds = (float(self.tr.get("logit_scale_min", 0.5)), float(self.tr.get("logit_scale_max", 3.0)))

        if self.mode == "tstep":
            core = ThreeCompTStep(in_dim, hidden, self.tr, seed=self.seed)
            head_in = hidden
        else:
            core = FPTBlock(in_dim, hidden, seed=self.seed)
            head_in = hidden

        head = ReadoutMLP(head_in, head_hidden, num_classes,
                          seed=self.seed,
                          learn_logit_scale=learn_scale,
                          logit_scale_init=logit_scale_init,
                          scale_bounds=scale_bounds)

        # Optimizer / Scheduler
        head_lr = float(self.tr.get("head_lr", self.tr.get("lr", 1e-3)))
        rec_lr  = float(self.tr.get("rec_lr",  self.tr.get("lr", 1e-3)))
        head_only = bool(self.tr.get("head_only", True))
        unfreeze_at_conf = float(self.tr.get("unfreeze_at_conf", 0.13))
        weight_decay = float(self.tr.get("weight_decay", 1e-4))
        grad_clip = float(self.tr.get("grad_clip", 1.0))
        total_steps = int(self.tr.get("total_steps", 0))
        warmup_ratio = float(self.tr.get("warmup_ratio", 0.05))
        min_lr_ratio = float(self.tr.get("min_lr_ratio", 0.1))

        params = head.params()
        lrs    = [head_lr for _ in params]

        # Add recurrent params when not head_only
        rec_params = []
        if self.mode == "fpt" and not head_only:
            rec_params = [core.Wxh, core.Whh, core.bh]
            params.extend(rec_params)
            lrs.extend([rec_lr, rec_lr, rec_lr])

        opt = AdamW(params, lrs, betas=(0.9,0.999), eps=1e-8, weight_decay=weight_decay)

        # steps_per_epoch
        batch = int(self.tr.get("batch_size", 64))
        steps_per_epoch = self.tr.get("steps_per_epoch", None)
        if steps_per_epoch is None or int(steps_per_epoch) <= 0:
            steps_per_epoch = math.ceil(Xtr.shape[0] / batch)
        total_steps = total_steps if total_steps > 0 else steps_per_epoch * int(self.tr.get("epochs", 1))
        sch = WarmupCosine(opt.lrs, total_steps, warmup_ratio=warmup_ratio, min_lr_ratio=min_lr_ratio)

        augment = bool(self.tr.get("augment", True))

        await send_log(self.js, self.sb["logs"],
                       f"trainer start mode={self.mode} hidden={hidden} head_hidden={head_hidden} head_only={head_only} "
                       f"lr_head={head_lr} lr_rec={rec_lr} logit_scale={float(head.logit_scale[0])} steps/ep={steps_per_epoch}")

        best_acc = 0.0
        best_loss = 1e9
        global_step = 0

        epochs = int(self.tr.get("epochs", 1))
        rate_target = float(self.tr.get("rate_target", 0.20))
        rate_lambda = float(self.tr.get("rate_reg_lambda", 0.0))

        for ep in range(1, epochs+1):
            t0 = time.time()
            seen = 0; corr = 0; loss_sum = 0.0; steps = 0
            top5_acc_sum = 0.0
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
            for step, (xb, yb) in enumerate(iter_minibatches(Xtr, ytr, batch, steps_per_epoch, augment=augment), start=1):
                global_step += 1
                # forward (core)
                if self.mode == "tstep":
                    S, s_mean = core.forward_states(xb)
                    Z = s_mean
                else:
                    # FPT solve
                    K_max = int(self.tr.get("K", 4))
                    tol = float(self.tr.get("tol", 1e-5))
                    solver = self.tr.get("solver", "anderson").lower()
                    if solver == "anderson":
                        h0 = np.zeros((xb.shape[0], hidden), np.float32)
                        hK, k_eff, residual = FPTBlock.anderson_solve(xb, h0, core.Wxh, core.Whh, core.bh,
                                                                      K=K_max, tol=tol,
                                                                      m=int(self.tr.get("anderson_m",4)),
                                                                      beta=float(self.tr.get("anderson_beta",0.5)),
                                                                      logger=log_fpt)
                    else:
                        # plain fixed-point
                        h_prev = np.zeros((xb.shape[0], hidden), np.float32)
                        residual = 0.0; k_eff = K_max
                        log_fpt.info(f"固定点迭代开始：步数={xb.shape[0]}, 迭代次数={K_max}, 阈值={tol:.2e}, 阻尼=1.00")
                        for k in range(1, K_max+1):
                            h_next = FPTBlock.phi(xb, h_prev, core.Wxh, core.Whh, core.bh)
                            d = (h_next - h_prev).reshape(h_next.shape[0], -1)
                            residual = float(np.sqrt((d*d).mean()))
                            log_fpt.info(f"迭代 {k}/{K_max}，残差={residual:.3e}")
                            h_prev = h_next
                            if residual < tol:
                                log_fpt.info(f"残差 {residual:.3e} 已低于阈值 {tol:.3e}，提前停止迭代")
                                k_eff = k
                                break
                        hK = h_prev
                    Z = hK
                    await js_publish(self.js, self.sb["train_iter"] if "train_iter" in self.sb else self.sb["logs"],
                                     {"epoch": ep, "step": step, "k": int(k_eff), "residual": float(residual), "solver": solver, "ts": time.time()})

                # forward (head)
                logits, cache = head.forward(Z)
                nll, probs, log_probs = softmax_nll(logits, yb)
                pred = np.argmax(probs, axis=1)
                acc_b = float(np.mean(pred == yb))
                top5_b = topk_acc(probs, yb, k=5)
                conf_b = float(np.mean(probs[np.arange(yb.shape[0]), yb]))
                entropy_b = float(-np.mean(np.sum(probs * np.log(np.clip(probs,1e-9,None)), axis=1)))
                logit_mean = float(np.mean(logits))
                logit_std  = float(np.std(logits))

                # build dlogits
                dlogits = probs
                dlogits[np.arange(yb.shape[0]), yb] -= 1.0
                dlogits /= yb.shape[0]

                # backward head
                head.backward(cache, dlogits)
                grads = head.grads()

                # rate regularization (T-step only; monitor; no gradient to head)
                rate = 0.0
                reg_loss = 0.0
                if self.mode == "tstep":
                    rate = float(np.mean(Z))
                    if rate_lambda > 0.0:
                        reg_loss = rate_lambda * (rate - rate_target) ** 2
                        nll += reg_loss  # add to scalar loss, no head gradient

                # set grads
                opt.set_grads(grads)
                # scheduler lrs (per param)
                cur_lrs = sch.get_lrs()
                opt.lrs = cur_lrs[:len(opt.lrs)]
                stats = opt.step(grad_clip=grad_clip)
                sch.step()

                # optional: unfreeze recurrent after conf high enough (FPT)
                if self.mode == "fpt" and head_only and conf_b >= unfreeze_at_conf:
                    head_only = False
                    opt.params.extend([core.Wxh, core.Whh, core.bh])
                    opt.m.extend([np.zeros_like(core.Wxh), np.zeros_like(core.Whh), np.zeros_like(core.bh)])
                    opt.v.extend([np.zeros_like(core.Wxh), np.zeros_like(core.Whh), np.zeros_like(core.bh)])
                    opt.lrs.extend([float(self.tr.get("rec_lr", 1e-3))]*3)
                    await send_log(self.js, self.sb["logs"], f"unfreeze recurrent at conf={conf_b:.3f}")

                # meters
                seen += xb.shape[0]
                corr += int((pred == yb).sum())
                loss_sum += float(nll)
                top5_acc_sum += top5_b
                steps += 1
                tps = seen / max(1e-6, time.time()-t0)
                # publish metrics batch
                payload = {
                    "epoch": ep, "step": step,
                    "loss": float(nll), "acc": acc_b, "top5": top5_b,
                    "throughput": tps, "lr": float(opt.lrs[0]),
                    "nll": float(nll), "conf": conf_b, "entropy": entropy_b,
                    "logit_mean": logit_mean, "logit_std": logit_std,
                    "s_rate": rate if self.mode == "tstep" else None,
                    "grad_norm": stats["grad_norm"], "delta_norm": stats["delta_norm"],
                    "ts": time.time()
                }
                await js_publish(self.js, self.sb["metrics"], payload)

                # optional spikes (visual)
                if step % 10 == 0:
                    await self._emit_spikes(Z)

                # clamp logit scale if learnable
                head.clamp_scale()

            # validation
            v_seen=v_cor=v_steps=0; v_loss=0.0; v_top5=0.0; v_conf=0.0
            for xb, yb in iter_minibatches(Xte, yte, batch, steps_per_epoch=min(50, math.ceil(Xte.shape[0]/batch)), augment=False):
                if self.mode == "tstep":
                    _, s_mean = core.forward_states(xb)
                    Z = s_mean
                else:
                    h0 = np.zeros((xb.shape[0], hidden), np.float32)
                    hK, _, _ = FPTBlock.anderson_solve(xb, h0, core.Wxh, core.Whh, core.bh,
                                                       K=int(self.tr.get("K",4)),
                                                       tol=float(self.tr.get("tol",1e-5)),
                                                       m=int(self.tr.get("anderson_m",4)),
                                                       beta=float(self.tr.get("anderson_beta",0.5)))
                    Z = hK
                logits, _ = head.forward(Z)
                nll, probs, _ = softmax_nll(logits, yb)
                v_loss += float(nll); v_steps += 1
                v_seen += xb.shape[0]
                pred = np.argmax(probs, axis=1)
                v_cor += int((pred == yb).sum())
                v_top5 += topk_acc(probs, yb, k=5)
                v_conf += float(np.mean(probs[np.arange(yb.shape[0]), yb]))

            train_loss = loss_sum/max(1,steps)
            train_acc  = corr/max(1,seen)
            val_loss = v_loss/max(1,v_steps)
            val_acc  = v_cor/max(1,v_seen)
            top5_epoch = v_top5/max(1,v_steps)
            v_conf_epoch = v_conf/max(1,v_steps)

            best_acc = max(best_acc, val_acc)
            best_loss = min(best_loss, val_loss)

            # epoch metrics
            await js_publish(self.js, self.sb["metrics"], {
                "epoch": ep, "step": 0,
                "loss": float(val_loss), "acc": float(val_acc), "top5": float(top5_epoch),
                "throughput": 0.0, "lr": float(opt.lrs[0]),
                "nll": float(val_loss), "conf": float(v_conf_epoch), "entropy": None,
                "logit_mean": None, "logit_std": None,
                "s_rate": None,
                "best_acc": float(best_acc), "best_loss": float(best_loss),
                "phase": "val", "ts": time.time()
            })

            # log line
            sec = time.time()-t0
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
    nats_url = os.getenv("NATS_URL", cfg["nats"]["url"])
    nc = await nats.connect(nats_url)
    js = nc.jetstream()

    # subjects map; allow missing train_iter -> fallback to logs
    subjects = {
        "metrics": cfg["subjects"].get("metrics", "snn.metrics.training"),
        "spikes":  cfg["subjects"].get("spikes",  "snn.spikes.layer.1"),
        "logs":    cfg["subjects"].get("logs",    "snn.ui.log.training"),
        "train_iter": cfg["subjects"].get("train_iter", cfg["subjects"].get("logs","snn.ui.log.training"))
    }

    trainer = NPTrainer(cfg, js, subjects)
    await trainer.train()
    await nc.drain()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
