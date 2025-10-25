#!/usr/bin/env python3
"""NATS-based training worker using NumPy for three-compartment SNN models.

This worker supports two training modes:
    - T-step: sequential unrolling across T steps, updating only the output layer.
    - FPT: truncated BPTT (length K) that updates recurrent and output weights.

Configuration is resolved from ``config.yaml``. Metrics, spike summaries, and logs
are published to NATS JetStream subjects defined in the configuration. All payloads
are JSON-encoded with standard library ``json`` and deduplicated via ``Nats-Msg-Id``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from nats.aio.client import Client as NATS
from nats.errors import Error as NatsError


LOGGER = logging.getLogger("training_worker_np")


# ---------------------------------------------------------------------------
# Configuration structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train_file: str
    val_file: str
    inputs_key: str = "x"
    labels_key: str = "y"


@dataclass(frozen=True)
class NetworkConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True)
class HyperParams:
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    truncation_steps: int = 4
    timesteps: int = 16
    seed: int = 42
    warmup_steps: int = 0
    grad_clip: Optional[float] = None
    temperature: float = 1.0
    learnable_temperature: bool = False
    solver: str = "plain"
    anderson_m: int = 4
    anderson_beta: float = 0.5
    K_schedule: Optional[str] = None
    logit_scale: float = 1.0
    rate_reg_lambda: float = 0.0
    rate_target: float = 0.0
    g_apical_start: float = 0.5
    g_apical_end: float = 0.2
    beta_start: Optional[float] = None
    beta_end: Optional[float] = None
    v_th_start: Optional[float] = None
    v_th_end: Optional[float] = None
    contraction_rho: float = 0.9
    contraction_lambda: float = 0.0


@dataclass(frozen=True)
class NeuronDynamics:
    dt: float = 1e-3
    tau_soma: float = 2e-2
    tau_apical: float = 3e-2
    tau_basal: float = 3e-2
    coupling_apical: float = 0.6
    coupling_basal: float = 0.6
    threshold: float = -0.054
    v_rest: float = -0.07

    @property
    def alpha(self) -> float:
        return np.clip(self.dt / max(self.tau_soma, 1e-8), 0.0, 1.0)

    @property
    def beta(self) -> float:
        return np.clip(self.dt / max(self.tau_basal, 1e-8), 0.0, 1.0)

    @property
    def gamma(self) -> float:
        return np.clip(self.dt / max(self.tau_apical, 1e-8), 0.0, 1.0)


@dataclass(frozen=True)
class NatsSubjects:
    metrics: str
    spikes: str
    logs: str


@dataclass(frozen=True)
class NatsConfig:
    servers: Tuple[str, ...]
    stream: str
    subjects: NatsSubjects
    timeout: float = 2.0


@dataclass(frozen=True)
class WorkerConfig:
    mode: str
    dataset: DatasetConfig
    network: NetworkConfig
    hyper: HyperParams
    neuron: NeuronDynamics
    nats: NatsConfig


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def _parse_schedule(raw: Any) -> Tuple[Optional[float], Optional[float]]:
    """Extract optional (start, end) floats from a config value."""

    if raw is None:
        return None, None
    start_value: Optional[float]
    end_value: Optional[float]
    if isinstance(raw, dict):
        raw_start = raw.get("start")
        raw_end = raw.get("end", raw_start)
        start_value = float(raw_start) if raw_start is not None else None
        end_value = float(raw_end) if raw_end is not None else start_value
    else:
        start_value = float(raw)
        end_value = float(raw)
    return start_value, end_value


def resolve_worker_config(raw: Dict[str, Any], override_mode: Optional[str]) -> WorkerConfig:
    worker_raw = raw.get("training_worker", {})
    if not isinstance(worker_raw, dict):
        raise ValueError("`training_worker` block must be a mapping in config.yaml")

    mode = (override_mode or worker_raw.get("mode") or "tstep").lower()
    if mode not in {"tstep", "fpt"}:
        raise ValueError("training mode must be `tstep` or `fpt`")

    dataset_raw = worker_raw.get("data", {})
    if not isinstance(dataset_raw, dict):
        raise ValueError("`training_worker.data` must be a mapping")
    dataset = DatasetConfig(
        root=Path(dataset_raw.get("root", ".data/datasets")).expanduser(),
        train_file=str(dataset_raw.get("train", "train.npz")),
        val_file=str(dataset_raw.get("val", "val.npz")),
        inputs_key=str(dataset_raw.get("inputs_key", "x")),
        labels_key=str(dataset_raw.get("labels_key", "y")),
    )

    network_raw = worker_raw.get("network", {})
    if not isinstance(network_raw, dict):
        raise ValueError("`training_worker.network` must be a mapping")
    network = NetworkConfig(
        input_dim=int(network_raw.get("input_dim", 784)),
        hidden_dim=int(network_raw.get("hidden_dim", 128)),
        output_dim=int(network_raw.get("output_dim", 10)),
    )

    hyper_raw = worker_raw.get("hyperparams", {})
    if not isinstance(hyper_raw, dict):
        raise ValueError("`training_worker.hyperparams` must be a mapping")
    temp_raw = hyper_raw.get("temperature", 1.0)
    if isinstance(temp_raw, dict):
        temperature_value = float(temp_raw.get("value", 1.0))
        temperature_learnable = bool(temp_raw.get("learnable", False))
    else:
        temperature_value = float(temp_raw)
        temperature_learnable = bool(hyper_raw.get("temperature_learnable", False))
    grad_clip_raw = hyper_raw.get("grad_clip")
    grad_clip_value = float(grad_clip_raw) if grad_clip_raw is not None else None
    if grad_clip_value is not None and grad_clip_value <= 0:
        grad_clip_value = None
    logit_raw = hyper_raw.get("logit_scale", 1.0)
    if isinstance(logit_raw, dict):
        logit_scale_value = float(logit_raw.get("value", 1.0))
    else:
        logit_scale_value = float(logit_raw)
    rate_lambda = float(hyper_raw.get("rate_reg_lambda", 0.0))
    rate_target = float(hyper_raw.get("rate_target", 0.0))
    reg_block = hyper_raw.get("tstep_regularization")
    if isinstance(reg_block, dict):
        rate_lambda = float(reg_block.get("rate_reg_lambda", reg_block.get("lambda", rate_lambda)))
        rate_target = float(reg_block.get("rate_target", reg_block.get("target", rate_target)))
    g_apical_raw = hyper_raw.get("g_apical", {"start": 0.5, "end": 0.2})
    if isinstance(g_apical_raw, dict):
        g_apical_start = float(g_apical_raw.get("start", 0.5))
        g_apical_end = float(g_apical_raw.get("end", 0.2))
    else:
        g_apical_start = float(g_apical_raw)
        g_apical_end = float(hyper_raw.get("g_apical_end", g_apical_start))
    solver_block = hyper_raw.get("solver")
    solver_value = str(hyper_raw.get("solver", "plain")).lower() if not isinstance(solver_block, dict) else "plain"
    anderson_m_value = int(hyper_raw.get("anderson_m", 4))
    anderson_beta_value = float(hyper_raw.get("anderson_beta", 0.5))
    k_schedule_value = hyper_raw.get("K_schedule")
    if isinstance(solver_block, dict):
        solver_value = str(solver_block.get("type") or solver_block.get("name") or solver_value).lower()
        anderson_m_value = int(solver_block.get("anderson_m", anderson_m_value))
        anderson_beta_value = float(solver_block.get("anderson_beta", anderson_beta_value))
        k_schedule_value = solver_block.get("K_schedule", solver_block.get("schedule", k_schedule_value))
    warmup_steps_value = hyper_raw.get("warmup_steps")
    scheduler_block = hyper_raw.get("scheduler")
    if warmup_steps_value is None and isinstance(scheduler_block, dict):
        warmup_steps_value = scheduler_block.get("warmup_steps")
    warmup_steps_value = max(0, int(warmup_steps_value or 0))
    beta_start, beta_end = _parse_schedule(hyper_raw.get("beta"))
    v_th_start, v_th_end = _parse_schedule(hyper_raw.get("v_th"))
    contraction_rho = float(hyper_raw.get("contraction_rho", 0.9))
    contraction_lambda = float(hyper_raw.get("contraction_lambda", 0.0))
    if contraction_rho <= 0.0:
        contraction_rho = 0.0
    contraction_lambda = max(0.0, contraction_lambda)

    hyper = HyperParams(
        epochs=int(hyper_raw.get("epochs", 1)),
        batch_size=int(hyper_raw.get("batch_size", 32)),
        lr=float(hyper_raw.get("lr", 1e-3)),
        weight_decay=float(hyper_raw.get("weight_decay", 0.0)),
        truncation_steps=int(hyper_raw.get("truncation_steps", 4)),
        timesteps=int(hyper_raw.get("timesteps", 16)),
        seed=int(hyper_raw.get("seed", 42)),
        warmup_steps=warmup_steps_value,
        grad_clip=grad_clip_value,
        temperature=max(1e-6, temperature_value),
        learnable_temperature=temperature_learnable,
        solver=solver_value,
        anderson_m=anderson_m_value,
        anderson_beta=anderson_beta_value,
        K_schedule=k_schedule_value,
        logit_scale=max(1e-9, logit_scale_value),
        rate_reg_lambda=max(0.0, rate_lambda),
        rate_target=rate_target,
        g_apical_start=g_apical_start,
        g_apical_end=g_apical_end,
        beta_start=beta_start,
        beta_end=beta_end,
        v_th_start=v_th_start,
        v_th_end=v_th_end,
        contraction_rho=contraction_rho,
        contraction_lambda=contraction_lambda,
    )

    neuron_raw = worker_raw.get("neuron", {})
    if not isinstance(neuron_raw, dict):
        raise ValueError("`training_worker.neuron` must be a mapping")
    neuron = NeuronDynamics(
        dt=float(neuron_raw.get("dt", 1e-3)),
        tau_soma=float(neuron_raw.get("tau_soma", 2e-2)),
        tau_apical=float(neuron_raw.get("tau_apical", 3e-2)),
        tau_basal=float(neuron_raw.get("tau_basal", 3e-2)),
        coupling_apical=float(neuron_raw.get("coupling_apical", 0.6)),
        coupling_basal=float(neuron_raw.get("coupling_basal", 0.6)),
        threshold=float(neuron_raw.get("threshold", -0.054)),
        v_rest=float(neuron_raw.get("v_rest", -0.07)),
    )

    nats_raw = worker_raw.get("nats", {})
    if not isinstance(nats_raw, dict):
        raise ValueError("`training_worker.nats` must be a mapping")
    subjects_raw = nats_raw.get("subjects", {})
    if not isinstance(subjects_raw, dict):
        raise ValueError("`training_worker.nats.subjects` must be a mapping")
    subjects = NatsSubjects(
        metrics=str(subjects_raw.get("metrics", "training.metrics")),
        spikes=str(subjects_raw.get("spikes", "training.spikes")),
        logs=str(subjects_raw.get("logs", "training.logs")),
    )
    servers = nats_raw.get("servers")
    if not servers:
        servers_tuple: Tuple[str, ...] = ("nats://127.0.0.1:4222",)
    elif isinstance(servers, (list, tuple)):
        servers_tuple = tuple(str(entry) for entry in servers)
    else:
        raise ValueError("`training_worker.nats.servers` must be a list")
    nats_config = NatsConfig(
        servers=servers_tuple,
        stream=str(nats_raw.get("stream", "snn_training")),
        subjects=subjects,
        timeout=float(nats_raw.get("timeout", 2.0)),
    )

    return WorkerConfig(
        mode=mode,
        dataset=dataset,
        network=network,
        hyper=hyper,
        neuron=neuron,
        nats=nats_config,
    )


def current_millis() -> int:
    return int(time.time() * 1000)


def _cosine_decay(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    step = max(0, step)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _clip_gradient_norm(grads: Iterable[np.ndarray], max_norm: Optional[float]) -> None:
    if max_norm is None or max_norm <= 0:
        return
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float(np.sum(grad ** 2))
    norm = math.sqrt(max(total, 0.0))
    if norm == 0.0 or not math.isfinite(norm):
        return
    scale = min(1.0, max_norm / (norm + 1e-12))
    if scale >= 1.0:
        return
    for grad in grads:
        if grad is None:
            continue
        grad *= scale


def _apply_decoupled_weight_decay(param: np.ndarray, lr: float, weight_decay: float) -> None:
    if weight_decay <= 0.0 or lr <= 0.0:
        return
    factor = max(0.0, 1.0 - lr * weight_decay)
    param *= factor


def log_softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array")
    dtype = logits.dtype
    logits64 = logits.astype(np.float64, copy=False)
    max_logits = np.max(logits64, axis=1, keepdims=True)
    shifted = logits64 - max_logits
    logsumexp = np.log(np.sum(np.exp(shifted, dtype=np.float64), axis=1, keepdims=True))
    log_probs = shifted - logsumexp
    return log_probs.astype(dtype, copy=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    log_probs = log_softmax(logits)
    probs = np.exp(log_probs.astype(np.float64, copy=False))
    return probs.astype(logits.dtype, copy=False)


def nll_from_logits(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array")
    if targets.dtype != np.int64:
        raise TypeError("labels must be np.int64 for indexing")
    batch_size = logits.shape[0]
    if targets.shape and targets.shape[0] not in {0, batch_size}:
        raise ValueError("targets must match logits batch dimension")
    logits64 = logits.astype(np.float64, copy=False)
    max_logits = np.max(logits64, axis=1, keepdims=True)
    shifted = logits64 - max_logits
    exp_shifted = np.exp(shifted, dtype=np.float64)
    denom = np.clip(np.sum(exp_shifted, axis=1, keepdims=True), 1e-12, None)
    log_probs = shifted - np.log(denom)
    probs = np.exp(log_probs).astype(logits.dtype, copy=False)
    if targets.size == 0:
        return 0.0, probs
    row_idx = np.arange(targets.shape[0])
    log_prob_targets = log_probs[row_idx, targets]
    nll_values = -log_prob_targets
    loss = float(np.mean(nll_values)) if nll_values.size else 0.0
    return loss, probs


def classification_stats(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float, float]:
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    batch = probs.shape[0]
    if batch == 0:
        return 0.0, 0.0, 0.0, 0.0

    probs64 = probs.astype(np.float64, copy=False)
    probs_safe = np.clip(probs64, 1e-12, 1.0)
    entropy_values = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    entropy = float(np.mean(entropy_values)) if entropy_values.size else 0.0

    if targets.size == 0:
        return 0.0, 0.0, 0.0, entropy
    if targets.shape[0] != batch:
        raise ValueError("targets length must match probabilities batch size")

    preds = np.argmax(probs, axis=1)
    top1 = float(np.mean(preds == targets))

    classes = probs.shape[1]
    k = min(5, classes)
    if k > 0:
        partition = np.argpartition(-probs, k - 1, axis=1)[:, :k]
        hits = np.any(partition == targets[:, None], axis=1)
        top5 = float(np.mean(hits))
    else:
        top5 = 0.0

    confidence = float(np.mean(probs[np.arange(batch), targets]))
    return top1, top5, confidence, entropy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_npz(path: Path, inputs_key: str, labels_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with np.load(path) as handle:
        try:
            inputs = handle[inputs_key]
            labels = handle[labels_key]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise KeyError(f"missing key {exc} in dataset file {path}") from exc
    if labels.dtype != np.int64:
        labels = labels.astype(np.int64, copy=False)
    inputs = inputs.astype(np.float32, copy=False)
    return inputs, labels


def load_datasets(cfg: DatasetConfig, timesteps: int, input_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = cfg.root / cfg.train_file
    val_path = cfg.root / cfg.val_file
    train_x, train_y = _resolve_npz(train_path, cfg.inputs_key, cfg.labels_key)
    val_x, val_y = _resolve_npz(val_path, cfg.inputs_key, cfg.labels_key)

    expected_shape = (timesteps, input_dim)
    if train_x.ndim != 3 or train_x.shape[1:] != expected_shape:
        raise ValueError(
            f"train inputs must be (N, {timesteps}, {input_dim}), got {train_x.shape}"
        )
    if val_x.ndim != 3 or val_x.shape[1:] != expected_shape:
        raise ValueError(
            f"val inputs must be (N, {timesteps}, {input_dim}), got {val_x.shape}"
        )
    return train_x, train_y, val_x, val_y


# ---------------------------------------------------------------------------
# NATS publisher
# ---------------------------------------------------------------------------


class NatsPublisher:
    """Lightweight JetStream publisher with JSON serialization."""

    def __init__(self, config: NatsConfig) -> None:
        self._config = config
        self._nc: Optional[NATS] = None
        self._js = None

    async def __aenter__(self) -> "NatsPublisher":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._nc is not None:
            return
        self._nc = NATS()
        await self._nc.connect(servers=list(self._config.servers))
        self._js = self._nc.jetstream()
        subjects = [
            self._config.subjects.metrics,
            f"{self._config.subjects.spikes}.*",
            self._config.subjects.logs,
        ]
        try:
            await self._js.add_stream(name=self._config.stream, subjects=subjects)
        except NatsError:
            # Stream may already exist; ignore and continue.
            pass
        LOGGER.info(
            "Connected to NATS JetStream: servers=%s stream=%s subjects=%s",
            self._config.servers,
            self._config.stream,
            subjects,
        )

    async def close(self) -> None:
        if self._nc is None:
            return
        try:
            await self._nc.drain()
        finally:
            self._nc = None
            self._js = None

    async def publish_json(self, subject: str, payload: Dict[str, Any]) -> None:
        if self._js is None:
            raise RuntimeError("JetStream client not initialized")
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        msg_id = uuid.uuid4().hex
        headers = {"Nats-Msg-Id": msg_id}
        await self._js.publish(
            subject,
            payload=body,
            headers=headers,
            msg_id=msg_id,
            timeout=self._config.timeout,
        )


# ---------------------------------------------------------------------------
# Three-compartment model implemented with NumPy
# ---------------------------------------------------------------------------


class ThreeCompartmentModel:
    """Vectorized three-compartment recurrent layer with linear dynamics."""

    def __init__(self, network: NetworkConfig, neuron: NeuronDynamics, rng: np.random.Generator) -> None:
        hidden = network.hidden_dim
        scale_in = 1.0 / np.sqrt(network.input_dim)
        scale_rec = 1.0 / np.sqrt(hidden)

        self.Wxh = rng.standard_normal((hidden, network.input_dim)) * scale_in
        self.Whh = rng.standard_normal((hidden, hidden)) * scale_rec
        self.bh = np.zeros(hidden, dtype=np.float64)
        self.Wo = rng.standard_normal((network.output_dim, hidden)) * (1.0 / np.sqrt(hidden))
        self.bo = np.zeros(network.output_dim, dtype=np.float64)

        self.network = network
        self.neuron = neuron

    def forward(
        self,
        inputs: np.ndarray,
        apical_coupling_override: Optional[float] = None,
        beta_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Execute forward pass across full sequence.

        Returns:
            logits: array of shape (batch, output_dim)
            h_list: soma states per step (length T+1 including initial state)
            basal_list: same for basal compartment
            apical_list: same for apical compartment
            spikes: boolean masks per step (length T)
        """

        batch, timesteps, _ = inputs.shape
        hidden = self.network.hidden_dim
        neuron = self.neuron

        alpha = neuron.alpha
        beta = float(np.clip(beta_override, 0.0, 1.0)) if beta_override is not None else neuron.beta
        gamma = neuron.gamma
        coupling_apical = (
            float(apical_coupling_override)
            if apical_coupling_override is not None
            else neuron.coupling_apical
        )
        threshold_value = float(threshold_override) if threshold_override is not None else neuron.threshold

        h = np.full((batch, hidden), neuron.v_rest, dtype=np.float64)
        basal = np.full_like(h, neuron.v_rest)
        apical = np.full_like(h, neuron.v_rest)

        h_list: List[np.ndarray] = [h.copy()]
        basal_list: List[np.ndarray] = [basal.copy()]
        apical_list: List[np.ndarray] = [apical.copy()]
        spikes: List[np.ndarray] = []

        for step in range(timesteps):
            x_t = inputs[:, step, :]
            z_t = x_t @ self.Wxh.T + h @ self.Whh.T + self.bh

            basal = (1.0 - beta) * basal + beta * z_t
            apical = (1.0 - gamma) * apical + gamma * h
            h = (1.0 - alpha) * h + alpha * (
                coupling_apical * apical + neuron.coupling_basal * basal + z_t
            )

            spike_mask = h >= threshold_value
            spikes.append(spike_mask.copy())

            h_list.append(h.copy())
            basal_list.append(basal.copy())
            apical_list.append(apical.copy())
        logits = h @ self.Wo.T + self.bo
        return logits, h_list, basal_list, apical_list, spikes

# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------


def _aggregate_spikes(spikes: Iterable[np.ndarray]) -> Tuple[int, List[int]]:
    spike_list = list(spikes)
    if not spike_list:
        return 0, []
    stacked = np.stack(spike_list, axis=0)
    total_spikes = int(np.sum(stacked))
    per_neuron = np.sum(stacked, axis=(0, 1)).astype(int)
    return total_spikes, per_neuron.tolist()


@dataclass
class TrainStepResult:
    loss: float
    nll: float
    acc: float
    top5: float
    confidence: float
    entropy: float
    spikes_total: int
    spikes_per_neuron: List[int]
    residual: Optional[float] = None
    iterations: Optional[int] = None
    s_rate: Optional[float] = None
    logit_mean: Optional[float] = None
    logit_std: Optional[float] = None


class Trainer:
    def __init__(self, config: WorkerConfig, model: ThreeCompartmentModel) -> None:
        self.cfg = config
        self.model = model
        self.weight_decay = max(0.0, self.cfg.hyper.weight_decay)
        self.grad_clip = self.cfg.hyper.grad_clip if (self.cfg.hyper.grad_clip and self.cfg.hyper.grad_clip > 0) else None
        self.learnable_temperature = self.cfg.hyper.learnable_temperature
        self._temperature = max(1e-6, self.cfg.hyper.temperature)
        self.solver = self.cfg.hyper.solver if self.cfg.mode == "fpt" else "plain"
        self.anderson_m = max(1, self.cfg.hyper.anderson_m)
        self.anderson_beta = float(self.cfg.hyper.anderson_beta)
        self.K_schedule = self.cfg.hyper.K_schedule
        self.rate_reg_lambda = max(0.0, self.cfg.hyper.rate_reg_lambda)
        self.rate_target = float(self.cfg.hyper.rate_target)
        self.g_apical_start = float(self.cfg.hyper.g_apical_start)
        self.g_apical_end = float(self.cfg.hyper.g_apical_end)
        self.logit_scale = float(self.cfg.hyper.logit_scale)
        self.beta_start = self.cfg.hyper.beta_start
        self.beta_end = self.cfg.hyper.beta_end
        self.v_th_start = self.cfg.hyper.v_th_start
        self.v_th_end = self.cfg.hyper.v_th_end
        self.contraction_rho = float(self.cfg.hyper.contraction_rho)
        self.contraction_lambda = max(0.0, float(self.cfg.hyper.contraction_lambda))
        self._base_beta = self.model.neuron.beta
        self._base_threshold = self.model.neuron.threshold
        self._spectral_rng = np.random.default_rng(self.cfg.hyper.seed)
        self._spectral_vec: Optional[np.ndarray] = None
        self._probe_interval = 1  # log every batch for quick diagnosis

    @property
    def temperature(self) -> float:
        return float(self._temperature)

    def _forward_batch(
        self,
        batch_x: np.ndarray,
        *,
        apical_coupling_override: Optional[float] = None,
        beta_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> Tuple[
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        return self.model.forward(
            batch_x,
            apical_coupling_override,
            beta_override,
            threshold_override,
        )

    @staticmethod
    def _interp(start: float, end: float, progress: float) -> float:
        progress_clamped = float(np.clip(progress, 0.0, 1.0))
        return start + (end - start) * progress_clamped

    def resolve_annealed_params(self, progress: float) -> Tuple[float, Optional[float], Optional[float]]:
        g_apical = float(self._interp(self.g_apical_start, self.g_apical_end, progress))
        beta_override: Optional[float] = None
        if self.beta_start is not None or self.beta_end is not None:
            start = self.beta_start if self.beta_start is not None else self._base_beta
            end = self.beta_end if self.beta_end is not None else self._base_beta
            beta_override = float(np.clip(self._interp(start, end, progress), 0.0, 1.0))
        threshold_override: Optional[float] = None
        if self.v_th_start is not None or self.v_th_end is not None:
            start = self.v_th_start if self.v_th_start is not None else self._base_threshold
            end = self.v_th_end if self.v_th_end is not None else self._base_threshold
            threshold_override = float(self._interp(start, end, progress))
        return g_apical, beta_override, threshold_override

    @staticmethod
    def _mean_soma_states(h_list: List[np.ndarray]) -> np.ndarray:
        if len(h_list) <= 1:
            raise ValueError("soma history must include at least one timestep")
        soma_stack = np.stack(h_list[1:], axis=0)
        return np.mean(soma_stack, axis=0)

    def _scale_output_features(self, features: np.ndarray) -> np.ndarray:
        if self.logit_scale == 1.0:
            return features
        return features * self.logit_scale

    def _truth_probe(
        self,
        logits: np.ndarray,
        probs: np.ndarray,
        *,
        top1: float,
        top5: float,
        confidence: float,
        entropy: float,
        s_rate: Optional[float],
    ) -> Tuple[float, float]:
        if logits.size == 0:
            LOGGER.debug("[probe] empty logits tensor; skipping stats")
            return 0.0, 0.0
        logit_std = float(np.std(logits))
        logit_mean = float(np.mean(logits))
        message = (
            "[probe] logit_std=%.4e logit_mean=%.4e conf=%.4f entropy=%.4f "
            "top1=%.4f top5=%.4f"
        ) % (logit_std, logit_mean, confidence, entropy, top1, top5)
        if s_rate is not None:
            message += f" s_rate={s_rate:.4f}"
        LOGGER.debug(message)
        if not np.isfinite(logit_std) or not np.isfinite(logit_mean):
            raise AssertionError("logit statistics became non-finite")
        if not np.isfinite(confidence) or not np.isfinite(entropy):
            raise AssertionError("probability statistics became non-finite")
        if probs.size and (
            np.min(probs) < -1e-6 or np.max(probs) > 1.0 + 1e-6 or not np.all(np.isfinite(probs))
        ):
            raise AssertionError("probabilities outside [0,1] or non-finite")
        return logit_mean, logit_std

    def _estimate_spectral_norm(self, matrix: np.ndarray, iterations: int = 6) -> float:
        if matrix.size == 0:
            return 0.0
        dim = matrix.shape[1]
        vec = self._spectral_vec
        if vec is None or vec.shape[0] != dim:
            vec = self._spectral_rng.standard_normal(dim)
        v = vec.astype(np.float64, copy=True)
        for _ in range(max(1, iterations)):
            mv = matrix @ v
            norm_mv = np.linalg.norm(mv)
            if norm_mv == 0.0 or not np.isfinite(norm_mv):
                v = self._spectral_rng.standard_normal(dim)
                continue
            v = (matrix.T @ (mv / norm_mv))
            norm_v = np.linalg.norm(v)
            if norm_v == 0.0 or not np.isfinite(norm_v):
                v = self._spectral_rng.standard_normal(dim)
            else:
                v /= norm_v
        self._spectral_vec = v
        mv_final = matrix @ v
        return float(np.linalg.norm(mv_final))

    def enforce_contraction(self) -> None:
        if self.cfg.mode != "fpt":
            return
        rho = max(0.0, self.contraction_rho)
        if rho <= 0.0:
            return
        spectral = self._estimate_spectral_norm(self.model.Whh)
        if spectral <= 0.0 or spectral <= rho:
            return
        scale = rho / spectral
        self.model.Whh *= scale
        LOGGER.info("Rescaled Whh spectral norm from %.4f to %.4f via contraction", spectral, rho)

    def _train_step_tstep(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        lr: float,
        apical_coupling: Optional[float] = None,
        beta_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> TrainStepResult:
        num_classes = self.model.network.output_dim
        if batch_y.dtype != np.int64:
            raise ValueError("labels must be int64")
        if batch_y.size > 0:
            y_min = int(batch_y.min())
            y_max = int(batch_y.max())
            if y_min < 0 or y_max >= num_classes:
                raise ValueError(f"labels out of range [{y_min}, {y_max}) for {num_classes} classes")

        _, h_list, _, _, spikes = self._forward_batch(
            batch_x,
            apical_coupling_override=apical_coupling,
            beta_override=beta_override,
            threshold_override=threshold_override,
        )
        features = self._mean_soma_states(h_list)
        s_rate = float(np.mean(features)) if features.size else 0.0
        scaled_features = self._scale_output_features(features)
        pre_logits = scaled_features @ self.model.Wo.T + self.model.bo
        temp_value = self.temperature
        logits = pre_logits / temp_value
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(f"logits must be (batch,{num_classes}), got {logits.shape}")

        nll, probs = nll_from_logits(logits, batch_y)
        acc, top5, confidence, entropy = classification_stats(probs, batch_y)
        logit_mean, logit_std = self._truth_probe(
            logits,
            probs,
            top1=acc,
            top5=top5,
            confidence=confidence,
            entropy=entropy,
            s_rate=s_rate,
        )
        row_sums = np.sum(probs, axis=1)
        if not np.all(np.isfinite(row_sums)) or np.max(np.abs(row_sums - 1.0)) > 1e-4:
            raise ValueError("probabilities must sum to 1 per row")
        loss = nll
        contraction_lambda = self.contraction_lambda if self.cfg.mode == "fpt" else 0.0
        if contraction_lambda > 0.0:
            loss += contraction_lambda * float(np.sum(self.model.Whh ** 2))
        s_rate = float(np.mean(features)) if features.size else 0.0
        if self.rate_reg_lambda > 0.0:
            rate_error = s_rate - self.rate_target
            loss += self.rate_reg_lambda * (rate_error ** 2)

        grad_logits = probs.copy()
        if batch_y.size:
            grad_logits[np.arange(batch_y.shape[0]), batch_y] -= 1.0
        grad_logits /= max(batch_y.shape[0], 1)

        grad_pre = grad_logits / temp_value
        grad_Wo = grad_pre.T @ scaled_features
        grad_bo = np.sum(grad_pre, axis=0)
        temp_grad_arr = None
        if self.learnable_temperature:
            temp_grad = -np.sum(grad_logits * pre_logits) / (temp_value ** 2)
            temp_grad_arr = np.array([temp_grad], dtype=np.float64)
        grads = [grad_Wo, grad_bo]
        if temp_grad_arr is not None:
            grads.append(temp_grad_arr)
        _clip_gradient_norm(grads, self.grad_clip)
        if temp_grad_arr is not None:
            temp_update = float(temp_grad_arr[0])
        else:
            temp_update = None

        _apply_decoupled_weight_decay(self.model.Wo, lr, self.weight_decay)
        self.model.Wo -= lr * grad_Wo
        self.model.bo -= lr * grad_bo
        if temp_update is not None:
            self._temperature = max(1e-6, self._temperature - lr * temp_update)

        total_spikes, per_neuron = _aggregate_spikes(spikes)
        return TrainStepResult(
            loss=loss,
            nll=nll,
            acc=acc,
            top5=top5,
            confidence=confidence,
            entropy=entropy,
            spikes_total=total_spikes,
            spikes_per_neuron=per_neuron,
            s_rate=s_rate,
            logit_mean=logit_mean,
            logit_std=logit_std,
        )

    def _train_step_fpt(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        lr: float,
        apical_coupling: Optional[float] = None,
        beta_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> TrainStepResult:
        num_classes = self.model.network.output_dim
        assert batch_y.dtype == np.int64, "labels must be int64"
        if batch_y.size > 0:
            assert int(batch_y.min()) >= 0 and int(batch_y.max()) < num_classes, "labels out of range"

        _, h_list, _, _, spikes = self._forward_batch(
            batch_x,
            apical_coupling_override=apical_coupling,
            beta_override=beta_override,
            threshold_override=threshold_override,
        )
        s_mean = self._mean_soma_states(h_list)
        s_rate = float(np.mean(s_mean)) if s_mean.size else 0.0
        temp_value = self.temperature
        final_state = h_list[-1]
        scaled_state = self._scale_output_features(final_state)
        pre_logits = scaled_state @ self.model.Wo.T + self.model.bo
        logits = pre_logits / temp_value
        assert logits.shape[1] == num_classes, "logit dimension mismatch"

        nll, probs = nll_from_logits(logits, batch_y)
        acc, top5, confidence, entropy = classification_stats(probs, batch_y)
        logit_mean, logit_std = self._truth_probe(
            logits,
            probs,
            top1=acc,
            top5=top5,
            confidence=confidence,
            entropy=entropy,
            s_rate=s_rate,
        )
        loss = nll
        if self.rate_reg_lambda > 0.0:
            rate_error = s_rate - self.rate_target
            loss += self.rate_reg_lambda * (rate_error ** 2)

        grad_logits = probs.copy()
        if batch_y.size:
            grad_logits[np.arange(batch_y.shape[0]), batch_y] -= 1.0
        grad_logits /= max(batch_y.shape[0], 1)

        neuron = self.cfg.neuron
        beta_value = (
            float(np.clip(beta_override, 0.0, 1.0))
            if beta_override is not None
            else neuron.beta
        )
        max_steps = max(1, self.cfg.hyper.truncation_steps)
        solver = self.solver
        anderson_depth = max(1, self.anderson_m)
        anderson_beta = np.clip(self.anderson_beta, 0.0, 1.0)
        h_current = h_list[-1].copy()
        h_prev = h_list[-2].copy()
        residual_value = float(np.linalg.norm(h_current - h_prev) / np.sqrt(h_current.size))
        iteration_used = 1
        if solver == "anderson":
            diffs: List[np.ndarray] = []
            residuals: List[np.ndarray] = []
            for iteration in range(1, max_steps + 1):
                if iteration > 1:
                    delta = h_current - h_prev
                    residual_value = float(np.linalg.norm(delta) / np.sqrt(delta.size))
                    if residual_value <= self.cfg.hyper.tol if hasattr(self.cfg.hyper, "tol") else False:
                        iteration_used = iteration
                        break
                h_prev = h_current.copy()
                # compute next h via plain update
                grad_tmp = grad_logits / temp_value
                grad_h = (grad_tmp @ self.model.Wo) * self.logit_scale
                grad_basal = np.zeros_like(grad_h)
                grad_apical = np.zeros_like(grad_h)
                grad_Wxh = np.zeros_like(self.model.Wxh)
                grad_Whh = np.zeros_like(self.model.Whh)
                grad_bh = np.zeros_like(self.model.bh)
                alpha = neuron.alpha
                beta = beta_value
                gamma = neuron.gamma
                timesteps = batch_x.shape[1]
                for idx in range(timesteps - 1, -1, -1):
                    x_t = batch_x[:, idx, :]
                    h_prev_step = h_list[idx]
                    grad_basal += alpha * neuron.coupling_basal * grad_h
                    grad_apical += alpha * neuron.coupling_apical * grad_h
                    grad_z = alpha * grad_h + beta * grad_basal
                    grad_Wxh += grad_z.T @ x_t
                    grad_Whh += grad_z.T @ h_prev_step
                    grad_bh += np.sum(grad_z, axis=0)
                    grad_h_prev = (1.0 - alpha) * grad_h + gamma * grad_apical + grad_z @ self.model.Whh
                    grad_basal = (1.0 - beta) * grad_basal
                    grad_apical = (1.0 - gamma) * grad_apical
                    grad_h = grad_h_prev
                plain_update = h_prev - lr * grad_h
                residual_vector = plain_update - h_prev
                diffs.append(plain_update.reshape(plain_update.shape[0], -1))
                residuals.append(residual_vector.reshape(residual_vector.shape[0], -1))
                if len(diffs) > anderson_depth:
                    diffs.pop(0)
                    residuals.pop(0)
                stacked_diff = np.stack(diffs, axis=0)
                stacked_res = np.stack(residuals, axis=0)
                mat = stacked_res.reshape(len(residuals), -1)
                try:
                    coeffs, *_ = np.linalg.lstsq(mat.T, np.zeros(mat.shape[1]), rcond=None)
                except np.linalg.LinAlgError:
                    coeffs = np.ones(len(residuals)) / len(residuals)
                coeffs = coeffs / max(np.sum(coeffs), 1e-8)
                h_mix = np.tensordot(coeffs, stacked_diff, axes=(0, 0)).reshape(h_prev.shape)
                h_current = (1.0 - anderson_beta) * h_prev + anderson_beta * h_mix
                iteration_used = iteration
        else:
            truncation = max_steps

        grad_pre = grad_logits / temp_value
        grad_Wo = grad_pre.T @ scaled_state
        grad_bo = np.sum(grad_pre, axis=0)

        grad_h = (grad_pre @ self.model.Wo) * self.logit_scale
        grad_basal = np.zeros_like(grad_h)
        grad_apical = np.zeros_like(grad_h)

        grad_Wxh = np.zeros_like(self.model.Wxh)
        grad_Whh = np.zeros_like(self.model.Whh)
        grad_bh = np.zeros_like(self.model.bh)

        alpha = neuron.alpha
        beta = beta_value
        gamma = neuron.gamma

        timesteps = batch_x.shape[1]
        steps = 0
        residual_accum = 0.0
        residual_count = 0
        for idx in range(timesteps - 1, -1, -1):
            x_t = batch_x[:, idx, :]
            h_prev = h_list[idx]

            grad_basal += alpha * neuron.coupling_basal * grad_h
            grad_apical += alpha * neuron.coupling_apical * grad_h
            grad_z = alpha * grad_h + beta * grad_basal

            grad_Wxh += grad_z.T @ x_t
            grad_Whh += grad_z.T @ h_prev
            grad_bh += np.sum(grad_z, axis=0)

            grad_h_prev = (1.0 - alpha) * grad_h + gamma * grad_apical + grad_z @ self.model.Whh

            grad_basal = (1.0 - beta) * grad_basal
            grad_apical = (1.0 - gamma) * grad_apical
            grad_h = grad_h_prev

            if idx > 0:
                delta = np.abs(h_list[idx] - h_list[idx - 1])
                residual_accum += float(np.mean(delta))
                residual_count += 1

            steps += 1
            if steps >= truncation:
                break

        residual_value = residual_accum / residual_count if residual_count > 0 else 0.0
        if contraction_lambda > 0.0:
            grad_Whh += 2.0 * contraction_lambda * self.model.Whh

        temp_grad_arr = None
        if self.learnable_temperature:
            temp_grad = -np.sum(grad_logits * pre_logits) / (temp_value ** 2)
            temp_grad_arr = np.array([temp_grad], dtype=np.float64)
        grads = [grad_Wo, grad_bo, grad_Wxh, grad_Whh, grad_bh]
        if temp_grad_arr is not None:
            grads.append(temp_grad_arr)
        _clip_gradient_norm(grads, self.grad_clip)
        if temp_grad_arr is not None:
            temp_update = float(temp_grad_arr[0])
        else:
            temp_update = None

        _apply_decoupled_weight_decay(self.model.Wxh, lr, self.weight_decay)
        _apply_decoupled_weight_decay(self.model.Whh, lr, self.weight_decay)
        _apply_decoupled_weight_decay(self.model.Wo, lr, self.weight_decay)
        self.model.Wxh -= lr * grad_Wxh
        self.model.Whh -= lr * grad_Whh
        self.model.Wo -= lr * grad_Wo
        self.model.bh -= lr * grad_bh
        self.model.bo -= lr * grad_bo
        if temp_update is not None:
            self._temperature = max(1e-6, self._temperature - lr * temp_update)

        total_spikes, per_neuron = _aggregate_spikes(spikes)
        return TrainStepResult(
            loss=loss,
            nll=nll,
            acc=acc,
            top5=top5,
            confidence=confidence,
            entropy=entropy,
            spikes_total=total_spikes,
            spikes_per_neuron=per_neuron,
            residual=residual_value,
            s_rate=s_rate,
            logit_mean=logit_mean,
            logit_std=logit_std,
        )

    def evaluate(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        batch_size: int,
        *,
        apical_coupling: Optional[float] = None,
        beta_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float, float, float]:
        total_loss = 0.0
        total_acc = 0.0
        total_top5 = 0.0
        total_conf = 0.0
        total_entropy = 0.0
        logit_sum = 0.0
        logit_sq_sum = 0.0
        logit_count = 0
        count = 0
        temp_value = self.temperature
        for start in range(0, data_x.shape[0], batch_size):
            end = start + batch_size
            batch_x = data_x[start:end]
            batch_y = data_y[start:end]
            _, h_list, _, _, _ = self._forward_batch(
                batch_x,
                apical_coupling_override=apical_coupling,
                beta_override=beta_override,
                threshold_override=threshold_override,
            )
            if self.cfg.mode == "tstep":
                features = self._mean_soma_states(h_list)
                scaled_features = self._scale_output_features(features)
                logits = scaled_features @ self.model.Wo.T + self.model.bo
            else:
                final_state = h_list[-1]
                scaled_state = self._scale_output_features(final_state)
                logits = scaled_state @ self.model.Wo.T + self.model.bo
            logits = logits / temp_value
            loss, probs = nll_from_logits(logits, batch_y)
            acc, top5, confidence, entropy = classification_stats(probs, batch_y)
            batch_count = batch_x.shape[0]
            total_loss += loss * batch_count
            total_acc += acc * batch_count
            total_top5 += top5 * batch_count
            total_conf += confidence * batch_count
            total_entropy += entropy * batch_count
            logit_sum += float(np.sum(logits))
            logit_sq_sum += float(np.sum(logits ** 2))
            logit_count += logits.size
            count += batch_count
        denom = max(count, 1)
        logit_count = max(logit_count, 1)
        logit_mean = logit_sum / logit_count
        logit_var = max(logit_sq_sum / logit_count - logit_mean ** 2, 0.0)
        logit_std = math.sqrt(logit_var)
        return (
            total_loss / denom,
            total_acc / denom,
            total_top5 / denom,
            total_conf / denom,
            total_entropy / denom,
            logit_mean,
            logit_std,
        )

    def train_batch(
        self,
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        lr: float,
        *,
        apical_coupling: Optional[float] = None,
        beta_override: Optional[float] = None,
        threshold_override: Optional[float] = None,
    ) -> TrainStepResult:
        if self.cfg.mode == "tstep":
            return self._train_step_tstep(
                batch_x,
                batch_y,
                lr,
                apical_coupling,
                beta_override,
                threshold_override,
            )
        return self._train_step_fpt(
            batch_x,
            batch_y,
            lr,
            apical_coupling,
            beta_override,
            threshold_override,
        )


# ---------------------------------------------------------------------------
# Worker orchestration
# ---------------------------------------------------------------------------


def _batch_iterator(data_x: np.ndarray, data_y: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    total = data_x.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield data_x[start:end], data_y[start:end]


async def run_worker(worker_cfg: WorkerConfig, override_epochs: Optional[int] = None) -> None:
    rng = np.random.default_rng(worker_cfg.hyper.seed)
    model = ThreeCompartmentModel(worker_cfg.network, worker_cfg.neuron, rng)

    train_x, train_y, val_x, val_y = load_datasets(
        worker_cfg.dataset,
        worker_cfg.hyper.timesteps,
        worker_cfg.network.input_dim,
    )

    epochs = override_epochs or worker_cfg.hyper.epochs
    trainer = Trainer(worker_cfg, model)

    steps_per_epoch = math.ceil(train_x.shape[0] / worker_cfg.hyper.batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    async with NatsPublisher(worker_cfg.nats) as publisher:
        LOGGER.info("Starting training: mode=%s epochs=%d batches=%d", worker_cfg.mode, epochs, len(train_x))
        global_step = 0
        last_lr = worker_cfg.hyper.lr
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_acc = 0.0
            epoch_top5 = 0.0
            epoch_conf = 0.0
            epoch_entropy = 0.0
            epoch_rate_sum = 0.0
            epoch_rate_count = 0
            epoch_examples = 0
            epoch_throughputs: List[float] = []
            epoch_residuals: List[float] = []
            epoch_logit_sum = 0.0
            epoch_logit_sq_sum = 0.0
            epoch_logit_count = 0
            if epochs <= 1:
                progress = 1.0
            else:
                progress = epoch / max(1, epochs - 1)
            g_apical_value, beta_override, threshold_override = trainer.resolve_annealed_params(progress)
            for batch_x, batch_y in _batch_iterator(train_x, train_y, worker_cfg.hyper.batch_size):
                step_start = time.perf_counter()
                global_step += 1
                current_lr = _cosine_decay(
                    global_step,
                    worker_cfg.hyper.lr,
                    worker_cfg.hyper.warmup_steps,
                    total_steps,
                )
                last_lr = current_lr
                try:
                    result = trainer.train_batch(
                        batch_x,
                        batch_y,
                        current_lr,
                        apical_coupling=g_apical_value,
                        beta_override=beta_override,
                        threshold_override=threshold_override,
                    )
                except AssertionError as exc:
                    await publisher.publish_json(
                        worker_cfg.nats.subjects.logs,
                        {
                            "level": "ERROR",
                            "msg": f"[TSTEP] epoch={epoch} step={step + 1} failed: {exc}",
                            "time_unix": current_millis(),
                        },
                    )
                    break
                step_duration = max(time.perf_counter() - step_start, 1e-6)
                batch_size = batch_x.shape[0]
                epoch_loss += result.loss * batch_size
                epoch_nll += result.nll * batch_size
                epoch_acc += result.acc * batch_size
                epoch_top5 += result.top5 * batch_size
                epoch_conf += result.confidence * batch_size
                epoch_entropy += result.entropy * batch_size
                epoch_examples += batch_size
                throughput = float(batch_size / step_duration)
                epoch_throughputs.append(throughput)
                if result.residual is not None:
                    epoch_residuals.append(result.residual)
                if result.logit_mean is not None and result.logit_std is not None:
                    logits_per_batch = batch_size * worker_cfg.network.output_dim
                    epoch_logit_sum += result.logit_mean * logits_per_batch
                    epoch_logit_sq_sum += (
                        (result.logit_std ** 2 + result.logit_mean ** 2) * logits_per_batch
                    )
                    epoch_logit_count += logits_per_batch
                if result.s_rate is not None:
                    epoch_rate_sum += float(result.s_rate) * batch_size
                    epoch_rate_count += batch_size

                if global_step % 10 == 0:
                    metrics_payload: Dict[str, Any] = {
                        "phase": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": result.loss,
                        "nll": result.nll,
                        "acc": result.acc,
                        "top5": result.top5,
                        "conf": result.confidence,
                        "entropy": result.entropy,
                        "lr": current_lr,
                        "temperature": trainer.temperature,
                        "mode": worker_cfg.mode,
                        "examples": batch_size,
                        "throughput": throughput,
                        "step_ms": step_duration * 1000.0,
                        "time_unix": current_millis(),
                    }
                    if result.residual is not None:
                        metrics_payload["residual"] = result.residual
                    metrics_payload["s_rate"] = float(result.s_rate) if result.s_rate is not None else 0.0
                    metrics_payload["rate_target"] = trainer.rate_target
                    if result.logit_mean is not None:
                        metrics_payload["logit_mean"] = result.logit_mean
                    if result.logit_std is not None:
                        metrics_payload["logit_std"] = result.logit_std
                    spikes_payload = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "mode": worker_cfg.mode,
                        "layer": 0,
                        "total": result.spikes_total,
                        "per_neuron": result.spikes_per_neuron,
                        "time_unix": current_millis(),
                    }
                    await publisher.publish_json(worker_cfg.nats.subjects.metrics, metrics_payload)
                    spike_subject = f"{worker_cfg.nats.subjects.spikes}.0"
                    await publisher.publish_json(spike_subject, spikes_payload)

            denom = max(epoch_examples, 1)
            train_avg_loss = epoch_loss / denom
            train_avg_nll = epoch_nll / denom
            train_avg_acc = epoch_acc / denom
            train_avg_top5 = epoch_top5 / denom
            train_avg_conf = epoch_conf / denom
            train_avg_entropy = epoch_entropy / denom
            train_avg_rate = epoch_rate_sum / max(epoch_rate_count, 1)
            if epoch_logit_count > 0:
                train_logit_mean = epoch_logit_sum / epoch_logit_count
                train_logit_var = max(epoch_logit_sq_sum / epoch_logit_count - train_logit_mean ** 2, 0.0)
                train_logit_std = math.sqrt(train_logit_var)
            else:
                train_logit_mean = 0.0
                train_logit_std = 0.0
            (
                val_loss,
                val_acc,
                val_top5,
                val_conf,
                val_entropy,
                val_logit_mean,
                val_logit_std,
            ) = trainer.evaluate(
                val_x,
                val_y,
                worker_cfg.hyper.batch_size,
                apical_coupling=g_apical_value,
                beta_override=beta_override,
                threshold_override=threshold_override,
            )
            if worker_cfg.mode == "fpt":
                trainer.enforce_contraction()
            avg_throughput = float(np.mean(epoch_throughputs)) if epoch_throughputs else None
            residual_mean = float(np.mean(epoch_residuals)) if epoch_residuals else None

            metrics_payload = {
                "phase": "val",
                "epoch": epoch + 1,
                "loss": val_loss,
                "nll": val_loss,
                "acc": val_acc,
                "top5": val_top5,
                "conf": val_conf,
                "entropy": val_entropy,
                "lr": last_lr,
                "temperature": trainer.temperature,
                "train_loss": train_avg_loss,
                "train_nll": train_avg_nll,
                "train_acc": train_avg_acc,
                "train_top5": train_avg_top5,
                "train_conf": train_avg_conf,
                "train_entropy": train_avg_entropy,
                "train_s_rate": train_avg_rate,
                "train_logit_mean": train_logit_mean,
                "train_logit_std": train_logit_std,
                "mode": worker_cfg.mode,
                "avg_throughput": avg_throughput,
                "time_unix": current_millis(),
            }
            if residual_mean is not None:
                metrics_payload["residual"] = residual_mean
            metrics_payload["logit_mean"] = val_logit_mean
            metrics_payload["logit_std"] = val_logit_std
            metrics_payload["rate_target"] = trainer.rate_target
            await publisher.publish_json(worker_cfg.nats.subjects.metrics, metrics_payload)

            log_metric = {
                "train_loss": train_avg_loss,
                "train_nll": train_avg_nll,
                "train_acc": train_avg_acc,
                "train_top5": train_avg_top5,
                "train_conf": train_avg_conf,
                "train_entropy": train_avg_entropy,
                "train_s_rate": train_avg_rate,
                "train_logit_mean": train_logit_mean,
                "train_logit_std": train_logit_std,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_top5": val_top5,
                "val_conf": val_conf,
                "val_entropy": val_entropy,
                "val_logit_mean": val_logit_mean,
                "val_logit_std": val_logit_std,
                "rate_target": trainer.rate_target,
            }
            if avg_throughput is not None:
                log_metric["avg_throughput"] = avg_throughput
            if residual_mean is not None:
                log_metric["residual"] = residual_mean
            log_payload = {
                "level": "INFO",
                "msg": (
                    f"[epoch {epoch + 1}/{epochs}] mode={worker_cfg.mode} "
                    f"train_loss={train_avg_loss:.4f} train_acc={train_avg_acc:.4f} train_top5={train_avg_top5:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_top5={val_top5:.4f} "
                    f"conf={val_conf:.4f} entropy={val_entropy:.4f} rate={train_avg_rate:.3f}->{trainer.rate_target:.2f}"
                    + (f" residual={residual_mean:.6f}" if residual_mean is not None else "")
                    + (f" avg_throughput={avg_throughput:.2f}" if avg_throughput is not None else "")
                ),
                "time_unix": current_millis(),
                "metric": log_metric,
            }
            await publisher.publish_json(worker_cfg.nats.subjects.logs, log_payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy three-compartment training worker")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["tstep", "fpt"], help="Override training mode")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (default from config)")
    return parser.parse_args()


def setup_logging(level: Optional[str], config_data: Dict[str, Any]) -> None:
    logging_cfg = config_data.get("logging") if isinstance(config_data.get("logging"), dict) else {}
    level_name = level or logging_cfg.get("level", "INFO")
    resolved_level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=resolved_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    args = parse_args()
    config_path: Path = args.config
    config_data = load_yaml_config(config_path)
    setup_logging(args.log_level, config_data)

    worker_config = resolve_worker_config(config_data, args.mode)
    try:
        asyncio.run(run_worker(worker_config, override_epochs=args.epochs))
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted by user")
    except Exception as exc:  # pragma: no cover - runtime safety net
        LOGGER.exception("Training worker crashed: %s", exc)
        raise


if __name__ == "__main__":
    main()
