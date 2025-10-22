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
    hyper = HyperParams(
        epochs=int(hyper_raw.get("epochs", 1)),
        batch_size=int(hyper_raw.get("batch_size", 32)),
        lr=float(hyper_raw.get("lr", 1e-3)),
        weight_decay=float(hyper_raw.get("weight_decay", 0.0)),
        truncation_steps=int(hyper_raw.get("truncation_steps", 4)),
        timesteps=int(hyper_raw.get("timesteps", 16)),
        seed=int(hyper_raw.get("seed", 42)),
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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float64)
    sums = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(sums, 1e-12)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    if targets.dtype != np.int64:
        raise TypeError("labels must be np.int64 for indexing")
    probs = softmax(logits)
    idx = (np.arange(targets.shape[0]), targets)
    log_probs = -np.log(np.maximum(probs[idx], 1e-12))
    return float(np.mean(log_probs)), probs


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
        self, inputs: np.ndarray
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
        beta = neuron.beta
        gamma = neuron.gamma

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
                neuron.coupling_apical * apical + neuron.coupling_basal * basal + z_t
            )

            spike_mask = h >= neuron.threshold
            spikes.append(spike_mask.copy())

            h_list.append(h.copy())
            basal_list.append(basal.copy())
            apical_list.append(apical.copy())
        logits = h @ self.Wo.T + self.bo
        return logits, h_list, basal_list, apical_list, spikes

    def predict(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits, _, _, _, _, _ = self.forward(inputs)
        probs = softmax(logits)
        return logits, probs


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------


def _weight_decay_update(param: np.ndarray, grad: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    if weight_decay:
        grad = grad + weight_decay * param
    return param - lr * grad


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
    acc: float
    spikes_total: int
    spikes_per_neuron: List[int]
    residual: Optional[float] = None


class Trainer:
    def __init__(self, config: WorkerConfig, model: ThreeCompartmentModel) -> None:
        self.cfg = config
        self.model = model

    def _forward_batch(self, batch_x: np.ndarray) -> Tuple[
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        return self.model.forward(batch_x)

    def _train_step_tstep(self, batch_x: np.ndarray, batch_y: np.ndarray) -> TrainStepResult:
        num_classes = self.model.network.output_dim
        assert batch_y.dtype == np.int64, "labels must be int64"
        if batch_y.size > 0:
            assert int(batch_y.min()) >= 0 and int(batch_y.max()) < num_classes, "labels out of range"

        logits, h_list, _, _, spikes = self._forward_batch(batch_x)
        assert logits.shape[1] == num_classes, "logit dimension mismatch"

        loss, probs = cross_entropy(logits, batch_y)
        preds = np.argmax(probs, axis=1)
        assert preds.shape == (batch_y.shape[0],), "argmax must operate along axis=1"
        acc = float(np.mean(preds == batch_y))

        grad_logits = probs
        grad_logits[np.arange(batch_y.shape[0]), batch_y] -= 1.0
        grad_logits /= batch_y.shape[0]

        h_last = h_list[-1]
        grad_Wo = grad_logits.T @ h_last
        grad_bo = np.sum(grad_logits, axis=0)

        hp = self.cfg.hyper
        self.model.Wo = _weight_decay_update(self.model.Wo, grad_Wo, hp.lr, hp.weight_decay)
        self.model.bo = self.model.bo - hp.lr * grad_bo

        total_spikes, per_neuron = _aggregate_spikes(spikes)
        return TrainStepResult(loss=loss, acc=acc, spikes_total=total_spikes, spikes_per_neuron=per_neuron)

    def _train_step_fpt(self, batch_x: np.ndarray, batch_y: np.ndarray) -> TrainStepResult:
        num_classes = self.model.network.output_dim
        assert batch_y.dtype == np.int64, "labels must be int64"
        if batch_y.size > 0:
            assert int(batch_y.min()) >= 0 and int(batch_y.max()) < num_classes, "labels out of range"

        logits, h_list, _, _, spikes = self._forward_batch(batch_x)
        assert logits.shape[1] == num_classes, "logit dimension mismatch"

        loss, probs = cross_entropy(logits, batch_y)
        preds = np.argmax(probs, axis=1)
        assert preds.shape == (batch_y.shape[0],), "argmax must operate along axis=1"
        acc = float(np.mean(preds == batch_y))

        grad_logits = probs
        grad_logits[np.arange(batch_y.shape[0]), batch_y] -= 1.0
        grad_logits /= batch_y.shape[0]

        hp = self.cfg.hyper
        neuron = self.cfg.neuron
        truncation = max(1, hp.truncation_steps)

        grad_Wo = grad_logits.T @ h_list[-1]
        grad_bo = np.sum(grad_logits, axis=0)

        grad_h = grad_logits @ self.model.Wo
        grad_basal = np.zeros_like(grad_h)
        grad_apical = np.zeros_like(grad_h)

        grad_Wxh = np.zeros_like(self.model.Wxh)
        grad_Whh = np.zeros_like(self.model.Whh)
        grad_bh = np.zeros_like(self.model.bh)

        alpha = neuron.alpha
        beta = neuron.beta
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

        self.model.Wxh = _weight_decay_update(self.model.Wxh, grad_Wxh, hp.lr, hp.weight_decay)
        self.model.Whh = _weight_decay_update(self.model.Whh, grad_Whh, hp.lr, hp.weight_decay)
        self.model.Wo = _weight_decay_update(self.model.Wo, grad_Wo, hp.lr, hp.weight_decay)
        self.model.bh = self.model.bh - hp.lr * grad_bh
        self.model.bo = self.model.bo - hp.lr * grad_bo

        total_spikes, per_neuron = _aggregate_spikes(spikes)
        return TrainStepResult(
            loss=loss,
            acc=acc,
            spikes_total=total_spikes,
            spikes_per_neuron=per_neuron,
            residual=residual_value,
        )

    def evaluate(self, data_x: np.ndarray, data_y: np.ndarray, batch_size: int) -> Tuple[float, float]:
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        for start in range(0, data_x.shape[0], batch_size):
            end = start + batch_size
            batch_x = data_x[start:end]
            batch_y = data_y[start:end]
            logits, probs = self.model.predict(batch_x)
            loss, _ = cross_entropy(logits, batch_y)
            preds = np.argmax(probs, axis=1)
            acc = float(np.mean(preds == batch_y))
            total_loss += loss * batch_x.shape[0]
            total_acc += acc * batch_x.shape[0]
            count += batch_x.shape[0]
        return total_loss / max(count, 1), total_acc / max(count, 1)

    def train_batch(self, batch_x: np.ndarray, batch_y: np.ndarray) -> TrainStepResult:
        if self.cfg.mode == "tstep":
            return self._train_step_tstep(batch_x, batch_y)
        return self._train_step_fpt(batch_x, batch_y)


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

    async with NatsPublisher(worker_cfg.nats) as publisher:
        LOGGER.info("Starting training: mode=%s epochs=%d batches=%d", worker_cfg.mode, epochs, len(train_x))
        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_examples = 0
            epoch_throughputs: List[float] = []
            epoch_residuals: List[float] = []
            for batch_x, batch_y in _batch_iterator(train_x, train_y, worker_cfg.hyper.batch_size):
                step_start = time.perf_counter()
                result = trainer.train_batch(batch_x, batch_y)
                step_duration = max(time.perf_counter() - step_start, 1e-6)
                batch_size = batch_x.shape[0]
                epoch_loss += result.loss * batch_size
                epoch_acc += result.acc * batch_size
                epoch_examples += batch_size
                global_step += 1
                throughput = float(batch_size / step_duration)
                epoch_throughputs.append(throughput)
                if result.residual is not None:
                    epoch_residuals.append(result.residual)

                if global_step % 10 == 0:
                    metrics_payload: Dict[str, Any] = {
                        "phase": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": result.loss,
                        "acc": result.acc,
                        "mode": worker_cfg.mode,
                        "examples": batch_size,
                        "throughput": throughput,
                        "step_ms": step_duration * 1000.0,
                        "time_unix": current_millis(),
                    }
                    if result.residual is not None:
                        metrics_payload["residual"] = result.residual
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

            train_avg_loss = epoch_loss / max(epoch_examples, 1)
            train_avg_acc = epoch_acc / max(epoch_examples, 1)
            val_loss, val_acc = trainer.evaluate(val_x, val_y, worker_cfg.hyper.batch_size)
            avg_throughput = float(np.mean(epoch_throughputs)) if epoch_throughputs else None
            residual_mean = float(np.mean(epoch_residuals)) if epoch_residuals else None

            metrics_payload = {
                "phase": "val",
                "epoch": epoch + 1,
                "loss": val_loss,
                "acc": val_acc,
                "train_loss": train_avg_loss,
                "train_acc": train_avg_acc,
                "mode": worker_cfg.mode,
                "avg_throughput": avg_throughput,
                "time_unix": current_millis(),
            }
            if residual_mean is not None:
                metrics_payload["residual"] = residual_mean
            await publisher.publish_json(worker_cfg.nats.subjects.metrics, metrics_payload)

            log_metric = {
                "train_loss": train_avg_loss,
                "train_acc": train_avg_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            if avg_throughput is not None:
                log_metric["avg_throughput"] = avg_throughput
            if residual_mean is not None:
                log_metric["residual"] = residual_mean
            log_payload = {
                "level": "INFO",
                "msg": (
                    f"[epoch {epoch + 1}/{epochs}] mode={worker_cfg.mode} "
                    f"train_loss={train_avg_loss:.4f} train_acc={train_avg_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
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
