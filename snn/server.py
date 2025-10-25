"""FastAPI 应用：提供与前端约定的 REST API 与 SSE 事件流。"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import gzip
import json
import logging
import math
import pickle
import random
import shutil
import struct
import tarfile
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple, Deque
from urllib.parse import urlparse
from collections import deque

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:  # pragma: no cover - 可选依赖，在测试环境下可能缺失
    from nats.aio.client import Client as NATSClient
except Exception:  # pragma: no cover - 未安装 nats-py
    NATSClient = None  # type: ignore

from .config import get_message_queue_config, load_config
from .data import DATASET_IMAGE_SHAPES, ensure_feature_stats, standardize_batch, augment_flat_batch
from .fpt import FixedPointConfig, fixed_point_parallel_solve
from .mq import InMemoryQueue, Message, MessageQueue, build_message_queue
from .neuron import ThreeCompartmentParams

logger = logging.getLogger(__name__)


def _current_millis() -> int:
    return int(time.time() * 1000)


def _cosine_with_warmup(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    if total_steps <= 0:
        return base_lr
    clamped_step = max(0, min(step, total_steps))
    if warmup_steps > 0 and clamped_step < warmup_steps:
        return base_lr * float(clamped_step) / float(max(1, warmup_steps))
    remaining = max(1, total_steps - warmup_steps)
    progress = (clamped_step - warmup_steps) / float(remaining)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _clip_gradients_inplace(
    grads: Dict[str, np.ndarray],
    max_norm: float,
    extra: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    if max_norm is None or max_norm <= 0.0:
        total = 0.0
        for grad in grads.values():
            total += float(np.sum(np.square(grad)))
        if extra:
            for value in extra.values():
                total += float(np.sum(np.square(value)))
        return math.sqrt(max(total, 0.0))
    total = 0.0
    for grad in grads.values():
        total += float(np.sum(np.square(grad)))
    if extra:
        for value in extra.values():
            total += float(np.sum(np.square(value)))
    norm = math.sqrt(max(total, 0.0))
    if math.isfinite(norm) and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for key in grads.keys():
            grads[key] *= scale
        if extra:
            for key in extra.keys():
                extra[key] *= scale
    return norm


HandlerFn = Callable[[Message], Awaitable[None]]


class TrainingInitRequest(BaseModel):
    dataset: str = Field(..., description="数据集名称")
    mode: str = Field("fpt", description="训练模式：tstep 或 fpt")
    network_size: int = Field(ge=1, description="整体神经元数量")
    layers: int = Field(ge=1, description="网络层数")
    lr: float = Field(gt=0.0, description="学习率")
    K: int = Field(ge=1, description="固定点迭代次数 / 近邻大小")
    tol: float = Field(gt=0.0, description="固定点容差")
    T: Optional[int] = Field(default=None, description="时间步长（tstep 模式）")
    epochs: int = Field(ge=1, description="训练轮次")
    solver: str = Field(default="plain", description="固定点求解器：plain 或 anderson")
    anderson_m: int = Field(default=4, ge=1, description="Anderson 深度")
    anderson_beta: float = Field(default=0.5, ge=0.0, le=1.0, description="Anderson 阻尼")
    K_schedule: Optional[str] = Field(default=None, description="可选的 K 调度策略（如 auto）")
    temperature: float = Field(default=1.0, gt=0.0, description="Logit temperature 缩放系数")
    logit_scale: float = Field(default=1.25, gt=0.0, description="Logit 幅度缩放系数")
    logit_scale_learnable: bool = Field(default=False, description="logit_scale 是否作为可学习参数")
    steps_per_epoch: Optional[int] = Field(default=None, ge=1, description="每个 epoch 的 batch 数（空则自动计算）")
    augment: bool = Field(default=True, description="是否对训练批次应用轻量数据增广")


class DatasetDownloadRequest(BaseModel):
    name: str = Field(..., description="要下载的数据集名称")


@dataclass
class DatasetRecord:
    name: str
    installed: bool = False
    message: Optional[str] = None
    progress: float = 0.0


DATASET_SOURCES: Dict[str, Dict[str, Any]] = {
    "MNIST": {
        "files": [
            {
                "filename": "mnist.npz",
                "sources": [
                    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                    "https://huggingface.co/datasets/mnist/resolve/main/mnist.npz?download=1",
                ],
            }
        ]
    },
    "FASHION": {
        "files": [
            {
                "filename": "train-images-idx3-ubyte.gz",
                "sources": [
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
                ],
            },
            {
                "filename": "train-labels-idx1-ubyte.gz",
                "sources": [
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
                ],
            },
            {
                "filename": "t10k-images-idx3-ubyte.gz",
                "sources": [
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
                ],
            },
            {
                "filename": "t10k-labels-idx1-ubyte.gz",
                "sources": [
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
                ],
            },
        ]
    },
    "CIFAR10": {
        "files": [
            {
                "filename": "cifar-10-python.tar.gz",
                "sources": [
                    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                    "https://huggingface.co/datasets/cifar10/resolve/main/cifar-10-batches-py.tar.gz?download=1",
                ],
            }
        ]
    },
}


@dataclass
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    input_dim: int
    num_classes: int
    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    image_shape: Optional[Tuple[int, int, int]] = None
    name: Optional[str] = None


@dataclass
class ModelState:
    W_basal: np.ndarray  # (num_classes, timesteps, input_dim)
    b_basal: np.ndarray  # (num_classes, timesteps)
    b_out: np.ndarray    # (num_classes,)
    logit_scale: float


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"无效的 IDX 图像文件头：magic={magic}")
    images = np.frombuffer(raw[16:], dtype=np.uint8).reshape(count, rows * cols)
    return images.astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"无效的 IDX 标签文件头：magic={magic}")
    labels = np.frombuffer(raw[8:], dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError("IDX 标签文件长度与头部声明不一致")
    return labels.astype(np.int64)


def _ensure_cifar_extracted(root: Path, tar_path: Path) -> Path:
    target_dir = root / "cifar-10-batches-py"
    if target_dir.exists():
        return target_dir
    if not tar_path.exists():
        raise FileNotFoundError(f"缺少 CIFAR-10 压缩包：{tar_path}")
    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(path=root)
    return target_dir


def _load_cifar_arrays(root: Path, tar_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    extracted = _ensure_cifar_extracted(root, tar_path)

    def _load_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="latin1")
        data = np.array(batch["data"], dtype=np.float32)
        labels = np.array(batch["labels"], dtype=np.int64)
        return data, labels

    train_parts: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    for idx in range(1, 6):
        data, labels = _load_batch(extracted / f"data_batch_{idx}")
        train_parts.append(data)
        train_labels.append(labels)

    test_data, test_labels = _load_batch(extracted / "test_batch")

    train_x = np.concatenate(train_parts, axis=0)
    train_y = np.concatenate(train_labels, axis=0)
    test_x = np.asarray(test_data, dtype=np.float32)
    test_y = np.asarray(test_labels, dtype=np.int64)

    return train_x, train_y, test_x, test_y


def _compute_basal_kernel(
    params: ThreeCompartmentParams,
    timesteps: int,
    config: FixedPointConfig,
) -> np.ndarray:
    """预计算单位基底电流对最终膜电位的影响。"""

    if timesteps <= 0:
        raise ValueError("timesteps 必须大于 0")
    apical_zero = [0.0] * timesteps
    kernel: List[float] = []
    for idx in range(timesteps):
        basal = [0.0] * timesteps
        basal[idx] = 1.0
        result = fixed_point_parallel_solve(
            params,
            apical_zero,
            basal,
            config=config,
        )
        final_soma = result.states[-1].soma if result.states else 0.0
        kernel.append(final_soma)
    return np.asarray(kernel, dtype=np.float32)

class EventBroker:
    """负责 SSE 订阅管理与广播。"""

    def __init__(self, message_queue: Optional[MessageQueue] = None, subject: str = "training.events") -> None:
        self._message_queue = message_queue
        self._subject = subject
        self._subscribers: Set[asyncio.Queue[Tuple[str, Dict[str, Any]]]] = set()
        self._lock = asyncio.Lock()
        self._pending: List[Tuple[str, Dict[str, Any]]] = []
        self._history: List[Tuple[str, Dict[str, Any]]] = []

    async def publish(self, event: str, payload: Dict[str, Any]) -> None:
        entry = (event, payload)
        if self._message_queue is not None:
            try:
                data = json.dumps({"event": event, "payload": payload}, ensure_ascii=False).encode("utf-8")
                self._message_queue.publish(self._subject, data, headers={"event": event})
            except Exception as exc:  # pragma: no cover - 队列异常仅记录日志
                logger.warning("消息队列发布失败：%s", exc)
        async with self._lock:
            self._history.append(entry)
            if not self._subscribers:
                self._pending.append(entry)
                return
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(entry)
                except asyncio.QueueFull:
                    logger.warning("订阅队列已满，丢弃事件 %s", event)

    async def subscribe(self) -> asyncio.Queue[Tuple[str, Dict[str, Any]]]:
        queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue()
        async with self._lock:
            self._subscribers.add(queue)
            if self._pending:
                for entry in self._pending:
                    queue.put_nowait(entry)
                self._pending.clear()
        return queue

    async def history(self) -> List[Tuple[str, Dict[str, Any]]]:
        async with self._lock:
            return list(self._history)

    async def unsubscribe(self, queue: asyncio.Queue[Tuple[str, Dict[str, Any]]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)


class LogRingBuffer:
    """线程安全的环形日志缓冲区。"""

    def __init__(self, capacity: int = 200) -> None:
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()

    async def append(self, entry: Dict[str, Any]) -> None:
        async with self._lock:
            self._buffer.append(entry)

    async def snapshot(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            data = list(self._buffer)
        if limit is None or limit >= len(data):
            return data
        return data[-limit:]


class ExternalQueueRelay:
    """从外部消息源消费指标/脉冲/日志并写入 SSE。"""

    def __init__(
        self,
        queue: MessageQueue,
        broker: EventBroker,
        log_buffer: LogRingBuffer,
        nats_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._queue = queue
        self._broker = broker
        self._log_buffer = log_buffer
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        self._nats_settings = nats_settings or {}
        self._nc: Optional[NATSClient] = None
        self._subscriptions: List[int] = []

    async def start(self) -> None:
        self._stop_event.clear()
        started = False
        if NATSClient is not None and self._nats_settings:
            try:
                await self._start_nats()
                started = True
            except Exception as exc:  # pragma: no cover - 网络或鉴权异常
                logger.warning("初始化 NATS 订阅失败：%s", exc)
        if isinstance(self._queue, InMemoryQueue) and not self._tasks:
            self._tasks = [
                asyncio.create_task(self._poll_metrics(), name="mq-relay-metrics"),
                asyncio.create_task(self._poll_spikes(), name="mq-relay-spikes"),
                asyncio.create_task(self._poll_logs(), name="mq-relay-logs"),
            ]
            started = True
        if not started:
            logger.info("外部事件订阅未启用（缺少 NATS 配置或使用非内存队列）")

    async def stop(self) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        if self._nc is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - 关闭连接容错
                await self._nc.drain()
            self._nc = None
            self._subscriptions = []

    async def _start_nats(self) -> None:
        if NATSClient is None:  # pragma: no cover - 防御性检查
            raise RuntimeError("nats-py 未安装，无法建立订阅")
        options = self._build_nats_options()
        if not options.get("servers"):
            options["servers"] = ["nats://127.0.0.1:4222"]
        self._nc = NATSClient()
        await self._nc.connect(**options)
        subscriptions = [
            await self._nc.subscribe("snn.metrics.training", cb=self._on_metrics),
            await self._nc.subscribe("snn.spikes.layer.*", cb=self._on_spikes),
            await self._nc.subscribe("snn.ui.log.training", cb=self._on_logs),
        ]
        resolved_subs: List[int] = []
        for sub in subscriptions:
            sid = getattr(sub, "sid", None)
            if sid is None:
                sid = getattr(sub, "id", None)
            if isinstance(sid, int):
                resolved_subs.append(sid)
        self._subscriptions = resolved_subs
        logger.info("已连接 NATS 并订阅训练事件：metrics/spikes/logs")

    def _build_nats_options(self) -> Dict[str, Any]:
        servers_raw = self._nats_settings.get("servers")
        servers: List[str] = []
        if isinstance(servers_raw, str):
            servers = [servers_raw]
        elif isinstance(servers_raw, Iterable):
            servers = [str(item) for item in servers_raw]
        timeout = self._nats_settings.get("timeout")
        allow_reconnect = self._nats_settings.get("allow_reconnect")
        max_reconnect_attempts = self._nats_settings.get("max_reconnect_attempts")
        options: Dict[str, Any] = {"servers": servers}
        if isinstance(timeout, (int, float)):
            options["connect_timeout"] = float(timeout)
        if allow_reconnect is not None:
            options["allow_reconnect"] = bool(allow_reconnect)
        if isinstance(max_reconnect_attempts, int):
            options["max_reconnect_attempts"] = max(1, max_reconnect_attempts)
        return options

    async def _on_metrics(self, msg: Any) -> None:
        await self._handle_metrics(self._to_message(msg))

    async def _on_spikes(self, msg: Any) -> None:
        await self._handle_spikes(self._to_message(msg))

    async def _on_logs(self, msg: Any) -> None:
        await self._handle_logs(self._to_message(msg))

    @staticmethod
    def _to_message(msg: Any) -> Message:
        headers_raw = getattr(msg, "headers", None)
        headers = dict(headers_raw) if headers_raw else {}
        data = msg.data if isinstance(msg.data, bytes) else bytes(msg.data)
        return Message(subject=getattr(msg, "subject", ""), data=data, headers=headers)

    async def _poll_metrics(self) -> None:
        await self._poll_subject("snn.metrics.training", self._handle_metrics)

    async def _poll_spikes(self) -> None:
        await self._poll_subject("snn.spikes.layer.*", self._handle_spikes)

    async def _poll_logs(self) -> None:
        await self._poll_subject("snn.ui.log.training", self._handle_logs)

    async def _poll_subject(
        self,
        subject: str,
        handler: HandlerFn,
        *,
        idle_sleep: float = 0.5,
        batch_size: int = 10,
    ) -> None:
        while not self._stop_event.is_set():
            try:
                messages = await asyncio.to_thread(self._queue.pull, subject, max_messages=batch_size)
            except Exception as exc:  # pragma: no cover - 网络或消息队列异常
                logger.warning("拉取消息失败 subject=%s: %s", subject, exc)
                await asyncio.sleep(1.0)
                continue
            if not messages:
                await asyncio.sleep(idle_sleep)
                continue
            for message in messages:
                if self._stop_event.is_set():
                    return
                try:
                    await handler(message)
                except asyncio.CancelledError:  # pragma: no cover - 任务取消
                    raise
                except Exception as exc:  # pragma: no cover - 单条消息处理异常
                    logger.warning("处理消息失败 subject=%s: %s", message.subject, exc)

    async def _handle_metrics(self, message: Message) -> None:
        payload = self._decode_payload(message)
        if payload is None:
            return
        if "time_unix" not in payload or not isinstance(payload["time_unix"], (int, float)):
            payload["time_unix"] = _current_millis()
        phase = payload.get("phase")
        phase_text = str(phase).lower() if isinstance(phase, str) else ""
        if phase_text in {"val", "validation"}:
            payload["phase"] = "val"
            event_name = "metrics_epoch"
        else:
            payload["phase"] = "train"
            event_name = "metrics_batch"
        await self._broker.publish(event_name, payload)

    async def _handle_spikes(self, message: Message) -> None:
        payload = self._decode_payload(message)
        if payload is None:
            return
        layer = payload.get("layer")
        if not isinstance(layer, int):
            subject_layer = self._extract_layer_from_subject(message.subject)
            if subject_layer is not None:
                payload["layer"] = subject_layer
        if "time_unix" not in payload or not isinstance(payload["time_unix"], (int, float)):
            payload["time_unix"] = _current_millis()
        await self._broker.publish("spike", payload)

    async def _handle_logs(self, message: Message) -> None:
        payload = self._decode_payload(message)
        if payload is None:
            return
        level_raw = payload.get("level", "INFO")
        level = str(level_raw).upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            level = "INFO"
        msg_value = payload.get("msg") or payload.get("message") or ""
        msg_text = str(msg_value)
        time_unix = payload.get("time_unix")
        if not isinstance(time_unix, (int, float)):
            time_unix = _current_millis()
        metric_payload = payload.get("metric") if isinstance(payload.get("metric"), dict) else None
        sse_payload: Dict[str, Any] = {
            "level": level,
            "msg": msg_text,
            "time_unix": time_unix,
        }
        if metric_payload is not None:
            sse_payload["metric"] = metric_payload
        await self._broker.publish("log", sse_payload)
        log_entry: Dict[str, Any] = {
            "ts": int(time_unix // 1000),
            "level": level,
            "message": msg_text,
        }
        if metric_payload is not None:
            log_entry["metric"] = metric_payload
        await self._log_buffer.append(log_entry)

    @staticmethod
    def _extract_layer_from_subject(subject: str) -> Optional[int]:
        for part in reversed(subject.split(".")):
            if part.isdigit():
                return int(part)
        return None

    def _decode_payload(self, message: Message) -> Optional[Dict[str, Any]]:
        try:
            text = message.data.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("无法解码消息主体 subject=%s", message.subject)
            return None
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("消息 JSON 解析失败 subject=%s", message.subject)
            return None
        if isinstance(decoded, dict):
            if "payload" in decoded and isinstance(decoded["payload"], dict):
                return decoded["payload"]
            return decoded
        logger.debug("忽略非对象消息 subject=%s", message.subject)
        return None


class DatasetService:
    """数据集下载与列表管理。"""

    def __init__(self, broker: EventBroker) -> None:
        self._broker = broker
        self._data_root = Path(__file__).resolve().parent.parent / ".data" / "datasets"
        self._data_root.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, DatasetRecord] = {
            "MNIST": DatasetRecord(name="MNIST", installed=False, progress=0.0),
            "FASHION": DatasetRecord(name="FASHION", installed=False, progress=0.0),
            "CIFAR10": DatasetRecord(name="CIFAR10", installed=False, progress=0.0),
        }
        self._download_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        self._refresh_installation()

    def list_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        self._refresh_installation()
        payload = [
            {
                "name": record.name,
                "installed": record.installed,
                "progress": round(record.progress * 100.0, 1),
                "message": record.message,
            }
            for record in self._records.values()
        ]
        return {"datasets": payload}

    async def start_download(self, name: str) -> None:
        async with self._lock:
            if self._download_task and not self._download_task.done():
                raise HTTPException(status.HTTP_409_CONFLICT, "已有数据集正在下载")
            normalized = name.upper()
            record = self._records.setdefault(normalized, DatasetRecord(name=normalized, installed=False, progress=0.0))
            record.progress = 0.0
            record.installed = False
            record.message = None
            self._download_task = asyncio.create_task(self._run_download(record))

    async def _run_download(self, record: DatasetRecord) -> None:
        start_payload = {
            "name": record.name,
            "state": "start",
            "progress": 0.0,
            "time_unix": _current_millis(),
        }
        await self._broker.publish("dataset_download", start_payload)
        dataset_dir = self._dataset_path(record.name)
        try:
            await self._download_dataset(record, dataset_dir)
            record.installed = True
            record.message = None
            record.progress = 1.0
            complete_payload = {
                "name": record.name,
                "state": "complete",
                "progress": record.progress,
                "time_unix": _current_millis(),
            }
            await self._broker.publish("dataset_download", complete_payload)
        except Exception as exc:
            logger.exception("下载数据集 %s 失败：%s", record.name, exc)
            record.installed = False
            record.message = str(exc)
            record.progress = max(0.0, record.progress)
            error_payload = {
                "name": record.name,
                "state": "error",
                "progress": record.progress,
                "message": record.message,
                "time_unix": _current_millis(),
            }
            await self._broker.publish("dataset_download", error_payload)
        finally:
            async with self._lock:
                self._download_task = None

    def _dataset_path(self, name: str) -> Path:
        safe = "".join((ch.lower() if ch.isalnum() or ch in {"-", "_"} else "_") for ch in name.strip())
        safe = safe or "dataset"
        return self._data_root / safe

    def _materialize_dataset(self, record: DatasetRecord) -> None:
        # Deprecated placeholder implementation retained for API compatibility.
        pass

    def _refresh_installation(self) -> None:
        for record in self._records.values():
            dataset_dir = self._dataset_path(record.name)
            if dataset_dir.exists():
                metadata_path = dataset_dir / "metadata.json"
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    files = metadata.get("files") if isinstance(metadata, dict) else None
                except (FileNotFoundError, json.JSONDecodeError):
                    files = None
                if files:
                    all_present = True
                    for entry in files:
                        filename = entry.get("filename")
                        if not filename:
                            continue
                        if not (dataset_dir / filename).exists():
                            all_present = False
                            break
                    if all_present:
                        record.installed = True
                        record.progress = 1.0
                        record.message = None
                        continue
                record.installed = False
                record.progress = 0.0
                record.message = "未检测到完整的数据集文件，请重新下载"
            else:
                record.installed = False
                record.progress = 0.0
                record.message = None

    async def _download_dataset(self, record: DatasetRecord, dataset_dir: Path) -> None:
        sources = DATASET_SOURCES.get(record.name.upper())
        if not sources:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"暂不支持数据集 {record.name}")

        dataset_dir.mkdir(parents=True, exist_ok=True)
        placeholder = dataset_dir / "sample.txt"
        if placeholder.exists():
            try:
                placeholder.unlink()
            except OSError:
                logger.warning("无法移除占位符文件 %s", placeholder)

        downloaded: List[Dict[str, Any]] = []
        files = sources.get("files", [])
        total = max(len(files), 1)

        try:
            for index, file_info in enumerate(files, start=1):
                path, source_url = await asyncio.to_thread(self._download_remote_file, dataset_dir, file_info)
                downloaded.append(
                    {
                        "filename": path.name,
                        "url": source_url,
                        "size_bytes": path.stat().st_size if path.exists() else None,
                    }
                )
                record.progress = index / total
                progress_payload = {
                    "name": record.name,
                    "state": "progress",
                    "progress": record.progress,
                    "time_unix": _current_millis(),
                }
                await self._broker.publish("dataset_download", progress_payload)
        except Exception as exc:
            logger.warning("下载数据集 %s 失败：%s", record.name, exc)
            record.message = str(exc)
            record.progress = max(0.0, record.progress)
            record.installed = False
            raise

        self._write_metadata(dataset_dir, record.name, downloaded)
        placeholder = dataset_dir / "sample.txt"
        try:
            placeholder.write_text(
                "Dataset placeholder generated by DatasetService.\n", encoding="utf-8"
            )
        except OSError:
            logger.warning("无法写入占位符文件 %s", placeholder)

        record.installed = True
        record.message = None
        record.progress = 1.0

    def _download_remote_file(self, dataset_dir: Path, file_info: Dict[str, Any]) -> Tuple[Path, str]:
        filename = file_info.get("filename")
        urls: List[str] = []
        if "sources" in file_info and isinstance(file_info["sources"], list):
            urls.extend(str(url) for url in file_info["sources"] if url)
        url = file_info.get("url")
        if url:
            urls.append(str(url))
        if not filename and urls:
            parsed = urlparse(urls[0])
            filename = Path(parsed.path).name or "dataset.bin"
        if not filename:
            raise ValueError("缺少文件名")

        destination = dataset_dir / filename
        if destination.exists():
            return destination, f"file://{destination}"

        temp_path = destination.parent / f"{destination.name}.tmp"
        last_error: Optional[BaseException] = None
        for candidate in urls:
            logger.info("下载数据集文件 %s -> %s", candidate, destination)
            request = urllib.request.Request(candidate, headers={"User-Agent": "snn-downloader/0.1"})
            try:
                with urllib.request.urlopen(request) as response, temp_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except Exception as exc:  # pragma: no cover - 网络异常
                logger.warning("下载失败，尝试下一个源：%s", exc)
                last_error = exc
                with contextlib.suppress(FileNotFoundError):
                    temp_path.unlink()
                continue
            temp_path.replace(destination)
            return destination, candidate

        if last_error:
            raise last_error
        raise RuntimeError("未能找到可用的下载源")

    def _write_metadata(self, dataset_dir: Path, name: str, files: List[Dict[str, Any]]) -> None:
        metadata = {
            "name": name,
            "downloaded_at": time.time(),
            "files": files,
        }
        metadata_path = dataset_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class TrainingService:
    """使用三隔室神经元 + 固定点求解器的训练服务。"""

    def __init__(self, broker: EventBroker, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._broker = broker
        self._rng = random.Random(42)
        self._lock = asyncio.Lock()
        self._status: str = "Idle"
        base_config: Dict[str, Any] = {
            "dataset": "MNIST",
            "mode": "fpt",
            "network_size": 128,
            "layers": 2,
            "lr": 1e-3,
            "K": 4,
            "tol": 1e-5,
            "T": 12,
            "epochs": 20,
            "solver": "anderson",
            "anderson_m": 4,
            "anderson_beta": 0.5,
            "K_schedule": None,
            "temperature": 1.0,
            "logit_scale": 1.25,
            "logit_scale_learnable": False,
            "epochs": 20,
            "warmup_steps": 200,
            "scheduler": "warmup_cosine",
            "min_lr": None,
            "min_lr_scale": 0.1,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "rate_reg_lambda": 1e-3,
            "rate_target": 0.2,
            "steps_per_epoch": None,
            "augment": True,
        }
        self._config = self._apply_overrides(base_config, defaults or {})
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._data_root = Path(__file__).resolve().parent.parent / ".data" / "datasets"
        self._dataset_cache: Dict[str, DatasetBundle] = {}
        self._model_state: Optional[ModelState] = None
        self._np_seed = 1234
        self._params = ThreeCompartmentParams()
        self._kernel_cache: Dict[Tuple[int, int, float, float], np.ndarray] = {}

    @property
    def status(self) -> str:
        return self._status

    def get_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    def _apply_overrides(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        if not isinstance(overrides, dict):
            return merged
        simple_fields = [
            "dataset",
            "mode",
            "network_size",
            "layers",
            "lr",
            "K",
            "tol",
            "T",
            "epochs",
            "weight_decay",
            "grad_clip",
            "warmup_steps",
            "min_lr",
            "min_lr_scale",
            "scheduler",
            "solver",
            "anderson_m",
            "anderson_beta",
            "K_schedule",
            "temperature",
            "logit_scale",
            "logit_scale_learnable",
            "rate_reg_lambda",
            "rate_target",
            "steps_per_epoch",
            "augment",
        ]
        for field in simple_fields:
            if field in overrides and not isinstance(overrides[field], dict):
                merged[field] = overrides[field]

        scheduler_block = overrides.get("scheduler")
        if isinstance(scheduler_block, dict):
            merged["scheduler"] = scheduler_block.get("type", scheduler_block.get("name", merged.get("scheduler", "cosine")))
            if "warmup_steps" in scheduler_block:
                merged["warmup_steps"] = int(scheduler_block["warmup_steps"])
            if "min_lr" in scheduler_block:
                merged["min_lr"] = float(scheduler_block["min_lr"])
            if "min_lr_scale" in scheduler_block:
                merged["min_lr_scale"] = float(scheduler_block["min_lr_scale"])

        solver_block = overrides.get("solver")
        if isinstance(solver_block, dict):
            merged["solver"] = str(
                solver_block.get("type", solver_block.get("name", merged.get("solver", "plain")))
            ).lower()
            if "anderson_m" in solver_block:
                merged["anderson_m"] = int(solver_block["anderson_m"])
            if "anderson_beta" in solver_block:
                merged["anderson_beta"] = float(solver_block["anderson_beta"])
            if "K_schedule" in solver_block or "schedule" in solver_block:
                merged["K_schedule"] = solver_block.get("K_schedule", solver_block.get("schedule"))

        logit_block = overrides.get("logit_scale")
        if isinstance(logit_block, dict):
            if "value" in logit_block:
                merged["logit_scale"] = float(logit_block["value"])
            if "learnable" in logit_block:
                merged["logit_scale_learnable"] = bool(logit_block["learnable"])

        reg_block = overrides.get("tstep_regularization")
        if isinstance(reg_block, dict):
            if "rate_reg_lambda" in reg_block or "lambda" in reg_block:
                merged["rate_reg_lambda"] = float(
                    reg_block.get("rate_reg_lambda", reg_block.get("lambda", merged["rate_reg_lambda"]))
                )
            if "rate_target" in reg_block or "target" in reg_block:
                merged["rate_target"] = float(
                    reg_block.get("rate_target", reg_block.get("target", merged["rate_target"]))
                )

        return merged

    def _prepare_inputs(
        self,
        batch: np.ndarray,
        dataset: DatasetBundle,
        *,
        rng: Optional[np.random.Generator],
        augment: bool,
    ) -> np.ndarray:
        prepared = batch.astype(np.float32, copy=False)
        if augment and dataset.image_shape is not None:
            if rng is None:
                raise ValueError("rng must be provided when augmentation is enabled")
            prepared = augment_flat_batch(prepared, dataset.image_shape, rng)
        if dataset.feature_mean is not None and dataset.feature_std is not None:
            prepared = standardize_batch(prepared, dataset.feature_mean, dataset.feature_std)
        if not np.all(np.isfinite(prepared)):
            raise FloatingPointError("检测到非有限的输入特征，标准化或增广流程可能异常")
        return prepared

    async def _log_dataset_stats(self, dataset: DatasetBundle) -> None:
        if dataset.feature_mean is None or dataset.feature_std is None:
            return
        mean = dataset.feature_mean
        std = dataset.feature_std
        def _fmt(values: np.ndarray) -> List[float]:
            return [round(float(v), 4) for v in values.tolist()]
        mean_head = _fmt(mean[:3])
        mean_tail = _fmt(mean[-3:])
        std_head = _fmt(std[:3])
        std_tail = _fmt(std[-3:])
        mean_span = float(np.max(mean) - np.min(mean))
        std_min = float(np.min(std))
        message = (
            f"[data] dataset={dataset.name or 'unknown'} mean_head={mean_head} "
            f"mean_tail={mean_tail} std_head={std_head} std_tail={std_tail} "
            f"span={mean_span:.4f} std_min={std_min:.4f}"
        )
        metric = {
            "dataset": dataset.name,
            "mean_head": mean_head,
            "mean_tail": mean_tail,
            "std_head": std_head,
            "std_tail": std_tail,
            "mean_span": mean_span,
            "std_min": std_min,
        }
        await self._emit_log("INFO", message, metric=metric)

    async def emit_config_summary(self, reason: str = "startup") -> None:
        """Publish a log event summarizing the active training configuration."""
        config = self.get_config()
        dataset = config.get("dataset")
        mode = config.get("mode")
        solver = config.get("solver")
        scheduler = config.get("scheduler")
        warmup_steps = config.get("warmup_steps")
        steps_per_epoch = config.get("steps_per_epoch")
        lr = config.get("lr")
        weight_decay = config.get("weight_decay")
        grad_clip = config.get("grad_clip")
        logit_scale = config.get("logit_scale")
        rate_reg_lambda = config.get("rate_reg_lambda")
        rate_target = config.get("rate_target")
        anderson_m = config.get("anderson_m")
        anderson_beta = config.get("anderson_beta")
        network_size = config.get("network_size")
        layers = config.get("layers")
        epochs = config.get("epochs")

        def _fmt(value: Any, fmt: str = ".4f") -> str:
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return format(value, fmt)
            return str(value)

        message = (
            f"[config:{reason}] dataset={dataset} mode={mode} epochs={epochs} layers={layers} "
            f"neurons={network_size} lr={_fmt(lr)} solver={solver} anderson_m={anderson_m} "
            f"anderson_beta={_fmt(anderson_beta)} scheduler={scheduler} warmup={warmup_steps} "
            f"steps_per_epoch={steps_per_epoch} weight_decay={_fmt(weight_decay)} "
            f"grad_clip={_fmt(grad_clip)} logit_scale={_fmt(logit_scale)} "
            f"rate_reg_lambda={_fmt(rate_reg_lambda)} rate_target={_fmt(rate_target)}"
        )
        metric = {
            "dataset": dataset,
            "mode": mode,
            "epochs": epochs,
            "layers": layers,
            "network_size": network_size,
            "lr": lr,
            "solver": solver,
            "anderson_m": anderson_m,
            "anderson_beta": anderson_beta,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
            "steps_per_epoch": steps_per_epoch,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "logit_scale": logit_scale,
            "rate_reg_lambda": rate_reg_lambda,
            "rate_target": rate_target,
        }
        await self._emit_log("INFO", message, metric=metric)

    async def _emit_log(
        self,
        level: str,
        message: str,
        *,
        metric: Optional[Dict[str, Any]] = None,
        subject: Optional[str] = None,
    ) -> None:
        level_upper = level.upper()
        log_level = getattr(logging, level_upper, logging.INFO)
        logger.log(log_level, message)
        payload: Dict[str, Any] = {"level": level_upper, "msg": message, "time_unix": _current_millis()}
        if metric is not None:
            payload["metric"] = metric
        await self._broker.publish("log", payload)
        if subject:
            queue = getattr(self._broker, "_message_queue", None)
            if queue is not None:
                try:
                    queue.publish(subject, json.dumps(payload, ensure_ascii=True).encode("utf-8"))
                except Exception as exc:  # pragma: no cover - 外部队列可能未配置
                    logger.debug("发布训练日志到 %s 失败：%s", subject, exc)

    async def init_training(self, payload: TrainingInitRequest) -> None:
        async with self._lock:
            self._config = self._apply_overrides(self._config, payload.model_dump())
        logger.info(
            "初始化训练参数：dataset=%s, mode=%s, epochs=%d, layers=%d, network_size=%d",
            payload.dataset,
            payload.mode,
            payload.epochs,
            payload.layers,
            payload.network_size,
        )
        await self._broker.publish(
            "log",
            {
                "level": "INFO",
                "msg": (
                    f"初始化训练参数 dataset={payload.dataset} mode={payload.mode} "
                    f"epochs={payload.epochs} layers={payload.layers} neurons={payload.network_size}"
                ),
                "time_unix": _current_millis(),
            },
        )
        await self._broker.publish(
            "train_init",
            {
                "dataset": payload.dataset,
                "epochs": payload.epochs,
                "batch_size": max(1, payload.network_size // max(payload.layers, 1)),
                "timesteps": payload.T,
                "fixed_point_K": payload.K,
                "fixed_point_tol": payload.tol,
                "solver": payload.solver,
                "anderson_m": payload.anderson_m,
                "anderson_beta": payload.anderson_beta,
                "K_schedule": payload.K_schedule,
                "hidden": payload.network_size,
                "layers": payload.layers,
                "lr": payload.lr,
                "temperature": payload.temperature,
                "logit_scale": payload.logit_scale,
                "logit_scale_learnable": payload.logit_scale_learnable,
                "steps_per_epoch": self._config.get("steps_per_epoch"),
                "scheduler": self._config.get("scheduler"),
                "warmup_steps": self._config.get("warmup_steps"),
                "weight_decay": self._config.get("weight_decay"),
                "grad_clip": self._config.get("grad_clip"),
                "rate_reg_lambda": self._config.get("rate_reg_lambda"),
                "rate_target": self._config.get("rate_target"),
                "augment": bool(self._config.get("augment", True)),
                "time_unix": _current_millis(),
            },
        )

    async def start_training(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                raise HTTPException(status.HTTP_409_CONFLICT, "训练已经在进行中")
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self._run_training())
            self._status = "Training"
        await self._broker.publish("train_status", {"status": "Training", "time_unix": _current_millis()})
        await self._broker.publish(
            "log",
            {"level": "INFO", "msg": "训练已启动", "time_unix": _current_millis()},
        )

    async def stop_training(self) -> None:
        async with self._lock:
            if not self._task or self._task.done():
                raise HTTPException(status.HTTP_409_CONFLICT, "当前没有正在运行的训练")
            self._stop_event.set()
            task = self._task
        await task
        async with self._lock:
            self._status = "Stopped"
        await self._broker.publish("train_status", {"status": "Stopped", "time_unix": _current_millis()})

    async def _run_training(self) -> None:
        config = self.get_config()
        train_cfg = config.get("train") if isinstance(config.get("train"), dict) else None

        def _train_param(key: str, default: Any) -> Any:
            if isinstance(train_cfg, dict) and key in train_cfg:
                return train_cfg[key]
            return config.get(key, default)

        dataset_name = config.get("dataset", "MNIST")
        mode_text = str(config.get("mode", "fpt")).lower()
        use_residual_metric = mode_text == "fpt"
        is_tstep_mode = mode_text == "tstep"
        np_rng = np.random.default_rng(self._np_seed)
        logger.info("加载数据集：%s", dataset_name)
        try:
            dataset = await asyncio.to_thread(self._load_dataset_bundle, dataset_name)
        except Exception as exc:
            logger.exception("加载数据集 %s 失败：%s", dataset_name, exc)
            await self._broker.publish(
                "log",
                {
                    "level": "ERROR",
                    "msg": f"加载数据集 {dataset_name} 失败：{exc}",
                    "time_unix": _current_millis(),
                },
            )
            async with self._lock:
                self._status = "Idle"
            await self._broker.publish("train_status", {"status": "Idle", "time_unix": _current_millis()})
            return

        timesteps = int(config.get("T") or 12)
        base_iterations = max(1, int(config.get("K", 3)))
        tolerance = float(_train_param("tol", config.get("tol", 1e-5)))
        solver = str(_train_param("solver", config.get("solver", "anderson"))).lower()
        anderson_m = max(1, int(_train_param("anderson_m", config.get("anderson_m", 4))))
        anderson_beta = float(_train_param("anderson_beta", config.get("anderson_beta", 0.5)))
        k_schedule_raw = _train_param("K_schedule", config.get("K_schedule"))
        k_schedule_values = self._normalize_k_schedule(k_schedule_raw, base_iterations)
        fp_kernel_config = FixedPointConfig(
            iterations=base_iterations,
            tolerance=tolerance,
            solver=solver,
            anderson_m=anderson_m,
            anderson_beta=anderson_beta,
        )
        kernel = self._get_basal_kernel(timesteps, base_iterations, tolerance, self._params.dt, fp_kernel_config)
        scheduler_name = "warmup_cosine"
        grad_clip = float(max(0.0, _train_param("grad_clip", config.get("grad_clip", 1.0))))
        base_lr = float(max(1e-6, _train_param("lr", config.get("lr", 1e-3))))
        min_lr_override = _train_param("min_lr", config.get("min_lr"))
        min_lr_scale = float(_train_param("min_lr_scale", config.get("min_lr_scale", 0.1)))
        min_lr = float(max(1e-9, min_lr_override if min_lr_override is not None else base_lr * min_lr_scale))
        weight_decay = float(max(0.0, _train_param("weight_decay", config.get("weight_decay", 1e-4))))
        temperature = float(max(1e-6, _train_param("temperature", 1.0)))
        initial_logit_scale = float(
            max(1e-6, _train_param("logit_scale", config.get("logit_scale", 1.25)))
        )
        logit_scale_learnable = bool(_train_param("logit_scale_learnable", False))
        k_schedule_label = k_schedule_raw or "none"
        rate_reg_lambda = float(
            max(0.0, _train_param("rate_reg_lambda", config.get("rate_reg_lambda", 1e-3)))
        )
        rate_target = float(_train_param("rate_target", config.get("rate_target", 0.2)))
        rate_reg_value = rate_reg_lambda
        augment_enabled = bool(_train_param("augment", config.get("augment", True)))
        self._config["augment"] = augment_enabled
        num_classes = dataset.num_classes
        batch_size = int(np.clip(config.get("network_size", 256), 32, 256))
        layers_count = max(1, int(config.get("layers", 1)))
        epochs = max(1, int(config.get("epochs", 1)))
        ema_decay = 0.9

        model = self._init_model(dataset.input_dim, timesteps, num_classes, np_rng, initial_logit_scale)
        self._model_state = model

        train_x = dataset.train_x
        train_y = dataset.train_y
        total_examples = train_x.shape[0]
        if total_examples == 0:
            raise RuntimeError("训练集样本数量为 0，无法开始训练")
        steps_per_epoch = math.ceil(total_examples / batch_size)
        requested_steps = self._config.get("steps_per_epoch")
        if requested_steps not in (None, steps_per_epoch):
            await self._emit_log(
                "INFO",
                (
                    f"steps_per_epoch 覆盖值 {requested_steps} 已被忽略，"
                    f"根据样本量改用 {steps_per_epoch}"
                ),
            )
        self._config["steps_per_epoch"] = steps_per_epoch
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = max(1, int(math.ceil(total_steps * 0.05)))
        await self._emit_log(
            "INFO",
            (
                f"using solver={solver} K={base_iterations} T={timesteps} "
                f"temperature={temperature} K_schedule={k_schedule_label} "
                f"scheduler={scheduler_name} warmup={warmup_steps} steps_per_epoch={steps_per_epoch} "
                f"rate_reg={rate_reg_value} grad_clip={grad_clip} augment={'on' if augment_enabled else 'off'}"
            ),
            metric={
                "solver": solver,
                "K": base_iterations,
                "T": timesteps,
                "K_schedule": k_schedule_label,
                "scheduler": scheduler_name,
                "temperature": temperature,
                "rate_reg_lambda": rate_reg_lambda,
                "rate_target": rate_target,
                "warmup_steps": warmup_steps,
                "min_lr": min_lr,
                "grad_clip": grad_clip,
                "logit_scale": initial_logit_scale,
                "logit_scale_learnable": logit_scale_learnable,
                "weight_decay": weight_decay,
                "steps_per_epoch": steps_per_epoch,
                "augment": augment_enabled,
            },
        )
        await self._emit_log(
            "INFO",
            (
                "config summary: mode=%s lr=%.5f min_lr=%.5f weight_decay=%.4f grad_clip=%.2f "
                "scheduler=%s warmup_steps=%d steps_per_epoch=%d logit_scale=%.3f rate_target=%s "
                "augment=%s"
            )
            % (
                config.get("mode", "fpt"),
                base_lr,
                min_lr,
                weight_decay,
                grad_clip,
                scheduler_name,
                warmup_steps,
                steps_per_epoch,
                initial_logit_scale,
                rate_target,
                "on" if augment_enabled else "off",
            ),
        )

        train_shape = dataset.train_x.shape
        val_shape = dataset.val_x.shape
        train_counts = np.bincount(dataset.train_y, minlength=num_classes).tolist()
        val_counts = np.bincount(dataset.val_y, minlength=num_classes).tolist()
        await self._emit_log(
            "INFO",
            (
                f"数据集 {dataset_name} 已加载：train_samples={train_shape[0]} val_samples={val_shape[0]} "
                f"input_dim={dataset.input_dim} num_classes={num_classes}"
            ),
        )
        await self._log_dataset_stats(dataset)
        await self._emit_log(
            "DEBUG",
            f"训练集标签分布：{train_counts}; 验证集标签分布：{val_counts}",
        )

        global_step = 0
        best_acc = 0.0
        best_loss = float("inf")
        ema_loss: Optional[float] = None
        ema_acc: Optional[float] = None

        try:
            for epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                epoch_indices = np_rng.permutation(total_examples)
                batch_throughputs: List[float] = []
                epoch_loss_total = 0.0
                epoch_acc_total = 0.0
                epoch_examples = 0
                epoch_residuals: List[float] = []
                epoch_conf_total = 0.0
                epoch_entropy_total = 0.0
                epoch_logit_sum = 0.0
                epoch_logit_sq_sum = 0.0
                epoch_logit_count = 0
                epoch_s_rate = 0.0
                epoch_bin, epoch_k_limit = self._k_limit_for_epoch(
                    epoch, epochs, k_schedule_values, base_iterations
                )
                preview_step = min(global_step + 1, total_steps)
                preview_lr = _cosine_with_warmup(
                    preview_step,
                    base_lr,
                    warmup_steps,
                    total_steps,
                    min_lr,
                )
                await self._emit_log(
                    "INFO",
                    (
                        f"开始 epoch={epoch}/{epochs} steps={steps_per_epoch} "
                        f"batch_size={batch_size} lr={preview_lr:.12f}"
                    ),
                )
                zero_acc_logs = 0
                for step in range(steps_per_epoch):
                    if self._stop_event.is_set():
                        await self._broker.publish(
                            "log",
                            {
                                "level": "WARNING",
                                "msg": f"训练在 epoch={epoch} step={step + 1} 被停止",
                                "time_unix": _current_millis(),
                            },
                        )
                        await self._broker.publish("train_status", {"status": "Stopped", "time_unix": _current_millis()})
                        return
    
                    offset = step * batch_size
                    batch_idx = epoch_indices[offset : offset + batch_size]
                    if batch_idx.size == 0:
                        continue
                    batch_raw = train_x[batch_idx]
                    batch_y = train_y[batch_idx]
                    batch_x = self._prepare_inputs(
                        batch_raw,
                        dataset,
                        rng=np_rng,
                        augment=augment_enabled,
                    )

                    batch_size_actual = batch_idx.shape[0]
                    if batch_y.dtype != np.int64:
                        raise AssertionError(f"batch labels must be int64, got {batch_y.dtype}")
                    if batch_y.size > 0:
                        y_min = int(batch_y.min())
                        y_max = int(batch_y.max())
                        if y_min < 0 or y_max >= num_classes:
                            raise AssertionError(
                                f"batch labels must be within [0, {num_classes}), got [{y_min}, {y_max}]"
                            )
    
                    global_step += 1
                    current_lr = _cosine_with_warmup(global_step, base_lr, warmup_steps, total_steps, min_lr)
                    step_start = time.perf_counter()
                    logits_raw, basal_currents = self._forward_batch(model, batch_x, kernel)
                    if logits_raw.shape[1] != num_classes:
                        raise AssertionError(
                            f"logits second dimension must equal num_classes={num_classes}, got {logits_raw.shape[1]}"
                        )
                    batch_s_rate = 0.0
                    grad_kernel = kernel
                    logits_source = logits_raw
                    if is_tstep_mode:
                        s_mean = np.mean(basal_currents, axis=2)
                        logits_source = s_mean + model.b_out[None, :]
                        grad_kernel = np.full_like(kernel, 1.0 / max(1, timesteps), dtype=kernel.dtype)
                        batch_s_rate = float(np.mean(s_mean)) if s_mean.size else 0.0
                    scale_factor = model.logit_scale / temperature
                    logits = logits_source * scale_factor
                    nll, probs = self._nll_from_logits(logits, batch_y)
                    loss = nll
                    if is_tstep_mode and rate_reg_lambda > 0.0:
                        rate_error = batch_s_rate - rate_target
                        loss += rate_reg_lambda * (rate_error ** 2)
                    grad_logits_scaled = probs.copy()
                    if batch_y.size:
                        grad_logits_scaled[np.arange(batch_y.shape[0]), batch_y] -= 1.0
                    grad_logits_scaled /= max(batch_y.shape[0], 1)
                    grad_logits = grad_logits_scaled * scale_factor
                    probs_safe = np.clip(probs, 1e-12, 1.0)
                    entropy = (
                        float(np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))) if probs.size else 0.0
                    )
                    confidence = (
                        float(np.mean(probs[np.arange(batch_y.shape[0]), batch_y])) if batch_y.size else 0.0
                    )
                    logit_mean = float(np.mean(logits)) if logits.size else 0.0
                    logit_std = float(np.std(logits)) if logits.size else 0.0
                    logit_scale_grad = None
                    if logit_scale_learnable:
                        inv_temp = 1.0 / temperature
                        logit_scale_grad = float(np.sum(grad_logits_scaled * logits_source) * inv_temp)
                    predictions = np.argmax(probs, axis=1)
                    if predictions.shape != batch_y.shape:
                        raise AssertionError("np.argmax must operate along axis=1")
                    batch_acc = float(np.mean(predictions == batch_y))
                    top5_hits = np.any(np.argsort(probs, axis=1)[:, -5:] == batch_y[:, None], axis=1)
                    top5_acc = float(np.mean(top5_hits))
    
                    if batch_size_actual > 0:
                        epoch_loss_total += loss * batch_size_actual
                        epoch_acc_total += batch_acc * batch_size_actual
                        epoch_examples += batch_size_actual
                        epoch_conf_total += confidence * batch_size_actual
                        epoch_entropy_total += entropy * batch_size_actual
                        if is_tstep_mode:
                            epoch_s_rate += batch_s_rate * batch_size_actual
                    if logits.size:
                        epoch_logit_sum += float(np.sum(logits))
                        epoch_logit_sq_sum += float(np.sum(logits ** 2))
                        epoch_logit_count += logits.size
                    grads = self._compute_gradients(batch_x, grad_logits, grad_kernel)
                    extra_grads = None
                    if logit_scale_grad is not None:
                        extra_grads = {"logit_scale": np.array([logit_scale_grad], dtype=np.float32)}
                    grad_norm = _clip_gradients_inplace(grads, grad_clip, extra_grads)
                    if extra_grads is not None:
                        logit_scale_grad = float(extra_grads["logit_scale"][0])
                    if batch_acc == 0.0 and zero_acc_logs < 3:
                        label_hist = np.bincount(batch_y, minlength=num_classes).tolist()
                        pred_hist = np.bincount(predictions, minlength=num_classes).tolist()
                        true_prob = confidence
                        top_pred_prob = float(np.mean(np.max(probs, axis=1))) if probs.size else 0.0
                        msg = (
                            f"epoch={epoch} step={step + 1} 检测到 acc=0.0："
                            f"label_hist={label_hist} pred_hist={pred_hist} "
                            f"logits_mean={logit_mean:.4f} logits_std={logit_std:.4f} "
                            f"true_prob_mean={true_prob:.4f} top_prob_mean={top_pred_prob:.4f} "
                            f"grad_norm={grad_norm:.4f}"
                        )
                        await self._emit_log("DEBUG", msg)
                        zero_acc_logs += 1
    
                    self._apply_updates(
                        model,
                        grads,
                        current_lr,
                        weight_decay,
                        learnable_logit_scale=logit_scale_learnable,
                        logit_scale_grad=logit_scale_grad,
                    )
    
                    step_duration = max(time.perf_counter() - step_start, 1e-6)
                    throughput = float(batch_x.shape[0] / step_duration)
                    batch_throughputs.append(throughput)
    
                    ema_loss = loss if ema_loss is None else ema_decay * ema_loss + (1 - ema_decay) * loss
                    ema_acc = batch_acc if ema_acc is None else ema_decay * ema_acc + (1 - ema_decay) * batch_acc
    
                    batch_fp_config = FixedPointConfig(
                        iterations=epoch_k_limit,
                        tolerance=tolerance,
                        solver=solver,
                        anderson_m=anderson_m,
                        anderson_beta=anderson_beta,
                    )
                    residual_value, actual_iters, spike_payload = self._build_iteration_events(
                        model,
                        batch_x,
                        batch_y,
                        basal_currents,
                        epoch,
                        step,
                        layers_count,
                        timesteps,
                        batch_fp_config,
                    )
    
                    if use_residual_metric and batch_size_actual > 0:
                        epoch_residuals.append(residual_value)
    
                    metrics_payload = {
                        "phase": "train",
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss,
                        "nll": loss,
                        "acc": batch_acc,
                        "top5": top5_acc,
                        "conf": confidence,
                        "entropy": entropy,
                        "throughput": throughput,
                        "step_ms": step_duration * 1000.0,
                        "ema_loss": ema_loss,
                        "ema_acc": ema_acc,
                        "lr": current_lr,
                        "temperature": temperature,
                        "logit_scale": model.logit_scale,
                        "logit_mean": logit_mean,
                        "logit_std": logit_std,
                        "s_rate": batch_s_rate,
                        "rate_target": rate_target,
                        "residual": residual_value,
                        "k": actual_iters,
                        "max_k": epoch_k_limit,
                        "k_bin": epoch_bin,
                        "examples": int(batch_x.shape[0]),
                        "time_unix": _current_millis(),
                    }
                    iter_payload = {
                        "epoch": epoch,
                        "step": step + 1,
                        "k": actual_iters,
                        "max_k": epoch_k_limit,
                        "layer": step % layers_count,
                        "residual": residual_value,
                        "solver": solver,
                        "k_bin": epoch_bin,
                        "lr": current_lr,
                        "time_unix": _current_millis(),
                    }
                    iter_payload["fp"] = {
                        "solver": solver,
                        "k": actual_iters,
                        "residual": residual_value,
                    }
    
                    await self._broker.publish("train_iter", iter_payload)
                    await self._broker.publish("metrics_batch", metrics_payload)
                    if spike_payload is not None:
                        await self._broker.publish("spike", spike_payload)
    
                    ema_loss_str = f"{ema_loss:.4f}" if ema_loss is not None else "nan"
                    ema_acc_str = f"{ema_acc:.4f}" if ema_acc is not None else "nan"
                    logger.info(
                        "[BATCH] ep=%d st=%d loss=%.4f acc=%.4f top5=%.4f tps=%.1f step_ms=%.1f ema_loss=%s "
                        "ema_acc=%s lr=%.5f examples=%d",
                        epoch,
                        step + 1,
                        loss,
                        batch_acc,
                        top5_acc,
                        throughput,
                        step_duration * 1000.0,
                        ema_loss_str,
                        ema_acc_str,
                        current_lr,
                        batch_x.shape[0],
                    )
    
                    if step % 10 == 0:
                        await asyncio.sleep(0)
    
                (
                    val_loss,
                    val_acc,
                    val_top5,
                    val_conf,
                    val_entropy,
                    val_logit_mean,
                    val_logit_std,
                    val_s_rate,
                ) = self._evaluate(
                    model,
                    dataset,
                    kernel,
                    temperature,
                    batch_size=512,
                    is_tstep=is_tstep_mode,
                )
                best_acc = max(best_acc, val_acc)
                best_loss = min(best_loss, val_loss)
                epoch_duration = time.perf_counter() - epoch_start
                avg_throughput = float(np.mean(batch_throughputs)) if batch_throughputs else None
                denom = max(epoch_examples, 1)
                train_loss_epoch = epoch_loss_total / denom
                train_acc_epoch = epoch_acc_total / denom
                train_conf_epoch = epoch_conf_total / denom
                train_entropy_epoch = epoch_entropy_total / denom
                train_s_rate_epoch = epoch_s_rate / denom if is_tstep_mode else 0.0
                if epoch_logit_count > 0:
                    train_logit_mean = epoch_logit_sum / epoch_logit_count
                    train_logit_var = max(epoch_logit_sq_sum / epoch_logit_count - train_logit_mean ** 2, 0.0)
                    train_logit_std = math.sqrt(train_logit_var)
                else:
                    train_logit_mean = 0.0
                    train_logit_std = 0.0
                residual_mean = (
                    float(np.mean(epoch_residuals)) if use_residual_metric and epoch_residuals else None
                )
    
                epoch_payload = {
                    "phase": "val",
                    "epoch": epoch,
                    "loss": val_loss,
                    "nll": val_loss,
                    "acc": val_acc,
                    "conf": val_conf,
                    "entropy": val_entropy,
                    "best_acc": best_acc,
                    "best_loss": best_loss,
                    "avg_throughput": avg_throughput,
                    "epoch_sec": epoch_duration,
                    "top5": val_top5,
                    "time_unix": _current_millis(),
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "train_conf": train_conf_epoch,
                    "train_entropy": train_entropy_epoch,
                    "train_s_rate": train_s_rate_epoch,
                    "temperature": temperature,
                    "logit_scale": model.logit_scale,
                    "logit_mean": val_logit_mean,
                    "logit_std": val_logit_std,
                    "train_logit_mean": train_logit_mean,
                    "train_logit_std": train_logit_std,
                    "s_rate": val_s_rate,
                    "rate_target": rate_target,
                }
                if residual_mean is not None:
                    epoch_payload["residual"] = residual_mean
                await self._broker.publish("metrics_epoch", epoch_payload)
    
                avg_tps_str = f"{avg_throughput:.1f}" if avg_throughput is not None else "nan"
                residual_str = f"{residual_mean:.6f}" if residual_mean is not None else "n/a"
                logger.info(
                    "[EPOCH] epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f best_acc=%.4f "
                    "best_loss=%.4f avg_tps=%s epoch_sec=%.1f top5=%.4f residual=%s",
                    epoch,
                    train_loss_epoch,
                    train_acc_epoch,
                    val_loss,
                    val_acc,
                    best_acc,
                    best_loss,
                    avg_tps_str,
                    epoch_duration,
                    val_top5,
                    residual_str,
                )
    
                log_metric: Dict[str, Any] = {
                    "train_loss": train_loss_epoch,
                    "train_acc": train_acc_epoch,
                    "train_conf": train_conf_epoch,
                    "train_entropy": train_entropy_epoch,
                    "train_s_rate": train_s_rate_epoch,
                    "train_logit_mean": train_logit_mean,
                    "train_logit_std": train_logit_std,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_conf": val_conf,
                    "val_entropy": val_entropy,
                    "val_s_rate": val_s_rate,
                    "val_logit_mean": val_logit_mean,
                    "val_logit_std": val_logit_std,
                    "temperature": temperature,
                    "logit_scale": model.logit_scale,
                    "rate_target": rate_target,
                    "steps_per_epoch": steps_per_epoch,
                }
                if avg_throughput is not None:
                    log_metric["avg_throughput"] = avg_throughput
                if residual_mean is not None:
                    log_metric["residual"] = residual_mean
                log_message = (
                    f"[epoch {epoch}/{epochs}] "
                    f"train_loss={train_loss_epoch:.4f} train_acc={train_acc_epoch:.4f} "
                    f"train_conf={train_conf_epoch:.4f} val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_conf={val_conf:.4f} "
                    f"rate={train_s_rate_epoch:.3f}→target={rate_target:.2f}"
                )
                log_message += (
                    f" residual={residual_mean:.6f}" if residual_mean is not None else " residual=n/a"
                )
                log_message += (
                    f" avg_throughput={avg_throughput:.2f}" if avg_throughput is not None else " avg_throughput=n/a"
                )
                await self._emit_log("INFO", log_message, metric=log_metric, subject="snn.ui.log.training")
    
                if self._stop_event.is_set():
                    await self._broker.publish("train_status", {"status": "Stopped", "time_unix": _current_millis()})
                    return
    
            await self._broker.publish(
                "log",
                {"level": "INFO", "msg": "训练完成", "time_unix": _current_millis()},
            )
            logger.info("训练完成")
        finally:
            async with self._lock:
                self._status = "Idle"
            await self._broker.publish("train_status", {"status": "Idle", "time_unix": _current_millis()})

    def _get_basal_kernel(
        self,
        timesteps: int,
        iterations: int,
        tolerance: float,
        dt: float,
        config: FixedPointConfig,
    ) -> np.ndarray:
        key = (
            timesteps,
            iterations,
            tolerance,
            dt,
            config.solver,
            config.anderson_m,
            config.anderson_beta,
        )
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = _compute_basal_kernel(self._params, timesteps, config)
            self._kernel_cache[key] = kernel
        return kernel

    def _load_dataset_bundle(self, name: str) -> DatasetBundle:
        normalized = name.upper()
        if normalized in self._dataset_cache:
            return self._dataset_cache[normalized]
        loader_map = {
            "MNIST": self._load_mnist,
            "FASHION": self._load_fashion_mnist,
            "FASHION-MNIST": self._load_fashion_mnist,
            "CIFAR10": self._load_cifar10,
        }
        if normalized not in loader_map:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"暂不支持数据集 {name}")
        bundle = loader_map[normalized]()
        self._dataset_cache[normalized] = bundle
        return bundle

    def _load_mnist(self) -> DatasetBundle:
        dataset_dir = self._data_root / "mnist"
        npz_path = dataset_dir / "mnist.npz"
        if not npz_path.exists():
            logger.warning("未找到 MNIST 数据文件，自动生成占位数据集：%s", npz_path)
            self._generate_placeholder_mnist(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"未找到 MNIST 数据文件：{npz_path}")
        with np.load(npz_path) as data:
            train_x = data["x_train"].astype(np.float32) / 255.0
            train_y = data["y_train"].astype(np.int64)
            test_x = data["x_test"].astype(np.float32) / 255.0
            test_y = data["y_test"].astype(np.int64)
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        feature_mean, feature_std = ensure_feature_stats(train_x, dataset_dir)
        num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)
        image_shape = DATASET_IMAGE_SHAPES["MNIST"]
        return DatasetBundle(
            train_x,
            train_y,
            test_x,
            test_y,
            train_x.shape[1],
            num_classes,
            feature_mean=feature_mean,
            feature_std=feature_std,
            image_shape=image_shape,
            name="MNIST",
        )

    def _generate_placeholder_mnist(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self._np_seed)
        train_samples = 512
        test_samples = 128
        feature_dim = 28 * 28
        x_train = rng.random((train_samples, feature_dim), dtype=np.float32)
        y_train = rng.integers(0, 10, size=train_samples, endpoint=False).astype(np.int64)
        x_test = rng.random((test_samples, feature_dim), dtype=np.float32)
        y_test = rng.integers(0, 10, size=test_samples, endpoint=False).astype(np.int64)
        np.savez(target, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        metadata = {
            "name": "MNIST",
            "generated": True,
            "files": [
                {
                    "filename": target.name,
                    "size_bytes": target.stat().st_size,
                    "source": "generated://placeholder",
                }
            ],
        }
        metadata_path = target.parent / "metadata.json"
        sample_path = target.parent / "sample.txt"
        try:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            sample_path.write_text("Dataset placeholder generated locally.\n", encoding="utf-8")
        except OSError:
            logger.warning("无法写入 MNIST 占位符元数据：%s", metadata_path)

    def _load_fashion_mnist(self) -> DatasetBundle:
        dataset_dir = self._data_root / "fashion"
        train_images = dataset_dir / "train-images-idx3-ubyte.gz"
        train_labels = dataset_dir / "train-labels-idx1-ubyte.gz"
        test_images = dataset_dir / "t10k-images-idx3-ubyte.gz"
        test_labels = dataset_dir / "t10k-labels-idx1-ubyte.gz"
        required = [train_images, train_labels, test_images, test_labels]
        if not all(path.exists() for path in required):
            missing = [path.name for path in required if not path.exists()]
            raise FileNotFoundError(f"缺少 Fashion-MNIST 文件：{', '.join(missing)}")
        train_x = _read_idx_images(train_images)
        train_y = _read_idx_labels(train_labels)
        test_x = _read_idx_images(test_images)
        test_y = _read_idx_labels(test_labels)
        feature_mean, feature_std = ensure_feature_stats(train_x, dataset_dir)
        num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)
        image_shape = DATASET_IMAGE_SHAPES["FASHION"]
        return DatasetBundle(
            train_x,
            train_y,
            test_x,
            test_y,
            train_x.shape[1],
            num_classes,
            feature_mean=feature_mean,
            feature_std=feature_std,
            image_shape=image_shape,
            name="FASHION",
        )

    def _load_cifar10(self) -> DatasetBundle:
        dataset_dir = self._data_root / "cifar10"
        tar_path = dataset_dir / "cifar-10-python.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(f"未找到 CIFAR10 数据集压缩包：{tar_path}")
        train_x, train_y, test_x, test_y = _load_cifar_arrays(dataset_dir, tar_path)
        train_x = (train_x / 255.0).astype(np.float32)
        test_x = (test_x / 255.0).astype(np.float32)
        train_count = train_x.shape[0]
        test_count = test_x.shape[0]
        train_x = np.transpose(train_x.reshape(train_count, 3, 32, 32), (0, 2, 3, 1)).reshape(train_count, -1)
        test_x = np.transpose(test_x.reshape(test_count, 3, 32, 32), (0, 2, 3, 1)).reshape(test_count, -1)
        feature_mean, feature_std = ensure_feature_stats(train_x, dataset_dir)
        num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)
        image_shape = DATASET_IMAGE_SHAPES["CIFAR10"]
        return DatasetBundle(
            train_x,
            train_y,
            test_x,
            test_y,
            train_x.shape[1],
            num_classes,
            feature_mean=feature_mean,
            feature_std=feature_std,
            image_shape=image_shape,
            name="CIFAR10",
        )

    def _init_model(
        self,
        input_dim: int,
        timesteps: int,
        num_classes: int,
        rng: np.random.Generator,
        logit_scale: float,
    ) -> ModelState:
        scale = math.sqrt(2.0 / max(1, input_dim))
        W_basal = rng.normal(0.0, scale, size=(num_classes, timesteps, input_dim)).astype(np.float32)
        b_basal = np.zeros((num_classes, timesteps), dtype=np.float32)
        b_out = np.zeros(num_classes, dtype=np.float32)
        return ModelState(W_basal=W_basal, b_basal=b_basal, b_out=b_out, logit_scale=float(logit_scale))

    def _forward_batch(
        self,
        model: ModelState,
        inputs: np.ndarray,
        kernel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        basal_currents = np.tensordot(inputs, model.W_basal, axes=([1], [2]))  # (batch, classes, timesteps)
        basal_currents += model.b_basal[None, :, :]
        logits = np.sum(basal_currents * kernel[None, None, :], axis=2) + model.b_out[None, :]
        return logits.astype(np.float32, copy=False), basal_currents.astype(np.float32, copy=False)

    @staticmethod
    def _nll_from_logits(
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        if logits.ndim != 2:
            raise ValueError("logits tensor must be 2D (batch, classes)")
        if labels.dtype != np.int64:
            raise TypeError(f"labels must be int64 for indexing, got {labels.dtype}")
        logits64 = logits.astype(np.float64, copy=False)
        max_logits = np.max(logits64, axis=1, keepdims=True)
        shifted = logits64 - max_logits
        exp_shifted = np.exp(shifted, dtype=np.float64)
        denom = np.clip(np.sum(exp_shifted, axis=1, keepdims=True), 1e-12, None)
        log_probs = shifted - np.log(denom)
        probs = np.exp(log_probs).astype(logits.dtype, copy=False)
        if labels.size == 0:
            return 0.0, probs
        row_idx = np.arange(labels.shape[0])
        log_prob_true = log_probs[row_idx, labels]
        loss = float(-np.mean(log_prob_true)) if log_prob_true.size else 0.0
        return loss, probs

    def _compute_gradients(
        self,
        inputs: np.ndarray,
        grad_logits: np.ndarray,
        kernel: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        grad_basal = grad_logits[:, :, None] * kernel[None, None, :]
        grad_W = np.einsum("bct,bi->cti", grad_basal, inputs)
        grad_b_basal = grad_basal.sum(axis=0)
        grad_b_out = grad_logits.sum(axis=0)
        return {
            "W_basal": grad_W,
            "b_basal": grad_b_basal,
            "b_out": grad_b_out,
        }

    def _apply_updates(
        self,
        model: ModelState,
        grads: Dict[str, np.ndarray],
        lr: float,
        weight_decay: float,
        *,
        learnable_logit_scale: bool = False,
        logit_scale_grad: Optional[float] = None,
    ) -> None:
        if weight_decay > 0.0 and lr > 0.0:
            decay = max(0.0, 1.0 - lr * weight_decay)
            model.W_basal *= decay
            model.b_basal *= decay
            model.b_out *= decay
            if learnable_logit_scale:
                model.logit_scale = float(model.logit_scale * decay)
        model.W_basal -= lr * grads["W_basal"]
        model.b_basal -= lr * grads["b_basal"]
        model.b_out -= lr * grads["b_out"]
        if learnable_logit_scale and logit_scale_grad is not None:
            updated = model.logit_scale - lr * logit_scale_grad
            model.logit_scale = float(np.clip(updated, 0.5, 3.0))

    @staticmethod
    def _normalize_k_schedule(raw: Any, fallback: int) -> Optional[List[int]]:
        values: List[int] = []
        if isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    value = int(item)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    values.append(value)
        elif isinstance(raw, str):
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            for part in parts:
                try:
                    value = int(part)
                except ValueError:
                    continue
                if value > 0:
                    values.append(value)
        elif isinstance(raw, dict):
            ordered = sorted(raw.items(), key=lambda entry: str(entry[0]))
            for _, entry_value in ordered:
                try:
                    value = int(entry_value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    values.append(value)
        if not values:
            return None
        return values

    @staticmethod
    def _k_limit_for_epoch(
        epoch_index: int,
        epochs: int,
        schedule: Optional[List[int]],
        default_value: int,
    ) -> Tuple[int, int]:
        if not schedule:
            return 0, default_value
        bin_size = max(1, math.ceil(epochs / len(schedule)))
        bin_index = min(len(schedule) - 1, max(0, (epoch_index - 1) // bin_size))
        return bin_index, schedule[bin_index]

    def _build_iteration_events(
        self,
        model: ModelState,
        inputs: np.ndarray,
        labels: np.ndarray,
        basal_currents: np.ndarray,
        epoch: int,
        step: int,
        layers_count: int,
        timesteps: int,
        fp_config: FixedPointConfig,
    ) -> Tuple[float, int, Optional[Dict[str, Any]]]:
        if inputs.shape[0] == 0:
            return 0.0, 0, None
        sample_idx = 0
        class_idx = int(labels[sample_idx]) if labels.size > 0 else 0
        class_idx = max(0, min(class_idx, model.W_basal.shape[0] - 1))
        sample_currents = basal_currents[sample_idx, class_idx, :].tolist()
        result = fixed_point_parallel_solve(
            self._params,
            [0.0] * timesteps,
            sample_currents,
            config=fp_config,
        )
        residual = result.residuals[-1] if result.residuals else 0.0
        iterations_used = len(result.residuals)
        spike_payload: Optional[Dict[str, Any]] = None
        if result.states:
            spikes = [idx for idx, state in enumerate(result.states) if state.spike]
            soma_values = [state.soma for state in result.states]
            if not spikes:
                spikes = list(np.argsort(soma_values)[-3:])
            else:
                spikes = spikes[:3]
            power = float(np.mean([soma_values[idx] for idx in spikes])) if spikes else float(np.mean(soma_values))
            apical_trace = [float(state.apical) for state in result.states]
            basal_trace = [float(state.basal) for state in result.states]
            spike_payload = {
                "layer": int(step % layers_count),
                "t": int(epoch * 1000 + step),
                "neurons": [int(value) for value in spikes],
                "power": power,
                "apical_trace": apical_trace,
                "basal_trace": basal_trace,
            }
        return float(residual), iterations_used, spike_payload

    def _evaluate(
        self,
        model: ModelState,
        dataset: DatasetBundle,
        kernel: np.ndarray,
        temperature: float,
        batch_size: int = 512,
        *,
        is_tstep: bool = False,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        inputs = dataset.val_x
        labels = dataset.val_y
        total = inputs.shape[0]
        losses: List[float] = []
        accs: List[float] = []
        top5_list: List[float] = []
        conf_sum = 0.0
        entropy_sum = 0.0
        logit_sum = 0.0
        logit_sq_sum = 0.0
        logit_count = 0
        s_rate_sum = 0.0
        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_x = inputs[start:end]
            batch_y = labels[start:end]
            if batch_x.size == 0:
                continue
            batch_x = self._prepare_inputs(batch_x, dataset, rng=None, augment=False)
            logits_raw, basal_currents = self._forward_batch(model, batch_x, kernel)
            logits_source = logits_raw
            batch_s_rate = 0.0
            if is_tstep:
                s_mean = np.mean(basal_currents, axis=2)
                logits_source = s_mean + model.b_out[None, :]
                batch_s_rate = float(np.mean(s_mean)) if s_mean.size else 0.0
            logits = logits_source * (model.logit_scale / temperature)
            loss, probs = self._nll_from_logits(logits, batch_y)
            pred = np.argmax(probs, axis=1)
            losses.append(loss)
            accs.append(float(np.mean(pred == batch_y)))
            top5_hits = np.any(np.argsort(probs, axis=1)[:, -5:] == batch_y[:, None], axis=1)
            top5_list.append(float(np.mean(top5_hits)))
            batch_size_actual = batch_x.shape[0]
            probs_safe = np.clip(probs, 1e-12, 1.0)
            entropy = (
                float(np.mean(-np.sum(probs_safe * np.log(probs_safe), axis=1))) if probs.size else 0.0
            )
            confidence = (
                float(np.mean(probs[np.arange(batch_y.shape[0]), batch_y])) if batch_y.size else 0.0
            )
            conf_sum += confidence * batch_size_actual
            entropy_sum += entropy * batch_size_actual
            logit_sum += float(np.sum(logits))
            logit_sq_sum += float(np.sum(logits ** 2))
            logit_count += logits.size
            if is_tstep:
                s_rate_sum += batch_s_rate * batch_size_actual
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_acc = float(np.mean(accs)) if accs else 0.0
        avg_top5 = float(np.mean(top5_list)) if top5_list else 0.0
        denom = max(total, 1)
        avg_conf = conf_sum / denom
        avg_entropy = entropy_sum / denom
        logit_count = max(logit_count, 1)
        logit_mean = logit_sum / logit_count
        logit_var = max(logit_sq_sum / logit_count - logit_mean ** 2, 0.0)
        logit_std = math.sqrt(logit_var)
        avg_s_rate = (s_rate_sum / denom) if is_tstep else 0.0
        return avg_loss, avg_acc, avg_top5, avg_conf, avg_entropy, logit_mean, logit_std, avg_s_rate


def _build_broker(message_queue_config: Optional[Dict[str, Any]]) -> Tuple[EventBroker, MessageQueue]:
    mq: MessageQueue = build_message_queue(message_queue_config)
    broker = EventBroker(mq)
    return broker, mq


def create_app(message_queue_config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """构建 FastAPI 应用，可通过参数覆盖消息队列配置。"""

    mq_config = message_queue_config or get_message_queue_config()
    broker, queue = _build_broker(mq_config)
    dataset_service = DatasetService(broker)
    global_config = load_config()
    training_service = TrainingService(broker, defaults=global_config.get("training_service"))
    log_buffer = LogRingBuffer(capacity=200)
    worker_nats_cfg = {}
    training_worker_cfg = global_config.get("training_worker") if isinstance(global_config, dict) else {}
    if isinstance(training_worker_cfg, dict):
        nats_cfg = training_worker_cfg.get("nats")
        if isinstance(nats_cfg, dict):
            worker_nats_cfg = nats_cfg
    relay = ExternalQueueRelay(queue, broker, log_buffer, worker_nats_cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await relay.start()
        await training_service.emit_config_summary(reason="startup")
        try:
            yield
        finally:
            await relay.stop()
            if isinstance(queue, InMemoryQueue):
                queue.close()

    app = FastAPI(title="SNN Training Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.broker = broker
    app.state.dataset_service = dataset_service
    app.state.training_service = training_service
    app.state.log_buffer = log_buffer
    app.state.queue_relay = relay

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        return {
            "training": training_service.get_config(),
            "training_worker": copy.deepcopy(global_config.get("training_worker")),
            "training_service": copy.deepcopy(global_config.get("training_service")),
            "status": training_service.status,
        }

    @app.get("/api/datasets")
    async def list_datasets() -> Dict[str, Any]:
        return dataset_service.list_datasets()

    @app.post("/api/datasets/download", status_code=status.HTTP_202_ACCEPTED)
    async def download_dataset(request: DatasetDownloadRequest) -> JSONResponse:
        await dataset_service.start_download(request.name)
        return JSONResponse({"ok": True, "message": "download scheduled"}, status_code=status.HTTP_202_ACCEPTED)

    @app.post("/api/train/init")
    async def init_training(request: TrainingInitRequest) -> JSONResponse:
        logger.info("接收到训练初始化请求：dataset=%s", request.dataset)
        await training_service.init_training(request)
        return JSONResponse({"ok": True})

    @app.post("/api/train/start", status_code=status.HTTP_202_ACCEPTED)
    async def start_training() -> JSONResponse:
        await training_service.start_training()
        return JSONResponse({"ok": True}, status_code=status.HTTP_202_ACCEPTED)

    @app.post("/api/train/stop")
    async def stop_training() -> JSONResponse:
        await training_service.stop_training()
        return JSONResponse({"ok": True})

    @app.get("/api/logs/recent")
    async def recent_logs(limit: int = 200) -> Dict[str, Any]:
        clamp = max(1, min(limit, 500))
        entries = await log_buffer.snapshot(clamp)
        return {"logs": entries}

    @app.get("/events")
    async def events() -> StreamingResponse:
        queue = await broker.subscribe()

        async def event_stream() -> Iterable[bytes]:
            try:
                while True:
                    event, payload = await queue.get()
                    data = json.dumps(payload, ensure_ascii=False)
                    message = f"event: {event}\ndata: {data}\n\n"
                    yield message.encode("utf-8")
            except asyncio.CancelledError:  # pragma: no cover - 连接主动关闭
                raise
            finally:
                await broker.unsubscribe(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


__all__ = ["create_app"]
