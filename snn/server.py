"""FastAPI 应用：提供与前端约定的 REST API 与 SSE 事件流。"""

from __future__ import annotations

import asyncio
import contextlib
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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_message_queue_config
from .fpt import FixedPointConfig, fixed_point_parallel_solve
from .mq import InMemoryQueue, MessageQueue, build_message_queue
from .neuron import ThreeCompartmentParams

logger = logging.getLogger(__name__)


def _current_millis() -> int:
    return int(time.time() * 1000)


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


@dataclass
class ModelState:
    W_basal: np.ndarray  # (num_classes, timesteps, input_dim)
    b_basal: np.ndarray  # (num_classes, timesteps)
    b_out: np.ndarray    # (num_classes,)


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

    def __init__(self, broker: EventBroker) -> None:
        self._broker = broker
        self._rng = random.Random(42)
        self._lock = asyncio.Lock()
        self._status: str = "Idle"
        self._config: Dict[str, Any] = {
            "dataset": "MNIST",
            "mode": "fpt",
            "network_size": 256,
            "layers": 3,
            "lr": 5e-3,
            "K": 4,
            "tol": 1e-5,
            "T": 12,
            "epochs": 6,
        }
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
        return dict(self._config)

    async def _emit_log(self, level: str, message: str) -> None:
        level_upper = level.upper()
        log_level = getattr(logging, level_upper, logging.INFO)
        logger.log(log_level, message)
        await self._broker.publish(
            "log",
            {"level": level_upper, "msg": message, "time_unix": _current_millis()},
        )

    async def init_training(self, payload: TrainingInitRequest) -> None:
        async with self._lock:
            self._config = payload.model_dump()
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
                "hidden": payload.network_size,
                "layers": payload.layers,
                "lr": payload.lr,
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
        dataset_name = config.get("dataset", "MNIST")
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
        iterations = max(1, int(config.get("K", 3)))
        tolerance = float(config.get("tol", 1e-5))
        fp_config = FixedPointConfig(iterations=iterations, tolerance=tolerance)
        kernel = self._get_basal_kernel(timesteps, iterations, tolerance, self._params.dt, fp_config)

        num_classes = dataset.num_classes
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
        await self._emit_log(
            "DEBUG",
            f"训练集标签分布：{train_counts}; 验证集标签分布：{val_counts}",
        )

        batch_size = int(np.clip(config.get("network_size", 256), 32, 256))
        layers_count = max(1, int(config.get("layers", 1)))
        lr = float(max(1e-5, config.get("lr", 1e-3)))
        epochs = max(1, int(config.get("epochs", 1)))
        ema_decay = 0.9
        weight_decay = 1e-4

        model = self._init_model(dataset.input_dim, timesteps, num_classes, np_rng)
        self._model_state = model

        train_x = dataset.train_x
        train_y = dataset.train_y
        total_examples = train_x.shape[0]
        steps_per_epoch = math.ceil(total_examples / batch_size)
        best_acc = 0.0
        best_loss = float("inf")
        ema_loss: Optional[float] = None
        ema_acc: Optional[float] = None

        try:
            for epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                indices = np_rng.permutation(total_examples)
                batch_throughputs: List[float] = []
                await self._emit_log(
                    "INFO",
                    (
                        f"开始 epoch={epoch}/{epochs} steps={steps_per_epoch} "
                        f"batch_size={batch_size} lr={lr:.5f}"
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

                    batch_idx = indices[step * batch_size : (step + 1) * batch_size]
                    batch_x = train_x[batch_idx]
                    batch_y = train_y[batch_idx]

                    step_start = time.perf_counter()
                    logits, basal_currents = self._forward_batch(model, batch_x, kernel)
                    loss, probs, grad_logits = self._cross_entropy(logits, batch_y)
                    predictions = np.argmax(probs, axis=1)
                    batch_acc = float(np.mean(predictions == batch_y))
                    top5_hits = np.any(np.argsort(probs, axis=1)[:, -5:] == batch_y[:, None], axis=1)
                    top5_acc = float(np.mean(top5_hits))

                    grads = self._compute_gradients(batch_x, grad_logits, kernel)
                    if batch_acc == 0.0 and zero_acc_logs < 3:
                        label_hist = np.bincount(batch_y, minlength=num_classes).tolist()
                        pred_hist = np.bincount(predictions, minlength=num_classes).tolist()
                        logits_mean = float(np.mean(logits)) if logits.size else 0.0
                        logits_std = float(np.std(logits)) if logits.size else 0.0
                        true_prob = float(np.mean(probs[np.arange(batch_y.shape[0]), batch_y])) if batch_y.size else 0.0
                        top_pred_prob = float(np.mean(np.max(probs, axis=1))) if probs.size else 0.0
                        grad_norm = float(np.linalg.norm(grads["W_basal"]))
                        msg = (
                            f"epoch={epoch} step={step + 1} 检测到 acc=0.0："
                            f"label_hist={label_hist} pred_hist={pred_hist} "
                            f"logits_mean={logits_mean:.4f} logits_std={logits_std:.4f} "
                            f"true_prob_mean={true_prob:.4f} top_prob_mean={top_pred_prob:.4f} "
                            f"grad_norm={grad_norm:.4f}"
                        )
                        await self._emit_log("DEBUG", msg)
                        zero_acc_logs += 1

                    self._apply_updates(model, grads, lr, weight_decay)

                    step_duration = max(time.perf_counter() - step_start, 1e-6)
                    throughput = float(batch_x.shape[0] / step_duration)
                    batch_throughputs.append(throughput)

                    ema_loss = loss if ema_loss is None else ema_decay * ema_loss + (1 - ema_decay) * loss
                    ema_acc = batch_acc if ema_acc is None else ema_decay * ema_acc + (1 - ema_decay) * batch_acc

                    residual_value, spike_payload = self._build_iteration_events(
                        model,
                        batch_x,
                        batch_y,
                        basal_currents,
                        epoch,
                        step,
                        layers_count,
                        timesteps,
                        fp_config,
                    )

                    metrics_payload = {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss,
                        "acc": batch_acc,
                        "top5": top5_acc,
                        "throughput": throughput,
                        "step_ms": step_duration * 1000.0,
                        "ema_loss": ema_loss,
                        "ema_acc": ema_acc,
                        "lr": lr,
                        "examples": int(batch_x.shape[0]),
                        "time_unix": _current_millis(),
                    }
                    iter_payload = {
                        "epoch": epoch,
                        "step": step + 1,
                        "k": iterations,
                        "layer": step % layers_count,
                        "residual": residual_value,
                        "time_unix": _current_millis(),
                    }

                    await self._broker.publish("train_iter", iter_payload)
                    await self._broker.publish("metrics_batch", metrics_payload)
                    if spike_payload is not None:
                        await self._broker.publish("spike", spike_payload)

                    if step % 10 == 0:
                        await asyncio.sleep(0)

                val_loss, val_acc, val_top5 = self._evaluate(model, dataset.val_x, dataset.val_y, kernel, batch_size=512)
                best_acc = max(best_acc, val_acc)
                best_loss = min(best_loss, val_loss)
                epoch_duration = time.perf_counter() - epoch_start
                avg_throughput = float(np.mean(batch_throughputs)) if batch_throughputs else None

                epoch_payload = {
                    "epoch": epoch,
                    "loss": val_loss,
                    "acc": val_acc,
                    "best_acc": best_acc,
                    "best_loss": best_loss,
                    "avg_throughput": avg_throughput,
                    "epoch_sec": epoch_duration,
                    "top5": val_top5,
                    "time_unix": _current_millis(),
                }
                await self._broker.publish("metrics_epoch", epoch_payload)

                await self._broker.publish(
                    "log",
                    {
                        "level": "INFO",
                        "msg": (
                            f"[EPOCH {epoch}/{epochs}] loss={loss:.4f} "
                            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                        ),
                        "time_unix": _current_millis(),
                    },
                )

                if self._stop_event.is_set():
                    await self._broker.publish("train_status", {"status": "Stopped", "time_unix": _current_millis()})
                    return

            await self._broker.publish(
                "log",
                {"level": "INFO", "msg": "训练完成", "time_unix": _current_millis()},
            )
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
        key = (timesteps, iterations, tolerance, dt)
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
            raise FileNotFoundError(f"未找到 MNIST 数据文件：{npz_path}")
        with np.load(npz_path) as data:
            train_x = data["x_train"].astype(np.float32) / 255.0
            train_y = data["y_train"].astype(np.int64)
            test_x = data["x_test"].astype(np.float32) / 255.0
            test_y = data["y_test"].astype(np.int64)
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)
        return DatasetBundle(train_x, train_y, test_x, test_y, train_x.shape[1], num_classes)

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
        num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)
        return DatasetBundle(train_x, train_y, test_x, test_y, train_x.shape[1], num_classes)

    def _load_cifar10(self) -> DatasetBundle:
        dataset_dir = self._data_root / "cifar10"
        tar_path = dataset_dir / "cifar-10-python.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(f"未找到 CIFAR10 数据集压缩包：{tar_path}")
        train_x, train_y, test_x, test_y = _load_cifar_arrays(dataset_dir, tar_path)
        train_x = (train_x / 255.0).astype(np.float32)
        test_x = (test_x / 255.0).astype(np.float32)
        num_classes = int(np.max(np.concatenate([train_y, test_y])) + 1)
        return DatasetBundle(train_x, train_y, test_x, test_y, train_x.shape[1], num_classes)

    def _init_model(
        self,
        input_dim: int,
        timesteps: int,
        num_classes: int,
        rng: np.random.Generator,
    ) -> ModelState:
        scale = math.sqrt(2.0 / max(1, input_dim))
        W_basal = rng.normal(0.0, scale, size=(num_classes, timesteps, input_dim)).astype(np.float32)
        b_basal = np.zeros((num_classes, timesteps), dtype=np.float32)
        b_out = np.zeros(num_classes, dtype=np.float32)
        return ModelState(W_basal=W_basal, b_basal=b_basal, b_out=b_out)

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

    def _cross_entropy(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits_stable).astype(np.float32, copy=False)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        eps = 1e-9
        loss = float(-np.log(np.clip(probs[np.arange(labels.shape[0]), labels], eps, 1.0)).mean())
        grad_logits = probs
        grad_logits[np.arange(labels.shape[0]), labels] -= 1.0
        grad_logits /= labels.shape[0]
        return loss, probs, grad_logits

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
    ) -> None:
        if weight_decay > 0:
            model.W_basal *= (1.0 - lr * weight_decay)
        model.W_basal -= lr * grads["W_basal"]
        model.b_basal -= lr * grads["b_basal"]
        model.b_out -= lr * grads["b_out"]

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
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        if inputs.shape[0] == 0:
            return 0.0, None
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
        return float(residual), spike_payload

    def _evaluate(
        self,
        model: ModelState,
        inputs: np.ndarray,
        labels: np.ndarray,
        kernel: np.ndarray,
        batch_size: int = 512,
    ) -> Tuple[float, float, float]:
        total = inputs.shape[0]
        losses: List[float] = []
        accs: List[float] = []
        top5_list: List[float] = []
        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_x = inputs[start:end]
            batch_y = labels[start:end]
            logits, _ = self._forward_batch(model, batch_x, kernel)
            loss, probs, _ = self._cross_entropy(logits, batch_y)
            pred = np.argmax(probs, axis=1)
            losses.append(loss)
            accs.append(float(np.mean(pred == batch_y)))
            top5_hits = np.any(np.argsort(probs, axis=1)[:, -5:] == batch_y[:, None], axis=1)
            top5_list.append(float(np.mean(top5_hits)))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_acc = float(np.mean(accs)) if accs else 0.0
        avg_top5 = float(np.mean(top5_list)) if top5_list else 0.0
        return avg_loss, avg_acc, avg_top5


def _build_broker(message_queue_config: Optional[Dict[str, Any]]) -> Tuple[EventBroker, MessageQueue]:
    mq: MessageQueue = build_message_queue(message_queue_config)
    broker = EventBroker(mq)
    return broker, mq


def create_app(message_queue_config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """构建 FastAPI 应用，可通过参数覆盖消息队列配置。"""

    mq_config = message_queue_config or get_message_queue_config()
    broker, queue = _build_broker(mq_config)
    dataset_service = DatasetService(broker)
    training_service = TrainingService(broker)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
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

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        return {"training": training_service.get_config(), "status": training_service.status}

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
