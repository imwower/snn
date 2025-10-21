"""FastAPI 应用：提供与前端约定的 REST API 与 SSE 事件流。"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_message_queue_config
from .mq import MessageQueue, build_message_queue, InMemoryQueue

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
    progress: float = 100.0


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
            "MNIST": DatasetRecord(name="MNIST", installed=True),
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
                "progress": record.progress,
                "message": record.message,
            }
            for record in self._records.values()
        ]
        return {"datasets": payload}

    async def start_download(self, name: str) -> None:
        async with self._lock:
            if self._download_task and not self._download_task.done():
                raise HTTPException(status.HTTP_409_CONFLICT, "已有数据集正在下载")
            record = self._records.setdefault(name, DatasetRecord(name=name, installed=False, progress=0.0))
            record.progress = 0.0
            record.installed = False
            record.message = None
            self._download_task = asyncio.create_task(self._run_download(record))

    async def _run_download(self, record: DatasetRecord) -> None:
        await self._broker.publish(
            "dataset_download",
            {"name": record.name, "state": "start", "progress": 0.0, "time_unix": _current_millis()},
        )
        try:
            for progress in [0.25, 0.5, 0.75, 1.0]:
                await asyncio.sleep(0.1)
                record.progress = round(progress * 100.0, 1)
                event_payload = {
                    "name": record.name,
                    "state": "progress" if progress < 1.0 else "complete",
                    "progress": record.progress,
                    "time_unix": _current_millis(),
                }
                if progress >= 1.0:
                    record.installed = True
                    self._materialize_dataset(record)
                await self._broker.publish("dataset_download", event_payload)
        finally:
            async with self._lock:
                self._download_task = None
        if record.progress < 100.0:
            await self._broker.publish(
                "dataset_download",
                {
                    "name": record.name,
                    "state": "complete",
                    "progress": record.progress,
                    "time_unix": _current_millis(),
                },
            )

    def _dataset_path(self, name: str) -> Path:
        safe = "".join((ch.lower() if ch.isalnum() or ch in {"-", "_"} else "_") for ch in name.strip())
        safe = safe or "dataset"
        return self._data_root / safe

    def _materialize_dataset(self, record: DatasetRecord) -> None:
        dataset_dir = self._dataset_path(record.name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": record.name,
            "downloaded_at": time.time(),
            "description": f"Placeholder dataset for {record.name}",
        }
        metadata_path = dataset_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        sample_path = dataset_dir / "sample.txt"
        if not sample_path.exists():
            sample_path.write_text(
                "This file represents downloaded data for dataset "
                f"{record.name}. Replace with actual dataset contents as needed.\n",
                encoding="utf-8",
            )
        record.installed = True
        record.progress = 100.0

    def _refresh_installation(self) -> None:
        for record in self._records.values():
            dataset_dir = self._dataset_path(record.name)
            if dataset_dir.exists():
                record.installed = True
                record.progress = 100.0
                record.message = None
            else:
                record.installed = False
                record.progress = 0.0


class TrainingService:
    """负责模拟训练流程并向 SSE 推送指标。"""

    def __init__(self, broker: EventBroker) -> None:
        self._broker = broker
        self._rng = random.Random(42)
        self._lock = asyncio.Lock()
        self._status: str = "Idle"
        self._config: Dict[str, Any] = {
            "dataset": "MNIST",
            "mode": "fpt",
            "network_size": 1024,
            "layers": 3,
            "lr": 1e-3,
            "K": 4,
            "tol": 1e-5,
            "T": None,
            "epochs": 3,
        }
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    @property
    def status(self) -> str:
        return self._status

    def get_config(self) -> Dict[str, Any]:
        return dict(self._config)

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
                "batch_size": payload.network_size // max(payload.layers, 1),
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
        epochs = config.get("epochs", 1)
        layers = config.get("layers", 1)
        iterations = max(1, config.get("K", 1))

        try:
            for epoch in range(1, epochs + 1):
                for step in range(1, 4):
                    if self._stop_event.is_set():
                        await self._broker.publish(
                            "log",
                            {
                                "level": "WARNING",
                                "msg": f"训练在 epoch={epoch} step={step} 被停止",
                                "time_unix": _current_millis(),
                            },
                        )
                        return
                    await asyncio.sleep(0.05)
                    residual = self._rng.random() * config["tol"] * iterations
                    metric_payload = {
                        "epoch": epoch,
                        "step": step,
                        "loss": max(0.05, self._rng.random()),
                        "acc": min(0.99, 0.5 + self._rng.random() * 0.5),
                        "throughput": 2500 + self._rng.random() * 500,
                        "lr": config["lr"],
                        "examples": config["network_size"],
                        "time_unix": _current_millis(),
                    }
                    iter_payload = {
                        "epoch": epoch,
                        "step": step,
                        "k": iterations,
                        "layer": self._rng.randrange(layers),
                        "residual": residual,
                        "time_unix": _current_millis(),
                    }
                    spike_payload = {
                        "layer": self._rng.randrange(layers),
                        "t": step,
                        "neurons": [self._rng.randrange(config["network_size"]) for _ in range(3)],
                        "power": self._rng.random(),
                    }
                    await self._broker.publish("train_iter", iter_payload)
                    await self._broker.publish("metrics_batch", metric_payload)
                    await self._broker.publish("spike", spike_payload)

                await asyncio.sleep(0.05)
                epoch_payload = {
                    "epoch": epoch,
                    "loss": max(0.05, self._rng.random()),
                    "acc": min(0.99, 0.5 + self._rng.random() * 0.5),
                    "best_acc": min(0.99, 0.7 + self._rng.random() * 0.25),
                    "best_loss": max(0.02, self._rng.random() * 0.2),
                    "avg_throughput": 2600 + self._rng.random() * 300,
                    "epoch_sec": 0.2 + self._rng.random() * 0.3,
                    "time_unix": _current_millis(),
                }
                await self._broker.publish("metrics_epoch", epoch_payload)

            await self._broker.publish(
                "log",
                {"level": "INFO", "msg": "训练完成", "time_unix": _current_millis()},
            )
        finally:
            async with self._lock:
                self._status = "Idle"
            await self._broker.publish("train_status", {"status": "Idle", "time_unix": _current_millis()})


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
