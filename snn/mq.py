"""事件驱动训练使用的消息队列抽象与工厂函数。"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Coroutine, Dict, Iterable, List, Optional

from .config import get_message_queue_config

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 可选依赖，测试环境可能缺失
    from nats.aio.client import Client as NATS  # type: ignore
    from nats.errors import Error as NatsError  # type: ignore
    from nats.errors import NoRespondersError  # type: ignore
    from nats.js.errors import (
        NoStreamResponseError as JetStreamNoStreamResponseError,  # type: ignore
        ServiceUnavailableError as JetStreamServiceUnavailableError,  # type: ignore
    )

    NATS_AVAILABLE = True
except Exception:  # pragma: no cover - 未安装 nats-py
    NATS = None  # type: ignore
    NatsError = Exception  # type: ignore
    NoRespondersError = Exception  # type: ignore
    JetStreamServiceUnavailableError = Exception  # type: ignore
    JetStreamNoStreamResponseError = Exception  # type: ignore
    NATS_AVAILABLE = False


@dataclass(frozen=True)
class Message:
    """统一封装队列中的事件消息。"""

    subject: str
    data: bytes
    headers: Dict[str, str] = field(default_factory=dict)


class MessageQueue(ABC):
    """消息队列抽象基类。"""

    @abstractmethod
    def publish(self, subject: str, data: bytes, *, headers: Optional[Dict[str, str]] = None) -> None:
        """发布消息。"""

    @abstractmethod
    def pull(self, subject: str, *, max_messages: int = 1) -> List[Message]:
        """拉取指定主题的消息。"""

    @abstractmethod
    def close(self) -> None:
        """释放底层资源。"""


class InMemoryQueue(MessageQueue):
    """基于内存列表的轻量实现，主要用于测试或离线模式。"""

    def __init__(self) -> None:
        self._messages: Dict[str, List[Message]] = {}
        self._closed = False

    def publish(self, subject: str, data: bytes, *, headers: Optional[Dict[str, str]] = None) -> None:
        if self._closed:
            raise RuntimeError("消息队列已关闭，无法发布消息")
        bucket = self._messages.setdefault(subject, [])
        bucket.append(Message(subject=subject, data=data, headers=headers or {}))
        logger.debug("内存队列已写入 subject=%s, size=%d", subject, len(bucket))

    def pull(self, subject: str, *, max_messages: int = 1) -> List[Message]:
        if self._closed:
            return []
        if max_messages <= 0:
            raise ValueError("max_messages 必须为正整数")
        bucket = self._messages.get(subject, [])
        if not bucket:
            return []
        count = min(max_messages, len(bucket))
        messages = bucket[:count]
        del bucket[:count]
        if not bucket:
            self._messages.pop(subject, None)
        logger.debug("内存队列读取 subject=%s, count=%d", subject, len(messages))
        return messages

    def close(self) -> None:
        self._closed = True
        self._messages.clear()


class NatsJetStreamQueue(MessageQueue):
    """基于 NATS JetStream 的消息队列封装。"""

    def __init__(
        self,
        *,
        servers: Iterable[str],
        stream: str,
        subject: str,
        durable: Optional[str] = None,
        connect: bool = True,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        request_timeout: float = 2.0,
        allow_reconnect: bool = True,
    ) -> None:
        if not NATS_AVAILABLE:
            raise RuntimeError("未安装 nats-py，无法创建 NatsJetStreamQueue")

        self._servers = list(servers)
        if not self._servers:
            raise ValueError("需要至少配置一个 NATS 服务器地址")
        self._stream = stream
        self._subject = subject
        self._subjects = self._compute_subjects(subject)
        self._durable = durable
        self._request_timeout = request_timeout
        self._allow_reconnect = allow_reconnect

        self._loop = loop or asyncio.new_event_loop()
        self._owns_loop = loop is None
        self._loop_thread: Optional[threading.Thread] = None
        if self._owns_loop:
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()
        self._nats = NATS()
        self._jetstream = None
        self._connected = False

        if connect:
            self._ensure_connected()

    def _ensure_connected(self) -> None:
        if self._connected:
            return

        async def _connect() -> None:
            await self._nats.connect(
                servers=self._servers,
                connect_timeout=self._request_timeout,
                allow_reconnect=self._allow_reconnect,
                max_reconnect_attempts=1 if not self._allow_reconnect else 60,
            )
            self._jetstream = self._nats.jetstream()
            try:
                await self._jetstream.add_stream(name=self._stream, subjects=self._subjects)
            except NatsError:
                pass
            self._connected = True

        self._submit(_connect())
        logger.info("已连接 NATS JetStream：stream=%s, subject=%s", self._stream, self._subject)

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.result(timeout=self._request_timeout)

    @staticmethod
    def _compute_subjects(subject: str) -> List[str]:
        subjects = [subject]
        if "*" not in subject and ">" not in subject:
            subjects.append(f"{subject}.>")
        return list(dict.fromkeys(subjects))

    def _should_recover_stream(self, exc: Exception) -> bool:
        if isinstance(exc, JetStreamServiceUnavailableError):
            message = str(exc).lower()
            return "error opening msg block file" in message
        if isinstance(exc, (JetStreamNoStreamResponseError, NoRespondersError)):
            return True
        return False

    async def _recreate_stream(self) -> None:
        assert self._jetstream is not None
        try:
            subjects = self._subjects
            try:
                info = await self._jetstream.stream_info(self._stream)  # type: ignore[attr-defined]
                configured = getattr(getattr(info, "config", None), "subjects", None)
                if configured:
                    subjects = list(configured)
            except Exception:
                subjects = self._subjects
            await self._jetstream.delete_stream(self._stream)
        except Exception as exc:  # pragma: no cover - 仅记录日志
            logger.warning("删除 JetStream 流 %s 失败：%s", self._stream, exc)
            subjects = self._subjects
        await self._jetstream.add_stream(name=self._stream, subjects=subjects)
        logger.warning("JetStream 流 %s 已重建，主题 %s", self._stream, self._subject)

    def publish(self, subject: str, data: bytes, *, headers: Optional[Dict[str, str]] = None) -> None:
        self._ensure_connected()
        assert self._jetstream is not None

        async def _publish() -> None:
            attempts = 0
            while True:
                try:
                    await self._jetstream.publish(subject, payload=data, headers=headers or {})
                    return
                except Exception as exc:
                    attempts += 1
                    if attempts > 1 or not self._should_recover_stream(exc):
                        raise
                    logger.warning("检测到 JetStream 存储异常，尝试重建流 %s：%s", self._stream, exc)
                    await self._recreate_stream()

        self._submit(_publish())

    def pull(self, subject: str, *, max_messages: int = 1) -> List[Message]:
        self._ensure_connected()
        assert self._jetstream is not None

        async def _pull() -> List[Message]:
            attempts = 0
            while True:
                try:
                    consumer = await self._jetstream.pull_subscribe(
                        subject=subject,
                        durable=self._durable or f"{self._stream}_worker",
                        stream=self._stream,
                    )
                    msgs = await consumer.fetch(batch=max_messages, timeout=self._request_timeout)
                    result: List[Message] = []
                    for msg in msgs:
                        result.append(Message(subject=msg.subject, data=bytes(msg.data), headers=dict(msg.headers or {})))
                        await msg.ack()
                    return result
                except Exception as exc:
                    attempts += 1
                    if attempts > 1 or not self._should_recover_stream(exc):
                        raise
                    logger.warning("拉取 JetStream 消息时检测到存储异常，尝试重建流 %s：%s", self._stream, exc)
                    await self._recreate_stream()

        future = asyncio.run_coroutine_threadsafe(_pull(), self._loop)
        return future.result(timeout=self._request_timeout)

    def close(self) -> None:
        if not self._connected:
            return

        async def _close() -> None:
            await self._nats.drain()

        self._submit(_close())
        self._connected = False
        if self._owns_loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=self._request_timeout)


def build_message_queue(config: Optional[Dict[str, object]] = None) -> MessageQueue:
    """根据配置构建消息队列实例，默认使用 NATS JetStream。"""

    mq_config = config or get_message_queue_config()
    backend = str(mq_config.get("backend", "nats")).lower()

    if backend == "memory":
        return InMemoryQueue()

    if backend == "nats":
        if not NATS_AVAILABLE:
            message = "未检测到 nats-py 依赖，无法构建 NATS JetStream 队列。请执行 `pip install nats-py`。"
            logger.error(message)
            raise RuntimeError(message)

        nats_cfg = mq_config.get("nats") if isinstance(mq_config.get("nats"), dict) else {}
        servers = nats_cfg.get("servers") or ["nats://127.0.0.1:4222"]
        stream = nats_cfg.get("stream") or "snn_events"
        subject = nats_cfg.get("subject") or "training.events"
        durable = nats_cfg.get("durable")
        connect = bool(nats_cfg.get("connect", True))
        timeout = float(nats_cfg.get("timeout", 2.0))
        allow_reconnect = bool(nats_cfg.get("allow_reconnect", True))

        try:
            queue = NatsJetStreamQueue(
                servers=servers,
                stream=stream,
                subject=subject,
                durable=durable,
                connect=connect,
                request_timeout=timeout,
                allow_reconnect=allow_reconnect,
            )
            logger.info("已创建 NATS JetStream 队列：servers=%s, stream=%s, subject=%s", servers, stream, subject)
            return queue
        except Exception as exc:  # pragma: no cover - 网络或配置错误
            logger.error("创建 NATS JetStream 队列失败：%s", exc)
            raise

    raise ValueError(f"不支持的消息队列类型：{backend}")


__all__ = [
    "Message",
    "MessageQueue",
    "InMemoryQueue",
    "NatsJetStreamQueue",
    "build_message_queue",
    "JetStreamServiceUnavailableError",
    "JetStreamNoStreamResponseError",
    "NoRespondersError",
]
