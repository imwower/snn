import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
import logging
import threading
import time
import unittest
from unittest import mock
from typing import Dict, List

from snn import InMemoryQueue, build_message_queue
from snn.mq import (
    NATS_AVAILABLE,
    JetStreamServiceUnavailableError,
    JetStreamNoStreamResponseError,
    NatsJetStreamQueue,
    NoRespondersError,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.propagate = False


class InMemoryQueueTests(unittest.TestCase):
    def test_publish_and_pull(self) -> None:
        logger.info("开始测试：内存队列发布与拉取消息")
        queue = InMemoryQueue()
        queue.publish("metrics", b"payload-1")
        queue.publish("metrics", b"payload-2", headers={"k": "v"})

        messages = queue.pull("metrics", max_messages=5)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].data, b"payload-1")
        self.assertEqual(messages[1].headers["k"], "v")
        logger.info("内存队列返回消息数量=%d", len(messages))

        # 再次读取应为空。
        self.assertEqual(queue.pull("metrics", max_messages=1), [])

    def test_pull_with_invalid_batch(self) -> None:
        queue = InMemoryQueue()
        with self.assertRaises(ValueError):
            queue.pull("metrics", max_messages=0)


class MessageQueueFactoryTests(unittest.TestCase):
    def test_build_memory_backend(self) -> None:
        logger.info("开始测试：工厂应返回内存队列实现")
        queue = build_message_queue({"backend": "memory"})
        self.assertIsInstance(queue, InMemoryQueue)

    def test_build_nats_without_dependency_raises(self) -> None:
        logger.info("开始测试：缺少 nats 依赖时应抛出错误")
        with mock.patch("snn.mq.NATS_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                build_message_queue({"backend": "nats", "nats": {"stream": "s", "subject": "sub"}})

    def test_build_nats_with_dependency(self) -> None:
        logger.info("开始测试：具备 nats 依赖时应实例化 NatsJetStreamQueue")
        dummy_queue = object()
        with mock.patch("snn.mq.NATS_AVAILABLE", True), mock.patch(
            "snn.mq.NatsJetStreamQueue", return_value=dummy_queue
        ) as mocked_cls:
            queue = build_message_queue(
                {
                    "backend": "nats",
                    "nats": {
                        "servers": ["nats://example:4222"],
                        "stream": "streamA",
                        "subject": "events.test",
                        "durable": "worker",
                        "connect": False,
                        "timeout": 5,
                    },
                }
            )
            self.assertIs(queue, dummy_queue)
            mocked_cls.assert_called_once_with(
                servers=["nats://example:4222"],
                stream="streamA",
                subject="events.test",
                durable="worker",
                connect=False,
                request_timeout=5.0,
                allow_reconnect=True,
            )

    def test_nats_roundtrip_when_server_available(self) -> None:
        logger.info("开始测试：验证 NATS 服务器可用性并完成消息往返")
        if not NATS_AVAILABLE:
            self.skipTest("本地环境未安装 nats-py，跳过 NATS 集成测试")

        queue = None
        try:
            queue = build_message_queue({"backend": "nats", "nats": {"timeout": 1.0, "allow_reconnect": False}})
        except Exception as exc:
            should_skip = isinstance(exc, FuturesTimeoutError) or "Operation not permitted" in str(exc)
            if should_skip:
                self.skipTest(f"NATS 服务器不可用或连接失败：{exc}")
                return
            self.fail(f"NATS 服务器不可用或连接失败：{exc}")
            return

        messages = []
        try:
            payload = b"NATS-roundtrip"
            try:
                queue.publish("training.events.test", payload, headers={"integration": "true"})
            except Exception as exc:
                self.fail(f"NATS 发送消息失败：{exc}")
                return
            for _ in range(3):
                try:
                    messages = queue.pull("training.events.test", max_messages=1)
                except Exception as exc:
                    self.fail(f"NATS 拉取消息失败：{exc}")
                    return
                if messages:
                    break
                time.sleep(0.1)
            if not messages:
                self.fail("已连接 NATS 服务器但未读取到测试消息")
            self.assertEqual(messages[0].data, payload)
            self.assertEqual(messages[0].headers.get("integration"), "true")
            logger.info("NATS 集成测试成功，收到消息数量=%d", len(messages))
        finally:
            if queue is not None:
                queue.close()


class NatsJetStreamQueueRecoveryTests(unittest.TestCase):
    def _make_storage_error(self) -> JetStreamServiceUnavailableError:
        message = (
            "nats: ServiceUnavailableError: error opening msg block file "
            '["/data/jetstream/jetstream/$G/streams/snn_events/msgs/2.blk"]'
        )
        try:
            return JetStreamServiceUnavailableError(
                code=503,
                err_code=10077,
                description=message,
            )
        except TypeError:
            return JetStreamServiceUnavailableError(message)

    def _make_no_stream_error(self) -> JetStreamNoStreamResponseError:
        try:
            return JetStreamNoStreamResponseError()
        except TypeError:
            return JetStreamNoStreamResponseError("nats: no response from stream")

    def _make_no_responder_error(self) -> NoRespondersError:
        try:
            return NoRespondersError()
        except TypeError:
            return NoRespondersError("nats: no responders available for request")

    def _build_queue_with_loop(self) -> NatsJetStreamQueue:
        queue = NatsJetStreamQueue.__new__(NatsJetStreamQueue)
        queue._stream = "snn_events"
        queue._subject = "training.events"
        queue._subjects = ["training.events", "training.events.>"]
        queue._durable = None
        queue._connected = True
        queue._request_timeout = 0.1
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        def stop_loop() -> None:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()

        self.addCleanup(stop_loop)
        queue._loop = loop
        queue._owns_loop = True
        queue._loop_thread = thread

        def submit(coro) -> None:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result(timeout=queue._request_timeout)

        queue._submit = submit
        queue._ensure_connected = mock.Mock()
        return queue

    def test_publish_recovers_from_missing_block_error(self) -> None:
        logger.info("开始测试：JetStream publish 遇到缺失块文件时应重建流")
        queue = self._build_queue_with_loop()

        error_factory = self._make_storage_error

        class FakeJetStream:
            def __init__(self) -> None:
                self.publish_calls = 0
                self.delete_calls = 0
                self.add_calls = 0
                self.last_subjects: List[str] = []

            async def publish(self, subject: str, payload: bytes, headers: Dict[str, str]) -> None:
                self.publish_calls += 1
                if self.publish_calls == 1:
                    raise error_factory()

            async def delete_stream(self, name: str) -> None:
                self.delete_calls += 1

            async def add_stream(self, name: str, subjects: List[str]) -> None:
                self.add_calls += 1
                self.last_subjects = list(subjects)

        fake = FakeJetStream()
        queue._jetstream = fake

        queue.publish("training.events", b"payload", headers={"k": "v"})

        self.assertEqual(fake.publish_calls, 2)
        self.assertEqual(fake.delete_calls, 1)
        self.assertEqual(fake.add_calls, 1)
        self.assertIn("training.events.>", fake.last_subjects)

    def test_pull_recovers_from_missing_block_error(self) -> None:
        logger.info("开始测试：JetStream pull 遇到缺失块文件时应重建流")
        queue = self._build_queue_with_loop()

        error_factory = self._make_storage_error

        class FakeMessage:
            def __init__(self) -> None:
                self.subject = "training.events"
                self.data = b"payload"
                self.headers = {"h": "1"}
                self.acked = False

            async def ack(self) -> None:
                self.acked = True

        class FakeConsumer:
            async def fetch(self, batch: int, timeout: float) -> List[FakeMessage]:
                return [FakeMessage()]

        class FakeJetStream:
            def __init__(self) -> None:
                self.subscribe_calls = 0
                self.delete_calls = 0
                self.add_calls = 0
                self.last_subjects: List[str] = []

            async def pull_subscribe(self, subject: str, durable: str, stream: str) -> FakeConsumer:
                self.subscribe_calls += 1
                if self.subscribe_calls == 1:
                    raise error_factory()
                return FakeConsumer()

            async def delete_stream(self, name: str) -> None:
                self.delete_calls += 1

            async def add_stream(self, name: str, subjects: List[str]) -> None:
                self.add_calls += 1
                self.last_subjects = list(subjects)

        fake = FakeJetStream()
        queue._jetstream = fake

        messages = queue.pull("training.events", max_messages=1)

        self.assertEqual(len(messages), 1)
        self.assertEqual(fake.subscribe_calls, 2)
        self.assertEqual(fake.delete_calls, 1)
        self.assertEqual(fake.add_calls, 1)
        self.assertIn("training.events.>", fake.last_subjects)

    def test_publish_recovers_when_stream_missing(self) -> None:
        logger.info("开始测试：JetStream publish 遇到缺失流时应重建")
        queue = self._build_queue_with_loop()

        class FakeJetStream:
            def __init__(self, error_factory) -> None:
                self.error_factory = error_factory
                self.publish_calls = 0
                self.delete_calls = 0
                self.add_calls = 0
                self.last_subjects: List[str] = []

            async def publish(self, subject: str, payload: bytes, headers: Dict[str, str]) -> None:
                self.publish_calls += 1
                if self.publish_calls == 1:
                    raise self.error_factory()

            async def delete_stream(self, name: str) -> None:
                self.delete_calls += 1

            async def add_stream(self, name: str, subjects: List[str]) -> None:
                self.add_calls += 1
                self.last_subjects = list(subjects)

        for factory in (self._make_no_stream_error, self._make_no_responder_error):
            fake = FakeJetStream(factory)
            queue._jetstream = fake
            queue.publish("training.events", b"payload", headers={"k": "v"})
            self.assertEqual(fake.publish_calls, 2)
            self.assertEqual(fake.delete_calls, 1)
            self.assertEqual(fake.add_calls, 1)
            self.assertIn("training.events.>", fake.last_subjects)


if __name__ == "__main__":
    unittest.main()
