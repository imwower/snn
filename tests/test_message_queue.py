import logging
import time
import unittest
from unittest import mock

from snn import InMemoryQueue, build_message_queue
from snn.mq import NATS_AVAILABLE


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

        cfg = {
            "backend": "nats",
            "nats": {
                "servers": ["nats://127.0.0.1:4222"],
                "stream": "snn_events_test",
                "subject": "training.events.test",
                "durable": "snn-test-worker",
                "connect": True,
                "timeout": 1.0,
                "allow_reconnect": False,
            },
        }

        queue = None
        try:
            queue = build_message_queue(cfg)
        except Exception as exc:
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


if __name__ == "__main__":
    unittest.main()
