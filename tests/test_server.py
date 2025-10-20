import asyncio
import unittest
from typing import List, Tuple

from httpx import ASGITransport, AsyncClient

from snn.server import create_app


class ServerInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        # 使用内存消息队列以便在测试环境下运行。
        self.app = create_app({"backend": "memory"})

    def test_dataset_and_training_flow(self) -> None:
        asyncio.run(self._async_dataset_and_training_flow())

    async def _async_dataset_and_training_flow(self) -> None:
        request_transport = ASGITransport(app=self.app)
        async with AsyncClient(transport=request_transport, base_url="http://testserver") as request_client:
            resp = await request_client.get("/api/datasets")
            self.assertEqual(resp.status_code, 200)
            datasets = resp.json()
            self.assertIn("datasets", datasets)
            self.assertTrue(any(entry["name"] == "MNIST" for entry in datasets["datasets"]))

            training_payload = {
                "dataset": "MNIST",
                "mode": "fpt",
                "network_size": 512,
                "layers": 3,
                "lr": 1e-3,
                "K": 4,
                "tol": 1e-5,
                "T": None,
                "epochs": 1,
            }

            resp = await request_client.post("/api/datasets/download", json={"name": "MNIST"})
            self.assertEqual(resp.status_code, 202)

            resp = await request_client.post("/api/train/init", json=training_payload)
            self.assertEqual(resp.status_code, 200)

            resp = await request_client.post("/api/train/start", json={})
            self.assertEqual(resp.status_code, 202)

            dataset_service = self.app.state.dataset_service
            training_service = self.app.state.training_service
            broker = self.app.state.broker

            if dataset_service._download_task is not None:
                await asyncio.wait_for(dataset_service._download_task, timeout=5.0)
            if training_service._task is not None:
                await asyncio.wait_for(training_service._task, timeout=5.0)

            events: List[Tuple[str, dict]] = await broker.history()
            event_names = [name for name, _ in events]
            self.assertIn("dataset_download", event_names)
            self.assertIn("train_init", event_names)
            self.assertIn("metrics_batch", event_names)
            self.assertIn("metrics_epoch", event_names)
            status_updates = [payload.get("status") for name, payload in events if name == "train_status"]
            self.assertIn("Training", status_updates)
            self.assertIn("Idle", status_updates)

            resp = await request_client.get("/api/config")
            self.assertEqual(resp.status_code, 200)
            config_payload = resp.json()
            self.assertEqual(config_payload.get("status"), "Idle")
            self.assertEqual(config_payload["training"]["dataset"], "MNIST")


if __name__ == "__main__":
    unittest.main()
