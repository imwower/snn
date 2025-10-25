import math
import unittest

import numpy as np

from snn.server import _clip_gradients_inplace, _cosine_with_warmup


class SchedulerAndGradientTests(unittest.TestCase):
    def test_cosine_with_warmup_progression(self) -> None:
        base_lr = 1e-3
        warmup = 5
        total_steps = 20
        min_lr = base_lr * 0.1
        warmup_values = [
            _cosine_with_warmup(step, base_lr, warmup, total_steps, min_lr) for step in range(warmup + 1)
        ]
        self.assertAlmostEqual(warmup_values[0], 0.0)
        self.assertAlmostEqual(warmup_values[-1], base_lr, places=9)
        post_warmup = _cosine_with_warmup(15, base_lr, warmup, total_steps, min_lr)
        self.assertGreater(post_warmup, min_lr)
        self.assertLess(post_warmup, base_lr)
        self.assertAlmostEqual(
            _cosine_with_warmup(total_steps, base_lr, warmup, total_steps, min_lr), min_lr, places=9
        )

    def test_clip_gradients_inplace_scales_extra_buffers(self) -> None:
        grads = {"w": np.array([3.0, 4.0], dtype=np.float32)}
        extra = {"b": np.array([12.0], dtype=np.float32)}
        original_norm = math.sqrt(3**2 + 4**2 + 12**2)
        reported_norm = _clip_gradients_inplace(grads, max_norm=5.0, extra=extra)
        self.assertAlmostEqual(reported_norm, original_norm)
        combined = np.concatenate([grads["w"], extra["b"]])
        self.assertLessEqual(np.linalg.norm(combined), 5.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
