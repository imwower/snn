from __future__ import annotations

import math
from typing import List, Optional, Sequence


class WarmupCosineScheduler:
    """Warmup + cosine decay scheduler with optional restarts (SGDR-style)."""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        *,
        min_lr: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        restarts: Optional[Sequence[int]] = None,
    ) -> None:
        if total_steps <= 0 and not restarts:
            raise ValueError("total_steps must be positive when no restarts are provided")
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr if min_lr is not None else self.base_lr * 0.1)
        self.total_steps = max(1, int(total_steps))
        self.global_step = 0
        self.restart_lengths: List[int] = (
            [int(length) for length in restarts if int(length) > 0] if restarts else [self.total_steps]
        )
        self.restart_index = 0
        self.cycle_step = 0
        self._explicit_warmup = warmup_steps

    def _current_cycle_length(self) -> int:
        if self.restart_index < len(self.restart_lengths):
            return self.restart_lengths[self.restart_index]
        return self.restart_lengths[-1]

    def _current_warmup(self, cycle_length: int) -> int:
        if self._explicit_warmup is not None:
            return max(1, int(self._explicit_warmup))
        return max(1, int(math.ceil(0.05 * cycle_length)))

    def step(self) -> float:
        cycle_length = self._current_cycle_length()
        warmup_steps = min(self._current_warmup(cycle_length), cycle_length)
        if self.cycle_step < warmup_steps:
            progress = self.cycle_step / max(1, warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * progress
        else:
            denom = max(1, cycle_length - warmup_steps)
            progress = (self.cycle_step - warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        self._advance_cycle(cycle_length)
        self.global_step += 1
        return lr

    def _advance_cycle(self, cycle_length: int) -> None:
        self.cycle_step += 1
        if self.cycle_step >= cycle_length:
            self.cycle_step = 0
            if self.restart_index + 1 < len(self.restart_lengths):
                self.restart_index += 1
