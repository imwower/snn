"""固定点并行训练（Fixed-point Parallel Training, FPT）求解器。

该模块实现 README 中描述的并行固定点迭代思路：对整段时间序列的膜电位进行迭代更新，
通过有限次迭代逼近顺序积分的结果。实现采用纯 Python 列表运算，可在 CPU 上并行更新
所有时间步，从而将复杂度从 O(T) 近似压缩到 O(K)，其中 K 为迭代次数。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from .neuron import CompartmentState, ThreeCompartmentParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixedPointConfig:
    """固定点迭代配置。"""

    iterations: int = 3
    tolerance: float = 1e-6
    damping: float = 1.0
    solver: str = "plain"
    anderson_m: int = 4
    anderson_beta: float = 0.5


@dataclass
class FixedPointResult:
    """固定点迭代结果，包含状态序列与残差列表。"""

    states: List[CompartmentState]
    residuals: List[float]


def _to_list(seq: Sequence[float], *, name: str) -> List[float]:
    """将序列转换为浮点列表并校验长度。"""

    if not isinstance(seq, Sequence):
        raise ValueError(f"{name} 必须是序列类型")
    return [float(value) for value in seq]


def _initial_vector(length: int, value: float) -> List[float]:
    """构造长度一致的初始向量。"""

    return [float(value) for _ in range(length)]


def _apply_damped_update(prev: np.ndarray, candidate: np.ndarray, damping: float) -> np.ndarray:
    return (1.0 - damping) * prev + damping * candidate


def _anderson_mix(
    prev: np.ndarray,
    state_history: Sequence[np.ndarray],
    residual_history: Sequence[np.ndarray],
    beta: float,
    damping: float,
) -> np.ndarray:
    if len(state_history) < 2:
        return _apply_damped_update(prev, state_history[-1], damping)
    stacked_res = np.stack(residual_history, axis=1)  # (dim, m)
    m = stacked_res.shape[1]
    gram = stacked_res.T @ stacked_res
    ones = np.ones((m, 1), dtype=np.float64)
    system = np.block([[gram, ones], [ones.T, np.zeros((1, 1), dtype=np.float64)]])
    rhs = np.zeros(m + 1, dtype=np.float64)
    rhs[-1] = 1.0
    try:
        solution = np.linalg.solve(system + 1e-10 * np.eye(m + 1, dtype=np.float64), rhs)
    except np.linalg.LinAlgError:
        return _apply_damped_update(prev, state_history[-1], damping)
    coeffs = solution[:m]
    if not np.all(np.isfinite(coeffs)):
        return _apply_damped_update(prev, state_history[-1], damping)
    mixed = np.zeros_like(prev)
    for weight, state in zip(coeffs, state_history):
        mixed += weight * state
    beta_clamped = float(np.clip(beta, 0.0, 1.0))
    return (1.0 - beta_clamped) * prev + beta_clamped * mixed


def fixed_point_parallel_solve(
    params: ThreeCompartmentParams,
    apical_currents: Sequence[float],
    basal_currents: Sequence[float],
    *,
    soma_currents: Optional[Sequence[float]] = None,
    config: FixedPointConfig = FixedPointConfig(),
    initial_state: Optional[Tuple[float, float, float]] = None,
) -> FixedPointResult:
    """执行固定点并行迭代，返回膜电位序列。

    Args:
        params: 三腔室神经元参数。
        apical_currents: 顶端树突输入电流序列，长度为 T。
        basal_currents: 基底树突输入电流序列，长度为 T。
        soma_currents: 可选的胞体输入电流，默认全零。
        config: 固定点迭代配置，控制迭代次数、残差阈值与阻尼系数。
        initial_state: 可选的初始膜电位 (soma, apical, basal)，默认使用静息电位。

    Returns:
        FixedPointResult，包含迭代后的 `CompartmentState` 列表与残差序列。
    """

    if config.iterations <= 0:
        raise ValueError("iterations 必须为正整数")
    if not (0.0 < config.damping <= 1.0):
        raise ValueError("damping 需位于 (0, 1] 区间")

    apical = _to_list(apical_currents, name="apical_currents")
    basal = _to_list(basal_currents, name="basal_currents")
    if len(apical) != len(basal):
        raise ValueError("apical_currents 与 basal_currents 的长度必须一致")

    if soma_currents is None:
        soma_input = [0.0] * len(apical)
    else:
        soma_input = _to_list(soma_currents, name="soma_currents")
        if len(soma_input) != len(apical):
            raise ValueError("soma_currents 的长度需要匹配树突输入序列")

    steps = len(apical)
    if steps == 0:
        return FixedPointResult(states=[], residuals=[])

    init_soma, init_apical, init_basal = (
        initial_state
        if initial_state is not None
        else (params.v_rest, params.v_rest, params.v_rest)
    )

    soma = _initial_vector(steps, init_soma)
    apical_state = _initial_vector(steps, init_apical)
    basal_state = _initial_vector(steps, init_basal)

    residuals: List[float] = []

    dt = params.dt
    damping = config.damping
    solver_name = (config.solver or "plain").lower()
    use_anderson = solver_name == "anderson"
    history_states: Deque[np.ndarray] = deque(maxlen=max(1, config.anderson_m))
    history_residuals: Deque[np.ndarray] = deque(maxlen=max(1, config.anderson_m))
    beta_mix = float(np.clip(config.anderson_beta, 0.0, 1.0))

    logger.info(
        "固定点迭代开始：步数=%d, 迭代次数=%d, 阈值=%.2e, 阻尼=%.2f",
        steps,
        config.iterations,
        config.tolerance,
        damping,
    )

    for iteration in range(1, config.iterations + 1):
        prev_soma = soma[:]
        prev_apical = apical_state[:]
        prev_basal = basal_state[:]

        soma_tm1 = [init_soma] + prev_soma[:-1]
        apical_tm1 = [init_apical] + prev_apical[:-1]
        basal_tm1 = [init_basal] + prev_basal[:-1]

        apical_candidate: List[float] = []
        basal_candidate: List[float] = []
        soma_candidate: List[float] = []

        for idx in range(steps):
            apical_leak = (params.v_rest - apical_tm1[idx]) / params.tau_apical
            apical_coupling = params.coupling_apical * (soma_tm1[idx] - apical_tm1[idx])
            apical_candidate.append(
                apical_tm1[idx] + dt * (apical_leak + apical_coupling + apical[idx])
            )

            basal_leak = (params.v_rest - basal_tm1[idx]) / params.tau_basal
            basal_coupling = params.coupling_basal * (soma_tm1[idx] - basal_tm1[idx])
            basal_candidate.append(
                basal_tm1[idx] + dt * (basal_leak + basal_coupling + basal[idx])
            )

        for idx in range(steps):
            soma_leak = (params.v_rest - soma_tm1[idx]) / params.tau_soma
            soma_coupling = (
                params.coupling_apical * (apical_candidate[idx] - soma_tm1[idx])
                + params.coupling_basal * (basal_candidate[idx] - soma_tm1[idx])
            )
            soma_candidate.append(
                soma_tm1[idx] + dt * (soma_leak + soma_coupling + soma_input[idx])
            )

        for idx in range(steps):
            apical_state[idx] = (1.0 - damping) * prev_apical[idx] + damping * apical_candidate[idx]
            basal_state[idx] = (1.0 - damping) * prev_basal[idx] + damping * basal_candidate[idx]

        prev_soma_arr = np.asarray(prev_soma, dtype=np.float64)
        soma_candidate_arr = np.asarray(soma_candidate, dtype=np.float64)
        if use_anderson:
            residual_vec = soma_candidate_arr - prev_soma_arr
            history_states.append(soma_candidate_arr.copy())
            history_residuals.append(residual_vec.copy())
            updated_soma_arr = _anderson_mix(
                prev_soma_arr,
                list(history_states),
                list(history_residuals),
                beta_mix,
                damping,
            )
        else:
            updated_soma_arr = _apply_damped_update(prev_soma_arr, soma_candidate_arr, damping)
        soma = updated_soma_arr.tolist()

        soma_diff = float(np.max(np.abs(updated_soma_arr - prev_soma_arr)))
        apical_diff = float(
            np.max(np.abs(np.asarray(apical_state, dtype=np.float64) - np.asarray(prev_apical, dtype=np.float64)))
        )
        basal_diff = float(
            np.max(np.abs(np.asarray(basal_state, dtype=np.float64) - np.asarray(prev_basal, dtype=np.float64)))
        )
        residual = max(soma_diff, apical_diff, basal_diff)
        residuals.append(residual)
        logger.info("迭代 %d/%d，残差=%.3e", iteration, config.iterations, residual)

        if residual <= config.tolerance:
            logger.info("残差 %.3e 已低于阈值 %.3e，提前停止迭代", residual, config.tolerance)
            break

    times = [dt * (idx + 1.0) for idx in range(steps)]
    spike_flags = [value >= params.threshold for value in soma]

    states = [
        CompartmentState(
            time=times[idx],
            soma=soma[idx],
            apical=apical_state[idx],
            basal=basal_state[idx],
            spike=spike_flags[idx],
        )
        for idx in range(steps)
    ]

    return FixedPointResult(states=states, residuals=residuals)
