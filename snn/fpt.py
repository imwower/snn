"""固定点并行训练（Fixed-point Parallel Training, FPT）求解器。

该模块实现 README 中描述的并行固定点迭代思路：对整段时间序列的膜电位进行迭代更新，
通过有限次迭代逼近顺序积分的结果。实现采用纯 Python 列表运算，可在 CPU 上并行更新
所有时间步，从而将复杂度从 O(T) 近似压缩到 O(K)，其中 K 为迭代次数。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Sequence, Tuple

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
    anderson_ridge: float = 1e-4
    line_search: bool = True
    fp_err_guard: float = 5.0


@dataclass
class FixedPointResult:
    """固定点迭代结果，包含状态序列与残差列表。"""

    states: List[CompartmentState]
    residuals: List[float]
    iter_errors: List[float] = field(default_factory=list)
    effective_iterations: int = 0
    final_fp_error: float = 0.0
    final_iter_error: float = 0.0
    solver_used: str = "anderson"


def _to_list(seq: Sequence[float], *, name: str) -> List[float]:
    """将序列转换为浮点列表并校验长度。"""

    if not isinstance(seq, Sequence):
        raise ValueError(f"{name} 必须是序列类型")
    return [float(value) for value in seq]


def _initial_vector(length: int, value: float) -> List[float]:
    """构造长度一致的初始向量。"""

    return [float(value) for _ in range(length)]


def _stack_state(soma: np.ndarray, apical: np.ndarray, basal: np.ndarray) -> np.ndarray:
    return np.stack([soma, apical, basal], axis=0)


def _split_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state.shape[0] != 3:
        raise ValueError("state 需要包含 [soma, apical, basal]")
    return state[0], state[1], state[2]


def _rms(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(flat * flat)))


def _phi_eval(phi: Callable[[np.ndarray], np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    phi_state = np.asarray(phi(state), dtype=np.float64)
    delta = phi_state - state
    fp_err = _rms(delta)
    return phi_state, delta, fp_err


def _anderson_ridge(delta_history: Sequence[np.ndarray], ridge: float) -> Optional[np.ndarray]:
    if len(delta_history) < 2:
        return None
    mat = np.stack(delta_history, axis=1)  # (dim, n)
    gram = mat.T @ mat
    n = gram.shape[0]
    lam = float(max(ridge, 0.0))
    ones = np.ones((n, 1), dtype=np.float64)
    system = np.block(
        [
            [gram + lam * np.eye(n, dtype=np.float64), ones],
            [ones.T, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[-1] = 1.0
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None
    coeffs = solution[:n]
    if not np.all(np.isfinite(coeffs)):
        return None
    return mat @ coeffs


def _backtracking_step(
    state: np.ndarray,
    direction: np.ndarray,
    phi: Callable[[np.ndarray], np.ndarray],
    reference_err: float,
    enabled: bool,
) -> Tuple[np.ndarray, float, float]:
    if not enabled:
        new_state = state + direction
        _, _, fp_err = _phi_eval(phi, new_state)
        return new_state, fp_err, 1.0

    for alpha in (1.0, 0.5, 0.25):
        candidate = state + alpha * direction
        _, _, fp_err = _phi_eval(phi, candidate)
        if reference_err <= 0.0 or fp_err <= 0.99 * reference_err:
            return candidate, fp_err, alpha
    return state, reference_err, 0.0


def fpt_solve(
    phi: Callable[[np.ndarray], np.ndarray],
    h0: np.ndarray,
    K_max: int,
    tol: float,
    *,
    solver: str = "anderson",
    m: int = 4,
    beta: float = 0.5,
    ridge: float = 1e-4,
    line_search: bool = True,
    fp_err_guard: float = 5.0,
    damping: float = 1.0,
    logger: Optional[logging.Logger] = None,
    trace: Optional[List[Tuple[int, float, float]]] = None,
    allow_fallback: bool = True,
) -> Tuple[np.ndarray, int, float, float, str]:
    """Generic fixed-point solver with optional Anderson acceleration."""

    if K_max <= 0:
        raise ValueError("K_max must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    state = np.asarray(h0, dtype=np.float64)
    solver_name = str(solver or "plain").lower()
    use_anderson = solver_name == "anderson" and m > 1
    history: Deque[np.ndarray] = deque(maxlen=max(1, int(m)))
    ridge_value = float(max(ridge, 0.0))
    guard = float(max(fp_err_guard, 0.0))
    damping = float(np.clip(damping, 1e-4, 1.0))
    beta_mix = float(np.clip(beta, 0.0, 1.0))
    fp_err_final = float("inf")
    iter_err_final = 0.0
    effective_k = 0
    solver_used = solver_name
    if logger:
        logger.info(
            "固定点迭代开始：dim=%d, 迭代次数=%d, 阈值=%.2e, solver=%s",
            int(state.size),
            K_max,
            tol,
            solver_name,
        )

    for iteration in range(1, K_max + 1):
        phi_state, delta, fp_err = _phi_eval(phi, state)
        if trace is not None:
            trace.append((iteration, fp_err, 0.0))
        if fp_err <= tol:
            fp_err_final = fp_err
            iter_err_final = 0.0
            effective_k = iteration - 1
            break

        step = delta * damping
        if use_anderson:
            history.append(delta.reshape(-1).copy())
            mixed = _anderson_ridge(list(history), ridge_value)
            if mixed is not None:
                mixed = mixed.reshape(state.shape)
                step = mixed
                if beta_mix > 0.0:
                    step = beta_mix * step
                step *= damping

        if not np.any(step):
            fp_err_final = fp_err
            iter_err_final = 0.0
            effective_k = iteration - 1
            break

        new_state, fp_err_next, alpha = _backtracking_step(
            state, step, phi, fp_err, bool(line_search)
        )
        iter_err = _rms(new_state - state)
        if trace is not None:
            trace[-1] = (iteration, fp_err, iter_err)
        if logger:
            logger.info(
                "迭代 %d/%d，fp_err=%.3e iter_err=%.3e alpha=%.2f",
                iteration,
                K_max,
                fp_err,
                iter_err,
                alpha,
            )

        if not np.isfinite(fp_err_next) or fp_err_next > guard:
            if use_anderson and allow_fallback:
                if logger:
                    logger.warning("ANDERSON->PLAIN fallback (fp_err=%.3e)", fp_err_next)
                fallback_state, fallback_k, fallback_fp_err, fallback_iter_err, fallback_solver = fpt_solve(
                    phi,
                    state,
                    max(K_max, 6),
                    tol,
                    solver="plain",
                    m=1,
                    beta=0.2,
                    ridge=ridge_value,
                    line_search=line_search,
                    fp_err_guard=guard,
                    damping=damping,
                    logger=logger,
                    trace=trace,
                    allow_fallback=False,
                )
                return (
                    fallback_state,
                    fallback_k,
                    fallback_fp_err,
                    fallback_iter_err,
                    fallback_solver,
                )
            fp_err_final = fp_err_next
            iter_err_final = iter_err
            effective_k = iteration
            return new_state.astype(np.float64, copy=False), effective_k, fp_err_final, iter_err_final, solver_used

        state = new_state
        fp_err_final = fp_err_next
        iter_err_final = iter_err
        effective_k = iteration
        if fp_err_next <= tol:
            break

    return state.astype(np.float64, copy=False), effective_k, fp_err_final, iter_err_final, solver_used


def _build_three_compartment_phi(
    params: ThreeCompartmentParams,
    apical_input: np.ndarray,
    basal_input: np.ndarray,
    soma_input: np.ndarray,
    init_soma: float,
    init_apical: float,
    init_basal: float,
) -> Callable[[np.ndarray], np.ndarray]:
    steps = apical_input.shape[0]
    dt = float(params.dt)

    def _phi(state: np.ndarray) -> np.ndarray:
        soma_prev, ap_prev, ba_prev = _split_state(state)
        soma_tm1 = np.empty_like(soma_prev)
        ap_tm1 = np.empty_like(ap_prev)
        ba_tm1 = np.empty_like(ba_prev)
        soma_tm1[0] = init_soma
        ap_tm1[0] = init_apical
        ba_tm1[0] = init_basal
        if steps > 1:
            soma_tm1[1:] = soma_prev[:-1]
            ap_tm1[1:] = ap_prev[:-1]
            ba_tm1[1:] = ba_prev[:-1]

        apical_leak = (params.v_rest - ap_tm1) / params.tau_apical
        apical_coupling = params.coupling_apical * (soma_tm1 - ap_tm1)
        ap_next = ap_tm1 + dt * (apical_leak + apical_coupling + apical_input)

        basal_leak = (params.v_rest - ba_tm1) / params.tau_basal
        basal_coupling = params.coupling_basal * (soma_tm1 - ba_tm1)
        ba_next = ba_tm1 + dt * (basal_leak + basal_coupling + basal_input)

        soma_leak = (params.v_rest - soma_tm1) / params.tau_soma
        soma_coupling = (
            params.coupling_apical * (ap_next - soma_tm1)
            + params.coupling_basal * (ba_next - soma_tm1)
        )
        soma_next = soma_tm1 + dt * (soma_leak + soma_coupling + soma_input)

        return _stack_state(soma_next, ap_next, ba_next)

    return _phi


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

    apical_arr = np.asarray(apical, dtype=np.float64)
    basal_arr = np.asarray(basal, dtype=np.float64)
    soma_arr = np.asarray(soma_input, dtype=np.float64)
    h0 = _stack_state(
        np.full(steps, init_soma, dtype=np.float64),
        np.full(steps, init_apical, dtype=np.float64),
        np.full(steps, init_basal, dtype=np.float64),
    )

    phi_fn = _build_three_compartment_phi(
        params,
        apical_arr,
        basal_arr,
        soma_arr,
        init_soma,
        init_apical,
        init_basal,
    )

    trace: List[Tuple[int, float, float]] = []
    h_final, k_eff, fp_err, iter_err, solver_used = fpt_solve(
        phi_fn,
        h0,
        config.iterations,
        config.tolerance,
        solver=config.solver,
        m=config.anderson_m,
        beta=config.anderson_beta,
        ridge=config.anderson_ridge,
        line_search=config.line_search,
        fp_err_guard=config.fp_err_guard,
        damping=config.damping,
        logger=logger,
        trace=trace,
    )

    soma_state, apical_state, basal_state = _split_state(h_final)
    times = [params.dt * (idx + 1.0) for idx in range(steps)]
    spike_flags = [float(value) >= params.threshold for value in soma_state]

    states = [
        CompartmentState(
            time=times[idx],
            soma=float(soma_state[idx]),
            apical=float(apical_state[idx]),
            basal=float(basal_state[idx]),
            spike=bool(spike_flags[idx]),
        )
        for idx in range(steps)
    ]

    residuals = [entry[1] for entry in trace]
    iter_history = [entry[2] for entry in trace]

    return FixedPointResult(
        states=states,
        residuals=residuals,
        iter_errors=iter_history,
        effective_iterations=k_eff,
        final_fp_error=fp_err,
        final_iter_error=iter_err,
        solver_used=solver_used,
    )
