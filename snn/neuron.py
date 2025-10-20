"""Three-compartment leaky integrate-and-fire neuron implementation.

The module implements a minimalist three-compartment neuron (soma, apical and
basal dendrites) that can be stepped explicitly or driven by pre-computed
currents.  The equations are written in discrete time using an Euler scheme so
they can be plugged into fixed-point solvers or classic time-stepped training
loops.

The model keeps the number of parameters intentionally small: the coupling
constants determine how strongly dendritic compartments pull the soma membrane,
and the provided currents are interpreted as voltage change rates per unit time.
This makes it straightforward to connect the neuron to either analytical
derivations or empirical event streams, as described in the project README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List

import yaml


# 读取配置文件中的日志级别，默认值为 INFO。
def _load_log_level(default_level: int = logging.INFO) -> int:
    """Load log level from config.yaml, falling back to the provided default."""

    # config.yaml 与仓库根目录对齐，因此需要向上一级目录查找。
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not config_path.exists():
        return default_level

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except Exception:
        return default_level

    level_name = (
        (config.get("logging") or {}).get("level") if isinstance(config, dict) else None
    )
    if not isinstance(level_name, str):
        return default_level

    level_value = getattr(logging, level_name.upper(), None)
    if isinstance(level_value, int):
        return level_value
    return default_level


_LOG_LEVEL = _load_log_level()
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

@dataclass(frozen=True)
class ThreeCompartmentParams:
    """Hyper-parameters that govern the neuron dynamics.

    All membrane potentials are modelled in volts while time is expressed in
    seconds.  Input currents are used as voltage change rates (volt/second) to
    keep the implementation dependency-free; callers can rescale their inputs to
    match the desired physical setting.
    """

    dt: float = 1e-3
    tau_soma: float = 2e-2
    tau_apical: float = 3e-2
    tau_basal: float = 3e-2
    coupling_apical: float = 0.6
    coupling_basal: float = 0.6
    v_rest: float = -0.07
    threshold: float = -0.054
    reset_potential: float = -0.065
    refractory_period: float = 2e-3


@dataclass
class CompartmentState:
    """Snapshot of the membrane potentials after a simulation step."""

    time: float
    soma: float
    apical: float
    basal: float
    spike: bool


class ThreeCompartmentNeuron:
    """Explicit Euler solver for a three-compartment leaky integrate-and-fire neuron."""

    def __init__(
        self,
        params: ThreeCompartmentParams = ThreeCompartmentParams(),
        *,
        initial_potentials: Optional[dict] = None
    ) -> None:
        self.params = params
        self._refractory_time_remaining = 0.0
        self.time = 0.0
        self._set_initial_potentials(initial_potentials)
        # 神经元初始化完成后记录当前状态，便于调试。
        logger.debug(
            "神经元初始化完成，dt=%.4e, 阈值=%.3f, 初始电位=(soma=%.3f, apical=%.3f, basal=%.3f)",
            self.params.dt,
            self.params.threshold,
            self.v_soma,
            self.v_apical,
            self.v_basal,
        )

    def _set_initial_potentials(self, initial_potentials: Optional[dict]) -> None:
        defaults = initial_potentials or {}
        self.v_soma = float(defaults.get("soma", self.params.v_rest))
        self.v_apical = float(defaults.get("apical", self.params.v_rest))
        self.v_basal = float(defaults.get("basal", self.params.v_rest))
        # 记录初始电位，用于追踪启动条件。
        logger.debug(
            "设置初始膜电位：soma=%.3f, apical=%.3f, basal=%.3f",
            self.v_soma,
            self.v_apical,
            self.v_basal,
        )

    def reset(self, *, initial_potentials: Optional[dict] = None) -> None:
        """Reset membrane potentials and internal timers."""

        self._refractory_time_remaining = 0.0
        self.time = 0.0
        self._set_initial_potentials(initial_potentials)
        # 重置神经元时输出调试日志。
        logger.debug("神经元状态已重置，时间轴归零。")

    def step(
        self,
        apical_current: float,
        basal_current: float,
        soma_current: float = 0.0
    ) -> CompartmentState:
        """Advance the neuron by a single time step.

        Args:
            apical_current: Input current driving the apical dendrite.
            basal_current: Input current driving the basal dendrite.
            soma_current: Optional direct current applied to the soma.

        Returns:
            CompartmentState with updated membrane potentials and spike flag.
        """

        p = self.params
        dt = p.dt
        # 记录输入电流，帮助定位异常输入。
        logger.debug(
            "步进开始：t=%.4f, 输入电流(apical=%.4f, basal=%.4f, soma=%.4f)",
            self.time,
            apical_current,
            basal_current,
            soma_current,
        )

        # Update dendrites first so the soma can observe their fresh potentials.
        apical_leak = (p.v_rest - self.v_apical) / p.tau_apical
        apical_coupling = p.coupling_apical * (self.v_soma - self.v_apical)
        new_apical = self.v_apical + dt * (apical_leak + apical_coupling + apical_current)

        basal_leak = (p.v_rest - self.v_basal) / p.tau_basal
        basal_coupling = p.coupling_basal * (self.v_soma - self.v_basal)
        new_basal = self.v_basal + dt * (basal_leak + basal_coupling + basal_current)

        spike = False

        if self._refractory_time_remaining > 0.0:
            self._refractory_time_remaining = max(0.0, self._refractory_time_remaining - dt)
            new_soma = p.reset_potential
        else:
            soma_leak = (p.v_rest - self.v_soma) / p.tau_soma
            soma_coupling = (
                p.coupling_apical * (new_apical - self.v_soma)
                + p.coupling_basal * (new_basal - self.v_soma)
            )
            new_soma = self.v_soma + dt * (soma_leak + soma_coupling + soma_current)
            if new_soma >= p.threshold:
                spike = True
                new_soma = p.reset_potential
                self._refractory_time_remaining = p.refractory_period
                # 记录触发脉冲的详细信息。
                logger.debug(
                    "检测到脉冲：t=%.4f, 膜电位超过阈值 %.3f，进入不应期 %.4e",
                    self.time + dt,
                    p.threshold,
                    p.refractory_period,
                )

        self.v_apical = new_apical
        self.v_basal = new_basal
        self.v_soma = new_soma
        self.time += dt
        # 输出此次步进后的膜电位变化。
        logger.debug(
            "步进结束：t=%.4f, 膜电位(soma=%.3f, apical=%.3f, basal=%.3f), spike=%s",
            self.time,
            self.v_soma,
            self.v_apical,
            self.v_basal,
            spike,
        )

        return CompartmentState(
            time=self.time,
            soma=self.v_soma,
            apical=self.v_apical,
            basal=self.v_basal,
            spike=spike,
        )

    def run(
        self,
        apical_currents: Sequence[float],
        basal_currents: Sequence[float],
        soma_currents: Optional[Sequence[float]] = None
    ) -> List[CompartmentState]:
        """Simulate the neuron for a sequence of inputs.

        Args:
            apical_currents: Sequence of apical currents (length T).
            basal_currents: Sequence of basal currents (length T).
            soma_currents: Optional sequence of soma currents; defaults to zeros.

        Returns:
            List of CompartmentState snapshots (length T).

        Raises:
            ValueError: if the input sequences do not share the same length.
        """

        apical = list(apical_currents)
        basal = list(basal_currents)
        if len(apical) != len(basal):
            raise ValueError("apical_currents and basal_currents must have the same length")

        if soma_currents is None:
            soma_seq = [0.0] * len(apical)
        else:
            soma_seq = list(soma_currents)
            if len(soma_seq) != len(apical):
                raise ValueError("soma_currents must match the length of dendritic inputs")

        history: List[CompartmentState] = []
        for ap_current, ba_current, so_current in zip(apical, basal, soma_seq):
            history.append(self.step(ap_current, ba_current, so_current))

        # 运行结束后记录总步数，便于追踪模拟长度。
        logger.debug("运行结束，总步数=%d，最终时间=%.4f", len(history), self.time)

        return history
