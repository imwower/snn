"""三腔室泄漏积分发放 (LIF) 神经元实现。

本模块提供一个最小化的三腔室模型（胞体、树突顶端与树突基底），可按步推进或接受预先计算的电流序列。
模型采用离散时间的欧拉显式求解器，可直接嵌入固定步长的训练循环或定点求解流程。

参数数量保持精简：耦合常数控制树突对胞体电位的影响，输入电流被解释为单位时间的电位变化率。
因此可以更方便地衔接解析推导或事件流数据，具体细节参考项目 README。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, List

from .config import get_logging_level


_LOG_LEVEL = get_logging_level()
logging.basicConfig(level=_LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(_LOG_LEVEL)

@dataclass(frozen=True)
class ThreeCompartmentParams:
    """定义神经元动力学的超参数。

    所有膜电位使用伏特表示，时间使用秒。输入电流表示单位时间内的电位变化率（伏/秒），
    以保持实现的轻量化；调用方可根据物理场景自行缩放。
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
    """记录单步仿真后的膜电位快照。"""

    time: float
    soma: float
    apical: float
    basal: float
    spike: bool


class ThreeCompartmentNeuron:
    """三腔室泄漏积分发放神经元的欧拉显式求解器。"""

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
        """重置膜电位与内部计时器。"""

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
        """按单个时间步推进神经元状态。

        Args:
            apical_current: 施加在树突顶端的输入电流。
            basal_current: 施加在树突基底的输入电流。
            soma_current: 直接作用于胞体的附加电流，默认为 0。

        Returns:
            返回更新后的膜电位及放电标记。
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

        # 先更新树突电位，使胞体读取到最新状态。
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
            logger.debug(
                "处于不应期：剩余时间=%.4e，胞体电位保持在重置值 %.3f",
                self._refractory_time_remaining,
                new_soma,
            )
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
        """使用输入序列驱动神经元仿真。

        Args:
            apical_currents: 树突顶端电流序列，长度为 T。
            basal_currents: 树突基底电流序列，长度为 T。
            soma_currents: 可选的胞体电流序列，默认为全零。

        Returns:
            返回长度为 T 的 `CompartmentState` 列表。

        Raises:
            ValueError: 当输入序列长度不一致时抛出。
        """

        apical = list(apical_currents)
        basal = list(basal_currents)
        if len(apical) != len(basal):
            raise ValueError("apical_currents 与 basal_currents 的长度必须一致")

        if soma_currents is None:
            soma_seq = [0.0] * len(apical)
        else:
            soma_seq = list(soma_currents)
            if len(soma_seq) != len(apical):
                raise ValueError("soma_currents 的长度需要匹配树突输入序列")

        history: List[CompartmentState] = []
        for ap_current, ba_current, so_current in zip(apical, basal, soma_seq):
            history.append(self.step(ap_current, ba_current, so_current))

        # 运行结束后记录总步数，便于追踪模拟长度。
        logger.debug("运行结束，总步数=%d，最终时间=%.4f", len(history), self.time)

        return history
