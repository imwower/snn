import logging
import unittest

from snn import ThreeCompartmentNeuron, ThreeCompartmentParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.propagate = False


class ThreeCompartmentNeuronTests(unittest.TestCase):
    def test_strong_basal_input_triggers_spike(self) -> None:
        logger.info("开始测试：强基底输入应触发脉冲")
        params = ThreeCompartmentParams(
            dt=1.0,
            tau_soma=5.0,
            tau_apical=5.0,
            tau_basal=5.0,
            coupling_apical=0.4,
            coupling_basal=0.6,
            threshold=-0.05,
            reset_potential=-0.065,
            refractory_period=2.0,
        )
        neuron = ThreeCompartmentNeuron(params)

        spike_triggered = False
        for _ in range(20):
            state = neuron.step(apical_current=0.0, basal_current=0.02)
            if state.spike:
                spike_triggered = True
                logger.info("检测到脉冲，时间=%.2f，胞体电位=%.3f", state.time, state.soma)
                break

        self.assertTrue(spike_triggered, "基底输入足够强时应触发脉冲")
        self.assertLessEqual(neuron.v_soma, params.reset_potential + 1e-9)

    def test_refractory_period_prevents_immediate_retrigger(self) -> None:
        logger.info("开始测试：不应期应阻止立即再次放电")
        params = ThreeCompartmentParams(
            dt=1.0,
            tau_soma=5.0,
            tau_apical=5.0,
            tau_basal=5.0,
            coupling_apical=0.5,
            coupling_basal=0.5,
            threshold=-0.05,
            reset_potential=-0.065,
            refractory_period=3.0,
        )
        neuron = ThreeCompartmentNeuron(params)

        # 先驱动一次脉冲。
        while True:
            state = neuron.step(apical_current=0.015, basal_current=0.02)
            if state.spike:
                logger.info("首次脉冲产生于时间 %.2f", state.time)
                break

        refractory_steps = int(round(params.refractory_period / params.dt))
        for _ in range(refractory_steps):
            state = neuron.step(apical_current=0.05, basal_current=0.05)
            self.assertFalse(state.spike, "不应期窗口内不应产生新的脉冲")

    def test_run_matches_manual_stepping(self) -> None:
        logger.info("开始测试：run 接口应与逐步调用保持一致")
        params = ThreeCompartmentParams(dt=0.5, tau_soma=5.0, tau_apical=4.0, tau_basal=4.0)

        apical = [0.01, 0.02, 0.0, 0.0, 0.01]
        basal = [0.025, 0.015, 0.0, 0.01, 0.02]
        soma = [0.0, 0.0, 0.0, 0.0, 0.0]

        neuron_a = ThreeCompartmentNeuron(params)
        run_history = neuron_a.run(apical, basal, soma)

        neuron_b = ThreeCompartmentNeuron(params)
        manual_history = [neuron_b.step(a, b, s) for a, b, s in zip(apical, basal, soma)]

        self.assertEqual(len(run_history), len(manual_history))
        for auto, manual in zip(run_history, manual_history):
            self.assertAlmostEqual(auto.soma, manual.soma, places=12)
            self.assertAlmostEqual(auto.apical, manual.apical, places=12)
            self.assertAlmostEqual(auto.basal, manual.basal, places=12)
            self.assertEqual(auto.spike, manual.spike)
        logger.info("run 接口仿真步数=%d，对比结果完全一致", len(run_history))

    def test_reset_restores_initial_state(self) -> None:
        logger.info("开始测试：reset 应恢复初始状态")
        params = ThreeCompartmentParams(dt=0.1, threshold=-0.055)
        neuron = ThreeCompartmentNeuron(params)

        # 通过一次脉冲改变内部状态。
        for _ in range(50):
            state = neuron.step(apical_current=0.02, basal_current=0.02)
            if state.spike:
                logger.info("触发脉冲后开始验证 reset 行为")
                break

        neuron.reset()
        self.assertEqual(neuron.time, 0.0)
        self.assertAlmostEqual(neuron.v_soma, params.v_rest, places=12)
        self.assertAlmostEqual(neuron.v_apical, params.v_rest, places=12)
        self.assertAlmostEqual(neuron.v_basal, params.v_rest, places=12)
        logger.info("reset 后膜电位恢复至静息值 %.3f", params.v_rest)


if __name__ == "__main__":
    unittest.main()
