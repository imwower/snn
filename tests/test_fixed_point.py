import logging
import unittest

from snn import (
    FixedPointConfig,
    ThreeCompartmentNeuron,
    ThreeCompartmentParams,
    fixed_point_parallel_solve,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.propagate = False


class FixedPointParallelTrainingTests(unittest.TestCase):
    def test_fixed_point_matches_sequential_subthreshold(self) -> None:
        logger.info("开始测试：固定点迭代应逼近顺序积分结果（无脉冲场景）")
        params = ThreeCompartmentParams(
            dt=0.5,
            tau_soma=10.0,
            tau_apical=8.0,
            tau_basal=9.0,
            threshold=1.0,
        )

        apical = [0.02, 0.015, 0.01, 0.005, 0.0, -0.002, 0.0, 0.003]
        basal = [0.01, 0.012, 0.02, 0.015, 0.005, 0.0, 0.0, 0.001]
        soma = [0.0] * len(apical)

        sequential = ThreeCompartmentNeuron(params)
        seq_history = sequential.run(apical, basal, soma)

        result = fixed_point_parallel_solve(
            params,
            apical,
            basal,
            soma_currents=soma,
            config=FixedPointConfig(iterations=12, tolerance=1e-7, damping=1.0),
        )

        self.assertGreaterEqual(len(result.residuals), 1)
        self.assertLess(result.residuals[-1], result.residuals[0])
        self.assertLessEqual(result.residuals[-1], 5e-3)
        logger.info(
            "固定点迭代完成，迭代次数=%d，最终残差=%.3e",
            len(result.residuals),
            result.residuals[-1],
        )

        for fp_state, seq_state in zip(result.states, seq_history):
            self.assertAlmostEqual(fp_state.soma, seq_state.soma, places=4)
            self.assertAlmostEqual(fp_state.apical, seq_state.apical, places=4)
            self.assertAlmostEqual(fp_state.basal, seq_state.basal, places=4)
            self.assertFalse(fp_state.spike)

    def test_fixed_point_zero_inputs_stays_at_rest(self) -> None:
        logger.info("开始测试：零输入时固定点迭代应保持静息电位")
        params = ThreeCompartmentParams(dt=0.1, tau_soma=5.0, tau_apical=4.0, tau_basal=4.5)
        steps = 6
        zeros = [0.0] * steps

        result = fixed_point_parallel_solve(
            params,
            zeros,
            zeros,
            soma_currents=zeros,
            config=FixedPointConfig(iterations=5, tolerance=1e-8),
        )

        self.assertEqual(len(result.states), steps)
        self.assertLessEqual(result.residuals[-1], 1e-8)

        rest = params.v_rest
        for idx, state in enumerate(result.states):
            self.assertAlmostEqual(state.soma, rest, places=8)
            self.assertAlmostEqual(state.apical, rest, places=8)
            self.assertAlmostEqual(state.basal, rest, places=8)
            self.assertFalse(state.spike)
        logger.info("固定点迭代保持静息电位，步数=%d", steps)


if __name__ == "__main__":
    unittest.main()
