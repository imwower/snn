import unittest

from snn import ThreeCompartmentNeuron, ThreeCompartmentParams


class ThreeCompartmentNeuronTests(unittest.TestCase):
    def test_strong_basal_input_triggers_spike(self) -> None:
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
                break

        self.assertTrue(spike_triggered, "Expected spike when basal input is sufficiently strong")
        self.assertLessEqual(neuron.v_soma, params.reset_potential + 1e-9)

    def test_refractory_period_prevents_immediate_retrigger(self) -> None:
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

        # Drive the neuron into a spike once.
        while True:
            state = neuron.step(apical_current=0.015, basal_current=0.02)
            if state.spike:
                break

        refractory_steps = int(round(params.refractory_period / params.dt))
        for _ in range(refractory_steps):
            state = neuron.step(apical_current=0.05, basal_current=0.05)
            self.assertFalse(state.spike, "Refractory window should clamp spikes")

    def test_run_matches_manual_stepping(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
