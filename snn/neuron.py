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

from dataclasses import dataclass
from typing import Optional, Sequence, List


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

    def _set_initial_potentials(self, initial_potentials: Optional[dict]) -> None:
        defaults = initial_potentials or {}
        self.v_soma = float(defaults.get("soma", self.params.v_rest))
        self.v_apical = float(defaults.get("apical", self.params.v_rest))
        self.v_basal = float(defaults.get("basal", self.params.v_rest))

    def reset(self, *, initial_potentials: Optional[dict] = None) -> None:
        """Reset membrane potentials and internal timers."""

        self._refractory_time_remaining = 0.0
        self.time = 0.0
        self._set_initial_potentials(initial_potentials)

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

        self.v_apical = new_apical
        self.v_basal = new_basal
        self.v_soma = new_soma
        self.time += dt

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

        return history
