"""Core simulation primitives for the three-compartment SNN reference implementation.

The package exposes the :class:`ThreeCompartmentNeuron` model together with its
configuration dataclass and state container.  The implementation focuses on
clarity and testability so the neuron dynamics can be reused by the training
pipeline or fixed-point solvers described in the project README.
"""

from .neuron import CompartmentState, ThreeCompartmentNeuron, ThreeCompartmentParams

__all__ = ["ThreeCompartmentNeuron", "ThreeCompartmentParams", "CompartmentState"]
