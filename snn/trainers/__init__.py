"""Training utilities for NumPy-based ThreeCompartment SNNs."""

try:
    from .train_numpy import main  # type: ignore
except Exception:  # pragma: no cover - allow module-level import without hard deps
    main = None  # type: ignore

__all__ = ["main"]
