"""Validated direction-independent options for the experimental Newton methods."""

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np


def _positive_finite(name, value):
    """Reject boolean, non-real, nonfinite or nonpositive numerical controls."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive.")


def _integer_at_least(name, value, minimum):
    """Validate a nonboolean finite integer count with its required lower bound."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class StepLimitOptions:
    """Bound scaled displacements and restore a reduced envelope gradually.

    A failed initial trial carries the accepted step length into the next
    iteration's envelope. A successful initial trial near the envelope permits
    growth toward max_limit. Numerical derivative probes have separate controls.
    """

    initial_limit: float = 1.0
    max_limit: float = 100.0
    growth_factor: float = 2.0

    def __post_init__(self):
        """Require finite ordered positive limits and growth of at least one."""
        for name in ("initial_limit", "max_limit", "growth_factor"):
            _positive_finite(name, getattr(self, name))
        if self.initial_limit > self.max_limit:
            raise ValueError("initial_limit must not exceed max_limit.")
        if self.growth_factor < 1:
            raise ValueError("growth_factor must be >= 1.")


@dataclass(frozen=True)
class BacktrackingOptions:
    """Configure Armijo backtracking and an optional scaled displacement envelope.

    step_limits=None disables only the envelope. max_backtracks=0 still checks
    the initial trial against Armijo, but does not retry a rejected point.
    """

    armijo_c1: float = 1e-4
    contraction_factor: float = 0.5
    max_backtracks: int = 30
    step_limits: StepLimitOptions | None = field(default_factory=StepLimitOptions)

    def __post_init__(self):
        """Validate sufficient decrease, contraction, retries and nested limits."""
        for name in ("armijo_c1", "contraction_factor"):
            _positive_finite(name, getattr(self, name))
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be < 1.")
        _integer_at_least("max_backtracks", self.max_backtracks, 0)
        if isinstance(self.step_limits, dict):
            object.__setattr__(
                self, "step_limits", StepLimitOptions(**self.step_limits)
            )
        if self.step_limits is not None and not isinstance(
            self.step_limits, StepLimitOptions
        ):
            raise TypeError(
                "step_limits must be StepLimitOptions, a dictionary, or None."
            )


@dataclass(frozen=True)
class NewtonOptions:
    """Configure Newton–Raphson or Gauss–Newton with optional backtracking.

    Omission enables protected backtracking. Explicit backtracking=None selects
    full valid steps, even if the residual increases. Root success always
    requires residual_tol. max_iterations counts trial attempts, max_nfev counts
    every residual call including undefined trials and finite-difference probes.
    For Gauss–Newton, gtol detects a negligible projected residual change
    ||A z|| <= gtol * min(1, ||F||), rather than testing an arbitrarily scaled
    gradient. Newton–Raphson has no gradient-based early-exit test.
    """

    x_scale: object = 1.0
    residual_tol: float = 1e-8
    xtol: float = 1e-12
    gtol: float = 1e-12
    max_iterations: int = 500
    max_nfev: int = 5000
    finite_diff_step: float = 1e-7
    history_level: str = "full"
    backtracking: BacktrackingOptions | None = field(
        default_factory=BacktrackingOptions
    )

    def __post_init__(self):
        """Validate common controls and normalize nested dictionaries consistently."""
        for name in ("residual_tol", "xtol", "gtol", "finite_diff_step"):
            _positive_finite(name, getattr(self, name))
        for name in ("max_iterations", "max_nfev"):
            _integer_at_least(name, getattr(self, name), 1)
        if self.history_level not in ("full", "summary", "none"):
            raise ValueError("history_level must be 'full', 'summary' or 'none'.")
        scale = np.asarray(self.x_scale, dtype=float)
        if (
            scale.ndim > 1
            or scale.size == 0
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError("x_scale must be a positive finite scalar or vector.")
        if isinstance(self.backtracking, dict):
            object.__setattr__(
                self, "backtracking", BacktrackingOptions(**self.backtracking)
            )
        if self.backtracking is not None and not isinstance(
            self.backtracking, BacktrackingOptions
        ):
            raise TypeError(
                "backtracking must be BacktrackingOptions, a dictionary, or None."
            )
