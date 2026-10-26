"""Validated options for trust-region and line-search root solvers."""

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


def _validate_derivatives(options):
    """Normalize derivative aliases and validate optional physical probe distances."""
    aliases = {
        "2-point": "2-point",
        "forward_finite_differences": "2-point",
        "3-point": "3-point",
        "central_finite_differences": "3-point",
        "cs": "cs",
        "complex_step": "cs",
    }
    if options.derivative_method not in aliases:
        raise ValueError("derivative_method must be '2-point', '3-point', or 'cs'.")
    object.__setattr__(options, "derivative_method", aliases[options.derivative_method])
    if options.derivative_abs_step is not None:
        step = np.asarray(options.derivative_abs_step, dtype=float)
        if (
            step.ndim > 1
            or not step.size
            or not np.all(np.isfinite(step))
            or np.any(step <= 0)
        ):
            raise ValueError(
                "derivative_abs_step must be a positive finite scalar or vector."
            )


def _validate_scaling(options):
    """Require an explicit boolean for automatic numerical column scaling."""
    if not isinstance(options.jacobian_scaling, (bool, np.bool_)):
        raise TypeError("jacobian_scaling must be a boolean.")
    object.__setattr__(options, "jacobian_scaling", bool(options.jacobian_scaling))


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
    """Configure Newton-Raphson or Gauss-Newton with optional backtracking.

    Omission enables protected backtracking. Explicit backtracking=None selects
    valid steps without a decrease test, even if the residual increases. With
    the default first_step_fraction=1 these are full Newton steps. Root success
    always requires residual_tol. max_iterations counts trial attempts, max_nfev counts
    every residual call including undefined trials and finite-difference probes.
    For Gauss-Newton, gtol detects a negligible projected residual change
    ||A z|| <= gtol * min(1, ||F||), rather than testing an arbitrarily scaled
    gradient. Newton-Raphson has no gradient-based early-exit test.

    jacobian_scaling=True equilibrates linear-system columns using running
    maximum norms at fresh Jacobians. This leaves x_scale, derivative probes,
    and physical backtracking limits unchanged. Full-rank Newton/Gauss-Newton
    directions are unchanged in exact arithmetic; rank-deficient minimum-norm
    Gauss-Newton directions can change. False disables numerical equilibration.

    first_step_fraction is a dimensionless initial multiplier in (0, 1] of
    the Newton direction. After each accepted move this cap grows by
    step_growth_factor (finite and >= 1), up to 1. It is independent of the
    absolute x_scale displacement envelope; both restrictions apply when
    backtracking is enabled, and Armijo may shorten the trial further. Rejected
    trials do not advance the schedule. A growth factor of 1 fixes the cap.
    The defaults leave existing full-step and backtracking behavior unchanged.
    """

    x_scale: object = 1.0
    residual_tol: float = 1e-8
    xtol: float = 1e-12
    gtol: float = 1e-12
    max_iterations: int = 500
    max_nfev: int = 5000
    finite_diff_step: float = 1e-7
    derivative_method: str = "2-point"
    derivative_abs_step: object = None
    history_level: str = "full"
    backtracking: BacktrackingOptions | None = field(
        default_factory=BacktrackingOptions
    )
    jacobian_scaling: bool = True
    first_step_fraction: float = 1.0
    step_growth_factor: float = 1.5

    def __post_init__(self):
        """Validate common controls and normalize nested dictionaries consistently."""
        _validate_derivatives(self)
        _validate_scaling(self)
        _positive_finite("first_step_fraction", self.first_step_fraction)
        _positive_finite("step_growth_factor", self.step_growth_factor)
        if self.first_step_fraction > 1:
            raise ValueError("first_step_fraction must be <= 1.")
        if self.step_growth_factor < 1:
            raise ValueError("step_growth_factor must be >= 1.")
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


@dataclass(frozen=True)
class StepControlOptions:
    """Control scaled trial distances and termination.

    ``x_scale`` contains fixed characteristic variable distances (not MINPACK
    ``diag``; the correspondence is ``diag = 1 / x_scale``). ``initial_radius``
    is an absolute scaled distance, independent of the initial point.
    ``max_radius`` bounds every trial. After a successful boundary step with
    ratio >= 0.75, the radius may grow by ``growth_factor``; poor/undefined
    trials shrink it by ``shrink_factor``. A growth factor of one prevents
    regrowth, including after recovery. Radii never exceed ``max_radius``.

    Success requires ``norm(fun(x)) <= residual_tol``. Small steps, a stationary
    least-squares point, and exhausted budgets are failures, not roots.
    ``max_iterations`` counts all trial attempts (accepted or rejected).
    ``max_nfev`` includes initial, trial, and numerical-derivative evaluations.
    Forward difference probes move ``finite_diff_step * x_scale[j]`` in x[j].

    jacobian_scaling=True adds numerical column equilibration without changing
    the physical meaning of x_scale or the radius cap. Numerical trust steps
    are shortened along their ray when required by the physical envelope.
    Scaling updates only at fresh Jacobians and stays fixed through Broyden
    updates and trial retries. False selects the original fixed-scale geometry.
    """

    initial_radius: float = 1.0
    max_radius: float = 100.0
    growth_factor: float = 2.0
    shrink_factor: float = 0.25
    acceptance_ratio: float = 0.1
    residual_tol: float = 1e-8
    xtol: float = 1e-12
    gtol: float = 1e-12
    max_iterations: int = 500
    max_nfev: int = 5000
    finite_diff_step: float = 1e-7
    derivative_method: str = "2-point"
    derivative_abs_step: object = None
    x_scale: object = 1.0
    history_level: str = "full"
    jacobian_scaling: bool = True

    def __post_init__(self):
        """Validate finite positive controls, consistent radii, integer budgets and the
        selected history mode.
        """
        _validate_derivatives(self)
        _validate_scaling(self)
        if self.history_level not in ("full", "summary", "none"):
            raise ValueError("history_level must be 'full', 'summary' or 'none'.")
        for name in (
            "initial_radius",
            "max_radius",
            "growth_factor",
            "shrink_factor",
            "acceptance_ratio",
            "residual_tol",
            "xtol",
            "gtol",
            "finite_diff_step",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if self.initial_radius > self.max_radius:
            raise ValueError("initial_radius must not exceed max_radius.")
        if self.growth_factor < 1:
            raise ValueError("growth_factor must be >= 1.")
        if self.shrink_factor >= 1:
            raise ValueError("shrink_factor must be < 1.")
        if self.acceptance_ratio >= 0.25:
            raise ValueError("acceptance_ratio must be < 0.25.")
        for name in ("max_iterations", "max_nfev"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        scale = np.asarray(self.x_scale, dtype=float)
        if (
            scale.ndim > 1
            or scale.size == 0
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError("x_scale must be a positive finite scalar or vector.")
