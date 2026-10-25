"""SLSQP evaluation values and validated options."""

from dataclasses import dataclass, field
from numbers import Real
import numpy as np


@dataclass
class Evaluation:
    """Return the objective and all constraints from one model evaluation.

    The objective is minimized. Equalities must be zero and inequalities must
    be nonnegative at a feasible point. Values must already use the caller's
    intended normalization; this container applies no physical scaling.
    """

    objective: float
    equalities: np.ndarray = field(default_factory=lambda: np.empty(0))
    inequalities: np.ndarray = field(default_factory=lambda: np.empty(0))

    def vector(self):
        """Pack objective, equalities, and nonnegative inequalities into one vector."""
        return np.r_[self.objective, self.equalities, self.inequalities]


@dataclass(frozen=True)
class SLSQPOptions:
    """Configure native stopping, trial recovery, derivatives, and final checks.

    ``ftol`` and ``maxiter`` control SciPy's native solve. Step fractions limit
    trial motion along its proposed direction; backtracking retries recognized
    domain failures, and ``step_growth_factor`` governs restoration of the cap
    after a shortened start or recovery. See ``minimize_slsqp`` for that sequence.

    ``finite_diff_step`` is an absolute distance in the caller's coordinates
    (after any scaling applied by the caller). Each derivative probe is evaluated once. ``feasibility_tol`` and ``optimality_tol`` are independent acceptance
    thresholds for scaled residuals, not aliases for native ``ftol``. Validation
    rejects unsupported values before a solve begins.

    ``check_kkt=False`` disables multiplier fits and the optimality acceptance
    gate. Native success and feasibility are still required; ``kkt_passed`` and
    stationarity diagnostics are then unavailable, not successful checks.
    """

    ftol: float = 1e-9
    maxiter: int = 200
    finite_diff_step: float = 1e-7
    first_step_fraction: float = 0.25
    max_step_fraction: float = 1.0
    step_growth_factor: float | None = 1.5
    domain_backtracking: bool = True
    backtrack_factor: float = 0.5
    max_backtracks: int = 20
    feasibility_tol: float = 1e-6
    optimality_tol: float = 1e-4
    check_kkt: bool = True

    def __post_init__(self):
        """Reject invalid stopping, recovery, and derivative controls."""
        if not isinstance(self.check_kkt, bool):
            raise TypeError("check_kkt must be a boolean.")
        for name in ("ftol", "finite_diff_step", "feasibility_tol", "optimality_tol"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("first_step_fraction", "max_step_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be in (0, 1].")
        if self.step_growth_factor is not None and (
            isinstance(self.step_growth_factor, (bool, np.bool_))
            or not isinstance(self.step_growth_factor, Real)
            or not np.isfinite(self.step_growth_factor)
            or self.step_growth_factor < 1
        ):
            raise ValueError("step_growth_factor must be None or finite and >= 1.")
        if (
            isinstance(self.backtrack_factor, (bool, np.bool_))
            or not isinstance(self.backtrack_factor, Real)
            or not np.isfinite(self.backtrack_factor)
            or not 0 < self.backtrack_factor < 1
        ):
            raise ValueError("backtrack_factor must be in (0, 1).")
        for name, minimum in (
            ("maxiter", 1),
            ("max_backtracks", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(self.domain_backtracking, bool):
            raise ValueError("domain_backtracking must be boolean.")
