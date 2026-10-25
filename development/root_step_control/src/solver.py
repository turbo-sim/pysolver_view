"""QR-based LM and Powell-style hybrid experiments controlling every trial radius.

These are independent trust-region implementations, not wrappers around or
bit-for-bit reproductions of MINPACK. Hybrid combines dogleg steps and scaled
Broyden QR updates, refreshing the Jacobian after poor model agreement.
Only public NumPy/SciPy interfaces are used. The
geometry is fixed by the caller: ``norm((x_trial - x_base) / x_scale) <= radius``.
The first radius, a persistent maximum, and permitted growth are independent.
All trial points, including rejected and undefined ones, obey the radius;
finite-difference probes are separate and use ``finite_diff_step`` instead.
"""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.linalg import norm

from .linear_model import LinearModel
from .newton_options import BacktrackingOptions, NewtonOptions, StepLimitOptions
from .newton_solver import solve_newton
from .solver_common import BudgetExhausted as _BudgetExhausted, DomainError
from scipy.optimize import OptimizeResult


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
    x_scale: object = 1.0
    history_level: str = "full"

    def __post_init__(self):
        """Validate finite positive controls, consistent radii, integer budgets and the
        selected history mode.
        """
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


def _solve_trust_region(fun, x0, jac=None, *, method="lm", options=None, callback=None):
    """Solve a square or overdetermined system using controlled LM or hybrid.

    ``fun(x)`` returns a nonempty residual vector and ``jac(x)`` its m-by-n
    Jacobian, with m >= n. Without ``jac`` forward differences are used.
    ``options`` is a :class:`StepControlOptions` or a dictionary of its fields.
    The optional callback receives a copy of each accepted history record,
    including the initial point; returning exactly ``False`` requests a stop.

    The OptimizeResult has ``history`` (initial and accepted iterates),
    ``trials`` (all attempted solver steps from their accepted bases), and
    ``evaluations`` (all residual calls, with initial/trial/finite_difference
    kinds). Copies of coordinates are retained. ``nit`` counts accepted moves;
    ``ntrials`` counts trial attempts. Failures return the last accepted point.
    No residual values are fabricated for undefined trials.

    ``history_level='full'`` retains coordinates and scalar diagnostics;
    ``'summary'`` retains scalar diagnostics without coordinate copies;
    ``'none'`` disables stored histories. Callbacks always receive full detached
    accepted records. Counts, maxima, and numerical behavior are independent of
    the history setting. ``linear_algebra`` reports factorization/update counts.

    ``method='hybr'`` uses dogleg steps and scaled Broyden rank-one Jacobian
    updates from valid trials, including rejected trials. Two consecutive poor
    agreements trigger a new Jacobian. A suspect approximate model is refreshed
    before declaring stationarity/stagnation. Unlike native MINPACK HYBRJ, this
    variant uses QR factors with SVD fallback, a configurable radius/acceptance policy, and
    residual-based success; it deliberately omits native slow-progress cutoffs
    that would penalize intentionally small steps. ``nupdates`` counts Broyden
    updates; ``njev`` counts full Jacobian evaluations. It accepts m >= n, whereas
    SciPy root('hybr') requires square systems.
    """
    if method not in ("lm", "hybr"):
        raise ValueError("method must be 'lm' or 'hybr'.")
    if options is None:
        options = StepControlOptions()
    elif isinstance(options, dict):
        options = StepControlOptions(**options)
    if not isinstance(options, StepControlOptions):
        raise TypeError("options must be StepControlOptions or a dictionary.")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite vector.")
    scale = np.asarray(options.x_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(x.size, scale.item())
    if scale.shape != x.shape:
        raise ValueError("x_scale must be scalar or have the same shape as x0.")
    scale = scale.copy()
    radius = options.initial_radius
    history, trials, evaluations = [], [], []
    nfev = njev = nit = nupdates = ntrials = nrejected = ninvalid = 0
    max_trial_step_scaled = max_radius_used = max_radius_ratio = 0.0
    linear_stats = {}
    store_history = options.history_level != "none"
    store_coordinates = options.history_level == "full"
    f = None
    f_norm = last_eval_norm = None
    residual_size = None

    def evaluate(point, kind):
        """Evaluate a copied point, enforce the call budget and validate residual shape and finiteness.

        Record the call when logging is enabled. Return a detached residual vector and
        cache its norm; raise DomainError for recognized undefined/overflowing residuals
        and let unexpected application errors propagate.
        """
        nonlocal nfev, residual_size, last_eval_norm
        if nfev >= options.max_nfev:
            raise _BudgetExhausted
        nfev += 1
        record = None
        if store_history:
            record = dict(kind=kind, valid=False, residual_norm=None, nfev=nfev)
            if store_coordinates:
                record["x"] = point.copy()
            evaluations.append(record)
        try:
            values = np.asarray(fun(point.copy()), dtype=float)
            if values.ndim != 1 or values.size < x.size:
                raise ValueError(
                    "fun must return a vector with at least len(x0) entries."
                )
            if residual_size is not None and values.size != residual_size:
                raise ValueError("Residual size changed during the solve.")
            if not np.all(np.isfinite(values)):
                raise DomainError("Nonfinite residual at model evaluation.")
            residual_norm = float(norm(values, check_finite=False))
            if residual_norm > np.sqrt(np.finfo(float).max):
                raise DomainError(
                    "Residual is too large for a finite squared-norm merit."
                )
        except DomainError as error:
            if record is not None:
                record["reason"] = str(error)
            raise
        residual_size = values.size
        last_eval_norm = residual_norm
        if record is not None:
            record.update(valid=True, residual_norm=residual_norm)
        return values.copy()

    def finish(status, message):
        """Assemble an OptimizeResult for the last accepted state, including counters,
        histories and step audits.
        """
        return OptimizeResult(
            x=x.copy(),
            fun=None if f is None else f.copy(),
            success=status == 0,
            status=status,
            message=message,
            method=method,
            nfev=nfev,
            njev=njev,
            nit=nit,
            nupdates=nupdates,
            ntrials=ntrials,
            nrejected=nrejected,
            ninvalid=ninvalid,
            history=history,
            trials=trials,
            evaluations=evaluations,
            radius=radius,
            residual_norm=f_norm,
            history_level=options.history_level,
            max_trial_step_scaled=max_trial_step_scaled,
            max_radius_used=max_radius_used,
            max_radius_ratio=max_radius_ratio,
            linear_algebra=dict(linear_stats),
        )

    def record_accepted(step_norm=0.0, scaled_step_norm=0.0):
        """Store the current accepted diagnostics and return True if the detached
        callback requests a stop.
        """
        if not store_history and callback is None:
            return False
        record = dict(
            iteration=nit,
            residual_norm=f_norm,
            step_norm=step_norm,
            scaled_step_norm=scaled_step_norm,
            radius=radius,
            nfev=nfev,
        )
        if store_coordinates:
            record["x"] = x.copy()
        if store_history:
            history.append(record)
        if callback is not None:
            callback_record = deepcopy(record)
            if "x" not in callback_record:
                callback_record["x"] = x.copy()
            return callback(callback_record) is False
        return False

    try:
        f = evaluate(x, "initial")
        f_norm = last_eval_norm
    except DomainError as error:
        return finish(4, f"Undefined initial point: {error}")
    if record_accepted():
        return finish(5, "Stopped by callback.")
    if f_norm <= options.residual_tol:
        return finish(0, "Residual tolerance satisfied.")

    x_norm = float(norm(x / scale))
    model = None
    model_is_fresh = False
    poor_agreements = 0
    # A pass attempts one trial. Rejection retains the accepted point; the
    # current model can still be reused or improved using a valid secant pair.
    while ntrials < options.max_iterations:
        trial_number = ntrials + 1
        if model is None:
            # Reevaluate derivatives only when no reusable local model remains.
            if nfev >= options.max_nfev:
                return finish(1, "Residual evaluation budget exhausted.")
            njev += 1
            try:
                if jac is not None:
                    jacobian = np.asarray(jac(x.copy()), dtype=float)
                else:
                    jacobian = np.empty((f.size, x.size))
                    for j in range(x.size):
                        probe = x.copy()
                        probe[j] += options.finite_diff_step * scale[j]
                        distance = probe[j] - x[j]
                        if distance == 0 or not np.isfinite(probe[j]):
                            return finish(
                                3, "Finite-difference probe cannot be represented."
                            )
                        jacobian[:, j] = (
                            evaluate(probe, "finite_difference") - f
                        ) / distance
            except _BudgetExhausted:
                return finish(
                    1, "Residual evaluation budget exhausted during derivatives."
                )
            except DomainError as error:
                return finish(4, f"Undefined derivative evaluation: {error}")
            if jacobian.shape != (f.size, x.size):
                raise ValueError(f"jac must have shape {(f.size, x.size)}.")
            if not np.all(np.isfinite(jacobian)):
                return finish(
                    4, "Nonfinite Jacobian; derivative recovery is not enabled."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_jacobian = jacobian * scale
                gradient = scaled_jacobian.T @ f
            if not np.all(np.isfinite(scaled_jacobian)) or not np.all(
                np.isfinite(gradient)
            ):
                return finish(
                    3,
                    "Scaled linear model overflowed; revise variable/residual scaling.",
                )
            if norm(gradient, np.inf) <= options.gtol:
                return finish(
                    2, "Stationary least-squares point above residual tolerance."
                )
            model = LinearModel(scaled_jacobian, f, stats=linear_stats)
            model_is_fresh = True
            poor_agreements = 0

        if norm(model.g, np.inf) <= options.gtol:
            if not model_is_fresh:
                model = None
                continue
            return finish(2, "Stationary least-squares point above residual tolerance.")

        scaled_step = model.step(method, radius)
        # Bound before evaluation, including rounding in the subproblem solve.
        size = norm(scaled_step)
        if size > radius:
            scaled_step *= np.nextafter(radius, 0.0) / size
        with np.errstate(over="ignore", invalid="ignore"):
            trial_x = x + scale * scaled_step
        if not np.all(np.isfinite(trial_x)):
            return finish(3, "Trial coordinates overflowed; revise x_scale or radius.")
        step = trial_x - x
        scaled_step = step / scale
        scaled_norm = float(norm(scaled_step))
        # Coordinate rounding can exceed the radius slightly. Shrink again;
        # if the smallest representable move exceeds it, report stagnation.
        for _ in range(3):
            if scaled_norm <= radius * (1 + 8 * np.finfo(float).eps):
                break
            # Leave room for coordinate addition, whose ulp depends on x,
            # not on the step. Merely nudging radius by one ulp can repeat
            # the identical out-of-radius point when |x/scale| is large.
            coordinate_ulp = np.spacing(np.maximum(np.abs(x), np.abs(trial_x)))
            reserve = 2 * norm(coordinate_ulp / scale)
            scaled_step *= max(0.0, radius - reserve) / scaled_norm
            trial_x = x + scale * scaled_step
            step = trial_x - x
            scaled_step = step / scale
            scaled_norm = float(norm(scaled_step))
        if scaled_norm > radius * (1 + 8 * np.finfo(float).eps):
            return finish(3, "No representable trial within the requested radius.")
        if scaled_norm <= options.xtol * (options.xtol + x_norm) or scaled_norm == 0:
            if not model_is_fresh:
                model = None
                continue
            return finish(3, "Step stagnation above residual tolerance.")
        model_change = model.a @ scaled_step
        predicted = float(
            -np.dot(f, model_change) - 0.5 * np.dot(model_change, model_change)
        )
        if not np.isfinite(predicted) or predicted <= 0:
            if not model_is_fresh:
                model = None
                continue
            return finish(
                3, "No positive predicted reduction above residual tolerance."
            )
        try:
            trial_f = evaluate(trial_x, "trial")
            # Difference of squared norms avoids overflowing an intermediate
            # dot product even when both individual merit values are finite.
            trial_norm = last_eval_norm
            actual = float((0.5 * (f_norm - trial_norm)) * (f_norm + trial_norm))
            ratio = actual / predicted
            valid, reason = True, "model reduction"
        except _BudgetExhausted:
            return finish(1, "Residual evaluation budget exhausted.")
        except DomainError as error:
            trial_f, trial_norm, actual, ratio = None, None, None, None
            valid, reason = False, str(error)
        accepted = valid and actual > 0 and ratio >= options.acceptance_ratio
        ntrials += 1
        nrejected += not accepted
        ninvalid += not valid
        max_trial_step_scaled = max(max_trial_step_scaled, scaled_norm)
        max_radius_used = max(max_radius_used, radius)
        max_radius_ratio = max(max_radius_ratio, scaled_norm / radius)
        physical_step_norm = (
            float(norm(step, check_finite=False))
            if store_history or callback is not None
            else 0.0
        )
        if store_history:
            record = dict(
                trial=trial_number,
                iteration=nit,
                accepted=bool(accepted),
                valid=valid,
                reason=reason,
                residual_norm=trial_norm,
                step_norm=physical_step_norm,
                scaled_step_norm=scaled_norm,
                radius=radius,
                ratio=ratio,
                predicted_reduction=predicted,
                actual_reduction=actual,
                nfev=nfev,
                njev=njev,
                jacobian_source="evaluated" if model_is_fresh else "broyden",
            )
            if store_coordinates:
                record.update(x_base=x.copy(), x=trial_x.copy())
            trials.append(record)
        old_f = f
        used_radius = radius
        if not valid or ratio < 0.25:
            radius *= options.shrink_factor
        elif ratio >= 0.75 and scaled_norm >= 0.8 * used_radius:
            radius = min(options.max_radius, options.growth_factor * used_radius)
        if accepted:
            x, f = trial_x, trial_f
            f_norm = trial_norm
            x_norm = float(norm(x / scale))
            nit += 1
            if record_accepted(physical_step_norm, scaled_norm):
                return finish(5, "Stopped by callback.")
            if f_norm <= options.residual_tol:
                return finish(0, "Residual tolerance satisfied.")
        if method == "hybr" and valid:
            # Broyden's secant update uses the displacement and residual
            # difference from the OLD accepted base, even on a rejected trial.
            # In fixed scaled coordinates this is MINPACK's weighted update.
            poor_agreements = poor_agreements + 1 if ratio < 0.1 else 0
            if poor_agreements >= 2:
                model = None
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    correction = (trial_f - old_f - model_change) / scaled_norm
                if model.update(correction, scaled_step / scaled_norm, f):
                    model_is_fresh = False
                    nupdates += 1
                else:
                    model = None
        elif method == "lm" and accepted:
            model = None
        if radius <= np.finfo(float).tiny:
            return finish(3, "Trust radius underflow above residual tolerance.")
    return finish(1, "Trial iteration budget exhausted.")


METHOD_ALIASES = {
    "lm": "lm",
    "levenberg_marquardt": "lm",
    "hybr": "hybr",
    "powell_hybrid": "hybr",
    "newton_raphson": "newton_raphson",
    "newton": "newton_raphson",
    "gauss_newton": "gauss_newton",
    "gn": "gauss_newton",
}

METHOD_DISPLAY_NAMES = {
    "lm": "Levenberg–Marquardt",
    "hybr": "Powell hybrid",
    "newton_raphson": "Newton–Raphson",
    "gauss_newton": "Gauss–Newton",
}


def normalize_method(method):
    """Resolve one documented root-method spelling to its canonical identifier."""
    if not isinstance(method, str) or method.strip().lower() not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown root method; choose from {', '.join(METHOD_ALIASES)}."
        )
    return METHOD_ALIASES[method.strip().lower()]


def solve_root(fun, x0, jac=None, *, method="lm", options=None, callback=None):
    """Solve a root problem using an experimental method and explicit step policy.

    lm/levenberg_marquardt and hybr/powell_hybrid use StepControlOptions and
    trust regions. newton_raphson/newton and gauss_newton/gn use NewtonOptions;
    backtracking is enabled by default and explicit backtracking=None selects
    full valid steps, including residual increases. Newton requires a square
    numerically full-rank Jacobian; the other methods allow m >= n.

    Dictionaries and matching option dataclasses are accepted. Numerical
    derivatives, residual budgets, copied callbacks, history levels and
    residual-based root success are consistent across drivers. Callbacks receive
    full detached initial/committed records and stop on an exact False return.
    Results preserve the last committed point and separately report attempted
    steps, model calls, canonical method, and requested spelling. Undefined
    initial points or derivative probes terminate; programming errors propagate.
    """
    canonical = normalize_method(method)
    driver = (
        solve_newton
        if canonical in ("newton_raphson", "gauss_newton")
        else _solve_trust_region
    )
    result = driver(fun, x0, jac, method=canonical, options=options, callback=callback)
    result.requested_method = method
    result.method_name = METHOD_DISPLAY_NAMES[canonical]
    return result
