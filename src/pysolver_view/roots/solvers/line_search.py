"""Newton-Raphson and Gauss-Newton with full steps or protected backtracking.

Both methods rebuild their Jacobian after each committed point and reuse the
same factors for rejected line-search trials. They never silently switch to LM
or Broyden. All accepted states, attempted steps and residual calls use the same
accounting conventions as the neighboring trust-region experiments.
"""

import numpy as np
from scipy.linalg import norm

from .trial_step import prepare_trial
from .linear_model import LinearModel
from .options import NewtonOptions
from .exceptions import BudgetExhausted, DomainError, UnrepresentableProbe
from .evaluation import ModelEvaluator, prepare_inputs
from .history import HistoryRecorder
from .scaling import JacobianScaling


def solve_newton_raphson(
    fun, x0, jac=None, *, options=None, callback=None, observer=None
):
    """Solve a square system with strict Newton directions and optional backtracking.

    A numerically singular Jacobian terminates rather than selecting a truncated
    least-squares direction. NewtonOptions (or a matching dictionary) selects
    protected backtracking by default; backtracking=None selects full steps.
    No least-squares stationarity test is applied to Newton directions.
    """
    return _solve_line_search(
        fun,
        x0,
        jac,
        method="newton_raphson",
        compute_direction=_newton_direction,
        requires_square=True,
        stationary=None,
        options=options,
        callback=callback,
        observer=observer,
    )


def solve_gauss_newton(
    fun, x0, jac=None, *, options=None, callback=None, observer=None
):
    """Solve a square or overdetermined system with least-squares directions.

    Guarded QR and minimum-norm SVD handle full and deficient numerical rank.
    NewtonOptions (or a matching dictionary) controls backtracking/full steps.
    A stationary least-squares point above the residual tolerance is a failure,
    not a root; its projected residual change supplies the stationarity test.
    """
    return _solve_line_search(
        fun,
        x0,
        jac,
        method="gauss_newton",
        compute_direction=_gauss_newton_direction,
        requires_square=False,
        stationary=_least_squares_stationary,
        options=options,
        callback=callback,
        observer=observer,
    )


def _newton_direction(model):
    """Copy a strict square full-rank Newton direction from the factored model."""
    return model.newton_raphson.copy()


def _gauss_newton_direction(model):
    """Copy the model's least-squares direction, including rank-deficient fallback."""
    return model.gauss_newton.copy()


def _least_squares_stationary(model_change, residual_norm, tolerance):
    """Detect negligible projected residual change without a scale-sensitive gradient."""
    return norm(model_change, check_finite=False) <= tolerance * min(1.0, residual_norm)


def _solve_line_search(
    fun,
    x0,
    jac,
    *,
    method,
    compute_direction,
    requires_square,
    stationary,
    options,
    callback,
    observer,
):
    """Execute full steps or Armijo recovery with supplied direction and rank rules.

    compute_direction(model) supplies the method's direction. requires_square
    controls the residual shape contract; an optional stationary function checks
    the projected change. method labels results only and selects no algorithm.

    Options may be NewtonOptions or a nested dictionary. Backtracking=None
    commits every valid full step; otherwise Armijo decrease is required.
    Only recognized DomainError and nonfinite model values permit trial retry.
    Failures return the last accepted point, not the last attempted point.
    """
    if options is None:
        options = NewtonOptions()
    elif isinstance(options, dict):
        options = NewtonOptions(**options)
    if not isinstance(options, NewtonOptions):
        raise TypeError("options must be NewtonOptions or a dictionary.")
    x, scale = prepare_inputs(x0, options.x_scale)
    scaling = JacobianScaling(x.size, options.jacobian_scaling)
    search = options.backtracking
    limits = None if search is None else search.step_limits
    step_limit = None if limits is None else limits.initial_limit
    step_policy = "full_step" if search is None else "backtracking"
    recorder = HistoryRecorder(options.history_level, callback, observer)
    evaluator = ModelEvaluator(
        fun, jac, scale, options, recorder, square=requires_square
    )
    evaluate = evaluator.residual
    store_history = options.history_level != "none"
    nit = ntrials = nrejected = ninvalid = nbacktracks = 0
    max_trial_step_scaled = max_step_limit_used = max_step_limit_ratio = 0.0
    linear_stats = {}
    f = f_norm = None
    last_rank = linear_residual_norm = None

    def finish(status, message):
        """Return detached accepted values and counters without fabricating a radius."""
        return recorder.result(
            x=x,
            fun=f,
            status=status,
            message=message,
            method=method,
            step_policy=step_policy,
            nfev=evaluator.nfev,
            njev=evaluator.njev,
            nit=nit,
            nupdates=0,
            ntrials=ntrials,
            nrejected=nrejected,
            ninvalid=ninvalid,
            nbacktracks=nbacktracks,
            radius=None,
            step_limit=step_limit,
            residual_norm=f_norm,
            max_trial_step_scaled=max_trial_step_scaled,
            max_step_limit_used=None if limits is None else max_step_limit_used,
            max_step_limit_ratio=None if limits is None else max_step_limit_ratio,
            numerical_rank=last_rank,
            linear_residual_norm=linear_residual_norm,
            linear_algebra=linear_stats,
            **scaling.diagnostics(),
        )

    def record_accepted(
        step_norm=0.0, scaled_step_norm=0.0, fraction=0.0, backtracks=0
    ):
        """Record a committed point and call an isolated full-record callback."""
        if not store_history and callback is None:
            return False
        return recorder.accepted(
            x,
            f,
            iteration=nit,
            residual_norm=f_norm,
            step_norm=step_norm,
            scaled_step_norm=scaled_step_norm,
            step_limit=step_limit,
            step_fraction=fraction,
            backtracks=backtracks,
            step_policy=step_policy,
            nfev=evaluator.nfev,
            njev=evaluator.njev,
        )

    try:
        f = evaluate(x, "initial")
        f_norm = evaluator.last_norm
    except DomainError as error:
        return finish(4, f"Undefined initial point: {error}")
    if record_accepted():
        return finish(5, "Stopped by callback.")
    if f_norm <= options.residual_tol:
        return finish(0, "Residual tolerance satisfied.")

    while ntrials < options.max_iterations:
        if evaluator.nfev >= options.max_nfev:
            return finish(1, "Residual evaluation budget exhausted.")
        try:
            jacobian = evaluator.jacobian(x, f)
        except UnrepresentableProbe as error:
            return finish(3, str(error))
        except BudgetExhausted:
            return finish(1, "Residual evaluation budget exhausted during derivatives.")
        except DomainError as error:
            return finish(4, f"Undefined derivative evaluation: {error}")
        if not np.all(np.isfinite(jacobian)):
            return finish(4, "Nonfinite Jacobian; derivative recovery is not enabled.")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_jacobian = jacobian * scale
        if not np.all(np.isfinite(scaled_jacobian)):
            return finish(3, "Scaled linear model overflowed; revise variable scaling.")
        factors = scaling.update(scaled_jacobian)
        try:
            model = LinearModel(
                scaled_jacobian * factors, f, stats=linear_stats, compute_gradient=False
            )
            last_rank = model.numerical_rank
            # Factor in equilibrated coordinates, then apply all line-search and
            # physical step controls to the original caller-scaled displacement.
            direction = factors * compute_direction(model)
        except np.linalg.LinAlgError as error:
            return finish(7, f"Newton linear solve failed: {error}")
        except ValueError as error:
            return finish(3, f"Unusable linear model: {error}")
        raw_length = float(norm(direction, check_finite=False))
        with np.errstate(over="ignore", invalid="ignore"):
            model_change = scaled_jacobian @ direction
        if not np.isfinite(raw_length) or not np.all(np.isfinite(model_change)):
            return finish(3, "Newton model change exceeds numerical range.")
        linear_residual_norm = float(norm(f + model_change, check_finite=False))
        # A nearly zero projection identifies least-squares stationarity without
        # letting small Jacobian units alone stop an otherwise useful direction.
        if stationary is not None and stationary(model_change, f_norm, options.gtol):
            return finish(2, "Stationary least-squares point above residual tolerance.")
        if raw_length == 0:
            return finish(3, "Zero direction above residual tolerance.")
        if search is not None:
            slope = float(np.dot(f, model_change))
            if not np.isfinite(slope) or slope >= 0:
                return finish(
                    3, "No finite descent direction above residual tolerance."
                )

        fraction = 1.0 if step_limit is None else min(1.0, step_limit / raw_length)
        reductions_allowed = 0 if search is None else search.max_backtracks
        for backtracks in range(reductions_allowed + 1):
            if ntrials >= options.max_iterations:
                return finish(1, "Trial iteration budget exhausted.")
            if evaluator.nfev >= options.max_nfev:
                return finish(1, "Residual evaluation budget exhausted.")
            if backtracks:
                nbacktracks += 1
            try:
                trial_x, step, scaled_step, scaled_norm, actual_fraction = (
                    prepare_trial(x, scale, direction, fraction, step_limit)
                )
            except ValueError as error:
                return finish(3, str(error))
            with np.errstate(over="ignore", invalid="ignore"):
                x_norm = float(norm(x / scale, check_finite=False))
            # Relative stagnation is meaningful only when its scale is finite.
            threshold = (
                options.xtol * (options.xtol + x_norm) if np.isfinite(x_norm) else 0.0
            )
            if scaled_norm == 0 or scaled_norm <= threshold:
                return finish(3, "Step stagnation above residual tolerance.")
            required = directional_derivative = None
            if search is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    directional_derivative = float(
                        np.dot(f, scaled_jacobian @ scaled_step)
                    )
                if (
                    not np.isfinite(directional_derivative)
                    or directional_derivative >= 0
                ):
                    return finish(3, "Rounded trial has no finite descent prediction.")
                required = -search.armijo_c1 * directional_derivative
            try:
                trial_f = evaluate(trial_x, "trial")
                trial_norm = evaluator.last_norm
                actual = float((0.5 * (f_norm - trial_norm)) * (f_norm + trial_norm))
                valid = True
                accepted = search is None or (actual > 0 and actual >= required)
                reason = (
                    "full step"
                    if search is None
                    else (
                        "Armijo decrease"
                        if accepted
                        else "insufficient Armijo decrease"
                    )
                )
            except BudgetExhausted:
                return finish(1, "Residual evaluation budget exhausted.")
            except DomainError as error:
                trial_f = trial_norm = actual = None
                valid = accepted = False
                reason = str(error)
            ntrials += 1
            nrejected += not accepted
            ninvalid += not valid
            max_trial_step_scaled = max(max_trial_step_scaled, scaled_norm)
            if step_limit is not None:
                max_step_limit_used = max(max_step_limit_used, step_limit)
                max_step_limit_ratio = max(
                    max_step_limit_ratio, scaled_norm / step_limit
                )
            physical_norm = float(norm(step, check_finite=False))
            if store_history:
                record = dict(
                    trial=ntrials,
                    iteration=nit,
                    accepted=bool(accepted),
                    valid=valid,
                    reason=reason,
                    residual_norm=trial_norm,
                    step_norm=physical_norm,
                    scaled_step_norm=scaled_norm,
                    step_limit=step_limit,
                    step_fraction=actual_fraction,
                    backtracks=backtracks,
                    step_policy=step_policy,
                    directional_derivative=directional_derivative,
                    armijo_required_reduction=required,
                    actual_reduction=actual,
                    nfev=evaluator.nfev,
                    njev=evaluator.njev,
                    jacobian_source="evaluated",
                )
                recorder.trial(x, trial_x, record)
            if accepted:
                if step_limit is not None:
                    if backtracks:
                        step_limit = min(step_limit, scaled_norm)
                    elif scaled_norm >= 0.8 * step_limit:
                        step_limit = min(
                            limits.max_limit, limits.growth_factor * step_limit
                        )
                x, f, f_norm = trial_x, trial_f, trial_norm
                nit += 1
                if record_accepted(
                    physical_norm, scaled_norm, actual_fraction, backtracks
                ):
                    return finish(5, "Stopped by callback.")
                if f_norm <= options.residual_tol:
                    return finish(0, "Residual tolerance satisfied.")
                break
            if search is None:
                return finish(4, f"Undefined full-step trial: {reason}")
            if ntrials >= options.max_iterations:
                return finish(1, "Trial iteration budget exhausted.")
            if evaluator.nfev >= options.max_nfev:
                return finish(1, "Residual evaluation budget exhausted.")
            if backtracks == reductions_allowed:
                return finish(6, "Backtracking exhausted above residual tolerance.")
            fraction = actual_fraction * search.contraction_factor
    return finish(1, "Trial iteration budget exhausted.")
