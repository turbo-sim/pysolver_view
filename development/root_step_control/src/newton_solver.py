"""Newton–Raphson and Gauss–Newton with full steps or protected backtracking.

Both methods rebuild their Jacobian after each committed point and reuse the
same factors for rejected line-search trials. They never silently switch to LM
or Broyden. All accepted states, attempted steps and residual calls use the same
accounting conventions as the neighboring trust-region experiments.
"""

from copy import deepcopy

import numpy as np
from scipy.linalg import norm
from scipy.optimize import OptimizeResult

from .line_search import prepare_trial
from .linear_model import LinearModel
from .newton_options import NewtonOptions
from .solver_common import BudgetExhausted, DomainError


def solve_newton(fun, x0, jac=None, *, method, options=None, callback=None):
    """Solve with a strict Newton or minimum-norm Gauss–Newton direction.

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
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite vector.")
    scale = np.asarray(options.x_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(x.size, scale.item())
    if scale.shape != x.shape:
        raise ValueError("x_scale must be scalar or have the same shape as x0.")
    scale = scale.copy()
    search = options.backtracking
    limits = None if search is None else search.step_limits
    step_limit = None if limits is None else limits.initial_limit
    step_policy = "full_step" if search is None else "backtracking"
    history, trials, evaluations = [], [], []
    store_history = options.history_level != "none"
    store_coordinates = options.history_level == "full"
    nfev = njev = nit = ntrials = nrejected = ninvalid = nbacktracks = 0
    max_trial_step_scaled = max_step_limit_used = max_step_limit_ratio = 0.0
    linear_stats = {}
    f = f_norm = last_eval_norm = residual_size = None
    last_rank = linear_residual_norm = None

    def evaluate(point, kind):
        """Count one copied model call and validate fixed residual shape and range."""
        nonlocal nfev, last_eval_norm, residual_size
        if nfev >= options.max_nfev:
            raise BudgetExhausted
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
            if method == "newton_raphson" and values.size != x.size:
                raise ValueError(
                    "Newton–Raphson requires a square residual/Jacobian pair."
                )
            if not np.all(np.isfinite(values)):
                raise DomainError("Nonfinite residual at model evaluation.")
            last_eval_norm = float(norm(values, check_finite=False))
            if last_eval_norm > np.sqrt(np.finfo(float).max):
                raise DomainError(
                    "Residual is too large for a finite squared-norm merit."
                )
        except DomainError as error:
            if record is not None:
                record["reason"] = str(error)
            raise
        residual_size = values.size
        if record is not None:
            record.update(valid=True, residual_norm=last_eval_norm)
        return values.copy()

    def finish(status, message):
        """Return detached accepted values and counters without fabricating a radius."""
        return OptimizeResult(
            x=x.copy(),
            fun=None if f is None else f.copy(),
            success=status == 0,
            status=status,
            message=message,
            method=method,
            step_policy=step_policy,
            nfev=nfev,
            njev=njev,
            nit=nit,
            nupdates=0,
            ntrials=ntrials,
            nrejected=nrejected,
            ninvalid=ninvalid,
            nbacktracks=nbacktracks,
            history=history,
            trials=trials,
            evaluations=evaluations,
            radius=None,
            step_limit=step_limit,
            residual_norm=f_norm,
            history_level=options.history_level,
            max_trial_step_scaled=max_trial_step_scaled,
            max_step_limit_used=None if limits is None else max_step_limit_used,
            max_step_limit_ratio=None if limits is None else max_step_limit_ratio,
            numerical_rank=last_rank,
            linear_residual_norm=linear_residual_norm,
            linear_algebra=dict(linear_stats),
        )

    def record_accepted(
        step_norm=0.0, scaled_step_norm=0.0, fraction=0.0, backtracks=0
    ):
        """Record a committed point and call an isolated full-record callback."""
        if not store_history and callback is None:
            return False
        record = dict(
            iteration=nit,
            residual_norm=f_norm,
            step_norm=step_norm,
            scaled_step_norm=scaled_step_norm,
            step_limit=step_limit,
            step_fraction=fraction,
            backtracks=backtracks,
            step_policy=step_policy,
            nfev=nfev,
            njev=njev,
        )
        if store_coordinates:
            record["x"] = x.copy()
        if store_history:
            history.append(record)
        if callback is not None:
            copied = deepcopy(record)
            copied["x"] = x.copy()
            return callback(copied) is False
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

    while ntrials < options.max_iterations:
        if nfev >= options.max_nfev:
            return finish(1, "Residual evaluation budget exhausted.")
        njev += 1
        try:
            if jac is not None:
                jacobian = np.asarray(jac(x.copy()), dtype=float)
            else:
                jacobian = np.empty((f.size, x.size))
                for column in range(x.size):
                    probe = x.copy()
                    probe[column] += options.finite_diff_step * scale[column]
                    distance = probe[column] - x[column]
                    if distance == 0 or not np.isfinite(probe[column]):
                        return finish(
                            3, "Finite-difference probe cannot be represented."
                        )
                    jacobian[:, column] = (
                        evaluate(probe, "finite_difference") - f
                    ) / distance
        except BudgetExhausted:
            return finish(1, "Residual evaluation budget exhausted during derivatives.")
        except DomainError as error:
            return finish(4, f"Undefined derivative evaluation: {error}")
        if jacobian.shape != (f.size, x.size):
            raise ValueError(f"jac must have shape {(f.size, x.size)}.")
        if not np.all(np.isfinite(jacobian)):
            return finish(4, "Nonfinite Jacobian; derivative recovery is not enabled.")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_jacobian = jacobian * scale
        if not np.all(np.isfinite(scaled_jacobian)):
            return finish(3, "Scaled linear model overflowed; revise variable scaling.")
        try:
            model = LinearModel(
                scaled_jacobian, f, stats=linear_stats, compute_gradient=False
            )
            last_rank = model.numerical_rank
            direction = (
                model.newton_raphson.copy()
                if method == "newton_raphson"
                else model.gauss_newton.copy()
            )
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
        if method == "gauss_newton" and norm(
            model_change, check_finite=False
        ) <= options.gtol * min(1.0, f_norm):
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
            if nfev >= options.max_nfev:
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
                trial_norm = last_eval_norm
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
                    nfev=nfev,
                    njev=njev,
                    jacobian_source="evaluated",
                )
                if store_coordinates:
                    record.update(x_base=x.copy(), x=trial_x.copy())
                trials.append(record)
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
            if nfev >= options.max_nfev:
                return finish(1, "Residual evaluation budget exhausted.")
            if backtracks == reductions_allowed:
                return finish(6, "Backtracking exhausted above residual tolerance.")
            fraction = actual_fraction * search.contraction_factor
    return finish(1, "Trial iteration budget exhausted.")
