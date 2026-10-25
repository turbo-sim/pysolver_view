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

import numpy as np
from scipy.linalg import norm

from ..options import StepControlOptions
from ..exceptions import BudgetExhausted, DomainError, UnrepresentableProbe
from .linear_model import LinearModel
from .evaluation import ModelEvaluator, prepare_inputs
from .history import HistoryRecorder
from .trial_step import prepare_trust_region_trial


def solve_levenberg_marquardt(fun, x0, jac=None, *, options=None, callback=None):
    """Solve with damped least-squares steps and a fresh Jacobian after acceptance.

    StepControlOptions (or a matching dictionary) controls the trust region.
    Rejected trials reuse the same factorized model; successful moves request
    a fresh Jacobian. Shared evaluation/history and residual-root contracts
    are implemented by _solve_trust_region().
    """
    return _solve_trust_region(
        fun,
        x0,
        jac,
        method="levenberg_marquardt",
        compute_step=LinearModel.levenberg_marquardt_step,
        update_jacobian=None,
        options=options,
        callback=callback,
    )


def solve_powell_hybrid(fun, x0, jac=None, *, options=None, callback=None):
    """Solve with dogleg steps and Broyden updates from valid trials.

    StepControlOptions (or a matching dictionary) controls the trust region.
    Valid rejected trials can also update the model. Two consecutive poor
    agreements or an unusable factor update request a fresh Jacobian; suspect
    models are refreshed before declaring stationarity or stagnation.
    """
    return _solve_trust_region(
        fun,
        x0,
        jac,
        method="powell_hybrid",
        compute_step=LinearModel.powell_hybrid_step,
        update_jacobian=_update_powell_jacobian,
        options=options,
        callback=callback,
    )


def _update_powell_jacobian(
    model,
    *,
    previous_residual,
    trial_residual,
    accepted_residual,
    model_change,
    scaled_step,
    scaled_norm,
    ratio,
    poor_agreements,
):
    """Apply Powell's secant update or request a fresh Jacobian after poor agreement.

    Both accepted and rejected valid trials use the displacement and residual
    difference from the previous accepted base. Return the reusable model (or
    None), the poor-agreement count, and the number of successful updates.
    """
    poor_agreements = poor_agreements + 1 if ratio < 0.1 else 0
    if poor_agreements >= 2:
        return None, poor_agreements, 0
    with np.errstate(over="ignore", invalid="ignore"):
        correction = (trial_residual - previous_residual - model_change) / scaled_norm
    if model.update(correction, scaled_step / scaled_norm, accepted_residual):
        return model, poor_agreements, 1
    return None, poor_agreements, 0


def _solve_trust_region(
    fun,
    x0,
    jac,
    *,
    method,
    compute_step,
    update_jacobian,
    options,
    callback,
):
    """Execute trust-region acceptance and recovery with supplied numerical choices.

    compute_step(model, radius) supplies the chosen subproblem solution.
    update_jacobian=None refreshes the Jacobian after each accepted move;
    otherwise the supplied update function processes valid secant pairs.
    method identifies results only; it does not select an algorithm here.

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

    Unlike native MINPACK HYBRJ, the Powell hybrid entry point uses QR factors
    with SVD fallback, configurable radius/acceptance policies and residual-based
    success. Native slow-progress cutoffs are omitted to allow deliberately small
    steps. nupdates counts Broyden updates; njev counts full Jacobian evaluations.
    Both entry points accept m >= n; SciPy root('hybr') requires square systems.
    """
    if options is None:
        options = StepControlOptions()
    elif isinstance(options, dict):
        options = StepControlOptions(**options)
    if not isinstance(options, StepControlOptions):
        raise TypeError("options must be StepControlOptions or a dictionary.")
    x, scale = prepare_inputs(x0, options.x_scale)
    radius = options.initial_radius
    recorder = HistoryRecorder(options.history_level, callback)
    evaluator = ModelEvaluator(fun, jac, scale, options, recorder)
    evaluate = evaluator.residual
    nit = nupdates = ntrials = nrejected = ninvalid = 0
    max_trial_step_scaled = max_radius_used = max_radius_ratio = 0.0
    linear_stats = {}
    store_history = options.history_level != "none"
    f = None
    f_norm = None

    def finish(status, message):
        """Assemble an OptimizeResult for the last accepted state, including counters,
        histories and step audits.
        """
        return recorder.result(
            x=x,
            fun=f,
            status=status,
            message=message,
            method=method,
            nfev=evaluator.nfev,
            njev=evaluator.njev,
            nit=nit,
            nupdates=nupdates,
            ntrials=ntrials,
            nrejected=nrejected,
            ninvalid=ninvalid,
            radius=radius,
            residual_norm=f_norm,
            max_trial_step_scaled=max_trial_step_scaled,
            max_radius_used=max_radius_used,
            max_radius_ratio=max_radius_ratio,
            linear_algebra=linear_stats,
        )

    def record_accepted(step_norm=0.0, scaled_step_norm=0.0):
        """Store the current accepted diagnostics and return True if the detached
        callback requests a stop.
        """
        if not store_history and callback is None:
            return False
        return recorder.accepted(
            x,
            iteration=nit,
            residual_norm=f_norm,
            step_norm=step_norm,
            scaled_step_norm=scaled_step_norm,
            radius=radius,
            nfev=evaluator.nfev,
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
            if evaluator.nfev >= options.max_nfev:
                return finish(1, "Residual evaluation budget exhausted.")
            try:
                jacobian = evaluator.jacobian(x, f)
            except UnrepresentableProbe as error:
                return finish(3, str(error))
            except BudgetExhausted:
                return finish(
                    1, "Residual evaluation budget exhausted during derivatives."
                )
            except DomainError as error:
                return finish(4, f"Undefined derivative evaluation: {error}")
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

        scaled_step = compute_step(model, radius)
        try:
            trial_x, step, scaled_step, scaled_norm = prepare_trust_region_trial(
                x, scale, scaled_step, radius
            )
        except ValueError as error:
            return finish(3, str(error))
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
            trial_norm = evaluator.last_norm
            actual = float((0.5 * (f_norm - trial_norm)) * (f_norm + trial_norm))
            ratio = actual / predicted
            valid, reason = True, "model reduction"
        except BudgetExhausted:
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
                nfev=evaluator.nfev,
                njev=evaluator.njev,
                jacobian_source="evaluated" if model_is_fresh else "broyden",
            )
            recorder.trial(x, trial_x, record)
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
        if update_jacobian is not None and valid:
            model, poor_agreements, updates = update_jacobian(
                model,
                previous_residual=old_f,
                trial_residual=trial_f,
                accepted_residual=f,
                model_change=model_change,
                scaled_step=scaled_step,
                scaled_norm=scaled_norm,
                ratio=ratio,
                poor_agreements=poor_agreements,
            )
            if updates:
                model_is_fresh = False
                nupdates += updates
        elif update_jacobian is None and accepted:
            model = None
        if radius <= np.finfo(float).tiny:
            return finish(3, "Trust radius underflow above residual tolerance.")
    return finish(1, "Trial iteration budget exhausted.")
