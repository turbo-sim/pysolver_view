"""Run SciPy's SLSQP with controlled trial steps and explicit domain recovery.

This driver manages the evaluation loop around SciPy's compiled optimizer. A
model can be undefined at a trial point even when every variable satisfies its
bounds. On a recognized ``DomainError``, the driver can shorten the pending step
and retry the actual model. Successful recovery can also limit the next step,
with gradual growth toward the configured maximum fraction. SciPy's own merit
line search remains active, and its accumulated solver state is retained.

In the supported SciPy version, ``minimize(method='SLSQP')`` exposes neither
these step fractions nor a hook to reject an undefined pending trial and retry
it with consistent internal state. An iteration callback cannot supply that
control, and an exception from a model evaluation would otherwise escape the
public wrapper. The private interaction is isolated in ``_slsqp_backend``;
this module chooses when to evaluate, shorten, recover, or stop. It uses the
existing SLSQP algorithm rather than implementing another optimizer.

Bound-aware finite differences use one probe per variable. An invalid probe
stops the solve with a diagnostic instead of retrying or inventing a derivative. Histories and callbacks describe the initial and accepted designs,
and ``kkt`` supplies independent first-order diagnostics at those points.
Final success requires native success and the configured feasibility and
optimality limits by default. Disabling check_kkt skips the optimality gate.
These derivative and reporting features could also surround
the public interface; pending-trial control is the reason for the private bridge.

The numerical driver requires NumPy and SciPy and has no application or UI assumptions.
The caller owns coordinates and scaling and returns an ``Evaluation`` containing
a minimized objective, equalities == 0, and inequalities >= 0. Only explicitly
identified ``DomainError`` failures are recoverable. No artificial objective
values or modifications to the model equations are used, and programming errors
propagate to the caller. Recovery has finite retry limits and cannot guarantee
that a feasible continuation exists.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult

from ._slsqp_backend import Backend
from ..diagnostics.kkt import assess_kkt_nonnegative as assess_kkt


class DomainError(Exception):
    """Identify an undefined physical model evaluation that may permit a retry.

    Raise this only when the model recognizes a domain failure at the supplied
    point. Metadata records its cause for reporting; it does not supply substitute
    objective or constraint values. Recovery means trying another point, and can
    still fail if no valid trial step is found within the limits. Derivative
    probes are not retried.
    """

    def __init__(self, message, *, metadata=None):
        """Store a recognized domain failure and detached application metadata."""
        super().__init__(message)
        self.metadata = {} if metadata is None else deepcopy(metadata)

    def to_dict(self):
        """Return detached diagnostic details without exposing internal metadata."""
        return {**self.metadata, "message": str(self), "recoverable": True}


class _CallbackStop(Exception):
    """Distinguish explicit callback cancellation from model exceptions."""


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


MESSAGES = {
    0: "Optimization terminated successfully",
    2: "More equality constraints than independent variables",
    3: "More than 3*n iterations in LSQ subproblem",
    4: "Inequality constraints incompatible",
    5: "Singular matrix E in LSQ subproblem",
    6: "Singular matrix C in LSQ subproblem",
    7: "Rank-deficient equality constraint subproblem HFTI",
    8: "Positive directional derivative for linesearch",
    9: "Iteration limit reached",
    10: "Initial point is outside the model domain",
    11: "No valid line-search trial found",
    12: "No valid finite-difference probe found",
    13: "Stopped by callback",
}


def _format_tolerance_pair(residual, tolerance):
    """Keep nearby failing residuals visibly distinct from their tolerance."""
    values = f"{residual:.6g}", f"{tolerance:.6g}"
    if residual != tolerance and values[0] == values[1]:
        values = f"{residual:.17g}", f"{tolerance:.17g}"
    return values


def _grow_step_fraction(options, fraction):
    """Restore a step cap without overflowing for a large growth factor."""
    cap = options.max_step_fraction
    growth = options.step_growth_factor
    if growth is None:
        return cap
    if fraction == 0:
        return 0.0
    if fraction >= cap / growth:
        return cap
    return fraction * growth


def _relative_step_fraction(step, original_step):
    """Measure cumulative shortening, including native merit backtracking.

    The stored step stays collinear with its original QP direction. Using its
    largest component avoids squaring/norm overflow and tiny denominators.
    SciPy's state alpha is only the last incremental shortening factor.
    """
    index = int(np.argmax(np.abs(original_step)))
    if original_step[index] == 0:
        return 0.0
    return float(step[index] / original_step[index])


def minimize_slsqp(
    evaluate, x0, bounds=None, *, jac=None, options=None, callback=None, logger=None
):
    """Minimize a combined model evaluation through SciPy's compiled SLSQP.

    ``jac`` optionally returns rows [objective, equalities, inequalities].
    Otherwise use bounded, one-sided differences with one probe per variable.
    Choose the direction from the available bound distance before evaluating;
    a DomainError stops differentiation without retrying. No artificial
    objective/constraint penalties are used.

    The two step fractions scale the QP direction before its first trial: the
    first major iteration uses min(first_step_fraction, max_step_fraction), and
    later initial caps grow by step_growth_factor up to max_step_fraction.
    After a domain-recovered iteration, growth restarts from the actual accepted
    fraction of that iteration's original QP direction, including any native
    merit shortening. Further recoveries reset it again. Growth 1 holds the current/recovered cap; None
    restores max_step_fraction immediately on each iteration after the first.
    Native merit backtracking remains active in both cases.
    Domain backtracking is applied to every native function request. A final
    first-order stationarity check prevents tiny imposed steps from being
    mistaken for convergence merely because SciPy's step tolerance is met.

    Callback/history contain initial and accepted points (not derivative probes
    or rejected trials). StopIteration from callback gives status 13. Failure
    during domain recovery returns the last accepted point with success=False.
    """
    options = SLSQPOptions() if options is None else options
    if isinstance(options, dict):
        options = SLSQPOptions(**options)
    if not isinstance(options, SLSQPOptions):
        raise TypeError("options must be SLSQPOptions or a dictionary.")
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite vector.")
    n = x.size
    if bounds is None:
        lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    else:
        pairs = np.asarray(bounds, dtype=float)
        if pairs.shape != (n, 2) or np.any(np.isnan(pairs)):
            raise ValueError(
                "bounds must have shape (n, 2), using infinities for absent bounds."
            )
        lower, upper = pairs[:, 0].copy(), pairs[:, 1].copy()
    if np.any(lower >= upper):
        raise ValueError(
            "Each lower bound must be below its upper bound; omit fixed variables."
        )
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("x0 must lie within its bounds.")
    started = perf_counter()
    history, events, cache = [], [], OrderedDict()
    counts = dict(nfev=0, njev=0, valid_evaluations=0)
    sizes = None
    backend = None
    last_x, last_eval = x.copy(), None
    derivative_cache = None
    next_step_fraction = min(options.first_step_fraction, options.max_step_fraction)
    iteration_step_fraction = next_step_fraction
    original_step = None
    recovered_iteration = False
    accepted_fraction = None

    def log(message, *args):
        """Send a formatted diagnostic to the optional application logger."""
        if logger is not None:
            logger.info(message, *args)

    log("Starting pysolver SLSQP optimization (SciPy backend)")
    log("Func-value is the signed, normalized minimized objective.")
    log(
        "Step-norm is the accepted displacement in solver coordinates; KKT uses derivatives at the reported point."
    )
    log(
        "Tolerances: native ftol=%g; feasibility=%g; KKT optimality=%g. KKT/tol <= 1 passes the independent check.",
        options.ftol,
        options.feasibility_tol,
        options.optimality_tol,
    )
    log(
        "%5s %9s %9s %14s %12s %12s %12s %12s %10s %8s",
        "Iter",
        "Grad-eval",
        "Func-eval",
        "Func-value",
        "Infeasibility",
        "Step-norm",
        "Optimality",
        "Complement.",
        "KKT/tol",
        "Recoveries",
    )

    def record_event(event):
        """Retain trial or derivative rejection metadata and log its cause."""
        events.append(event)
        if event["kind"] in ("trial_rejection", "derivative_rejection"):

            log(
                "Rejected %s: %s",
                "trial" if event["kind"] == "trial_rejection" else "derivative probe",
                event["message"],
            )

    def model(point):
        """Evaluate one exact point, validate responses, and cache detached values."""
        nonlocal sizes
        key = point.tobytes()
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        counts["nfev"] += 1
        raw = evaluate(point.copy())
        result = Evaluation(
            float(raw.objective),
            np.atleast_1d(raw.equalities).astype(float),
            np.atleast_1d(raw.inequalities).astype(float),
        )
        if result.equalities.ndim != 1 or result.inequalities.ndim != 1:
            raise ValueError("Constraints must be one-dimensional arrays.")
        if not np.all(np.isfinite(result.vector())):
            raise ValueError(
                "Model returned nonfinite values; use DomainError for known domain failures."
            )
        current_sizes = (result.equalities.size, result.inequalities.size)
        if sizes is not None and sizes != current_sizes:
            raise ValueError("Constraint sizes changed during optimization.")
        sizes = current_sizes
        counts["valid_evaluations"] += 1
        cache[key] = result
        if len(cache) > 128:
            cache.popitem(last=False)
        return result

    def violation(point, value):
        """Compute maximum equality, inequality, and finite-bound infeasibility."""
        if value is None:
            return float("inf")
        return max(
            float(np.max(np.abs(value.equalities), initial=0)),
            float(np.max(-value.inequalities, initial=0)),
            float(np.max(lower - point, initial=0)),
            float(np.max(point - upper, initial=0)),
        )

    kkt_stats = dict(evaluations=0, time=0.0)

    def timed_assessment(*args, **kwargs):
        """Measure independent KKT work separately from numerical solver work."""
        started_check = perf_counter()
        try:
            return assess_kkt(*args, **kwargs)
        finally:
            kkt_stats["evaluations"] += 1
            kkt_stats["time"] += perf_counter() - started_check

    def report(point, value, matrix=None):
        """Record one accepted point and its independent same-point KKT assessment."""
        nonlocal last_x, last_eval
        last_x, last_eval = point.copy(), value
        if history and np.array_equal(history[-1]["x"], point):
            return
        kkt = (
            None
            if matrix is None or not options.check_kkt
            else timed_assessment(
                point,
                matrix[0],
                value.equalities,
                matrix[1 : 1 + value.equalities.size],
                value.inequalities,
                matrix[1 + value.equalities.size :],
                lower,
                upper,
                feasibility_tol=options.feasibility_tol,
                optimality_tol=options.optimality_tol,
            )
        )
        available = kkt is not None and kkt["available"]
        entry = dict(
            iteration=0 if backend is None else int(backend.state["iter"]),
            x=point.copy(),
            objective=value.objective,
            fitness=value.vector().copy(),
            max_constraint_violation=violation(point, value),
            step_norm=(
                0.0 if not history else float(np.linalg.norm(point - history[-1]["x"]))
            ),
            objective_change=(
                0.0
                if not history
                else float(value.objective - history[-1]["objective"])
            ),
            diagnostics_available=available,
            kkt=kkt,
            **{
                key: kkt[key] if available else None
                for key in (
                    "optimality",
                    "stationarity",
                    "complementarity",
                    "dual_feasibility",
                    "kkt_residual",
                    "kkt_ratio",
                )
            },
            domain_rejections=sum(
                event["kind"] == "trial_rejection" for event in events
            ),
            derivative_rejections=sum(
                event["kind"] == "derivative_rejection" for event in events
            ),
            **counts,
        )
        history.append(entry)

        def display(key):
            """Format an available metric or explicitly label missing diagnostics."""
            return "unavailable" if entry[key] is None else f"{entry[key]:.4e}"

        log(
            "%5d %9d %9d %+14.6e %12.4e %12.4e %12s %12s %10s %8d",
            entry["iteration"],
            counts["njev"],
            counts["nfev"],
            value.objective,
            entry["max_constraint_violation"],
            entry["step_norm"],
            display("optimality"),
            display("complementarity"),
            display("kkt_ratio"),
            entry["domain_rejections"],
        )
        if callback is not None:
            try:
                callback(deepcopy(entry))
            except StopIteration as error:
                raise _CallbackStop from error

    def derivative(point, value):
        """Obtain one combined Jacobian with bounded, nonrecovering probes."""
        nonlocal derivative_cache
        if derivative_cache is not None and np.array_equal(derivative_cache[0], point):
            return derivative_cache[1].copy()
        counts["njev"] += 1
        expected_shape = (value.vector().size, n)
        if jac is not None:
            try:
                matrix = np.asarray(jac(point.copy()), dtype=np.float64)
            except DomainError as error:
                # An external derivative owner may include the exact failed
                # probe. Retain its provenance without retrying the derivative.
                record_event(
                    dict(
                        kind="derivative_rejection",
                        column=error.metadata.get("column"),
                        x=np.asarray(error.metadata.get("probe", point)).copy(),
                        message=str(error),
                        failure=error.to_dict(),
                    )
                )
                raise
        else:
            matrix = np.empty(expected_shape)
            base = value.vector()
            for column in range(n):
                length = options.finite_diff_step
                forward_room = upper[column] - point[column]
                backward_room = point[column] - lower[column]
                # Choose a bound-admissible direction before evaluating. This
                # is ordinary one-sided differentiation, not a domain retry.
                direction = (
                    1 if forward_room >= length or forward_room >= backward_room else -1
                )
                room = forward_room if direction == 1 else backward_room
                probe = point.copy()
                probe[column] = np.clip(
                    point[column] + direction * min(length, room),
                    lower[column],
                    upper[column],
                )
                step = probe[column] - point[column]
                if abs(step) <= 8 * np.finfo(float).eps * max(1, abs(point[column])):
                    raise DomainError(
                        f"No representable derivative perturbation for variable {column}."
                    )
                try:
                    sampled = model(probe)
                except DomainError as error:
                    record_event(
                        dict(
                            kind="derivative_rejection",
                            column=column,
                            x=probe.copy(),
                            message=str(error),
                            failure=error.to_dict(),
                        )
                    )
                    raise DomainError(
                        f"Derivative probe for variable {column} failed; probes are not retried: {error}",
                        metadata={
                            "column": column,
                            "probe": probe.tolist(),
                            "cause": error.to_dict(),
                        },
                    ) from error
                matrix[:, column] = (sampled.vector() - base) / step
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Jacobian must be finite with shape {expected_shape}.")
        derivative_cache = point.copy(), matrix.copy()
        return matrix

    def finish(status, detail=""):
        """Build the authoritative last-accepted result and strict acceptance status."""
        value = last_eval
        infeasibility = violation(last_x, value)
        feasible = bool(infeasibility <= options.feasibility_tol)
        message = MESSAGES.get(status, f"SLSQP exit mode {status}")
        if detail:
            message += f": {detail}"
        if status == 0 and not feasible:
            residual_text, tolerance_text = _format_tolerance_pair(
                infeasibility, options.feasibility_tol
            )
            message += (
                f"; final constraint violation {residual_text} exceeds tolerance "
                f"{tolerance_text}."
            )
        last_record = history[-1] if history else {}
        kkt = last_record.get("kkt")
        available = kkt is not None and kkt["available"]
        optimality = kkt["optimality"] if available else float("inf")
        stationarity = kkt["stationarity"] if available else float("inf")
        complementarity = kkt["complementarity"] if available else float("inf")
        stationary = (
            (available and optimality <= options.optimality_tol)
            if options.check_kkt
            else None
        )
        final_jacobian = (
            derivative_cache[1]
            if derivative_cache is not None
            and np.array_equal(derivative_cache[0], last_x)
            else None
        )
        if status == 0 and options.check_kkt:
            if not available:
                message += "; final KKT check unavailable"
                if kkt is not None:
                    message += f": {kkt['message']}"
            elif not stationary:
                residual_text, tolerance_text = _format_tolerance_pair(
                    optimality, options.optimality_tol
                )
                message += (
                    f"; final first-order residual {residual_text} exceeds "
                    f"{tolerance_text} (small steps alone do not establish convergence)."
                )
        result = OptimizeResult(
            x=last_x.copy(),
            fun=float("nan") if value is None else value.objective,
            fitness=None if value is None else value.vector().copy(),
            success=status == 0 and feasible and (not options.check_kkt or stationary),
            kkt_enabled=options.check_kkt,
            kkt_passed=bool(available and kkt["passed"]) if options.check_kkt else None,
            kkt_evaluations=kkt_stats["evaluations"],
            kkt_time=kkt_stats["time"],
            solver_success=status == 0,
            feasible=feasible,
            stationary=stationary,
            optimality=optimality if options.check_kkt else None,
            stationarity=stationarity if options.check_kkt else None,
            complementarity=complementarity if options.check_kkt else None,
            dual_feasibility=(
                kkt["dual_feasibility"]
                if available
                else (float("inf") if options.check_kkt else None)
            ),
            kkt_residual=(
                kkt["kkt_residual"]
                if available
                else (float("inf") if options.check_kkt else None)
            ),
            kkt_ratio=(
                kkt["kkt_ratio"]
                if available
                else (float("inf") if options.check_kkt else None)
            ),
            kkt=kkt,
            kkt_multipliers=kkt["multipliers"] if available else None,
            diagnostics_available=available,
            step_norm=last_record.get("step_norm", 0.0),
            objective_change=last_record.get("objective_change", 0.0),
            jac=None if final_jacobian is None else final_jacobian[0].copy(),
            status=status,
            message=message,
            nit=0 if backend is None else backend.state["iter"],
            history=history,
            events=events,
            max_constraint_violation=infeasibility,
            elapsed_time=perf_counter() - started,
            **counts,
            domain_rejections=sum(
                event["kind"] == "trial_rejection" for event in events
            ),
            derivative_rejections=sum(
                event["kind"] == "derivative_rejection" for event in events
            ),
            multipliers=(
                np.empty(0)
                if backend is None
                else backend.multipliers[: sum(sizes)].copy()
            ),
        )
        log("Success: %s", result.success)
        log("Exit message: %s", result.message)
        if available:
            log(
                "Final KKT: stationarity=%.4e; primal=%.4e; dual=%.4e; complementarity=%.4e; KKT/tol=%.4e (%s)",
                stationarity,
                kkt["primal_feasibility"],
                kkt["dual_feasibility"],
                complementarity,
                kkt["kkt_ratio"],
                "pass" if kkt["passed"] else "fail",
            )
        elif options.check_kkt:
            log("Final KKT: unavailable (no valid same-point derivative certificate)")
        else:
            log("Final KKT: disabled (native termination and feasibility only)")
        log("Solution time: %.3f seconds", result.elapsed_time)
        return result

    try:
        value = model(x)
    except DomainError as error:
        record_event(
            dict(
                kind="initial_rejection",
                x=x.copy(),
                message=str(error),
                failure=error.to_dict(),
            )
        )
        return finish(10, str(error))
    try:
        meq, mineq = sizes
        if meq > n:
            report(x, value)
            return finish(2)
        backend = Backend(
            x, lower, upper, meq, mineq, ftol=options.ftol, maxiter=options.maxiter
        )
        try:
            matrix = derivative(x, value)
        except DomainError as error:
            report(x, value)
            return finish(12, str(error))
        report(x, value, matrix)
        while True:
            mode = backend.advance(value, matrix)
            if mode == 1:
                if backend.state["line"] == 1:
                    fraction = next_step_fraction
                    iteration_step_fraction = fraction
                    original_step = backend.step.copy()
                    recovered_iteration = False
                    accepted_fraction = None
                    if fraction < 1:
                        before = backend.x.copy()
                        backend.shorten_trial(fraction)
                        record_event(
                            dict(
                                kind="step_cap",
                                iteration=backend.state["iter"],
                                fraction=fraction,
                                original_x=before,
                                x=backend.x.copy(),
                            )
                        )
                for attempt in range(options.max_backtracks + 1):
                    try:
                        value = model(backend.x)
                        if (
                            recovered_iteration
                            and options.step_growth_factor is not None
                        ):
                            # Snapshot before advance(), which may overwrite the
                            # workspace when it starts a subsequent QP iteration.
                            accepted_fraction = _relative_step_fraction(
                                backend.step, original_step
                            )
                        break
                    except DomainError as error:
                        record_event(
                            dict(
                                kind="trial_rejection",
                                iteration=backend.state["iter"],
                                native_line=backend.state["line"],
                                backtrack=attempt,
                                x=backend.x.copy(),
                                message=str(error),
                                failure=error.to_dict(),
                            )
                        )
                        if (
                            not options.domain_backtracking
                            or attempt == options.max_backtracks
                        ):
                            return finish(11, str(error))
                        before = backend.x.copy()
                        backend.shorten_trial(options.backtrack_factor)
                        recovered_iteration = True
                        if np.array_equal(before, backend.x) or np.array_equal(
                            backend.base, backend.x
                        ):
                            return finish(
                                11,
                                "Step cannot be shortened further at machine precision.",
                            )
            elif mode == -1:
                next_step_fraction = _grow_step_fraction(
                    options,
                    (
                        accepted_fraction
                        if accepted_fraction is not None
                        else iteration_step_fraction
                    ),
                )
                if accepted_fraction is not None:
                    record_event(
                        dict(
                            kind="step_restoration",
                            iteration=backend.state["iter"],
                            accepted_fraction=accepted_fraction,
                            next_fraction=next_step_fraction,
                        )
                    )
                try:
                    matrix = derivative(backend.x, value)
                except DomainError as error:
                    report(backend.x, value)
                    return finish(12, str(error))
                report(backend.x, value, matrix)
            else:
                if not options.check_kkt:
                    report(backend.x, value)
                    return finish(mode)
                # Native convergence may precede a gradient request at this
                # accepted point. Obtain its own derivatives for the final row.
                try:
                    matrix = derivative(backend.x, value)
                except DomainError as error:
                    report(backend.x, value)
                    return finish(mode, f"Final KKT derivative unavailable: {error}")
                report(backend.x, value, matrix)
                return finish(mode)
    except _CallbackStop:
        return finish(13)
