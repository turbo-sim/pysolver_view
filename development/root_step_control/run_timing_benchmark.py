"""Time native SciPy comparisons and four PySolverView root-finding methods.

Run directly so BLAS thread settings precede numerical-library imports::

    python development/root_step_control/run_timing_benchmark.py

The native comparison covers Levenberg--Marquardt and Powell hybrid. A separate
comparison covers those methods plus Newton--Raphson and Gauss--Newton within
PySolverView, with both unrestricted and protected steps. Protected cases share
initial scaled distance .1, maximum 1 and growth 1.5. Full histories are retained;
LM/hybrid summary-history runs are a separately labeled secondary experiment.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter_ns

THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

if __name__ == "__main__":
    if "numpy" in sys.modules:
        raise SystemExit(
            "Run this file directly so BLAS threads initialize before NumPy."
        )
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy
from scipy.optimize import OptimizeResult, root
from pysolver_view import NonlinearSystemProblem, NonlinearSystemSolver

from pysolver_view.solvers.root import solver as solver_module
from development.root_step_control.src.problems import (
    BroydenTridiagonalProblem,
    get_large_problems,
)
from development.root_step_control.src.problems.thermodynamics import (
    HeatedGasDuctProblem,
)

TARGET = 1e-8
# Preserve saved study identifiers; display names come from canonical solver names.
METHOD_NAMES = {
    method: solver_module.METHOD_DISPLAY_NAMES[solver_module.normalize_method(method)]
    for method in ("lm", "hybr", "newton_raphson", "gauss_newton")
}
SOLVERS = ("scipy_lm", "scipy_hybr", "pysolverview_lm", "pysolverview_hybr")
CUSTOM_SOLVERS = tuple(f"pysolverview_{method}" for method in METHOD_NAMES)
CONTROLLED_SOLVERS = tuple(
    f"pysolverview_{method}_controlled" for method in METHOD_NAMES
)
SUMMARY_SOLVERS = tuple(
    f"pysolverview_{method}_controlled_summary" for method in ("lm", "hybr")
)
KERNELS = ("svd", "qr", "qr_update", "triangular")


@dataclass(frozen=True)
class SolverConfiguration:
    """Identify one backend, algorithm, step policy and history mode explicitly."""

    backend: str
    method: str
    policy: str
    history_level: str = "full"


SOLVER_CONFIGURATIONS = {
    "scipy_lm": SolverConfiguration("scipy", "lm", "generous", "native_unavailable"),
    "scipy_hybr": SolverConfiguration(
        "scipy", "hybr", "generous", "native_unavailable"
    ),
    **{
        f"pysolverview_{method}": SolverConfiguration(
            "pysolverview",
            method,
            "generous" if method in ("lm", "hybr") else "full_step",
        )
        for method in METHOD_NAMES
    },
    **{
        f"pysolverview_{method}_controlled": SolverConfiguration(
            "pysolverview", method, "controlled"
        )
        for method in METHOD_NAMES
    },
    **{
        f"pysolverview_{method}_controlled_summary": SolverConfiguration(
            "pysolverview", method, "controlled", "summary"
        )
        for method in ("lm", "hybr")
    },
}


def solver_configuration(solver):
    """Look up a complete registered configuration without parsing its identifier."""
    try:
        return SOLVER_CONFIGURATIONS[solver]
    except KeyError as error:
        raise ValueError(f"Unknown solver {solver}") from error


@dataclass
class Measurements:
    """Accumulate one solve's callback and kernel times in nanoseconds, with call counts
    and first-target time.
    """

    residual_ns: int = 0
    jacobian_ns: int = 0
    svd_ns: int = 0
    qr_ns: int = 0
    qr_update_ns: int = 0
    triangular_ns: int = 0
    residual_calls: int = 0
    jacobian_calls: int = 0
    svd_calls: int = 0
    qr_calls: int = 0
    qr_update_calls: int = 0
    triangular_calls: int = 0
    first_target_ns: int | None = None
    first_target_call: int | None = None
    start_ns: int = 0

    def residual(self, function):
        """Wrap a residual function to count/time calls and record the first observed
        target crossing.
        """

        def measured(x):
            """Evaluate the residual, charge failed calls, and record target attainment
            after a valid return.
            """
            before = perf_counter_ns()
            try:
                value = function(x)
            finally:
                self.residual_ns += perf_counter_ns() - before
                self.residual_calls += 1
            if self.first_target_ns is None and np.linalg.norm(value) <= TARGET:
                self.first_target_ns = perf_counter_ns() - self.start_ns
                self.first_target_call = self.residual_calls
            return value

        return measured

    def jacobian(self, function):
        """Wrap an analytic Jacobian function so its elapsed time is measured separately
        from residual time.
        """

        def measured(x):
            """Return the Jacobian while counting its call and elapsed time, including
            exceptions.
            """
            before = perf_counter_ns()
            try:
                return function(x)
            finally:
                self.jacobian_ns += perf_counter_ns() - before
                self.jacobian_calls += 1

        return measured


@contextmanager
def measure_custom_kernels(measurements, solver, enabled):
    """Time module-local kernels and restore them, including after exceptions.

    Sequential use only. These disjoint kernel times are a subset of overhead;
    matrix products, norms, condition estimation, and Python logic remain in
    other overhead. No SciPy installation or native MINPACK symbol is patched.
    """
    replacements = []
    if enabled and solver_configuration(solver).backend == "pysolverview":
        from pysolver_view.solvers.root import linear_model

        bindings = [
            (linear_model, name, label)
            for name, label in (
                ("svd", "svd"),
                ("qr", "qr"),
                ("qr_update", "qr_update"),
                ("solve_triangular", "triangular"),
            )
        ]
    else:
        bindings = []

    def timed_kernel(original, label):
        """Build a timed replacement for one imported numerical kernel without changing
        its arguments.
        """

        def measured(*args, **kwargs):
            """Call the original kernel and add elapsed nanoseconds and a call count
            even when it raises.
            """
            before = perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                setattr(
                    measurements,
                    label + "_ns",
                    getattr(measurements, label + "_ns") + perf_counter_ns() - before,
                )
                setattr(
                    measurements,
                    label + "_calls",
                    getattr(measurements, label + "_calls") + 1,
                )

        return measured

    try:
        for module, name, label in bindings:
            original = getattr(module, name)
            replacements.append((module, name, original))
            setattr(module, name, timed_kernel(original, label))
        yield
    finally:
        for module, name, original in reversed(replacements):
            setattr(module, name, original)


def make_options(problem, solver):
    """Build matching scale/tolerance settings for a registered solver configuration.

    Native runs use factor=100. Generous custom trust radii follow that initial
    formula. Full-step Newton-family runs disable backtracking explicitly.
    Protected runs share scaled initial limit 0.1, maximum 1 and growth 1.5.
    """
    configuration = solver_configuration(solver)
    if configuration.backend == "scipy":
        result = dict(factor=100.0, diag=1.0 / problem.x_scale, xtol=1e-12)
        if configuration.method == "lm":
            result.update(ftol=1e-12, gtol=1e-12, maxiter=2000)
        else:
            result["maxfev"] = 2000
        return result
    common = dict(
        x_scale=problem.x_scale,
        residual_tol=TARGET,
        xtol=1e-12,
        gtol=1e-12,
        max_iterations=1500,
        max_nfev=2000,
        history_level=configuration.history_level,
    )
    if configuration.method in ("newton_raphson", "gauss_newton"):
        backtracking = None
        if configuration.policy == "controlled":
            backtracking = solver_module.BacktrackingOptions(
                step_limits=solver_module.StepLimitOptions(
                    initial_limit=0.1, max_limit=1.0, growth_factor=1.5
                )
            )
        return solver_module.NewtonOptions(**common, backtracking=backtracking)
    scaled_initial_norm = np.linalg.norm(problem.x0 / problem.x_scale)
    initial = 100.0 * scaled_initial_norm if scaled_initial_norm else 100.0
    controls = dict(
        initial_radius=float(initial),
        max_radius=max(1e12, float(initial)),
        growth_factor=2.0,
    )
    if configuration.policy == "controlled":
        controls.update(initial_radius=0.1, max_radius=1.0, growth_factor=1.5)
    return solver_module.StepControlOptions(**controls, **common)


class MeasuredRootProblem(NonlinearSystemProblem):
    """Delegate to the supplied measured callbacks without changing a benchmark model."""

    def __init__(self, residual, jacobian):
        """Retain residual/Jacobian callbacks for one timed solve."""
        self._residual = residual
        self._jacobian = jacobian

    def residual(self, x):
        """Evaluate the original problem through its optional timing wrapper."""
        return self._residual(x)

    def gradient(self, x):
        """Evaluate the supplied analytic Jacobian through its timing wrapper."""
        return self._jacobian(x)


def run_solver(problem, solver, options, residual, jacobian, *, interface="public"):
    """Dispatch one registered configuration to native root or solve_root using the
    supplied callbacks.
    """
    configuration = solver_configuration(solver)
    if configuration.backend == "scipy":
        return root(
            residual,
            problem.x0,
            jac=jacobian,
            method=configuration.method,
            options=options,
        )
    if interface == "core":
        return solver_module.solve_root(
            residual,
            problem.x0,
            jac=jacobian,
            method=configuration.method,
            options=options,
        )
    if interface != "public":
        raise ValueError("interface must be 'public' or 'core'.")
    facade = NonlinearSystemSolver(
        MeasuredRootProblem(residual, jacobian),
        method=configuration.method,
        library="pysolver",
        options=options,
        print_convergence=False,
        update_on="iteration",
    )
    facade.solve(problem.x0)
    return facade.result


def check_result(problem, solver, options, result, *, strict=True):
    """Independent residual verification and step audits outside solve timing.

    Failed convergence is recorded when strict=False. Step-bound violations remain
    hard errors, because a benchmark must not silently validate broken controls.
    """
    configuration = solver_configuration(solver)
    try:
        final_norm = float(np.linalg.norm(problem.residual(result.x)))
    except solver_module.DomainError:
        final_norm = float("inf")
    root_success = bool(np.isfinite(final_norm) and final_norm <= TARGET)
    if strict and not root_success:
        raise AssertionError(
            f"{problem.name}/{solver}: residual {final_norm:g}; {result.message}"
        )
    maximum_bound = maximum_step = maximum_ratio = 0.0
    trust_region = configuration.method in ("lm", "hybr")
    if configuration.backend == "pysolverview":
        for trial in result.get("trials", []):
            step = (
                np.linalg.norm((trial["x"] - trial["x_base"]) / problem.x_scale)
                if "x" in trial
                else trial["scaled_step_norm"]
            )
            bound = trial.get("radius") if trust_region else trial.get("step_limit")
            if bound is not None and step > bound * (1 + 1e-12):
                raise AssertionError(
                    "Experimental trial exceeded its intrinsic trust radius."
                    if trust_region
                    else "Newton-family trial exceeded its active step limit."
                )
            maximum_step = max(maximum_step, step)
            if bound is not None:
                maximum_bound = max(maximum_bound, bound)
                maximum_ratio = max(maximum_ratio, step / bound)
        maximum_bound = max(
            maximum_bound,
            result.get(
                "max_radius_used" if trust_region else "max_step_limit_used", 0.0
            )
            or 0.0,
        )
        maximum_step = max(maximum_step, result.get("max_trial_step_scaled", 0.0))
        maximum_ratio = max(
            maximum_ratio,
            result.get(
                "max_radius_ratio" if trust_region else "max_step_limit_ratio", 0.0
            )
            or 0.0,
        )
        maximum_allowed = (
            options.max_radius
            if trust_region
            else (
                options.backtracking.step_limits.max_limit
                if options.backtracking is not None
                and options.backtracking.step_limits is not None
                else None
            )
        )
        if maximum_ratio > 1 + 1e-12 or (
            maximum_allowed is not None
            and maximum_bound > maximum_allowed * (1 + 1e-12)
        ):
            raise AssertionError(
                "Experimental trial exceeded its configured maximum step bound."
            )
        if (
            trust_region
            and configuration.policy == "generous"
            and maximum_bound >= options.max_radius * (1 - 1e-12)
        ):
            raise AssertionError(
                "The user maximum radius bound became active in this timing case."
            )
    return dict(
        final_residual_norm=final_norm,
        root_success=root_success,
        native_success=bool(result.success),
        max_active_radius=maximum_bound if trust_region else None,
        max_active_step_bound=maximum_bound,
        max_trial_step_scaled=maximum_step,
        max_radius_ratio=maximum_ratio,
        accepted_iterations=(
            result.get("nit") if configuration.backend == "pysolverview" else None
        ),
        trial_attempts=result.get("ntrials"),
        backtracks=result.get("nbacktracks"),
        invalid_trials=result.get("ninvalid"),
        message=str(result.message).replace("\n", " "),
    )


def optional_maximum(values):
    """Return the maximum available value, or None when every value is missing."""
    available = [value for value in values if value is not None]
    return max(available) if available else None


def optional_mean(values):
    """Average available numeric values only when every solve supplies the metric."""
    values = list(values)
    return (
        float(np.mean(values)) if all(value is not None for value in values) else None
    )


def measure_batch(
    problem, solver, options, batch_size, instrumented, *, interface="public"
):
    """Average a sequential batch; retain failures instead of dropping samples."""
    totals = dict(
        total_ns=0,
        residual_ns=0,
        jacobian_ns=0,
        residual_calls=0,
        jacobian_calls=0,
        first_target_ns=0,
        first_target_call=0,
    )
    for label in KERNELS:
        totals[label + "_ns"] = totals[label + "_calls"] = 0
    checks = []
    last_result = None
    for _ in range(batch_size):
        measurements = Measurements()
        residual = (
            measurements.residual(problem.residual)
            if instrumented
            else problem.residual
        )
        jacobian = (
            measurements.jacobian(problem.gradient)
            if instrumented
            else problem.gradient
        )
        with measure_custom_kernels(measurements, solver, instrumented):
            before = perf_counter_ns()
            measurements.start_ns = before
            try:
                result = run_solver(
                    problem, solver, options, residual, jacobian, interface=interface
                )
            except solver_module.DomainError as error:
                result = OptimizeResult(
                    x=np.full(problem.n, np.nan),
                    success=False,
                    message=f"Aborted on DomainError: {error}",
                )
            elapsed = perf_counter_ns() - before
        totals["total_ns"] += elapsed
        if instrumented:
            for key in totals:
                if key != "total_ns":
                    value = getattr(measurements, key)
                    totals[key] = (
                        None
                        if value is None or totals[key] is None
                        else totals[key] + value
                    )
        checks.append(check_result(problem, solver, options, result, strict=False))
        last_result = result
    row = {
        key: value / batch_size if value is not None else None
        for key, value in totals.items()
    }
    row["solver_overhead_ns"] = (
        row["total_ns"] - row["residual_ns"] - row["jacobian_ns"]
        if instrumented
        else None
    )
    if instrumented:
        kernel_total = sum(row[label + "_ns"] for label in KERNELS)
        if row["solver_overhead_ns"] < 0 or kernel_total > row["solver_overhead_ns"]:
            raise AssertionError("Timing components are inconsistent.")
    else:
        for key in totals:
            if key != "total_ns":
                row[key] = None
    configuration = solver_configuration(solver)
    if configuration.backend == "scipy":
        for label in KERNELS:
            row[label + "_ns"] = row[label + "_calls"] = None
    row.update(
        problem=problem.name,
        dimension=problem.n,
        solver=solver,
        backend=configuration.backend,
        method=configuration.method,
        method_name=METHOD_NAMES[configuration.method],
        mode=configuration.policy,
        history_level=configuration.history_level,
        instrumented=instrumented,
        batch_size=batch_size,
        root_successes=sum(c["root_success"] for c in checks),
        all_root_success=all(c["root_success"] for c in checks),
        final_residual_norm=max(c["final_residual_norm"] for c in checks),
        all_native_success=all(c["native_success"] for c in checks),
        max_active_radius=optional_maximum(c["max_active_radius"] for c in checks),
        max_active_step_bound=max(c["max_active_step_bound"] for c in checks),
        max_trial_step_scaled=max(c["max_trial_step_scaled"] for c in checks),
        max_radius_ratio=max(c["max_radius_ratio"] for c in checks),
        messages=" | ".join(sorted({c["message"] for c in checks})),
    )
    for field in (
        "accepted_iterations",
        "trial_attempts",
        "backtracks",
        "invalid_trials",
    ):
        row[field] = optional_mean(check[field] for check in checks)
    return row, last_result


def summarize_batches(rows):
    """Aggregate batch rows into per-problem/per-solver timing summaries.

    Use clean total medians for ratios to native SciPy. Take every additive
    instrumented component from the same median-total batch, preserving the timing
    identity. Withhold ratios when either configuration misses the residual target.
    """
    summary = []
    for problem, solver in sorted({(r["problem"], r["solver"]) for r in rows}):
        group = [r for r in rows if r["problem"] == problem and r["solver"] == solver]
        measured = sorted(
            (r for r in group if r["instrumented"]), key=lambda r: r["total_ns"]
        )
        clean = np.array([r["total_ns"] / 1e6 for r in group if not r["instrumented"]])
        representative = measured[len(measured) // 2]
        result = dict(
            problem=problem,
            dimension=representative["dimension"],
            solver=solver,
            backend=solver_configuration(solver).backend,
            method=solver_configuration(solver).method,
            method_name=METHOD_NAMES[solver_configuration(solver).method],
            mode=representative["mode"],
            history_level=representative["history_level"],
            repetitions=len(measured),
            solves_per_batch=representative["batch_size"],
            clean_total_median_ms=float(np.median(clean)),
            clean_total_q25_ms=float(np.quantile(clean, 0.25)),
            clean_total_q75_ms=float(np.quantile(clean, 0.75)),
        )
        for name in (
            "total",
            "residual",
            "jacobian",
            "solver_overhead",
            "first_target",
        ):
            value = representative[name + "_ns"]
            result[name + "_ms"] = value / 1e6 if value is not None else None
        for name in ("residual_calls", "jacobian_calls", "first_target_call"):
            result[name] = representative[name]
        for name in ("residual", "jacobian"):
            result[name + "_us_per_call"] = (
                representative[name + "_ns"]
                / max(1, representative[name + "_calls"])
                / 1000
            )
        for label in KERNELS:
            value = representative[label + "_ns"]
            result["custom_" + label + "_ms"] = (
                value / 1e6 if value is not None else None
            )
            result["custom_" + label + "_calls"] = representative[label + "_calls"]
        kernel_times = [result["custom_" + label + "_ms"] for label in KERNELS]
        result["custom_kernel_subset_ms"] = (
            sum(kernel_times) if all(v is not None for v in kernel_times) else None
        )
        result["instrumented_to_clean_ratio"] = (
            result["total_ms"] / result["clean_total_median_ms"]
        )
        result["all_root_success"] = all(r["all_root_success"] for r in group)
        result["root_success_fraction"] = sum(r["root_successes"] for r in group) / sum(
            r["batch_size"] for r in group
        )
        result["max_final_residual_norm"] = max(r["final_residual_norm"] for r in group)
        result["all_native_success"] = all(r["all_native_success"] for r in group)
        for key in (
            "max_active_radius",
            "max_active_step_bound",
            "max_trial_step_scaled",
            "max_radius_ratio",
        ):
            result[key] = optional_maximum(r.get(key) for r in group)
        for key in (
            "accepted_iterations",
            "trial_attempts",
            "backtracks",
            "invalid_trials",
        ):
            result[key] = representative.get(key)
        result["messages"] = " | ".join(sorted({r["messages"] for r in group}))
        summary.append(result)
    lookup = {(r["problem"], r["solver"]): r for r in summary}
    for result in summary:
        method = solver_configuration(result["solver"]).method
        native = lookup.get((result["problem"], "scipy_" + method))
        result["clean_time_relative_to_native_method"] = (
            result["clean_total_median_ms"] / native["clean_total_median_ms"]
            if native and native["all_root_success"] and result["all_root_success"]
            else None
        )
        reference_id = (
            "pysolverview_hybr_controlled"
            if result["mode"] == "controlled"
            else "pysolverview_hybr"
        )
        reference = lookup.get((result["problem"], reference_id))
        result["clean_time_relative_to_custom_hybrid"] = (
            result["clean_total_median_ms"] / reference["clean_total_median_ms"]
            if reference
            and reference["all_root_success"]
            and result["all_root_success"]
            else None
        )
    return summary


def to_json_compatible(value):
    """Convert nested arrays and NumPy values to JSON-compatible values, mapping
    nonfinite floats to null.
    """
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(v) for v in value]
    return value


def write_csv(path, rows):
    """Write nonempty dictionaries as CSV rows using the first row's key order as the
    header.
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_optional_number(value, digits=4):
    """Format a measured number to the requested precision, or label missing
    measurements unavailable.
    """
    return "unavailable" if value is None else f"{value:.{digits}f}"


def comparison_groups(summary):
    """Return separately labeled native, custom and logging-cost comparisons."""
    groups = (
        (
            "timing_breakdown",
            "PySolverView versus SciPy/MINPACK",
            SOLVERS,
            "Levenberg–Marquardt and Powell hybrid use generous initial radii based on "
            "MINPACK factor=100. The custom maximum is at least $10^{12}$ and audited "
            "to remain inactive. Intrinsic trust-region adaptation remains enabled.",
        ),
        (
            "custom_timing_breakdown",
            "All PySolverView methods: unrestricted user step controls",
            CUSTOM_SOLVERS,
            "Levenberg–Marquardt and Powell hybrid retain intrinsic trust-region "
            "adaptation with generous user limits. Newton–Raphson and Gauss–Newton "
            "use full steps with backtracking=None. This compares explicit policies; "
            "it does not remove the trust-region algorithms' own damping.",
        ),
        (
            "controlled_timing_breakdown",
            "All PySolverView methods: protected steps",
            CONTROLLED_SOLVERS,
            "All four methods use scaled initial distance 0.1, maximum distance 1 "
            "and growth factor 1.5, with full histories. Newton-family methods use "
            "Armijo backtracking and carry forward a reduced limit after recovery. "
            "LM/Powell hybrid use their trust-region acceptance and radius policy; "
            "matching bounds do not make those policies identical.",
        ),
        (
            "summary_timing_breakdown",
            "Secondary comparison: history storage on 100-variable workloads",
            (
                "pysolverview_lm_controlled",
                "pysolverview_lm_controlled_summary",
                "pysolverview_hybr_controlled",
                "pysolverview_hybr_controlled_summary",
            ),
            "Full and summary records use the same protected LM/Powell hybrid "
            "settings. Summary mode omits coordinate copies only. These rows "
            "are a logging-cost experiment, separate from the four-method comparison.",
        ),
    )
    for stem, title, solver_ids, explanation in groups:
        rows = [
            row
            for row in summary
            if row["solver"] in solver_ids
            and (stem != "summary_timing_breakdown" or row["dimension"] == 100)
        ]
        order = {solver: index for index, solver in enumerate(solver_ids)}
        rows.sort(key=lambda row: (row["problem"], order[row["solver"]]))
        if rows:
            yield stem, title, explanation, rows


def solver_description(solver):
    """Describe a configuration with its full author-based name and explicit policy."""
    configuration = solver_configuration(solver)
    backend = "SciPy/MINPACK" if configuration.backend == "scipy" else "PySolverView"
    description = f"{backend} {METHOD_NAMES[configuration.method]}"
    if configuration.backend == "pysolverview":
        policy = {
            "generous": "generous radii",
            "full_step": "full step",
            "controlled": "protected",
        }[configuration.policy]
        description += f" ({policy}, {configuration.history_level} history)"
    return description


def write_report(output, summary, metadata):
    """Write separate comparison tables and additive timing details from saved data."""
    protocol = Path(__file__).resolve().parent / "notes/06_time_performance.md"
    try:
        protocol_link = os.path.relpath(protocol, output).replace(os.sep, "/")
    except ValueError:
        protocol_link = protocol.as_posix()
    lines = [
        "# Root-finder timing results",
        "",
        "Generated by `run_timing_benchmark.py`. Times are milliseconds per solve.",
        "",
        f"Python {metadata['python']}, NumPy {metadata['numpy']}, SciPy {metadata['scipy']}, "
        f"CoolProp {metadata['coolprop']}. {metadata['repetitions']} randomized rounds, "
        f"{metadata['batch_size']} solves per batch in each measurement mode, "
        f"{metadata['warmups']} warmups per configuration. One BLAS thread is requested "
        "before imports by the direct CLI.",
        "",
        "Clean medians contain no residual/Jacobian/kernel timers. Q25–Q75 describe "
        "batch-average variability, not confidence intervals. All trials and failures "
        "are retained. Speed ratios are withheld if either compared configuration "
        "fails the independently checked residual target.",
        "",
    ]
    for stem, title, explanation, rows in comparison_groups(summary):
        native_comparison = stem == "timing_breakdown"
        ratio_field = (
            "clean_time_relative_to_native_method"
            if native_comparison
            else "clean_time_relative_to_custom_hybrid"
        )
        ratio_label = (
            "Time/native same method"
            if native_comparison
            else "Time/custom Powell hybrid"
        )
        lines += [
            f"## {title}",
            "",
            explanation,
            "",
            f"| Problem | Solver | Clean median | Q25–Q75 | {ratio_label} | Root success |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['problem']} | {solver_description(row['solver'])} | "
                f"{row['clean_total_median_ms']:.4f} | "
                f"{row['clean_total_q25_ms']:.4f}–{row['clean_total_q75_ms']:.4f} | "
                f"{format_optional_number(row[ratio_field], 2)} | "
                f"{row['root_success_fraction']:.0%} |"
            )
        lines += ["", f"![{title}]({stem}.png)", ""]
    lines += [
        "## Additive timing breakdown",
        "",
        "Each row uses one instrumented median-total batch. Residual + Jacobian + "
        "solver overhead equals total before rounding. Overhead includes linear "
        "algebra, algorithm decisions, bookkeeping, timers, and histories; it is "
        "not pure linear-algebra time.",
        "",
        "| Problem | Solver | Residual | Jacobian | Solver overhead | Total | Residual/Jacobian calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['problem']} | {solver_description(row['solver'])} | "
            f"{row['residual_ms']:.4f} | {row['jacobian_ms']:.4f} | "
            f"{row['solver_overhead_ms']:.4f} | {row['total_ms']:.4f} | "
            f"{row['residual_calls']:.0f}/{row['jacobian_calls']:.0f} |"
        )
    lines += [
        "",
        "## Measured linear-algebra subsets",
        "",
        "These disjoint local-kernel timings are already included in overhead. "
        "Matrix products, norms, condition estimates, and scalar searches are outside "
        "them. Native internal kernel timings are unavailable through the public "
        "SciPy interface, not zero.",
        "",
        "| Problem | Solver | QR | QR update | SVD | Triangular | QR/update/SVD calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        calls = "/".join(
            format_optional_number(row["custom_" + name + "_calls"], 0)
            for name in ("qr", "qr_update", "svd")
        )
        lines.append(
            f"| {row['problem']} | {solver_description(row['solver'])} | "
            f"{format_optional_number(row['custom_qr_ms'])} | "
            f"{format_optional_number(row['custom_qr_update_ms'])} | "
            f"{format_optional_number(row['custom_svd_ms'])} | "
            f"{format_optional_number(row['custom_triangular_ms'])} | {calls} |"
        )
    lines += [
        "",
        "## Residual cost and first observed target",
        "",
        "The first target is the first evaluated point with residual 2-norm at most "
        "$10^{-8}$. It is not necessarily a native accepted iterate. Independent "
        "final verification is outside timing.",
        "",
        "| Problem | Solver | Residual microseconds/call | Jacobian microseconds/call | First target ms | First target call |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['problem']} | {solver_description(row['solver'])} | "
            f"{row['residual_us_per_call']:.2f} | {row['jacobian_us_per_call']:.2f} | "
            f"{format_optional_number(row['first_target_ms'])} | "
            f"{format_optional_number(row['first_target_call'], 0)} |"
        )
    failures = [row for row in summary if not row["all_root_success"]]
    lines += [
        "",
        "## Validation and limits",
        "",
        f"{len(summary) - len(failures)}/{len(summary)} configurations reached the "
        "independently checked residual target on every timed solve. Protected "
        "custom trials were audited against actual scaled displacement and their "
        "radius or line-search step limit. Full-step modes have no imposed limit "
        "and may increase the residual. Scalar-history variants use driver aggregates.",
        "",
    ]
    for row in failures:
        lines.append(
            f"- {row['problem']}, {solver_description(row['solver'])}: {row['messages']}"
        )
    lines += [
        "",
        "Five workloads include Broyden systems with 20 and 100 variables, "
        "a fully dense 100-variable discrete integral equation, and real CoolProp "
        "heated-duct models with 20 and 100 variables. These results do not establish "
        "universal speed or thermodynamic robustness.",
        "",
        "Imports, problem construction, warmups, verification, and plots are excluded from "
        "solve timing. Custom totals include NonlinearSystemSolver construction, option "
        "resolution, accepted-point reporting preparation, and detached result copies; "
        "printing and live plotting are disabled. SciPy uses scipy.optimize.root directly. "
        "Native stopping rules may perform additional work below the "
        "common residual target. Reported rankings depend on hardware, numerical "
        "library builds, logging, starts, and control policies.",
        "",
        "Complete data: [summary.csv](summary.csv), [raw_batches.csv](raw_batches.csv), "
        f"[metadata.json](metadata.json), and [timing protocol]({protocol_link}).",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def solver_plot_label(solver):
    """Use full author-based method names and backend identity in every plot label."""
    configuration = solver_configuration(solver)
    backend = "SciPy/MINPACK" if configuration.backend == "scipy" else "PySolverView"
    lines = [backend, METHOD_NAMES[configuration.method]]
    if configuration.history_level == "summary":
        lines.append("summary history")
    return "\n".join(lines)


def plot_results(output, summary):
    """Plot separate native/custom comparisons with additive breakdowns and clean totals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for stem, title, _, rows in comparison_groups(summary):
        names = list(dict.fromkeys(row["problem"] for row in rows))
        fig, axes = plt.subplots(
            len(names),
            1,
            figsize=(14, 4.0 * len(names)),
            squeeze=False,
            layout="constrained",
        )
        for axis, name in zip(axes[:, 0], names):
            group = [row for row in rows if row["problem"] == name]
            positions = np.arange(len(group))
            bottom = np.zeros(len(group))
            for field, label, color in (
                ("residual_ms", "Residual", "#4c78a8"),
                ("jacobian_ms", "Jacobian", "#f2a541"),
                (
                    "solver_overhead_ms",
                    "Solver overhead (includes linear algebra)",
                    "#7a5195",
                ),
            ):
                values = np.array([row[field] for row in group])
                bars = axis.bar(
                    positions, values, bottom=bottom, label=label, color=color
                )
                for bar, row in zip(bars, group):
                    if not row["all_root_success"]:
                        bar.set_hatch("//")
                bottom += values
            medians = np.array([row["clean_total_median_ms"] for row in group])
            lower = medians - np.array([row["clean_total_q25_ms"] for row in group])
            upper = np.array([row["clean_total_q75_ms"] for row in group]) - medians
            axis.errorbar(
                positions,
                medians,
                yerr=[lower, upper],
                fmt="D",
                color="black",
                markersize=4,
                capsize=3,
                label="Clean total median and quartiles",
            )
            labels = [solver_plot_label(row["solver"]) for row in group]
            labels = [
                label + ("\nFAILED target" if not row["all_root_success"] else "")
                for label, row in zip(labels, group)
            ]
            axis.set_xticks(positions, labels, fontsize=9)
            axis.set(title=name, ylabel="Milliseconds per solve")
            axis.grid(axis="y", alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
        fig.suptitle(
            title + "\nBars: additive instrumented median batch; diamonds: clean time"
        )
        for extension in ("png", "svg"):
            fig.savefig(output / f"{stem}.{extension}", dpi=160)
        plt.close(fig)


def run_benchmark(
    output,
    *,
    repetitions=21,
    batch_size=5,
    warmups=3,
    seed=20261025,
    problem_names=None,
):
    """Warm up, interleave and measure every native/custom workload configuration.

    Parameters choose the output directory, odd number of rounds, solves per batch,
    warmup count and reproducible shuffle seed. Both clean and instrumented solves
    are independently verified outside their timers. Return summary rows and save
    raw batches, metadata, Markdown and PNG/SVG plots. Optional problem_names
    selects named workloads for a quick functional run, preserving all methods.
    """
    if repetitions < 3 or repetitions % 2 != 1 or batch_size < 1 or warmups < 1:
        raise ValueError("Use odd repetitions >= 3, batch_size >= 1 and warmups >= 1.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems = [
        HeatedGasDuctProblem(n_cells=10),
        BroydenTridiagonalProblem(n=20),
        *get_large_problems(),
    ]
    if problem_names is not None:
        unknown = set(problem_names) - {problem.name for problem in problems}
        if unknown:
            raise ValueError(f"Unknown timing problems: {sorted(unknown)}")
        problems = [problem for problem in problems if problem.name in problem_names]
        if not problems:
            raise ValueError("Select at least one timing problem.")
    primary = ("scipy_lm", "scipy_hybr") + CUSTOM_SOLVERS + CONTROLLED_SOLVERS
    configurations = [
        (problem, solver, make_options(problem, solver))
        for problem in problems
        for solver in primary + (SUMMARY_SOLVERS if problem.n == 100 else ())
    ]
    for problem, solver, options in configurations:
        for _ in range(warmups):
            # Domain failures are recorded during timed solves instead of
            # deleting a difficult configuration from the comparison.
            try:
                result = run_solver(
                    problem, solver, options, problem.residual, problem.gradient
                )
            except solver_module.DomainError:
                continue
            check_result(problem, solver, options, result, strict=False)
    rows = []
    randomizer = random.Random(seed)
    for repetition in range(repetitions):
        jobs = [
            (configuration, measured)
            for configuration in configurations
            for measured in (False, True)
        ]
        randomizer.shuffle(jobs)
        for (problem, solver, options), measured in jobs:
            row, _ = measure_batch(problem, solver, options, batch_size, measured)
            row["repetition"] = repetition
            rows.append(row)
        print(f"Completed timing batch {repetition + 1}/{repetitions}", flush=True)
    summary = summarize_batches(rows)
    import CoolProp

    configuration_output = io.StringIO()
    with redirect_stdout(configuration_output):
        np.show_config()
        scipy.show_config()
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        coolprop=CoolProp.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
        thread_environment={k: os.environ.get(k) for k in THREAD_VARIABLES},
        library_configuration=configuration_output.getvalue(),
        repetitions=repetitions,
        batch_size=batch_size,
        warmups=warmups,
        seed=seed,
        residual_target=TARGET,
        configuration_count=len(configurations),
        timed_solve_count=len(configurations) * repetitions * batch_size * 2,
        solver_configurations={
            solver: asdict(solver_configuration(solver))
            for _, solver, _ in configurations
        },
        method_display_names=METHOD_NAMES,
        custom_interface="NonlinearSystemSolver(library='pysolver'), construction and reporting preparation included",
        scipy_interface="scipy.optimize.root directly",
        notes="Batch averages per solve. Analytic Jacobians. Clean totals determine ratios. Native linear-algebra timing unavailable. Failures retained.",
        problems=[
            dict(
                name=problem.name,
                dimension=problem.n,
                x0=problem.x0,
                x_scale=problem.x_scale,
                description=problem.description,
                source=problem.source,
            )
            for problem in problems
        ],
        solver_options={
            problem.name: {
                solver: (
                    options
                    if solver_configuration(solver).backend == "scipy"
                    else asdict(options)
                )
                for candidate, solver, options in configurations
                if candidate is problem
            }
            for problem in problems
        },
        thermodynamic_solutions={},
    )
    for problem in problems:
        if isinstance(problem, HeatedGasDuctProblem):
            result = run_solver(
                problem,
                "scipy_lm",
                make_options(problem, "scipy_lm"),
                problem.residual,
                problem.gradient,
            )
            metadata["thermodynamic_solutions"][problem.name] = (
                problem.solution_diagnostics(result.x)
            )
    write_csv(output / "raw_batches.csv", rows)
    write_csv(output / "summary.csv", summary)
    (output / "metadata.json").write_text(
        json.dumps(to_json_compatible(metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output, summary, metadata)
    plot_results(output, summary)
    print(f"Saved timing results to {output.resolve()}")
    return summary


def main():
    """Parse timing repetition, batch, warmup and output options, then execute the
    timing study.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_time_perf",
    )
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--problems", nargs="+", help="Optional workload names to run.")
    args = parser.parse_args()
    run_benchmark(
        args.output,
        repetitions=args.repetitions,
        batch_size=args.batch_size,
        warmups=args.warmups,
        problem_names=args.problems,
    )


if __name__ == "__main__":
    main()
