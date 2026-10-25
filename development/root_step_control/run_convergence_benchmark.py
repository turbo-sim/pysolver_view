"""Compare native/custom LM and Powell hybrid, then four custom root methods.

Run from the repository root::

    python development/root_step_control/run_convergence_benchmark.py

Two output folders separate native/custom LM and Powell hybrid from the
four-method PySolverView comparison. Newton-family full-step and uncapped
backtracking policies supplement the common controlled comparison. Every case
uses analytic Jacobians and fixed coordinate scales. SciPy does not expose
accepted iterates here: its consecutive evaluation displacements are not
interpreted as accepted steps or trust-region radii.
"""

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy
from scipy.optimize import root
from pysolver_view import NonlinearSystemSolver

from pysolver_view.roots.problems import (
    get_extension_problems,
    get_large_problems,
    get_problems,
)
from pysolver_view.roots.solvers.solver import (
    BacktrackingOptions,
    DomainError,
    METHOD_DISPLAY_NAMES,
    NewtonOptions,
    StepLimitOptions,
    StepControlOptions,
    normalize_method,
)

# Preserve saved study identifiers; display names come from canonical solver names.
METHOD_NAMES = {
    method: METHOD_DISPLAY_NAMES[normalize_method(method)]
    for method in ("lm", "hybr", "newton_raphson", "gauss_newton")
}

RESIDUAL_TOL = 1e-8
MAX_NFEV = 2000
MAX_ITERATIONS = 1500
CUSTOM_CASES = tuple((method, "controlled") for method in METHOD_NAMES) + tuple(
    (method, policy)
    for method in ("newton_raphson", "gauss_newton")
    for policy in ("backtracking", "full")
)
CONFIGURATIONS = {
    "generous": {"initial_radius": 100.0, "max_radius": 1e6, "growth_factor": 2.0},
    "initial_only": {"initial_radius": 0.1, "max_radius": 1e6, "growth_factor": 2.0},
    "controlled": {"initial_radius": 0.1, "max_radius": 1.0, "growth_factor": 1.5},
}


def euclidean_norm(value):
    """Return a Euclidean norm using hypot reduction to avoid squaring overflow."""
    # hypot reduction avoids squaring overflow for a finite, enormous residual.
    return float(np.hypot.reduce(np.ravel(value)))


def to_json_compatible(value):
    """Convert nested arrays and NumPy scalars to JSON values, replacing nonfinite
    floats with null.
    """
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    return value


def run_scipy_case(problem, method, factor):
    """Run one native SciPy case and retain its actual residual-call history.

    The problem supplies x0, x_scale, residual and gradient. The returned dictionary
    keeps native termination separate from an independent residual-target check;
    SciPy does not expose accepted-step histories for these methods.
    """
    evaluations = []
    jac_calls = 0

    def residual(x):
        """Record each requested point and residual norm, including recognized domain
        failures.
        """
        record = {"x": x.copy(), "kind": "scipy_function", "valid": False}
        evaluations.append(record)
        try:
            value = problem.residual(x)
        except DomainError as exc:
            record.update(reason=str(exc), residual_norm=np.inf)
            raise
        record.update(
            valid=bool(np.all(np.isfinite(value))), residual_norm=euclidean_norm(value)
        )
        return value

    def jacobian(x):
        """Evaluate the analytic problem Jacobian and count native derivative requests."""
        nonlocal jac_calls
        jac_calls += 1
        return problem.gradient(x)

    options = {
        "factor": factor,
        "diag": 1.0 / problem.x_scale,
        "xtol": 1e-12,
        "maxiter" if method == "lm" else "maxfev": MAX_NFEV,
    }
    if method == "lm":
        options.update(ftol=1e-12, gtol=1e-12)
    try:
        result = root(
            residual, problem.x0.copy(), jac=jacobian, method=method, options=options
        )
    except DomainError as exc:
        valid = [record for record in evaluations if record["valid"]]
        return {
            "label": f"scipy_{method}_factor_{factor:g}",
            "family": "scipy",
            "method": method,
            "options": options,
            "success": False,
            "native_success": None,
            "message": f"Aborted on DomainError: {exc}",
            "x": None,
            "residual_norm": None,
            "last_valid_residual_norm": valid[-1]["residual_norm"] if valid else None,
            "nfev": len(evaluations),
            "njev": jac_calls,
            "nit": None,
            "evaluations": evaluations,
            "history": [],
            "trials": [],
            "verification_nfev": 0,
        }
    verified_norm = euclidean_norm(problem.residual(result.x))
    return {
        "label": f"scipy_{method}_factor_{factor:g}",
        "family": "scipy",
        "method": method,
        "options": options,
        "success": bool(verified_norm <= RESIDUAL_TOL),
        "native_success": bool(result.success),
        "message": str(result.message),
        "x": result.x.copy(),
        "residual_norm": verified_norm,
        "nfev": len(evaluations),
        "native_nfev": int(result.nfev),
        "njev": jac_calls,
        "nit": None,
        "evaluations": evaluations,
        "history": [],
        "trials": [],
        "verification_nfev": 1,
    }


def run_custom_case(problem, method, configuration):
    """Run one custom method/configuration and return its verified solution, settings
    and full histories.
    """
    common = dict(
        residual_tol=RESIDUAL_TOL,
        xtol=1e-12,
        gtol=1e-12,
        max_iterations=MAX_ITERATIONS,
        max_nfev=MAX_NFEV,
        x_scale=problem.x_scale.copy(),
    )
    if method in ("lm", "hybr"):
        options = StepControlOptions(**CONFIGURATIONS[configuration], **common)
        policy = "trust_region"
    else:
        if configuration == "full":
            backtracking = None
        elif configuration == "backtracking":
            backtracking = BacktrackingOptions(step_limits=None)
        elif configuration == "controlled":
            backtracking = BacktrackingOptions(
                step_limits=StepLimitOptions(
                    initial_limit=0.1, max_limit=1.0, growth_factor=1.5
                )
            )
        else:
            raise ValueError(f"Unknown Newton-family configuration {configuration}")
        options = NewtonOptions(backtracking=backtracking, **common)
        policy = "full_step" if backtracking is None else "backtracking"
    if method == "newton_raphson" and getattr(problem, "m", problem.n) != problem.n:
        return dict(
            label=f"pysolverview_{method}_{configuration}",
            family="pysolverview",
            method=method,
            configuration=configuration,
            policy=policy,
            options=asdict(options),
            supported=False,
            success=False,
            native_success=None,
            message="Unsupported: Newton–Raphson requires a square system.",
            x=None,
            residual_norm=None,
            nfev=0,
            njev=0,
            nit=0,
            nupdates=0,
            evaluations=[],
            history=[],
            trials=[],
            verification_nfev=0,
        )
    solver = NonlinearSystemSolver(
        problem,
        method=method,
        library="pysolver",
        options=options,
        print_convergence=False,
        update_on="iteration",
    )
    solver.solve(problem.x0.copy())
    result = solver.result
    verified_norm = euclidean_norm(problem.residual(result.x))
    return {
        "label": f"pysolverview_{method}_{configuration}",
        "family": "pysolverview",
        "method": method,
        "configuration": configuration,
        "policy": policy,
        "supported": True,
        "options": asdict(options),
        "success": bool(result.success and verified_norm <= RESIDUAL_TOL),
        "native_success": bool(result.success),
        "message": str(result.message),
        "x": result.x.copy(),
        "residual_norm": verified_norm,
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "nit": int(result.nit),
        "nupdates": int(result.nupdates),
        "evaluations": result.evaluations,
        "history": result.history,
        "trials": result.trials,
        "verification_nfev": 1,
    }


def evaluation_displacements(evaluations, scale):
    """Return scaled distances between successive evaluations; these are not
    accepted-step lengths.
    """
    if len(evaluations) < 2:
        return np.empty(0)
    coordinates = np.array([record["x"] for record in evaluations])
    return np.array(
        [euclidean_norm(delta / scale) for delta in np.diff(coordinates, axis=0)]
    )


def summarize_case(problem, run):
    """Flatten one run into a CSV row of convergence, evaluation, rejection and
    step-limit diagnostics.
    """
    norms = [
        record.get("residual_norm", np.inf)
        for record in run["evaluations"]
        if record["valid"]
    ]
    displacement = evaluation_displacements(run["evaluations"], problem.x_scale)
    trials = run["trials"]
    return {
        "problem": problem.name,
        "dimension": problem.n,
        "residual_dimension": getattr(problem, "m", problem.n),
        "root_exists": getattr(problem, "root_exists", True),
        "supported": run.get("supported", True),
        "solver": run["label"],
        "method_name": METHOD_NAMES[run["method"]],
        "configuration": run.get(
            "configuration",
            f"factor={run['options']['factor']:g}" if run["family"] == "scipy" else "",
        ),
        "root_success": run["success"],
        "native_success": run["native_success"],
        "nfev": run["nfev"],
        "first_evaluation_below_residual_tol": next(
            (
                index
                for index, record in enumerate(run["evaluations"], start=1)
                if record["valid"] and record["residual_norm"] <= RESIDUAL_TOL
            ),
            None,
        ),
        "njev": run["njev"],
        "broyden_updates": run.get("nupdates"),
        "accepted_iterations": run["nit"],
        "final_residual_norm": run["residual_norm"],
        "max_evaluated_residual_norm": max(norms, default=np.nan),
        "invalid_evaluations": sum(
            not record["valid"] for record in run["evaluations"]
        ),
        "max_successive_evaluation_displacement_scaled": max(displacement, default=0.0),
        "max_trial_step_scaled": max(
            (record["scaled_step_norm"] for record in trials), default=None
        ),
        "max_accepted_step_scaled": max(
            (record["scaled_step_norm"] for record in run["history"]), default=None
        ),
        "rejected_trials": (
            sum(not record["accepted"] for record in trials) if trials else None
        ),
        "max_trial_bound_excess": max(
            (
                max(0.0, record["scaled_step_norm"] - trial_bound(record))
                for record in trials
                if trial_bound(record) is not None
            ),
            default=None,
        ),
        "max_configured_bound_excess": max(
            (
                max(0.0, record["scaled_step_norm"] - configured_bound(run))
                for record in trials
                if configured_bound(run) is not None
            ),
            default=None,
        ),
        "residual_increases_at_committed_points": sum(
            later["residual_norm"] > earlier["residual_norm"]
            for earlier, later in zip(run["history"], run["history"][1:])
        ),
        "message": run["message"].replace("\n", " "),
    }


def trial_bound(record):
    """Return the recorded radius or line-search step limit, with None meaning unbounded."""
    return record.get("radius", record.get("step_limit"))


def configured_bound(run):
    """Return the persistent configured cap for the run's algorithm and policy."""
    if run["method"] in ("lm", "hybr"):
        return run["options"].get("max_radius")
    backtracking = run["options"].get("backtracking")
    limits = backtracking.get("step_limits") if backtracking else None
    return limits["max_limit"] if limits else None


def audit_custom_steps(problem, run):
    """Check raw coordinates independently of the solver's recorded step norm."""
    if run["family"] != "pysolverview" or not run.get("supported", True):
        return
    for record in run["trials"]:
        observed_step = euclidean_norm(
            (record["x"] - record["x_base"]) / problem.x_scale
        )
        if not np.isclose(
            observed_step, record["scaled_step_norm"], rtol=1e-12, atol=0
        ):
            raise AssertionError(
                f"Incorrect recorded step in {problem.name}/{run['label']}"
            )
        bound = trial_bound(record)
        maximum = configured_bound(run)
        if bound is not None and observed_step > bound * (1 + 1e-12):
            raise AssertionError(
                f"Trial exceeded radius in {problem.name}/{run['label']}"
            )
        if bound is not None and maximum is not None and bound > maximum:
            raise AssertionError(
                f"Radius exceeded maximum in {problem.name}/{run['label']}"
            )
    norms = np.array([record["residual_norm"] for record in run["history"]])
    if run.get("policy") != "full_step" and np.any(
        np.diff(norms) > 1e-14 * np.maximum(1.0, norms[:-1])
    ):
        raise AssertionError(
            f"Accepted residual increased in {problem.name}/{run['label']}"
        )


def display_label(run):
    """Build a full author-name legend with backend, policy, and unsuccessful outcome."""
    backend = "SciPy/MINPACK" if run["family"] == "scipy" else "PySolverView"
    configuration = run.get("configuration")
    if configuration is None:
        configuration = f"factor {run['options']['factor']:g}"
    labels = {
        "controlled": "controlled",
        "full": "full step",
        "backtracking": "uncapped backtracking",
        "initial_only": "small initial radius",
        "generous": "generous radius",
    }
    label = f"{backend} {METHOD_NAMES[run['method']]} – {labels.get(configuration, configuration)}"
    if not run.get("supported", True):
        label += " (unsupported)"
    elif not run["success"]:
        label += " (no root)"
    return label


def plot_convergence(problem, runs, output, *, suffix="", title=None):
    """Save four-panel PNG/SVG comparisons of residual calls, accepted iterates and
    actual trial steps.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    colors = plt.get_cmap("tab10").colors

    def draw_history(axis, coordinates, values, **style):
        """Plot a trace with visible endpoints, including a singleton initial state."""
        endpoint_indices = sorted({0, len(coordinates) - 1}) if len(coordinates) else []
        axis.semilogy(
            coordinates,
            values,
            marker="o",
            markersize=3,
            markevery=endpoint_indices,
            **style,
        )

    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        label = display_label(run)
        if not run.get("supported", True):
            continue
        kwargs = dict(
            color=color,
            label=label,
            linewidth=1.6,
            linestyle="--" if run["method"] == "gauss_newton" else "-",
        )
        evaluations = run["evaluations"]
        residuals = [
            record.get("residual_norm", np.nan) if record["valid"] else np.nan
            for record in evaluations
        ]
        draw_history(
            axes[0, 0],
            np.arange(1, len(evaluations) + 1),
            np.maximum(residuals, 1e-16),
            **kwargs,
        )
        displacement = evaluation_displacements(evaluations, problem.x_scale)
        draw_history(
            axes[0, 1],
            np.arange(2, len(evaluations) + 1),
            np.maximum(displacement, 1e-16),
            **kwargs,
        )
        if run["family"] == "pysolverview":
            history = run["history"]
            draw_history(
                axes[1, 0],
                np.arange(len(history)),
                np.maximum([record["residual_norm"] for record in history], 1e-16),
                **kwargs,
            )
            trials = run["trials"]
            steps = np.maximum([record["scaled_step_norm"] for record in trials], 1e-16)
            draw_history(axes[1, 1], np.arange(1, len(trials) + 1), steps, **kwargs)
            rejected = [i for i, record in enumerate(trials) if not record["accepted"]]
            axes[1, 1].scatter(
                np.array(rejected) + 1, steps[rejected], color=color, marker="x", s=25
            )

    axes[0, 0].set(
        title="Every residual evaluation (invalid values omitted)",
        xlabel="Residual evaluation",
        ylabel="Residual 2-norm",
    )
    axes[0, 0].axhline(RESIDUAL_TOL, color="0.4", ls=":", lw=1)
    axes[0, 1].set(
        title="Consecutive evaluation displacement (not an accepted step)",
        xlabel="Residual evaluation",
        ylabel="Scaled displacement 2-norm",
    )
    axes[1, 0].set(
        title="Accepted iterates, custom solvers only",
        xlabel="Accepted iteration",
        ylabel="Residual 2-norm",
    )
    axes[1, 1].set(
        title="Trial from its accepted base; crosses mark rejection",
        xlabel="Trial number",
        ylabel="Scaled trial-step 2-norm",
    )
    axes[1, 1].axhline(
        CONFIGURATIONS["controlled"]["max_radius"], color="0.4", ls="--", lw=1
    )
    for ax in axes.flat:
        ax.grid(alpha=0.22, which="both")
    axes[0, 0].legend(fontsize=7, loc="best")
    axes[1, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{problem.name} (n={problem.n}, m={getattr(problem, 'm', problem.n)}): "
        + (title or "analytic Jacobians, fixed start and scales"),
        fontsize=13,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"{problem.name}{suffix}.{extension}", dpi=160)
    plt.close(fig)


def write_report(output, summary, artifacts, comparison):
    """Write an evidence-based Markdown report, retaining failures and unsupported cases."""
    eligible = [row for row in summary if row["supported"] and row["root_exists"]]
    negative = [row for row in summary if row["supported"] and not row["root_exists"]]
    skipped = [row for row in summary if not row["supported"]]
    title = (
        "SciPy/MINPACK versus PySolverView: Levenberg–Marquardt and Powell hybrid"
        if comparison == "backends"
        else "PySolverView: Levenberg–Marquardt, Powell hybrid, Newton–Raphson and Gauss–Newton"
    )
    lines = [
        f"# {title}",
        "",
        f"Verified roots: **{sum(row['root_success'] for row in eligible)}/{len(eligible)}** eligible runs on problems with roots.",
        f"Explicitly unsupported configurations: {len(skipped)}. Inconsistent-system controls: {len(negative)}, "
        f"of which {sum(not row['root_success'] for row in negative)} correctly did not report a root.",
        "",
        "## Protocol",
        "",
        "All cases use analytic Jacobians, fixed initial points and coordinate scales, and an independently verified residual 2-norm target of $10^{-8}$.",
        f"Custom budgets are {MAX_ITERATIONS} attempted trials and {MAX_NFEV} residual evaluations. A final verification evaluation is outside these solver counts.",
        "Native termination is retained separately. Native evaluation traces are not accepted-iterate histories; consecutive evaluation displacements are not solver trial lengths.",
        "The full labels in legends identify the method and backend. Raw records are in [traces.json](traces.json); flat metrics are in [summary.csv](summary.csv).",
        "",
    ]
    if comparison == "backends":
        lines += [
            "SciPy/MINPACK uses factors 100 and 0.1. PySolverView uses generous (100 initial, $10^6$ maximum, growth 2), small-initial-only (0.1 initial, $10^6$ maximum, growth 2), and controlled (0.1 initial, 1 maximum, growth 1.5) radii.",
            "These configurations reproduce the existing 100-case study when all ten original/large problems are selected. SciPy factor controls an initial scaled radius dependent on the starting point; it is not equivalent to an absolute PySolverView radius.",
        ]
    else:
        lines += [
            "The `_controlled` figures compare all four methods with initial scaled displacement limit 0.1, maximum 1, and growth factor 1.5. Trust-region and Armijo acceptance rules differ even when the displacement envelopes match.",
            "The `_newton_policies` figures compare controlled, uncapped backtracking, and full-step Newton–Raphson/Gauss–Newton. Full steps are allowed to increase residuals and terminate on an undefined trial. Uncapped backtracking starts each search at fraction one.",
            "Newton–Raphson requires square, numerically full-rank Jacobians; rectangular cases are recorded as unsupported rather than convergence failures. Gauss–Newton uses a minimum-norm least-squares direction when needed.",
            "The dense rectangular Rosenbrock controls are constructed extensions, not original MINPACK examples. The inconsistent case has an unavoidable constant residual of one and therefore no root.",
            "The coupled positive-domain logarithmic system deliberately exposes an undefined first full Newton step. It is a mathematical model-domain surrogate, not a thermodynamic property model. Actual property behavior is measured separately with CoolProp heated-duct cases.",
        ]
    domain_rows = [
        row
        for row in summary
        if row["problem"].startswith("coupled_logarithmic_domain")
    ]
    if domain_rows:
        lines += [
            "",
            "## Domain-recovery control",
            "",
            "The valid logarithmic-model start produces a negative-coordinate full Newton trial. The table distinguishes avoidance through finite displacement limits from recovery after an invalid evaluation. More conservative motion can require more calls.",
            "",
            "| Method | Policy | Root | Residual calls | Invalid calls | Largest scaled trial |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in domain_rows:
            lines.append(
                f"| {row['method_name']} | {row['configuration']} | {row['root_success']} | "
                f"{row['nfev']} | {row['invalid_evaluations']} | "
                f"{row['max_trial_step_scaled']:.3g} |"
            )
        lines += [
            "",
            f"[Controlled four-method histories]({domain_rows[0]['problem']}_controlled.png) and "
            f"[Newton-family policy histories]({domain_rows[0]['problem']}_newton_policies.png).",
        ]
    lines += [
        "",
        "## Outcomes by configuration",
        "",
        "| Method and configuration | Verified roots / eligible | Invalid evaluations | Rejected trials |",
        "| --- | --- | --- | --- |",
    ]
    for solver in dict.fromkeys(row["solver"] for row in summary):
        rows = [row for row in eligible if row["solver"] == solver]
        if not rows:
            continue
        first = rows[0]
        backend = "SciPy/MINPACK" if solver.startswith("scipy_") else "PySolverView"
        lines.append(
            f"| {backend} {first['method_name']} / {first['configuration']} | "
            f"{sum(row['root_success'] for row in rows)}/{len(rows)} | "
            f"{sum(row['invalid_evaluations'] for row in rows)} | "
            f"{sum(row['rejected_trials'] or 0 for row in rows)} |"
        )
    lines += [
        "",
        "## Individual outcomes",
        "",
        "| Problem | Method | Configuration | Root / outcome | Residual calls | Final residual norm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        outcome = (
            "unsupported"
            if not row["supported"]
            else (
                "root"
                if row["root_success"]
                else "expected no root" if not row["root_exists"] else "failed"
            )
        )
        residual = (
            "–"
            if row["final_residual_norm"] is None
            else f"{row['final_residual_norm']:.3e}"
        )
        lines.append(
            f"| {row['problem']} | {row['method_name']} | {row['configuration']} | {outcome} | {row['nfev']} | {residual} |"
        )
    lines += ["", "## Failure diagnostics", ""]
    failures = [row for row in eligible if not row["root_success"]]
    if failures:
        lines += [
            f"- **{row['problem']} / {row['method_name']} / {row['configuration']}**: {row['message']}"
            for row in failures
        ]
    else:
        lines.append("Every eligible run met the independently checked root target.")
    lines += [
        "",
        "## Audits and interpretation",
        "",
        "Every custom trial displacement was recomputed from its raw coordinates and checked against any active/configured bound. Monotonic accepted residuals were required only for protected policies. Full-step increases remain in the saved traces.",
        "A small limit can prevent an invalid initial step but can also increase work. Review failures and call counts alongside the residual history; no method is declared universally superior.",
        "Nominal thermodynamic runs without invalid property evaluations establish accuracy and displacement control, not demonstrated domain-recovery benefit. Physical solution diagnostics, when applicable, are stored with each run.",
        "These are convergence studies, not timings. Timing instrumentation and clean wall-clock comparisons are reported by the separate timing runner.",
        "",
        "## Environment",
        "",
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}. All benchmark classes implement NonlinearSystemProblem.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_benchmarks(
    output, *, problem_names=None, plots=True, suite="all", comparison="backends"
):
    """Run one explicit comparison family and save its summaries, traces, figures and report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if suite not in ("original", "large", "extension", "all"):
        raise ValueError("suite must be original, large, extension or all")
    if comparison not in ("backends", "custom"):
        raise ValueError("comparison must be backends or custom")
    if suite == "extension" and comparison == "backends":
        raise ValueError("extension suite is intended for the custom-method comparison")
    problems = get_problems() if suite in ("original", "all") else []
    if suite in ("large", "all"):
        problems.extend(get_large_problems())
    if comparison == "custom" and suite in ("extension", "all"):
        problems.extend(get_extension_problems())
    if problem_names:
        unknown = set(problem_names) - {problem.name for problem in problems}
        if unknown:
            raise ValueError(f"Unknown problem names: {sorted(unknown)}")
        problems = [problem for problem in problems if problem.name in problem_names]
    if not problems:
        raise ValueError("Select at least one benchmark problem")
    summary = []
    artifacts = {
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "residual_tol": RESIDUAL_TOL,
            "max_nfev": MAX_NFEV,
            "max_iterations_custom": MAX_ITERATIONS,
            "custom_configurations": CONFIGURATIONS,
            "custom_method_cases": CUSTOM_CASES,
            "method_names": METHOD_NAMES,
            "jacobians": "analytic",
            "suite": suite,
            "comparison": comparison,
            "implementation": "PySolverView controlled trust-region and Newton-family line-search methods",
            "trial_bound_audit_relative_tolerance": 1e-12,
            "notes": "Final verification and physical diagnostics are outside solver counters. Native evaluation displacements are not accepted steps. Full-step methods may increase residuals. Unsupported dimensions and inconsistent controls are reported separately.",
        },
        "problems": [],
    }
    for problem in problems:
        if comparison == "backends":
            runs = [
                run_scipy_case(problem, method, factor)
                for method in ("lm", "hybr")
                for factor in (100.0, 0.1)
            ]
            runs.extend(
                run_custom_case(problem, method, configuration)
                for method in ("lm", "hybr")
                for configuration in CONFIGURATIONS
            )
        else:
            runs = [
                run_custom_case(problem, method, configuration)
                for method, configuration in CUSTOM_CASES
            ]
        for run in runs:
            audit_custom_steps(problem, run)
            if run["x"] is not None and hasattr(problem, "solution_diagnostics"):
                run["physical_diagnostics"] = problem.solution_diagnostics(run["x"])
                run["verification_diagnostic_calls"] = 1
        summary.extend(summarize_case(problem, run) for run in runs)
        artifacts["problems"].append(
            {
                "name": problem.name,
                "class": type(problem).__name__,
                "dimension": problem.n,
                "residual_dimension": getattr(problem, "m", problem.n),
                "root_exists": getattr(problem, "root_exists", True),
                "description": problem.description,
                "source": problem.source,
                "x0": problem.x0,
                "x_scale": problem.x_scale,
                "reference_solution": problem.solution,
                "runs": runs,
            }
        )
        for run in runs:
            final = run["residual_norm"]
            final_text = "unavailable" if final is None else f"{final:.2e}"
            print(
                f"{problem.name:36} {run['label']:42} success={str(run['success']):5} nfev={run['nfev']:4} residual={final_text}"
            )
        if plots:
            if comparison == "backends":
                plot_convergence(problem, runs, output)
            else:
                plot_convergence(
                    problem,
                    [run for run in runs if run["configuration"] == "controlled"],
                    output,
                    suffix="_controlled",
                    title="four PySolverView methods, common displacement controls",
                )
                plot_convergence(
                    problem,
                    [
                        run
                        for run in runs
                        if run["method"] in ("newton_raphson", "gauss_newton")
                    ],
                    output,
                    suffix="_newton_policies",
                    title="Newton-family methods: controlled, uncapped backtracking, full steps",
                )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "traces.json").write_text(
        json.dumps(to_json_compatible(artifacts), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output, summary, artifacts, comparison)
    print(f"Saved results to {output.resolve()}")
    return summary


def main():
    """Parse explicit comparison families and output paths, then run selected convergence studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_convergence",
        help="Output for the SciPy/PySolverView comparison.",
    )
    parser.add_argument(
        "--custom-output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_custom_convergence",
        help="Output for the four-method PySolverView study.",
    )
    parser.add_argument(
        "--comparison", choices=("backends", "custom", "both"), default="both"
    )
    parser.add_argument(
        "--problems", nargs="+", help="Run only the named fixed problems."
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--suite", choices=("original", "large", "extension", "all"), default="all"
    )
    args = parser.parse_args()
    for comparison in ("backends", "custom"):
        if args.comparison not in (comparison, "both"):
            continue
        run_benchmarks(
            args.output if comparison == "backends" else args.custom_output,
            problem_names=args.problems,
            plots=not args.no_plots,
            suite=args.suite,
            comparison=comparison,
        )


if __name__ == "__main__":
    main()
