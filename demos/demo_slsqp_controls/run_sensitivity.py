"""Compare selected first-step fractions on constrained and unconstrained Rosenbrock.

Run from the repository root with --no-show for a headless reproducible study.
Study-specific execution, plots, and export code live in this script.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pysolver_view as psv
from pysolver_view.optimization.problems import StudyProblem
from pysolver_view.diagnostics.slsqp_audit import audit_solution


def configurations():
    """Return a compact empirically selected first-fraction comparison.

    Growth remains 1.5, the maximum fraction remains 1, and recovery is disabled.
    Fractions were selected after a logarithmic search and local refinement to
    show large overshoot, bounded trials, and a monotonic constrained trajectory.
    """
    profiles = [dict(name="scipy", library="scipy", group="baseline", controls={})]
    for fraction in [1.0, 0.1, 0.016, 0.005, 0.001]:
        profiles.append(
            dict(
                name=f"first_{fraction:g}",
                library="pysolver",
                group="first",
                controls=dict(
                    first_step_fraction=fraction,
                    max_step_fraction=1.0,
                    step_growth_factor=1.5,
                    domain_backtracking=False,
                ),
            )
        )
    return profiles


def objective_behavior(values):
    """Measure raw increases without smoothing or replacing data by best-so-far.

    Ignore only numerical changes <= 1e-10 * max(1, abs(previous objective)).
    Peak ratio is relative to the starting objective, not the final objective.
    """
    values = np.asarray(values)
    differences = np.diff(values)
    increases = int(
        np.count_nonzero(differences > 1e-10 * np.maximum(1.0, np.abs(values[:-1])))
    )
    return dict(
        monotonic=increases == 0,
        increases=increases,
        peak_ratio=float(np.max(values) / values[0]),
    )


def assess_history(problem, history):
    """Assess recorded points after solving, without adding model calls or events.

    This gives both backends the same diagnostic definition even when their
    automatic KKT schedules differ. Unavailable assessments remain NaN gaps.
    """
    lower, upper = problem.get_bounds()
    optimality, available = [], []
    for point in history["x"]:
        point = np.asarray(point)
        values, jac = problem.values(point), problem.gradient(point)
        assessment = psv.assess_kkt(
            point,
            jac[0],
            np.empty(0),
            np.empty((0, 2)),
            values[1:],
            jac[1:],
            lower,
            upper,
            feasibility_tol=1e-6,
            optimality_tol=1e-4,
        )
        available.append(assessment["available"])
        optimality.append(
            assessment["optimality"] if assessment["available"] else np.nan
        )
    return dict(
        model_calls=list(history["func_count_total"]),
        optimality=optimality,
        available=available,
    )


def run_case(case, profile):
    """Run one fresh model and retain failures, trial attempts, and independent audit."""
    problem = StudyProblem(case)
    options = (
        {"feasibility_tol": 1e-6, "optimality_tol": 1e-4, **profile["controls"]}
        if profile["library"] == "pysolver"
        else {}
    )
    solver = psv.OptimizationSolver(
        problem,
        library=profile["library"],
        method="slsqp",
        tolerance=1e-12,
        max_iterations=400,
        extra_options=options,
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
    )
    exception = None
    try:
        solver.solve(problem.initial)
    except psv.DomainError as error:
        # Native SciPy has no domain retry hook. Keep the failure in the study.
        exception = str(error)
    calls = list(problem.calls)
    points = np.array([call["x"] for call in calls])
    moved = [point for point in points if not np.array_equal(point, problem.initial)]
    row = dict(
        case=case,
        profile=profile["name"],
        library=profile["library"],
        group=profile["group"],
        controls=options,
        success=False,
        solver_success=False,
        kkt_passed=False,
        nfev=len(calls),
        nit=None,
        objective=None,
        kkt_ratio=None,
        first_trial_distance=(
            float(np.linalg.norm(moved[0] - problem.initial)) if moved else 0.0
        ),
        maximum_excursion=float(
            np.max(np.linalg.norm(points - problem.initial, axis=1))
        ),
        largest_objective=max(call["objective"] for call in calls if call["valid"]),
        invalid_trials=sum(not call["valid"] for call in calls),
        message=exception,
        calls=calls,
        history=solver.convergence_history,
        kkt=None,
        audit=None,
    )
    if solver.result is not None:
        x = solver.x_final
        values, jac = problem.values(x), problem.gradient(x)
        lower, upper = problem.get_bounds()
        assessment = psv.assess_kkt(
            x,
            jac[0],
            np.empty(0),
            np.empty((0, 2)),
            values[1:],
            jac[1:],
            lower,
            upper,
            feasibility_tol=1e-6,
            optimality_tol=1e-4,
        )
        audit = audit_solution(problem, solver, assessment)
        row.update(
            success=bool(solver.success),
            solver_success=bool(solver.result.solver_success),
            kkt_passed=assessment["passed"],
            nit=solver.iteration_count,
            objective=float(values[0]),
            kkt_ratio=assessment["kkt_ratio"],
            message=solver.message,
            x=x,
            kkt=assessment,
            audit=audit,
        )
        assert audit[
            "passed"
        ], f"Independent audit failed: {case}/{profile['name']}: {audit}"
        assert solver.result.nfev == len(
            calls
        ), "Audit must not be counted as solver work"
    raw_behavior = objective_behavior(
        [call["objective"] for call in calls if call["valid"]]
    )
    history_behavior = objective_behavior(solver.convergence_history["objective_value"])
    row.update(
        raw_monotonic=raw_behavior["monotonic"],
        raw_increases=raw_behavior["increases"],
        peak_ratio=raw_behavior["peak_ratio"],
        history_monotonic=history_behavior["monotonic"],
        history_increases=history_behavior["increases"],
    )
    row["optimality_history"] = assess_history(problem, solver.convergence_history)
    return row


def json_value(value):
    """Convert NumPy values and nonfinite unavailable quantities to portable JSON."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_case(rows, case, output):
    """Show marked objective and optimality histories using consistent colors."""
    psv.set_plot_options(grid=False)
    selected = [row for row in rows if row["case"] == case]
    colormap = plt.get_cmap("plasma")
    sweep_colors = iter(colormap(np.linspace(0.2, 0.8, len(selected) - 1)))
    colors = [
        "black" if row["profile"] == "scipy" else next(sweep_colors) for row in selected
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for row, color in zip(selected, colors):
        calls = row["calls"]
        label = (
            "SciPy"
            if row["library"] == "scipy"
            else f"First = {row['controls']['first_step_fraction']:.2g}"
        )
        xs = np.arange(1, len(calls) + 1)
        objectives = [call["objective"] if call["valid"] else np.nan for call in calls]
        axes[0, 0].plot(
            xs, objectives, "o-", markersize=3, label=label, linewidth=1, color=color
        )
        axes[0, 1].plot(
            xs[:12], objectives[:12], "o-", markersize=3, label=label, color=color
        )
        history = row["history"]
        axes[1, 0].plot(
            history["func_count_total"],
            history["objective_value"],
            "o-",
            markersize=3,
            label=label,
            color=color,
        )
        diagnostics = row["optimality_history"]
        axes[1, 1].plot(
            diagnostics["model_calls"],
            diagnostics["optimality"],
            "o-",
            markersize=3,
            label=label,
            color=color,
        )
    for ax in axes.flat[:3]:
        # Keep positive objectives on a log scale without a large unused region
        # between zero and the constrained minimum. Preserve exact zeros if any.
        plotted = np.concatenate([line.get_ydata() for line in ax.lines])
        if np.all(plotted > 0):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-16)
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Model calls (analytic derivatives; audits excluded)")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Objective at all valid model attempts")
    axes[0, 1].set_ylabel("Objective at all model attempts")
    axes[0, 1].set_title("First 12 model calls (no smoothing)")
    axes[1, 0].set_ylabel("Objective at recorded iteration points")
    axes[1, 0].set_title("Custom: accepted points; SciPy: major-iteration observations")
    axes[1, 1].set_yscale("symlog", linthresh=1e-12)
    axes[1, 1].axhline(1e-4, color="black", linestyle="--", label="Optimality target")
    axes[1, 1].set_ylabel("First-order optimality")
    axes[1, 1].set_xlabel("Model calls (post-solve diagnostics excluded)")
    axes[1, 1].set_title("max(stationarity, dual violation, complementarity)")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        case.replace("_", " ").title()
        + " - first step fraction sensitivity (growth 1.5)"
    )
    fig.savefig(output / f"{case}_first.png", dpi=160)
    return fig


def write_results(rows, output):
    """Save complete data, a compact CSV, and a factual Markdown outcome table."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(json_value(rows), indent=2, allow_nan=False), encoding="utf-8"
    )
    fields = [
        "case",
        "profile",
        "success",
        "solver_success",
        "kkt_passed",
        "nit",
        "nfev",
        "first_trial_distance",
        "maximum_excursion",
        "largest_objective",
        "peak_ratio",
        "raw_monotonic",
        "raw_increases",
        "history_monotonic",
        "history_increases",
        "invalid_trials",
        "objective",
        "kkt_ratio",
        "message",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Controlled SLSQP sensitivity results",
        "",
        "First-step fractions: 1, 0.1, 0.016, 0.005, 0.001. Growth factor 1.5; maximum fraction 1; domain recovery disabled.",
        "Only constrained and unconstrained Rosenbrock are solved. Colors: plasma 0.2-0.8; SciPy is black.",
        "Fractions were selected after a 121-point logarithmic search per problem and local refinement; this is an educational selection, not universal tuning.",
        "Monotonicity uses raw data with tolerance 1e-10*max(1,abs(previous objective)); no best-so-far curve is substituted.",
        "All objective and optimality curves have markers at every point. The optimality history is assessed after solving on each recorded point; these diagnostics do not add solver iterations/model calls.",
        "",
        "Analytical derivatives; native ftol=1e-12; 400-iteration budget; feasibility=1e-6; optimality=1e-4.",
        "Native success and independent KKT acceptance are shown separately. Failures are retained.",
        "First trial is the first distinct model point, not an accepted iterate. Excursion is distance from the start.",
        "All returned solutions are independently audited, including rejected outcomes; native domain exceptions have no final certificate.",
        "",
        "| Case | Profile | Success | KKT pass | Calls | Peak / initial | Raw rises | Recorded rises | KKT ratio |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = "unavailable" if row["kkt_ratio"] is None else f"{row['kkt_ratio']:.3g}"
        lines.append(
            f"| {row['case']} | {row['profile']} | {row['success']} | {row['kkt_passed']} | {row['nfev']} | {row['peak_ratio']:.3g} | {row['raw_increases']} | {row['history_increases']} | {ratio} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Run the full sensitivity matrix, export results, and show comparison figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show", action="store_true", help="Save figures without opening windows"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_sensitivity",
    )
    args = parser.parse_args()
    rows = []
    for case in ["rosenbrock", "constrained_rosenbrock"]:
        for profile in configurations():
            row = run_case(case, profile)
            rows.append(row)
            print(
                f"{case:24} {row['profile']:16} success={row['success']!s:5} KKT={row['kkt_passed']!s:5} calls={row['nfev']:4} peak/start={row['peak_ratio']:.3g} raw_rises={row['raw_increases']} recorded_rises={row['history_increases']}"
            )
    write_results(rows, args.output)
    for case in ["rosenbrock", "constrained_rosenbrock"]:
        plot_case(rows, case, args.output)
    print(f"Results: {args.output.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
