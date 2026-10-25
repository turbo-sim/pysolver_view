"""Measure optional KKT overhead on a constrained two-variable Rosenbrock problem.

Run from the repository root. Solver construction, plots, printing, and optional
post-solve reports are outside the timed region. Native solver settings match
within each enabled/disabled pair.
"""

import argparse
import csv
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pysolver_view as psv


class TimingProblem(psv.OptimizationProblem):
    """Minimize Rosenbrock subject to x+y<=1, with an active constraint at the solution."""

    def __init__(self):
        """Name the variables and record the actual model evaluation trajectory."""
        self.variable_names = ["x", "y"]
        self.calls = []

    def fitness(self, x):
        """Evaluate the objective and nonpositive linear constraint together."""
        self.calls.append(x.copy())
        a, b = x
        return np.array([100 * (b - a * a) ** 2 + (1 - a) ** 2, a + b - 1])

    def gradient(self, x):
        """Return exact objective and constraint derivative rows."""
        a, b = x
        return np.array(
            [[400 * a * (a * a - b) + 2 * (a - 1), 200 * (b - a * a)], [1.0, 1.0]]
        )

    def get_bounds(self):
        """Use unbounded coordinates so box clipping does not change the study."""
        return [-np.inf, -np.inf], [np.inf, np.inf]

    def get_nec(self):
        """Declare no equality constraints."""
        return 0

    def get_nic(self):
        """Declare the x+y<=1 inequality."""
        return 1


def known_solution():
    """Derive the positive boundary minimum from its cubic stationarity equation."""
    roots = np.roots([400.0, 600.0, -198.0, -202.0])
    a = next(
        root.real for root in roots if abs(root.imag) < 1e-12 and 0 < root.real < 1
    )
    x = np.array([a, 1 - a])
    return x, 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def timed_solve(library, check_kkt):
    """Time solve only and verify its result against the analytical minimum."""
    problem = TimingProblem()
    solver = psv.OptimizationSolver(
        problem,
        library=library,
        method="slsqp",
        tolerance=1e-12,
        max_iterations=200,
        check_kkt=check_kkt,
        update_on="gradient",
        print_convergence=False,
        plot_convergence=False,
        extra_options=(
            {
                "first_step_fraction": 1.0,
                "max_step_fraction": 1.0,
                "step_growth_factor": None,
                "domain_backtracking": False,
                "max_backtracks": 0,
                "feasibility_tol": 1e-6,
                "optimality_tol": 1e-6,
            }
            if library == "pysolver"
            else {}
        ),
    )
    started = perf_counter()
    solver.solve(np.array([-1.2, 1.0]))
    total = perf_counter() - started
    # No on-demand KKT call here: the disabled run must really avoid the checker.
    expected_x, expected_f = known_solution()
    np.testing.assert_allclose(solver.x_final, expected_x, atol=2e-6, rtol=0)
    assert solver.success, solver.message
    assert abs(solver.f_final - expected_f) < 2e-6
    assert (
        solver.iteration_count >= 10
    ), "Timing problem must require at least 10 iterations"
    result = solver.result
    if check_kkt:
        assert result.kkt_evaluations > 0, "Enabled timing must include actual KKT work"
    else:
        assert result.kkt_evaluations == 0 and result.kkt_time == 0
        assert result.kkt_passed is None
    # Actual model points are comparable across backends. Their callback streams
    # differ: native major-iteration observations need not be accepted points.
    trajectory = [point.tolist() for point in problem.calls]
    return dict(
        library=library,
        check_kkt=check_kkt,
        total_ms=1000 * total,
        kkt_ms=1000 * result.kkt_time,
        other_ms=1000 * (total - result.kkt_time),
        kkt_evaluations=result.kkt_evaluations,
        model_calls=result.nfev,
        gradient_calls=result.njev,
        iterations=result.nit,
        objective=float(result.fun),
        history_rows=len(solver.convergence_history["x"]),
        trajectory=trajectory,
    )


def print_summary(summary, paired, repeats):
    """Print aligned ASCII tables with clear timing and work-count labels."""
    print("\nSLSQP timing: automatic KKT checks on and off")
    print("Problem: constrained Rosenbrock, start (-1.2, 1.0), x+y <= 1.")
    print("Full steps; no step ramp; domain recovery disabled.")
    print(
        f"{repeats} warmed pairs per backend; all four model evaluation trajectories match."
    )
    print(
        "Times are medians in milliseconds. Checks and model/Jacobian calls are counts.\n"
    )
    header = (
        f"{'Backend':<12} {'KKT':<5} {'Total ms':>10} {'KKT ms':>10} "
        f"{'Other ms':>10} {'P25-P75 total ms':>21} {'Checks':>8} {'Model/Jac':>11} {'Iter':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        interval = f"{row['p25_ms']:.3f} - {row['p75_ms']:.3f}"
        calls = f"{row['model_calls']}/{row['gradient_calls']}"
        print(
            f"{row['library']:<12} {'on' if row['check_kkt'] else 'off':<5} "
            f"{row['total_ms']:>10.3f} {row['kkt_ms']:>10.3f} {row['other_ms']:>10.3f} "
            f"{interval:>21} {row['kkt_evaluations']:>8} {calls:>11} {row['iterations']:>6}"
        )
    print("-" * len(header))
    print("\nExtra solve time from enabling KKT (paired median):")
    for row in paired:
        print(
            f"  {row['library']:<10} {row['median_extra_ms']:>8.3f} ms "
            f"(middle 50%: {row['p25_extra_ms']:.3f} - {row['p75_extra_ms']:.3f} ms)"
        )
    print("\nKKT time includes multiplier estimation and residual reconstruction.")
    print("Other time includes model, solver, wrapper, and history/report bookkeeping.")
    print("KKT schedules can differ even when point trajectories match; see Checks.")
    print(
        "Trajectory comparison uses all model evaluations, not backend-specific callbacks."
    )


def main():
    """Warm up, alternate paired solves, and export measured overhead with plots."""
    psv.set_plot_options(grid=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "development/slsqp_controls/results_kkt_timing",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cold, rows = [], []
    for library in ["scipy", "pysolver"]:
        for enabled in [False, True]:
            cold.append(timed_solve(library, enabled))
        for _ in range(3):
            for enabled in [True, False]:
                timed_solve(library, enabled)
        for repeat in range(args.repeats):
            order = [False, True] if repeat % 2 == 0 else [True, False]
            for enabled in order:
                rows.append(dict(repeat=repeat, **timed_solve(library, enabled)))

    reference = cold[0]
    for sample in cold + rows:
        np.testing.assert_allclose(
            sample["trajectory"], reference["trajectory"], atol=1e-12, rtol=1e-12
        )
        assert sample["iterations"] == reference["iterations"]
        assert sample["model_calls"] == reference["model_calls"]
        # KKT may request a final derivative that the native algorithm did not
        # need. Count that work rather than requiring artificial equal counts.
        assert (
            reference["gradient_calls"]
            <= sample["gradient_calls"]
            <= reference["gradient_calls"] + 1
        )

    summary = []
    for library in ["scipy", "pysolver"]:
        for enabled in [False, True]:
            subset = [
                r for r in rows if r["library"] == library and r["check_kkt"] == enabled
            ]
            totals = [r["total_ms"] for r in subset]
            summary.append(
                dict(
                    library=library,
                    check_kkt=enabled,
                    total_ms=float(np.median(totals)),
                    p25_ms=float(np.percentile(totals, 25)),
                    p75_ms=float(np.percentile(totals, 75)),
                    kkt_ms=float(np.median([r["kkt_ms"] for r in subset])),
                    other_ms=float(np.median([r["other_ms"] for r in subset])),
                    kkt_evaluations=subset[0]["kkt_evaluations"],
                    iterations=subset[0]["iterations"],
                    model_calls=subset[0]["model_calls"],
                    gradient_calls=subset[0]["gradient_calls"],
                )
            )
    paired = []
    for library in ["scipy", "pysolver"]:
        differences = []
        for repeat in range(args.repeats):
            pair = {
                r["check_kkt"]: r
                for r in rows
                if r["library"] == library and r["repeat"] == repeat
            }
            differences.append(pair[True]["total_ms"] - pair[False]["total_ms"])
        paired.append(
            dict(
                library=library,
                median_extra_ms=float(np.median(differences)),
                p25_extra_ms=float(np.percentile(differences, 25)),
                p75_extra_ms=float(np.percentile(differences, 75)),
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        repeats=args.repeats,
        problem="constrained_rosenbrock",
        initial=[-1.2, 1.0],
        constraint="x+y<=1",
        controls=dict(
            first_step_fraction=1.0,
            max_step_fraction=1.0,
            step_growth_factor=None,
            domain_backtracking=False,
            max_backtracks=0,
        ),
        matched_trajectories=True,
        initial_runs=cold,
        summary=summary,
        paired_overhead=paired,
        samples=rows,
    )
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output / "samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Optional KKT timing on constrained Rosenbrock",
        "",
        "Full first/max fractions, no growth ramp, domain recovery disabled. All model trajectories, iteration counts, and model call counts were checked to match. KKT may require one additional final Jacobian.",
        "Actual model evaluations define trajectory comparison; callback observation streams have different semantics. Raw history row counts are also exported.",
        "Gradient observations enable native SciPy KKT checks. Custom SLSQP assesses initial/accepted points and may compute one extra final Jacobian; compare the assessment counts below.",
        "Problem: constrained Rosenbrock f=100*(y-x*x)^2+(1-x)^2, x+y<=1, start=(-1.2,1). Native ftol=1e-12; feasibility/optimality=1e-6.",
        f"{args.repeats} warmed paired repetitions per backend; alternating on/off order. All solutions checked analytically.",
        "No printing, plotting, constructor, or post-solve assessment is timed. Solve histories/results are timed.",
        "Initial runs are exported separately and excluded from the warmed summary; these are not isolated-process cold starts.",
        "",
        "| Backend | Checks | Median total ms | P25-P75 ms | Median KKT ms | Median other ms | KKT calls | Model/Jacobian calls |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['library']} | {'on' if row['check_kkt'] else 'off'} | {row['total_ms']:.3f} | {row['p25_ms']:.3f}-{row['p75_ms']:.3f} | {row['kkt_ms']:.3f} | {row['other_ms']:.3f} | {row['kkt_evaluations']} | {row['model_calls']}/{row['gradient_calls']} |"
        )
    lines += [
        "",
        "KKT time measures the numerical assessor, including multiplier estimation and residual reconstruction.",
        "Other time includes the model, derivatives, solver, Python wrapper, history copying, and report formatting.",
        "Paired extra time also includes any final derivatives needed only for KKT and changed bookkeeping; it need not equal assessor time.",
        "Medians of components need not add to the median total. A cheap quadratic exaggerates the relative overhead compared with costly physical models.",
        "",
    ]
    for row in paired:
        lines.append(
            f"- {row['library']}: median paired extra time {row['median_extra_ms']:.3f} ms (P25-P75 {row['p25_extra_ms']:.3f}-{row['p75_extra_ms']:.3f} ms)."
        )
    report = "\n".join(lines) + "\n"
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print_summary(summary, paired, args.repeats)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = [
        f"{r['library']}\nKKT {'on' if r['check_kkt'] else 'off'}" for r in summary
    ]
    for ax, metric, title in zip(
        axes, ["total_ms", "kkt_ms"], ["Total solve time", "Time inside KKT assessor"]
    ):
        values = [row[metric] for row in summary]
        bars = ax.bar(labels, values, color=["gray", "tab:blue", "gray", "tab:orange"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylabel("Median milliseconds")
        ax.set_title(title)
        ax.margins(y=0.2)
    fig.savefig(args.output / "timing.png", dpi=160)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
