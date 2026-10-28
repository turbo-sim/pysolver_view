"""Compare native Pounce NLP solvers with controlled PySolverView SLSQP."""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import pysolver_view as psv
from pysolver_view.optimization.problems import CoupledRosenbrockProblem


class NativeProblem:
    """Adapt combined model values to Pounce without redundant same-point work.

    The intermediate callback has no primal-vector argument. ``objective``
    retains its own most recent point, as Pounce's minimize facade does. We
    label these coordinates as reconstructed and verify objective agreement
    after the solve; they are not declared accepted-iterate events.
    """

    def __init__(self, problem):
        """Own caches, counters and detached native callback observations."""
        self.problem = problem
        self.value_x = None
        self.values = None
        self.jacobian_x = None
        self.derivatives = None
        self.objective_x = None
        self.model_calls = 0
        self.derivative_calls = 0
        self.history = []

    def _values(self, x):
        """Evaluate the combined fitness only when its exact input changes."""
        if self.value_x is None or not np.array_equal(x, self.value_x):
            self.value_x = np.array(x, copy=True)
            self.values = self.problem.fitness(x)
            self.model_calls += 1
        return self.values

    def _derivatives(self, x):
        """Evaluate the full analytic Jacobian once per distinct input."""
        if self.jacobian_x is None or not np.array_equal(x, self.jacobian_x):
            self.jacobian_x = np.array(x, copy=True)
            self.derivatives = self.problem.gradient(x)
            self.derivative_calls += 1
        return self.derivatives

    def objective(self, x):
        """Evaluate the objective and retain the objective callback's own input."""
        self.objective_x = np.array(x, copy=True)
        return self._values(x)[0]

    def gradient(self, x):
        """Return the analytic objective gradient."""
        return self._derivatives(x)[0]

    def constraints(self, x):
        """Return unchanged public equality and nonpositive inequality values."""
        return self._values(x)[1:]

    def jacobian(self, x):
        """Return the dense, row-major constraint Jacobian expected by Pounce."""
        return self._derivatives(x)[1:].ravel()

    def intermediate(self, **stats):
        """Store metrics and reconstructed coordinates without extra model calls."""
        self.history.append(
            {
                "iteration": int(stats["iter_count"]),
                "x": (
                    None
                    if self.objective_x is None or int(stats["alg_mod"]) != 0
                    else self.objective_x.copy()
                ),
                "event_kind": "native_iteration",
                "accepted": None,
                "coordinate_source": "last_objective_evaluation",
                "native_objective": float(stats["obj_value"]),
                "native_primal_infeasibility": float(stats["inf_pr"]),
                "native_dual_infeasibility": float(stats["inf_du"]),
                "barrier_parameter": float(stats["mu"]),
                "line_search_trials": int(stats["ls_trials"]),
                "native_phase": int(stats["alg_mod"]),
                "model_calls": self.model_calls,
                "derivative_calls": self.derivative_calls,
            }
        )
        return True


def audit_point(problem, x, feasibility_tol=1e-7, optimality_tol=1e-5):
    """Independently assess the original model with fresh exact derivatives."""
    values = problem.fitness(x)
    matrix = problem.gradient(x)
    equality_end = 1 + problem.get_nec()
    lower, upper = problem.get_bounds()
    assessment = psv.assess_kkt(
        x,
        matrix[0],
        values[1:equality_end],
        matrix[1:equality_end],
        values[equality_end:],
        matrix[equality_end:],
        lower,
        upper,
        feasibility_tol=feasibility_tol,
        optimality_tol=optimality_tol,
    )
    return {
        "objective": float(values[0]),
        "constraint_violation": float(assessment["primal_feasibility"]),
        "stationarity": assessment["stationarity"],
        "complementarity": assessment["complementarity"],
        "kkt_residual": assessment["kkt_residual"],
        "kkt_passed": bool(assessment["passed"]),
        "kkt_available": bool(assessment["available"]),
    }


def finish_run(problem, run):
    """Audit detached histories after solving and keep the returned point separate."""
    run["audit"] = audit_point(problem, run["x"])
    for row in run["history"]:
        if row["x"] is None:
            continue
        row.update(audit_point(problem, row["x"]))
        if "native_objective" in row:
            mismatch = abs(row["native_objective"] - row["objective"])
            row["objective_cache_mismatch"] = mismatch
            row["coordinate_verified"] = bool(
                mismatch <= 1e-9 * max(1.0, abs(row["objective"]))
                and row["native_phase"] == 0
            )
        else:
            row["coordinate_verified"] = True
    return run


def run_native(problem, method, tolerance, max_iterations):
    """Solve the original problem directly through Pounce's Problem API."""
    import pounce

    adapter = NativeProblem(problem)
    lower, upper = problem.get_bounds()
    equality_count, inequality_count = problem.get_nec(), problem.get_nic()
    native = pounce.Problem(
        n=len(lower),
        m=equality_count + inequality_count,
        problem_obj=adapter,
        lb=lower,
        ub=upper,
        cl=np.r_[np.zeros(equality_count), np.full(inequality_count, -np.inf)],
        cu=np.zeros(equality_count + inequality_count),
    )
    options = {
        "algorithm": "interior-point" if method == "ipm" else "active-set-sqp",
        "tol": tolerance,
        "max_iter": max_iterations,
        "print_level": 0,
        "hessian_approximation": "limited-memory",
        "bound_relax_factor": 0.0,
        "nlp_scaling_method": "none",
    }
    if method == "sqp":
        options.update(
            sqp_hessian="damped-bfgs",
            sqp_max_iter=max_iterations,
            sqp_tol=tolerance,
            sqp_constr_viol_tol=tolerance,
            sqp_dual_inf_tol=tolerance,
        )
    for name, value in options.items():
        native.add_option(name, value)
    x, info = native.solve(x0=problem.initial)
    return finish_run(
        problem,
        {
            "key": f"pounce_{method}",
            "label": (
                "Pounce interior-point" if method == "ipm" else "Pounce active-set SQP"
            ),
            "interface": "native pounce.Problem",
            "success": int(info["status"]) == 0,
            "status": int(info["status"]),
            "message": str(info["status_msg"]),
            "iterations": int(info["iter_count"]),
            "model_calls": adapter.model_calls,
            "derivative_calls": adapter.derivative_calls,
            "native_unscaled_kkt": info.get("final_unscaled_kkt_error"),
            "qp_solves": info.get("n_qp_solves"),
            "qp_working_set_changes": info.get("n_qp_ws_changes"),
            "options": options,
            "x": np.array(x, copy=True),
            "history": adapter.history,
        },
    )


def run_custom(problem, first_step_fraction, tolerance, max_iterations):
    """Run public custom SLSQP, recording its initial and accepted-point events."""
    history = []

    def accepted(entry):
        """Detach authoritative custom-driver coordinates and accounting."""
        history.append(
            {
                "iteration": int(entry["iteration"]),
                "x": np.array(entry["x"], copy=True),
                "event_kind": "accepted_iteration",
                "accepted": True,
                "coordinate_source": "custom_driver",
            }
        )

    solver = psv.OptimizationSolver(
        problem,
        library="pysolver",
        method="slsqp",
        tolerance=tolerance,
        max_iterations=max_iterations,
        print_convergence=False,
        update_on="iteration",
        backend_callback=accepted,
        extra_options={
            "first_step_fraction": first_step_fraction,
            "max_step_fraction": 1.0,
            "step_growth_factor": None,
            "feasibility_tol": 1e-7,
            "optimality_tol": 1e-5,
        },
    )
    x = solver.solve(problem.initial)
    return finish_run(
        problem,
        {
            "key": f"slsqp_{first_step_fraction:.2f}",
            "label": f"Custom SLSQP, first-step fraction {first_step_fraction:.2f}",
            "interface": "pysolver_view",
            "success": bool(solver.success),
            "status": solver.result.status,
            "message": str(solver.message),
            "iterations": int(solver.result.nit),
            "model_calls": int(solver.result.nfev),
            "derivative_calls": int(solver.result.njev),
            "options": dict(solver.result.effective_options),
            "events": solver.result.events,
            "x": np.array(x, copy=True),
            "history": history,
        },
    )


def json_safe(value):
    """Convert detached numerical records to portable JSON without NaN literals."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def environment_metadata():
    """Stamp reproducible numerical results with source and installed versions."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "pysolver_view_commit": commit,
        "pysolver_view_worktree_modified": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
            ).strip()
        ),
        "python": platform.python_version(),
        "pounce_solver": version("pounce-solver"),
        "pysolver_view": version("pysolver_view"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "timing_comparison": False,
    }


def plot_comparison(runs, output, show=False):
    """Plot actual recorded objectives, original feasibility and KKT residuals."""
    psv.set_plot_options(fontsize=11)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(runs)))
    for run, color in zip(runs, colors):
        rows = [row for row in run["history"] if row.get("coordinate_verified")]
        iterations = [row["iteration"] for row in rows]
        for axis, key in zip(
            axes, ("objective", "constraint_violation", "kkt_residual")
        ):
            values = [row[key] for row in rows]
            if key != "objective":
                values = [max(value, 1e-16) for value in values]
            axis.plot(
                iterations,
                values,
                color=color,
                marker="o",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=color,
                label=run["label"],
                zorder=3,
            )
    for axis, title in zip(
        axes, ("Objective", "Maximum original violation", "Independent KKT residual")
    ):
        axis.set_xlabel("Outer iteration")
        axis.set_ylabel(title)
        axis.grid(alpha=0.2)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[1].axhline(1e-7, color="0.5", linestyle="--", linewidth=0.8)
    axes[2].axhline(1e-5, color="0.5", linestyle="--", linewidth=0.8)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    figure.suptitle(
        f"{len(runs[0]['x'])}-variable coupled Rosenbrock NLP: Pounce and custom SLSQP"
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.94))
    figure.savefig(output / "convergence.png", dpi=180)
    figure.savefig(output / "convergence.pdf")
    if show:
        plt.show()
    plt.close(figure)


def write_results(problem, runs, output, mode="native"):
    """Export detailed detached histories, comparison table and metadata."""
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": environment_metadata(),
        "mode": mode,
        "model": {
            "name": type(problem).__name__,
            "variables": len(problem.initial),
            "equalities": problem.get_nec(),
            "inequalities": problem.get_nic(),
            "initial": problem.initial,
            "audit_feasibility_tolerance": 1e-7,
            "audit_optimality_tolerance": 1e-5,
        },
        "runs": runs,
    }
    (output / "comparison.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = []
    for run in runs:
        summary.append(
            {
                "solver": run["label"],
                "success": run["success"],
                "iterations": run["iterations"],
                "model_calls": run["model_calls"],
                "derivative_calls": run["derivative_calls"],
                "native_stopping_tolerance": run["options"].get(
                    "tol", run["options"].get("ftol")
                ),
                **run["audit"],
            }
        )
        rows = [
            {key: json_safe(value) for key, value in row.items() if key != "x"}
            for row in run["history"]
        ]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"history_{run['key']}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Pounce comparison results",
        "",
        "The recorded objectives are raw values, not best-so-far envelopes.",
        "Pounce rows are native outer callbacks with objective-cache-verified coordinates;",
        "custom SLSQP rows are initial/accepted points. The final returned points are",
        "audited separately. Model and audit work are counted separately; no timing ranking is made.",
        "",
        "Pounce uses native KKT tolerance `1e-8` by default; custom SLSQP uses",
        "native objective/step `ftol=1e-12`. Every returned point is assessed with",
        "the same original-model feasibility `1e-7` and KKT `1e-5` tolerances.",
        "Native stopping tolerances are recorded separately because their meanings differ.",
        "",
        "| Solver | Success | Iterations | Model calls | Objective | Violation | KKT residual | KKT passed |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['solver']} | {row['success']} | {row['iterations']} | "
            f"{row['model_calls']} | {row['objective']:.10g} | "
            f"{row['constraint_violation']:.3g} | {row['kkt_residual']:.3g} | "
            f"{row['kkt_passed']} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for row in summary:
        print(
            f"{row['solver']:45} success={row['success']} "
            f"iterations={row['iterations']:3} objective={row['objective']:.9g} "
            f"violation={row['constraint_violation']:.3g} "
            f"KKT={row['kkt_residual']:.3g} passed={row['kkt_passed']}"
        )


def make_parser(default_output):
    """Build the shared reproducible study command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--slsqp-ftol",
        type=float,
        default=1e-12,
        help="SLSQP objective/step stopping tolerance; not a KKT tolerance.",
    )
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv=None):
    """Run all four solvers and fail if any returned point fails the audit."""
    parser = make_parser(Path(__file__).resolve().parent / "results_native_comparison")
    args = parser.parse_args(argv)
    problem = CoupledRosenbrockProblem(blocks=args.blocks)
    runs = [
        run_native(problem, method, args.tolerance, args.max_iterations)
        for method in ("ipm", "sqp")
    ]
    runs.extend(
        run_custom(problem, fraction, args.slsqp_ftol, args.max_iterations)
        for fraction in (1.0, 0.20)
    )
    write_results(problem, runs, args.output)
    plot_comparison(runs, args.output, show=not args.no_show)
    for run in runs:
        if not run["success"] or not run["audit"]["kkt_passed"]:
            raise RuntimeError(f"{run['label']} did not pass: {run['message']}")
    return runs


if __name__ == "__main__":
    main()
