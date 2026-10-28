"""Verify the Pounce wrapper with the same four-solver native comparison."""

from pathlib import Path

import numpy as np

import pysolver_view as psv
from pysolver_view.optimization.problems import CoupledRosenbrockProblem

from development.pounce_solvers.run_native_solver_comparison import (
    finish_run,
    make_parser,
    plot_comparison,
    run_custom,
    run_native,
    write_results,
)


def run_wrapped(problem, method, tolerance, max_iterations):
    """Use the public backend and detach native observations for the shared audit."""
    history = []

    def native_observation(entry):
        """Record the exposed native callback without triggering model work."""
        metrics = entry["native_metrics"]
        history.append(
            {
                "iteration": int(entry["iteration"]),
                "x": None if entry["x"] is None else np.array(entry["x"], copy=True),
                "event_kind": entry["event_kind"],
                "accepted": entry["accepted"],
                "coordinate_source": entry["coordinate_source"],
                "native_objective": float(metrics["obj_value"]),
                "native_primal_infeasibility": float(metrics["inf_pr"]),
                "native_dual_infeasibility": float(metrics["inf_du"]),
                "barrier_parameter": float(metrics["mu"]),
                "line_search_trials": int(metrics["ls_trials"]),
                "native_phase": int(metrics["alg_mod"]),
            }
        )

    solver = psv.OptimizationSolver(
        problem,
        library="pounce",
        method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
        print_convergence=False,
        update_on="iteration",
        backend_callback=native_observation,
        extra_options={
            "nlp_scaling_method": "none",
            "bound_relax_factor": 0.0,
            "hessian_approximation": "limited-memory",
            **({"sqp_hessian": "damped-bfgs"} if method == "sqp" else {}),
        },
    )
    x = solver.solve(problem.initial)
    result = solver.result
    backend = solver.backend_result
    info = backend.info
    return finish_run(
        problem,
        {
            "key": f"pounce_{method}",
            "label": (
                "Pounce interior-point" if method == "ipm" else "Pounce active-set SQP"
            ),
            "interface": "pysolver_view Pounce backend",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "model_calls": int(result.nfev),
            "derivative_calls": int(result.njev),
            "diagnostic_model_calls": int(result.diagnostic_func_count),
            "native_unscaled_kkt": info.get("final_unscaled_kkt_error"),
            "qp_solves": info.get("n_qp_solves"),
            "qp_working_set_changes": info.get("n_qp_ws_changes"),
            "options": dict(backend.effective_options),
            "x": np.array(x, copy=True),
            "history": history,
        },
    )


def compare_native_reference(wrapped, native):
    """Check wrapper fidelity against a fresh direct solve of the same model.

    Status and iteration counts must agree. Returned coordinates/objectives and
    every regular observed coordinate are compared numerically; final-point
    feasibility and KKT have already been independently assessed by finish_run.
    This checks interface fidelity, not solver robustness across model classes.
    """
    coordinate_error = float(np.max(np.abs(wrapped["x"] - native["x"])))
    objective_error = abs(wrapped["audit"]["objective"] - native["audit"]["objective"])
    wrapped_rows = [row for row in wrapped["history"] if row["x"] is not None]
    native_rows = [row for row in native["history"] if row["x"] is not None]
    history_count_agrees = len(wrapped_rows) == len(native_rows)
    history_error = None
    if history_count_agrees:
        history_error = max(
            (
                float(np.max(np.abs(left["x"] - right["x"])))
                for left, right in zip(wrapped_rows, native_rows)
            ),
            default=0.0,
        )
    passed = bool(
        wrapped["status"] == native["status"]
        and wrapped["iterations"] == native["iterations"]
        and coordinate_error <= 1e-10
        and objective_error <= 1e-10
        and history_count_agrees
        and history_error <= 1e-10
    )
    return {
        "passed": passed,
        "native_status": native["status"],
        "native_iterations": native["iterations"],
        "max_coordinate_error": coordinate_error,
        "objective_error": objective_error,
        "history_count_agrees": history_count_agrees,
        "max_history_coordinate_error": history_error,
    }


def main(argv=None):
    """Run wrapped comparisons and fresh native references, export and verify."""
    parser = make_parser(Path(__file__).resolve().parent / "results_wrapper_comparison")
    args = parser.parse_args(argv)
    problem = CoupledRosenbrockProblem(blocks=args.blocks)
    runs = [
        run_wrapped(problem, method, args.tolerance, args.max_iterations)
        for method in ("ipm", "sqp")
    ]
    for method, run in zip(("ipm", "sqp"), runs):
        native = run_native(problem, method, args.tolerance, args.max_iterations)
        run["native_reference"] = compare_native_reference(run, native)
        print(f"{run['label']} native parity: {run['native_reference']}")
    runs.extend(
        run_custom(problem, fraction, args.slsqp_ftol, args.max_iterations)
        for fraction in (1.0, 0.20)
    )
    write_results(problem, runs, args.output, mode="wrapper")
    plot_comparison(runs, args.output, show=not args.no_show)
    for run in runs:
        if (
            not run["success"]
            or not run["audit"]["kkt_passed"]
            or not run.get("native_reference", {"passed": True})["passed"]
        ):
            raise RuntimeError(f"{run['label']} did not pass: {run['message']}")
    return runs


if __name__ == "__main__":
    main()
