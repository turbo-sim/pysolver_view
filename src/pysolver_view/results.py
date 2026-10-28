"""Detached public results assembled from authoritative backend state."""

import copy
from scipy.optimize import OptimizeResult
from .roots.methods import normalize_method, METHOD_DISPLAY_NAMES


def optimization_result(run, solution, final_values, final_snapshot):
    """Normalize a backend result without additional model work."""
    result = OptimizeResult(
        x=run.x_final.copy(),
        x_solver=run.x_final_norm.copy(),
        fun=run.f_final,
        fitness=final_values.copy(),
        optimality=final_snapshot["optimality"],
        constraint_violation=final_snapshot["constraint_violation"],
        success=run.success,
        solver_success=solution.get("solver_success", solution.success),
        status=solution.get("status"),
        message=run.message,
        nit=run.iteration_count,
        nfev=run.func_count_tot,
        njev=run.grad_count,
        derivative_func_count=run.derivative_func_count,
        diagnostic_func_count=run.diagnostic_func_count,
        diagnostic_grad_count=run.diagnostic_grad_count,
        failed_grad_count=run.failed_grad_count,
        func_count=run.func_count,
        func_count_total=run.func_count_tot,
        iteration_count=run.iteration_count,
        elapsed_time=run.elapsed_time,
        kkt_enabled=run.check_kkt,
        kkt_passed=None,
        kkt_evaluations=run.kkt_count,
        kkt_time=run.kkt_time,
    )
    if run.library == "pysolver":
        metadata, has_metadata = run._evaluation.final_constraint_metadata(
            run.x_final_norm
        )
        result.report_data = dict(
            variable_names=copy.deepcopy(run.problem.variable_names),
            bounds_physical=copy.deepcopy(run.problem.get_bounds()),
            bounds_solver=copy.deepcopy(run.problem.get_bounds_normalized()),
            problem_scale=run.problem.problem_scale,
            constraint_data=metadata,
            has_constraint_data=has_metadata,
        )
        for name in ("feasible", "stationary", "events", "effective_options"):
            result[name] = copy.deepcopy(solution.get(name))
        assessment = run._public_driver_kkt(solution.get("kkt"))
        result.kkt = assessment
        result.kkt_passed = (
            bool(assessment and assessment["passed"]) if run.check_kkt else None
        )
        result.kkt_evaluations = solution.get("kkt_evaluations", 0)
        result.kkt_time = solution.get("kkt_time", 0.0)
        result.optimality = solution.optimality if run.check_kkt else float("nan")
    if run.library == "pounce":
        result.native_info = copy.deepcopy(solution.info)
        result.native_multipliers = copy.deepcopy(solution.native_multipliers)
        result.native_multiplier_coordinates = "solver"
        result.effective_options = copy.deepcopy(solution.effective_options)
        result.native_x_solver = solution.native_x.copy()
        result.bound_projection = copy.deepcopy(solution.bound_projection)
        result.kkt_stop_requested = solution.kkt_stop_requested
        result.stopped_on_kkt = bool(solution.stopped_on_kkt and run.success)
        result.termination = (
            "kkt_converged"
            if result.stopped_on_kkt
            else (
                "kkt_check_failed"
                if solution.stopped_on_kkt
                else (
                    "cancelled"
                    if solution.status == 5
                    else (
                        "native_converged"
                        if solution.status in (0, 1)
                        else "native_failure"
                    )
                )
            )
        )
        result.kkt = copy.deepcopy(run._pounce_kkt)
        result.kkt_passed = (
            bool(run._pounce_kkt and run._pounce_kkt["passed"])
            if run.check_kkt
            else None
        )
        result.feasible = bool(
            result.constraint_violation <= run._pounce_controls["feasibility_tol"]
        )
        result.stationary = (
            bool(run._pounce_kkt and run._pounce_kkt["first_order_optimality_ok"])
            if run.check_kkt
            else None
        )
    return result


def root_result(run, solution, residual_norm, residual_tolerance):
    """Normalize a backend result without additional model work."""
    result = copy.deepcopy(solution)
    result.update(
        x=run.x_final.copy(),
        fun=None if run.f_final is None else run.f_final.copy(),
        success=run.success,
        solver_success=bool(solution.success),
        message=run.message,
        residual_norm=residual_norm,
        residual_tolerance=residual_tolerance,
        nfev=run.func_count_tot,
        njev=run.grad_count,
        nit=run.iteration_count,
        iteration_count=run.iteration_count,
        library=run.library,
        method=normalize_method(run.method),
        method_name=METHOD_DISPLAY_NAMES[normalize_method(run.method)],
        requested_method=run.requested_method,
        func_count=run.func_count,
        func_count_total=run.func_count_tot,
        derivative_func_count=run.derivative_func_count,
        failed_grad_count=run.failed_grad_count,
        elapsed_time=run.elapsed_time,
    )
    return result
