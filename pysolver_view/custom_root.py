"""Connect packaged root drivers to NonlinearSystemSolver observations and callbacks."""

from copy import deepcopy
from dataclasses import asdict

from .solvers.root import NewtonOptions, StepControlOptions, solve_root


def resolve_root_options(
    method, options, tolerance, max_iterations, derivative_method, derivative_abs_step
):
    """Resolve custom options, giving explicit dictionary/dataclass values precedence.

    Constructor tolerance/max_iterations fill missing residual/trial settings.
    Numerical derivatives belong to the numerical core. Explicit finite_diff_step
    uses scaled probes; otherwise the constructor's physical absolute step applies.
    """
    option_type = (
        NewtonOptions
        if method in ("newton_raphson", "gauss_newton")
        else StepControlOptions
    )
    if isinstance(options, option_type):
        return deepcopy(options)
    if options is not None and not isinstance(options, dict):
        raise TypeError(f"options must be {option_type.__name__} or a dictionary.")
    values = deepcopy(options) if options is not None else {}
    values.setdefault("residual_tol", tolerance)
    values.setdefault("max_iterations", max_iterations)
    values.setdefault("derivative_method", derivative_method)
    if "derivative_abs_step" not in values and "finite_diff_step" not in values:
        values["derivative_abs_step"] = derivative_abs_step
    return option_type(**values)


class RootSolverAdapter:
    """Translate numerical observations without evaluating the model or derivatives."""

    def __init__(self, solver):
        """Bind one facade run to its resolved custom options and history policy."""
        self.solver = solver
        self.options = solver.options
        self.active = (
            self.options.history_level != "none"
            or solver.display
            or solver.logger is not None
            or solver.plot
        )

    def update_counts(self, record):
        """Synchronize physical work, retaining probes as a subset of total calls."""
        solver = self.solver
        solver.func_count_tot = int(record["nfev"])
        solver.derivative_func_count = int(record.get("derivative_nfev", 0))
        solver.func_count = solver.func_count_tot - solver.derivative_func_count
        solver.grad_count = int(record["njev"])
        solver.failed_grad_count = int(record.get("failed_njev", 0))

    def observe(self, record):
        """Publish only the selected valid model/Jacobian stream, never probe rows."""
        self.update_counts(record)
        if not record["valid"]:
            return
        kind = record["kind"]
        selected = (
            self.solver.update_on == "function" and kind in ("initial", "trial")
        ) or (self.solver.update_on == "gradient" and kind == "jacobian")
        if selected:
            self.solver._print_convergence_progress(
                record["x"],
                record["fun"],
                event_kind="gradient" if kind == "jacobian" else "function",
                notify=False,
            )

    def accepted(self, record):
        """Record actual accepted iterations and forward detached cancellable callbacks."""
        solver = self.solver
        self.update_counts(record)
        solver.iteration_count = int(record["iteration"])
        solver.residuals_last = record["fun"].copy()
        solver._residual_x = record["x"].copy()
        solver._residual_value = record["fun"].copy()
        if self.options.history_level != "none":
            stored = {
                key: deepcopy(value)
                for key, value in record.items()
                if self.options.history_level == "full" or key not in ("x", "fun")
            }
            solver.iteration_history.append(stored)
        if self.active and solver.update_on == "iteration":
            solver._print_convergence_progress(
                record["x"],
                record["fun"],
                notify=False,
                event_kind="initial" if solver.iteration_count == 0 else "iteration",
                iteration=solver.iteration_count,
            )
        if solver.callback_func is not None:
            solver.callback_func_call_count += 1
            try:
                return solver.callback_func(
                    record["x"].copy(), solver.callback_func_call_count
                )
            except StopIteration:
                return False
        return None

    def solve(self, x0):
        """Run the packaged driver and return its authoritative result and options."""
        solver = self.solver
        result = solve_root(
            solver.problem.residual,
            x0,
            jac=getattr(solver.problem, "gradient", None),
            method=solver.method,
            options=self.options,
            callback=(
                self.accepted
                if self.active or solver.callback_func is not None
                else None
            ),
            observer=(
                self.observe
                if self.active and solver.update_on != "iteration"
                else None
            ),
        )
        self.update_counts(result)
        solver.iteration_count = int(result.nit)
        result.effective_options = asdict(self.options)
        result.requested_method = solver.requested_method
        return result
