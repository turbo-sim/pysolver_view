"""Connect packaged root drivers to NonlinearSystemSolver observations and callbacks."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

from ..solvers.options import NewtonOptions, StepControlOptions


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


@dataclass
class RootRunContext:
    """The model, counters and event sinks required by one custom root run."""

    residual: Callable
    jacobian: Callable | None
    method: str
    requested_method: str
    options: StepControlOptions | NewtonOptions
    update_on: str
    active: bool
    evaluation: object
    iteration_history: list
    record: Callable
    set_iteration: Callable
    callback: Callable | None

    @classmethod
    def from_solver(cls, solver):
        """Bind the public interface once, keeping it out of numerical adaptation."""

        def set_iteration(count):
            solver.iteration_count = int(count)

        def callback(point):
            solver.callback_func_call_count += 1
            return solver.callback_func(point, solver.callback_func_call_count)

        return cls(
            residual=solver.problem.residual,
            jacobian=getattr(solver.problem, "gradient", None),
            method=solver.method,
            requested_method=solver.requested_method,
            options=solver.options,
            update_on=solver.update_on,
            active=(
                solver.options.history_level != "none"
                or solver.display
                or solver.logger is not None
                or solver.plot
            ),
            evaluation=solver._evaluation,
            iteration_history=solver.iteration_history,
            record=solver._print_convergence_progress,
            set_iteration=set_iteration,
            callback=callback if solver.callback_func is not None else None,
        )


class RootSolverAdapter:
    """Translate numerical observations without evaluating the model or derivatives."""

    def __init__(self, context):
        """Bind one facade run to its resolved custom options and history policy."""
        # Retain the old adapter constructor for callers of the compatibility module.
        self.context = (
            context
            if isinstance(context, RootRunContext)
            else RootRunContext.from_solver(context)
        )
        self.options = self.context.options
        self.active = self.context.active

    def update_counts(self, record):
        """Synchronize physical work, retaining probes as a subset of total calls."""
        solver = self.context.evaluation
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
            self.context.update_on == "function" and kind in ("initial", "trial")
        ) or (self.context.update_on == "gradient" and kind == "jacobian")
        if selected:
            self.context.record(
                record["x"],
                record["fun"],
                event_kind="gradient" if kind == "jacobian" else "function",
                notify=False,
            )

    def accepted(self, record):
        """Record actual accepted iterations and forward detached cancellable callbacks."""
        context = self.context
        evaluation = context.evaluation
        self.update_counts(record)
        iteration = int(record["iteration"])
        context.set_iteration(iteration)
        evaluation.residuals_last = record["fun"].copy()
        evaluation._residual_x = record["x"].copy()
        evaluation._residual_value = record["fun"].copy()
        if self.options.history_level != "none":
            stored = {
                key: deepcopy(value)
                for key, value in record.items()
                if self.options.history_level == "full" or key not in ("x", "fun")
            }
            context.iteration_history.append(stored)
        if self.active and context.update_on == "iteration":
            context.record(
                record["x"],
                record["fun"],
                notify=False,
                event_kind="initial" if iteration == 0 else "iteration",
                iteration=iteration,
            )
        if context.callback is not None:
            try:
                return context.callback(record["x"].copy())
            except StopIteration:
                return False
        return None

    def solve(self, x0):
        """Run the packaged driver and return its authoritative result and options."""
        from ..solvers import solve_root

        context = self.context
        result = solve_root(
            context.residual,
            x0,
            jac=context.jacobian,
            method=context.method,
            options=self.options,
            callback=(
                self.accepted if self.active or context.callback is not None else None
            ),
            observer=(
                self.observe
                if self.active and context.update_on != "iteration"
                else None
            ),
        )
        self.update_counts(result)
        context.set_iteration(result.nit)
        result.effective_options = asdict(self.options)
        result.requested_method = context.requested_method
        return result
