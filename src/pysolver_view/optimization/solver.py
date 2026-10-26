from ..reporting.text import emit
from ..results import optimization_result
from ..reporting import plots, text
from ..history import append_observation
import os
import time
import copy
import logging
import numpy as np

from datetime import datetime

from .backends.registry import OPTIMIZATION_LIBRARIES, VALID_LIBRARIES_AND_METHODS
from .evaluation import BackendProblem, OptimizationEvaluation
from . import kkt_adapter
from ..reporting import tables

from .problems.base import OptimizationProblem as OptimizationProblem
from .constraints import (
    count_constraints as count_constraints,
    combine_objective_and_constraints as combine_objective_and_constraints,
    get_constraint_data,
    constraint_data_from_values,
)


class OptimizationSolver:
    r"""

    Solver class for general nonlinear programming problems.

    The solver is designed to handle constrained optimization problems of the form:

    Minimize:

    .. math::
        f(\mathbf{x}) \; \mathrm{with} \; \mathbf{x} \in \mathbb{R}^n

    Subject to:

    .. math::
        c_{\mathrm{eq}}(\mathbf{x}) = 0
    .. math::
        c_{\mathrm{in}}(\mathbf{x}) \leq 0
    .. math::
        \mathbf{x}_l \leq \mathbf{x} \leq \mathbf{x}_u

    where:

    - :math:`\mathbf{x}` is the vector of decision variables (i.e., degree of freedom).
    - :math:`f(\mathbf{x})` is the objective function to be minimized. Maximization problems can be casted into minimization problems by changing the sign of the objective function.
    - :math:`c_{\mathrm{eq}}(\mathbf{x})` are the equality constraints of the problem.
    - :math:`c_{\mathrm{in}}(\mathbf{x})` are the inequality constraints of the problem. Constraints of type :math:`c_{\mathrm{in}}(\mathbf{x}) \leq 0` can be casted into :math:`c_{\mathrm{in}}(\mathbf{x}) \geq 0` type by changing the sign of the constraint functions.
    - :math:`\mathbf{x}_l` and :math:`\mathbf{x}_u` are the lower and upper bounds on the decision variables.

    The class interfaces with various optimization methods provided by libraries such as `scipy` and `pygmo` to solve the problem and provides a structured framework for initialization, solution monitoring, and post-processing.

    This class employs a caching mechanism to avoid redundant evaluations. For a given set of independent variables, x, the optimizer requires the objective function, equality constraints, and inequality constraints to be provided separately. When working with complex models, these values are typically calculated all at once. If x hasn't changed from a previous evaluation, the caching system ensures that previously computed values are used, preventing unnecessary recalculations.

    Parameters
    ----------
    problem : OptimizationProblem
        An instance of the optimization problem to be solved. The problem should be defined
        in physical space, with its own bounds and (optionally) analytic derivatives.
    library : str, optional
        The library to use for solving the optimization problem (default is 'scipy').
    method : str, optional
        The optimization method to use from the specified library (default is 'slsqp').
    tolerance : float, optional
        Tolerance for termination. The minimization algorithm sets some solver-specific tolerances
        equal to tol. (default is 1e-6)
    max_iterations : int, optional
        Maximum number of iterations for the optimizer (default is 100).
    extra_options : dict, optional
        A dictionary of solver-specific options that prevails over 'tolerance' and 'max_iterations'
    derivative_method : str, optional
        Method to use for derivative calculation (default is '2-point').
    derivative_abs_step : float, optional
        Finite difference absolute step size to be used when the problem Jacobian is not provided. Default depends on calculation method.
    problem_scale : float or None, optional
        Scaling factor used to normalize the problem. This parameter controls the transformation
        of physical variables into a normalized domain. Specifically, for a physical variable x,
        with lower and upper bounds lb and ub, the normalized variable is computed as:

            x_norm = problem_scale * (x - lb) / (ub - lb)

        - If a numeric value is provided (e.g. 1.0, 10.0, etc.), the problem is scaled accordingly.
          Increasing problem_scale reduces the relative step sizes in the normalized space during
          line searches, which can improve convergence by making the optimization less aggresive.
          The rationale for the scaling is that the initial line search step size of many solvers is 1.0.
        - If set to None, no scaling is applied and the problem is solved in its original physical units.
          This might be preferred if the problem is already well-conditioned or if the user wishes to
          preserve the exact scale of the original formulation.

    print_convergence : bool, optional
        If True, displays the convergence progress (default is True).
    plot_convergence : bool, optional
        If True, plots the convergence progress (default is False).
    plot_scale_objective : str, optional
        Specifies the scale of the objective function axis in the convergence plot (default is 'linear').
    plot_scale_constraints : str, optional
        Specifies the scale of the constraint violation axis in the convergence plot (default is 'linear').
    logger : logging.Logger, optional
        Logger object to which logging messages will be directed. Logging is disabled if `logger` is None.
    update_on : str, optional
        Selects 'gradient' (default), 'function', or SLSQP 'iteration' events.
        SciPy callbacks denote major iterations with unknown acceptance; custom
        SLSQP records accepted points. Derivative probes never generate observations.
    callback_functions : list of callable or callable, optional
        Callbacks receive (copied_solver_coordinates, callback_count). The count
        is not a solver iteration number. SciPy uses the selected observation
        stream; custom SLSQP uses initial/accepted points only and supports
        cancellation with StopIteration or a False return. Final synchronization
        does not invoke callbacks.
    plot_improvement_only : bool, optional
        If True, plots only display iterations that improve the objective function value (useful for gradient-free optimizers) (default is False).
    check_kkt : bool, optional
        Enable automatic KKT diagnostics (default True). If False, histories have
        unavailable optimality and no multiplier estimation runs automatically.
        Custom SLSQP success then requires native success and feasibility only.
        Explicit evaluate_kkt_conditions or requested KKT/multiplier reports still
        perform an on-demand assessment, outside the automatic solve policy.
    backend_callback : callable, optional
        With ``library="pysolver"``, receive a detached copy of each initial or
        accepted SLSQP record, including KKT diagnostics and recovery counters.
        Records use normalized solver coordinates and the backend's constraint
        convention: equalities are zero and inequalities are nonnegative, as
        in ``backend_result.history``. This differs from the frontend's public
        nonpositive inequalities and equality multiplier signs. Rejected trials,
        derivative probes, and final synchronization do not invoke this hook.
        Return False or raise StopIteration to stop at the accepted point.

    Methods
    -------
    solve(x0):
        Solve the optimization problem using the specified initial guess `x0`.
    fitness(x):
        Evaluates the optimization problem objective function and constraints at a given point `x`.
    gradient(x):
        Evaluates the Jacobians of the optimization problem at a given point `x`.
    print_convergence_history():
        Print the final result and convergence history of the optimization problem.
    plot_convergence_history():
        Plot the convergence history of the optimization problem.
    """

    def __init__(
        self,
        problem,
        library="scipy",
        method="slsqp",
        tolerance=1e-6,
        max_iterations=100,
        extra_options={},
        derivative_method="2-point",
        derivative_abs_step=None,
        problem_scale=None,
        print_convergence=True,
        plot_convergence=False,
        plot_scale_objective="linear",
        plot_scale_constraints="linear",
        logger=None,
        update_on="gradient",
        callback_functions=None,
        plot_improvement_only=False,
        tolerance_check_cache=None,
        check_kkt=True,
        backend_callback=None,
    ):
        # Initialize class variables
        if not isinstance(check_kkt, bool):
            raise TypeError("check_kkt must be a boolean.")
        if "check_kkt" in extra_options:
            raise ValueError(
                "Pass check_kkt directly to OptimizationSolver, not in extra_options."
            )
        self.check_kkt = check_kkt
        self.problem = problem
        self.display = print_convergence
        self.plot = plot_convergence
        self.plot_scale_objective = plot_scale_objective
        self.plot_scale_constraints = plot_scale_constraints
        self.logger = logger
        self.library = library
        self.method = method
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
        self.callback_functions = self._validate_callback(callback_functions)
        if backend_callback is not None and not callable(backend_callback):
            raise TypeError("backend_callback must be callable or None.")
        if backend_callback is not None and library != "pysolver":
            raise ValueError("backend_callback requires library='pysolver'.")
        self.backend_callback = backend_callback
        self.callback_function_call_count = 0
        self.plot_improvement_only = plot_improvement_only

        # Tolerance to check if design variables are the same as in previous fitness call
        if tolerance_check_cache is None:
            self.tolerance_cache = 0.0
        else:
            self.tolerance_cache = tolerance_check_cache

        # Validate library and method
        self._validate_library_and_method()

        # Define options dictionary
        self.options = {"tolerance": tolerance, "max_iterations": max_iterations}
        self.options = self.options | extra_options

        # Set scaling for the optimization problem (define at the solver, rather than problem level)
        self.problem.problem_scale = problem_scale

        # Auto-generate design variable names if not provided
        if not hasattr(self.problem, "variable_names"):
            bounds = self.problem.get_bounds()
            self.problem.variable_names = (
                None
                if bounds is None
                else [f"design_variable_{i}" for i in range(len(bounds[0]))]
            )

        # Check for logger validity
        if self.logger is not None:
            if not isinstance(self.logger, logging.Logger):
                raise ValueError(
                    "The provided logger is not a valid logging.Logger instance."
                )

        # Check for valid display_on value
        self.update_on = update_on
        if update_on not in ["function", "gradient", "iteration"]:
            raise ValueError(
                "Invalid value for 'update_on'. Use 'function', 'gradient', or 'iteration'."
            )

        # Rename number of constraints
        self.N_eq = self.problem.get_nec()
        self.N_ineq = self.problem.get_nic()

        self._reset_run_state()

    def _reset_run_state(self):
        """Reset results, observations, and counters before an independent solve."""
        self._evaluation = OptimizationEvaluation(
            self.problem,
            library=self.library,
            options=self.options,
            derivative_method=self.derivative_method,
            derivative_abs_step=self.derivative_abs_step,
            fitness=self.fitness,
        )
        self.result = None
        self.backend_result = None
        self.x_final_norm = None
        self.iteration_count = None
        self.iteration_history = []
        self.callback_function_call_count = 0
        self._in_callback = False
        self._custom_initial_pending = self.library == "pysolver"
        self._custom_kkt = None
        # Initialize variables for convergence report
        self.f_final = None
        self.x_final = None
        self.x_last_norm = None
        self.success = None
        self.message = None
        self.convergence_report = []
        self.elapsed_time = None
        self.convergence_history = {
            "x": [],
            "event_kind": [],
            "iteration": [],
            "derivative_func_count": [],
            "diagnostic_func_count": [],
            "optimality": [],
            "grad_count": [],
            "func_count": [],
            "func_count_total": [],
            "objective_value": [],
            "constraint_violation": [],
            "norm_step": [],
        }

        # Initialize dictionary for cached variables

    def _validate_library_and_method(self):
        # Check if the library is valid
        if self.library not in VALID_LIBRARIES_AND_METHODS:
            error_message = (
                f"Invalid optimization library '{self.library}'. \nAvailable libraries:\n   - "
                + "\n   - ".join(VALID_LIBRARIES_AND_METHODS.keys())
                + "."
            )
            raise ValueError(error_message)

        # Check if the method is valid for the selected library
        if self.method and self.method not in VALID_LIBRARIES_AND_METHODS[self.library]:
            error_message = (
                f"Invalid method '{self.method}' for library '{self.library}'. \nValid methods are:\n   - "
                + "\n   - ".join(VALID_LIBRARIES_AND_METHODS[self.library])
                + "."
            )
            raise ValueError(error_message)

    def _validate_callback(self, callback):
        """Validate the callback functions argument."""
        if callback is None:
            return []
        if callable(callback):
            return [callback]
        elif isinstance(callback, list):
            non_callable_items = [item for item in callback if not callable(item)]
            if not non_callable_items:
                return callback
            else:
                error_msg = f"All elements in the callback list must be callable functions. Non-callable items: {non_callable_items}"
                raise TypeError(error_msg)
        else:
            error_msg = f"callback_func must be a function or a list of functions. Received type: {type(callback)} ({callback})"
            raise TypeError(error_msg)

    def solve(self, x0):
        """Solve from physical coordinates and return the backend's final point.

        ``backend_result`` retains native fields in solver coordinates. ``result``
        adds the physical solution, explicit solver coordinates, and facade call
        counts. Observation caches and callbacks never select the solution.
        """
        self._reset_run_state()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time = time.perf_counter()
        if self.plot:
            self._plot_convergence_callback([], [], initialize=True)
        x0 = np.asarray(x0, dtype=float).copy()
        if not self.problem.variable_names:
            self.problem.variable_names = [
                f"design_variable_{i}" for i in range(x0.size)
            ]
        if len(self.problem.variable_names) != x0.size:
            raise ValueError(
                "variable_names must contain one name per decision variable."
            )
        if self.library == "pysolver" and self.problem.get_bounds() is not None:
            lower, upper = map(np.asarray, self.problem.get_bounds())
            if np.any(lower >= upper):
                raise ValueError(
                    "Custom SLSQP requires lower < upper; remove fixed variables."
                )
        if self.problem.get_bounds() is not None:
            x0 = self.problem.clip_to_bounds(x0, logger=self.logger)
        x0 = self.problem.scale_physical_to_normalized(x0)
        self._write_header()
        if self.library in {"scipy", "pysolver"} and self.method == "slsqp":
            self.iteration_count = 0
        if self.update_on == "iteration" and self.iteration_count is None:
            raise ValueError("Iteration observations require SciPy or pysolver SLSQP.")
        if self.library != "pysolver":
            self._print_convergence_progress(x0, event_kind="initial")
        problem = BackendProblem(self)
        backend_options = dict(self.options)
        if self.library == "pysolver":
            backend_options["check_kkt"] = self.check_kkt
        solution = OPTIMIZATION_LIBRARIES[self.library](
            problem, x0, self.method, backend_options
        )
        self.backend_result = copy.deepcopy(solution)
        self.x_final_norm = np.asarray(solution.x, dtype=float).copy()
        self.x_final = np.array(
            self.problem.scale_normalized_to_physical(self.x_final_norm), copy=True
        )
        if self.library == "pysolver":
            final_values = solution.fitness
            self._custom_kkt = self._public_driver_kkt(solution.get("kkt"))
        else:
            final_values = self._evaluate_fitness(self.x_final_norm, diagnostic=True)
        if final_values is None:
            final_values = np.full(1 + self.N_eq + self.N_ineq, np.nan)
        self.f_final = copy.deepcopy(final_values[0])
        self.success = solution.success
        self.message = solution.message
        native_iterations = solution.get("nit")
        if native_iterations is not None:
            self.iteration_count = int(native_iterations)
        final_snapshot = (
            self._observation_snapshot(self.x_final_norm, "final", values=final_values)
            if np.all(np.isfinite(final_values))
            else dict(optimality=float("nan"), constraint_violation=float("inf"))
        )
        if "x" in final_snapshot and (
            self.x_last_norm is None
            or not np.array_equal(self.x_final_norm, self.x_last_norm)
        ):
            self._publish_observation(final_snapshot, notify=False)
        self.elapsed_time = time.perf_counter() - start_time
        self.result = optimization_result(self, solution, final_values, final_snapshot)
        self._write_footer()
        return self.x_final.copy()

    def fitness(self, x_norm, called_from_grad=False):
        """Evaluate combined responses; derivative probes never publish observations.

        Inputs use solver coordinates. Total model calls include numerical
        derivative work; ``func_count`` includes ordinary model calls only.
        """
        if called_from_grad:
            return self._evaluate_fitness(
                x_norm, derivative=True, use_cache=False, cache_result=False
            )
        before = self.func_count_tot
        values = self._evaluate_fitness(x_norm)
        if (
            self.func_count_tot != before
            and self.update_on == "function"
            and not self._custom_initial_pending
        ):
            self._print_convergence_progress(x_norm, event_kind="function")
        return values

    def _evaluate_fitness(
        self,
        x_norm,
        *,
        derivative=False,
        diagnostic=False,
        use_cache=True,
        cache_result=True,
    ):
        """Obtain a detached same-point value without recording or calling observers.

        Exact keys protect tiny representable steps and complex-step probes.
        ``tolerance_check_cache`` is retained as an input for compatibility but
        is not used to substitute values at a different point.
        """
        return self._evaluation._evaluate_fitness(
            x_norm,
            derivative=derivative,
            diagnostic=diagnostic,
            use_cache=use_cache,
            cache_result=cache_result,
        )

    def gradient(self, x_norm):
        """Obtain one full Jacobian and optionally record its base-point observation."""
        before = self.grad_count
        grad = self._evaluate_gradient(x_norm)
        if (
            self.grad_count != before
            and self.update_on == "gradient"
            and not self._custom_initial_pending
        ):
            self._print_convergence_progress(x_norm, event_kind="gradient")
        return grad

    def _evaluate_gradient(self, x_norm, *, diagnostic=False):
        """Compute/cache a full Jacobian without iteration events or probe callbacks."""
        return self._evaluation._evaluate_gradient(x_norm, diagnostic=diagnostic)

    def _handle_output(
        self, text: str, savefile: bool = False, filename: str = None, to_console=False
    ):
        """Emit prepared text using the existing display/logging policy."""
        return emit(
            text,
            logger=self.logger,
            to_console=self.display and to_console,
            savefile=savefile,
            filename=filename,
        )

    def _write_header(self):
        """Render and store the prepared header."""
        self.header, lines = text.optimization_header(
            type(self.problem).__name__, self.method
        )
        for line in lines:
            self._handle_output(line, to_console=True)
        self.convergence_report.extend(lines)

    def _observation_snapshot(self, x, event_kind, *, values=None):
        """Build one detached point-consistent record without forcing derivatives."""
        point = np.asarray(x, dtype=float).copy()
        if values is None:
            values = self._evaluate_fitness(point, diagnostic=True)
        equalities = values[1 : 1 + self.N_eq]
        inequalities = values[1 + self.N_eq :]
        violations = [0.0, *np.abs(equalities), *np.maximum(inequalities, 0.0)]
        bounds = self.problem.get_bounds_normalized()
        if bounds is not None:
            lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
            violations.extend(np.maximum(lower - point, 0.0))
            violations.extend(np.maximum(point - upper, 0.0))
        optimality = float("nan")
        if self.library == "pysolver":
            assessment = self._custom_kkt
            if (
                assessment is not None
                and assessment["available"]
                and np.array_equal(point, assessment["x"])
            ):
                optimality = assessment["stationarity"]
        elif (
            self.check_kkt
            and self.cache["x_jac"] is not None
            and np.array_equal(point, self.cache["x_jac"])
        ):
            # This is a pure reconstruction from already available values/derivatives.
            optimality = self._kkt_from_cache(point, 50 * self.options["tolerance"])[
                "first_order_optimality"
            ]
        return dict(
            x=point,
            event_kind=event_kind,
            iteration=self.iteration_count,
            grad_count=self.grad_count,
            func_count=self.func_count,
            func_count_total=self.func_count_tot,
            derivative_func_count=self.derivative_func_count,
            diagnostic_func_count=self.diagnostic_func_count,
            objective_value=values[0],
            constraint_violation=float(max(violations)),
            norm_step=(
                0.0
                if self.x_last_norm is None
                else float(np.linalg.norm(point - self.x_last_norm))
            ),
            optimality=optimality,
        )

    def _record_iteration(self, x):
        """Capture a native SLSQP major-iteration event, not assumed acceptance.

        SciPy's callback may precede further merit-search work. Its point is an
        observation, while the native returned result owns the final solution.
        """
        self.iteration_count = (self.iteration_count or 0) + 1
        snapshot = self._observation_snapshot(x, "iteration")
        snapshot["accepted"] = None
        previous = self.iteration_history[-1]["x"] if self.iteration_history else None
        snapshot["norm_step"] = (
            0.0 if previous is None else float(np.linalg.norm(snapshot["x"] - previous))
        )
        self.iteration_history.append(copy.deepcopy(snapshot))
        if self.update_on == "iteration":
            self._publish_observation(snapshot)

    @staticmethod
    def _public_driver_kkt(assessment):
        """Detach driver diagnostics and convert equality multipliers to public signs."""
        return kkt_adapter.public_driver_kkt(assessment)

    def _record_custom_iteration(self, entry):
        """Record accepted custom-driver points and notify cancellable callbacks.

        The driver invokes this only at the initial or accepted point. Function
        and gradient observation streams may include trials, but user callbacks
        for this backend always receive accepted points. Raising StopIteration
        or returning False stops at that point with a structured failed result.
        """
        self._evaluation.record_accepted_metadata(entry["x"])
        initial = self._custom_initial_pending
        self._custom_initial_pending = False
        self.iteration_count = int(entry["iteration"])
        self._custom_kkt = self._public_driver_kkt(entry.get("kkt"))
        values = entry["fitness"].copy()
        values[1 + self.N_eq :] *= -1
        snapshot = self._observation_snapshot(
            entry["x"], "initial" if initial else "iteration", values=values
        )
        snapshot["accepted"] = True
        snapshot["norm_step"] = entry["step_norm"]
        snapshot["kkt"] = copy.deepcopy(self._custom_kkt)
        if not initial:
            self.iteration_history.append(copy.deepcopy(snapshot))
        if initial or self.update_on == "iteration":
            self._publish_observation(snapshot, notify=False)
        if self.backend_callback is not None or self.callback_functions:
            self._in_callback = True
            try:
                if self.backend_callback is not None:
                    if self.backend_callback(copy.deepcopy(entry)) is False:
                        raise StopIteration("Stopped by backend callback")
                if self.callback_functions:
                    self.callback_function_call_count += 1
                for callback in self.callback_functions:
                    if (
                        callback(entry["x"].copy(), self.callback_function_call_count)
                        is False
                    ):
                        raise StopIteration("Stopped by user callback")
            finally:
                self._in_callback = False

    def _print_convergence_progress(self, x, event_kind="observation", notify=True):
        """Record and render a base-point observation without recursive evaluations."""
        self._publish_observation(
            self._observation_snapshot(x, event_kind), notify=notify
        )

    def _publish_observation(self, snapshot, notify=True):
        """Store a copied snapshot, render it, and notify legacy observation callbacks."""
        if self._in_callback:
            return
        record = copy.deepcopy(snapshot)
        self.x_last_norm = record["x"].copy()
        append_observation(self.convergence_history, record)
        status = text.optimization_progress(record)
        self._handle_output(status, to_console=True)
        self.convergence_report.append(status)
        if self.plot:
            self._plot_convergence_callback([], [])
        if notify and self.callback_functions and self.library != "pysolver":
            self.callback_function_call_count += 1
            self._in_callback = True
            try:
                for callback in self.callback_functions:
                    callback(record["x"].copy(), self.callback_function_call_count)
            finally:
                self._in_callback = False

    def _write_footer(self):
        """Render and store the prepared footer."""
        lines = text.optimization_footer(
            self.header,
            self.message,
            self.success,
            self.elapsed_time,
            self.result,
            self.check_kkt,
        )
        for line in lines:
            self._handle_output(line, to_console=True)
        self.convergence_report.extend(lines)

    def _has_constraint_plot(self):
        """Show feasibility whenever nonlinear constraints or finite bounds exist."""
        bounds = self.problem.get_bounds_normalized()
        has_bounds = bounds is not None and np.any(np.isfinite(np.asarray(bounds)))
        return self.N_eq > 0 or self.N_ineq > 0 or has_bounds

    def _plot_convergence_callback(self, x, iter, initialize=False, showfig=True):
        plots.update_optimization_plot(self._plot_view(), x, iter, initialize, showfig)
        self._sync_plot_handles()

    def plot_convergence_history(
        self,
        savefile=False,
        filename=None,
        output_dir="output",
        showfig=True,
    ):
        """
        Plot the convergence history of the problem.

        This method plots progress against the selected evaluation or native-iteration count:
            - Objective function value (left y-axis)
            - Maximum constraint violation (right y-axis)

        Constraint violation is displayed for nonlinear constraints or finite variable bounds.

        This method should be called only after the optimization problem has been solved, as it relies on data generated by the solving process.

        Parameters
        ----------
        savefile : bool, optional
            If True, the plot is saved to a file instead of being displayed. Default is False.
        filename : str, optional
            The name of the file to save the plot to. If not specified, the filename is automatically generated
            using the problem name and the start datetime. The file extension is not required.
        output_dir : str, optional
            The directory where the plot file will be saved if savefile is True. Default is "output".

        Returns
        -------
        matplotlib.figure.Figure
            The Matplotlib figure object for the plot. This can be used for further customization or display.

        Raises
        ------
        ValueError
            If this method is called before the problem has been solved.
        """
        return self._plot_history(savefile, filename, output_dir, showfig)

    def print_convergence_history(
        self, savefile=False, filename=None, output_dir="output", to_console=True
    ):
        """
        Print or save the convergence history of the optimization process.

        This function prints (or saves) a the report of the optimization convergence progress.
        It includes information collected at each iteration, including:

        - Number of gradient evaluations
        - Number of function evaluations
        - Objective function value
        - Maximum constraint violation
        - Two-norm of the update step

        It also includes a summary at the end of the run with:

        - Exit message
        - Success flag
        - Total solution time (in seconds)

        .. note::
            This method must be called **after** `solve()` has been executed. Otherwise, the convergence
            report is unavailable and a `ValueError` is raised.

        Parameters
        ----------
        savefile : bool, optional
            If True, the report is saved to a file. Otherwise, it is printed to the screen. Default is False.

        filename : str or None, optional
            The name of the file to save the report to. If None, a default name is generated based on the
            problem class name and the optimization start datetime.

        output_dir : str, optional
            Directory where the report file is saved if `savefile=True`. Default is "output".

        Raises
        ------
        ValueError
            If the method is called before the optimization problem has been solved.
        """
        if self.x_final is None:
            raise ValueError(
                "This method can only be used after invoking the 'solve()' method."
            )

        # Join convergence history lines
        full_text = "\n".join(self.convergence_report)

        # Decide filename if saving to disk
        if savefile:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                default = f"{type(self.problem).__name__}_{self.start_datetime}"
                filename = f"convergence_history_{default}.txt"
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = None

        # Output the report (either print or save to file)
        self._handle_output(
            full_text, savefile=savefile, filename=filepath, to_console=to_console
        )

    def print_optimization_report(
        self,
        x=None,
        tol=None,
        include_design_variables=True,
        include_constraints=True,
        include_kkt_conditions=False,
        include_multipliers=False,
        savefile=False,
        filename=None,
        output_dir="output",
        to_console=True,
        width=80,
    ):
        """Print or save selected variables, constraints, KKT and multipliers.

        With the custom ``pysolver`` backend, omitting both ``x`` and ``tol``
        renders detached data for the authoritative final point. Its cached KKT
        certificate retains the solve's separate feasibility and optimality
        tolerances. Unavailable or disabled diagnostics remain unassessed; no
        model evaluation, derivative probe, or multiplier fit is requested.

        Supplying physical ``x`` or an explicit ``tol`` requests a diagnostic
        assessment using the legacy common tolerance for feasibility and
        optimality. Those explicit requests may add counted diagnostic work.
        Other backends retain this diagnostic behavior. When needed, the
        default diagnostic tolerance is five times the native solver tolerance.

        Parameters
        ----------
        x : array-like or None
            Physical point to assess; None selects the final solution.
        tol : float or None
            Common tolerance for an explicitly requested diagnostic report.
        include_design_variables, include_constraints : bool
            Include physical/normalized variables and named constraints.
        include_kkt_conditions, include_multipliers : bool
            Include KKT residuals and the corresponding public-sign multipliers.
        savefile : bool
            Save the assembled report as text in ``output_dir``.
        filename : str or None
            Explicit filename, or None for a generated report filename.
        output_dir : str or path-like
            Directory receiving the saved report.
        to_console : bool
            Print the report independently of live convergence display settings.
        width : int, optional
            Maximum line width in characters, at least 60. Default is 80.
            Extra space expands name columns. Long names and notes wrap onto
            continuation lines instead of being shortened. Applies equally to
            console, logger, and saved final reports; live convergence is separate.
        """
        width = tables.validate_report_width(width)
        if x is None and self.x_final is None:
            raise ValueError("This report requires a completed solve or an explicit x.")
        cached_final = (
            x is None
            and tol is None
            and self.library == "pysolver"
            and self.result is not None
        )
        if cached_final:
            sections = self._final_report_sections(
                include_design_variables,
                include_constraints,
                include_kkt_conditions,
                include_multipliers,
                width=width,
            )
        else:
            x_phys = x if x is not None else self.x_final
            x_norm = self.problem.scale_physical_to_normalized(x_phys)
            tol = 5 * self.options["tolerance"] if tol is None else tol
            sections = []
            if include_kkt_conditions:
                sections.append(
                    self.make_kkt_optimality_report(x_norm, tol, width=width)
                )
            if include_design_variables:
                if self.problem.problem_scale is not None:
                    sections.append(
                        self.make_variables_report(x_norm, normalized=True, width=width)
                    )
                sections.append(
                    self.make_variables_report(x_phys, normalized=False, width=width)
                )
            if include_constraints:
                sections.append(self.make_constraint_report(x_norm, tol, width=width))
            if include_multipliers:
                sections.append(
                    self.make_lagrange_multipliers_report(x_norm, tol, width=width)
                )
        full_text = "\n".join(sections)
        if savefile:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                filename = (
                    f"optimization_report_{type(self.problem).__name__}"
                    f"_{self.start_datetime}.txt"
                )
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = None
        self._handle_output(
            full_text, savefile=savefile, filename=filepath, to_console=False
        )
        if to_console:
            print(full_text)

    def _final_report_sections(
        self,
        include_variables,
        include_constraints,
        include_kkt,
        include_multipliers,
        *,
        width=80,
    ):
        """Format detached final custom-solver data without requesting diagnostics."""
        result = self.result
        data = result["report_data"]
        assessment = result.get("kkt") or {}
        options = result.get("effective_options") or {}
        feasibility_tol = assessment.get(
            "feasibility_tol", options.get("feasibility_tol", self.options["tolerance"])
        )
        optimality_tol = assessment.get(
            "optimality_tol", options.get("optimality_tol", self.options["tolerance"])
        )
        enabled = bool(result.get("kkt_enabled", self.check_kkt))
        sections = []
        if include_kkt:
            sections.append(
                tables.final_kkt_report(
                    assessment,
                    feasibility_tol,
                    optimality_tol,
                    enabled=enabled,
                    width=width,
                )
            )
        if include_variables:
            if data["bounds_solver"] is not None and data["problem_scale"] is not None:
                sections.append(
                    tables.variables_report(
                        result.x_solver,
                        data["bounds_solver"],
                        data["variable_names"],
                        normalized=True,
                        width=width,
                    )
                )
            sections.append(
                tables.variables_report(
                    result.x,
                    data["bounds_physical"],
                    data["variable_names"],
                    normalized=False,
                    width=width,
                )
            )
        if include_constraints or include_multipliers:
            constraints = constraint_data_from_values(
                result.fitness,
                self.N_eq,
                self.N_ineq,
                feasibility_tol,
                data["constraint_data"],
                has_metadata=data["has_constraint_data"],
            )
            if include_constraints:
                sections.append(tables.constraint_report(constraints, width=width))
            if include_multipliers:
                sections.append(
                    tables.final_multipliers_report(
                        assessment,
                        constraints,
                        data["variable_names"],
                        data["bounds_solver"],
                        enabled=enabled,
                        width=width,
                    )
                )
        return sections

    def make_variables_report(self, x_norm, normalized=True, *, width=80):
        """
        Generate design variable report as a string.

        Parameters
        ----------
        x_norm : array-like
            Normalized design variables (input to the solver).
        normalized : bool, optional
            Whether to report in normalized or physical values. Default is True.
        width : int, optional
            Maximum line width, at least 60; complete names wrap as needed.

        Returns
        -------
        str
            Formatted string report.
        """
        width = tables.validate_report_width(width)
        return tables.variables_report(
            x_norm,
            (
                self.problem.get_bounds_normalized()
                if normalized
                else self.problem.get_bounds()
            ),
            self.problem.variable_names,
            normalized,
            width=width,
        )

    def get_constraint_data(self, x_norm, tol):
        """
        Return constraint evaluation data for the current optimization problem.

        This method returns a list of dictionaries describing the satisfaction
        status of all equality and inequality constraints at a given normalized
        design point.

        If `self.problem.constraint_data` exists and is valid, it is returned
        directly after key validation.

        Parameters
        ----------
        x_norm : array_like
            Normalized design variable vector.
        tol : float
            Tolerance for constraint satisfaction.

        Returns
        -------
        list of dict
            Each dictionary contains:

            - **name** : str
              Constraint identifier.

            - **type** : str
              Either "=" for equality or "<" for inequality constraints.

            - **target** : float
              Desired target value (0.0 for all constraints).

            - **value** : float
              Actual evaluated constraint value.

            - **satisfied** : bool
              Whether the constraint is satisfied (based on `tol`).

            Optionally, each dictionary may also include:

            - **normalized_mismatch** : float
              Mismatch scaled by the normalization factor.

            - **mismatch** : float
              Raw constraint violation (value - target).

            - **normalize** : float
              Normalization factor (if defined).

        Raises
        ------
        TypeError
            If `self.problem.constraint_data` exists but is not a list of dictionaries,
            or if required keys are missing from any entry.
        """
        return get_constraint_data(self._evaluation, x_norm, tol)

    def evaluate_kkt_conditions(self, x_norm, tol):
        """
        Evaluate the raw quantities required for KKT condition checks.

        This method performs all necessary calculations to evaluate:

            - Lagrangian gradient
            - Constraint violations (equality and inequality)
            - Lagrange multipliers for active constraints
            - Complementary slackness products

        Uses the shared sign-constrained multiplier estimate and applies ``tol``
        to feasibility and optimality. Metrics use solver coordinates and the
        supplied objective/constraint units; stationarity uses the infinity norm.

        Returns
        -------
        dict
            A dictionary containing raw values needed to assess the KKT conditions.
        """

        point = np.asarray(x_norm, dtype=float).copy()
        self._evaluate_fitness(point, diagnostic=True)
        self._evaluate_gradient(point, diagnostic=True)
        return self._kkt_from_cache(point, tol)

    def _kkt_from_cache(self, x_norm, tol):
        """Assess public-sign KKT residuals using only same-point cached arrays.

        The legacy dual-feasibility field remains a signed minimum multiplier.
        Shared diagnostics additionally expose the nonnegative dual residual and
        infinity-norm stationarity. Multiplier indices are user inequalities,
        then all lower-bound slots, then all upper-bound slots.
        """
        return kkt_adapter.kkt_from_cache(self._evaluation, x_norm, tol)

    def make_kkt_optimality_report(self, x_norm, tol, *, width=80):
        r"""
        Generate a detailed KKT condition satisfaction report.

        This report includes five key Karush-Kuhn-Tucker (KKT) checks:

        .. math::

            \|\nabla L(x, \lambda)\| \leq \mathrm{tol} \quad &\text{(First order optimality)} \\
            \max |c_{\text{eq}}(x)| \leq \mathrm{tol} \quad &\text{(Equality feasibility)} \\
            \max(0, c_{\text{ineq}}(x)) \leq \mathrm{tol} \quad &\text{(Inequality feasibility)} \\
            \min(\lambda_{\text{ineq}}) \geq 0 \quad &\text{(Dual feasibility)} \\
            \max |\lambda_i \cdot c_i| \leq \mathrm{tol} \quad &\text{(Complementary slackness)}

        For each condition, the report shows:

            - Actual computed value
            - Comparison direction and target (tolerance or 0)
            - Satisfaction status

        Parameters
        ----------
        x_norm : array-like
            Normalized decision variable vector to evaluate the KKT conditions at.
        tol : float
            Numerical tolerance used for comparisons in optimality and feasibility checks.
        width : int, optional
            Maximum line width, at least 60; notes wrap as needed.

        Returns
        -------
        str
            A formatted report string summarizing KKT condition satisfaction.
        """
        width = tables.validate_report_width(width)
        return tables.kkt_report(
            self.evaluate_kkt_conditions(x_norm, tol), tol, width=width
        )

    def make_constraint_report(self, x_norm, tol, *, width=80):
        """Format named constraints within ``width`` characters (minimum 60)."""
        width = tables.validate_report_width(width)
        return tables.constraint_report(
            self.get_constraint_data(x_norm, tol), width=width
        )

    def make_lagrange_multipliers_report(self, x_norm, tol, *, width=80):
        """
        Generate a report of all Lagrange multipliers:
        - Equalities: always included
        - Inequalities and finite bounds: all independently estimated multipliers.
        - Unavailable estimates: NaN with an explicit reason.

        ``width`` is the maximum line width (at least 60). Complete names wrap
        onto aligned continuation lines instead of being shortened.

        Returns
        -------
        str
            The formatted multipliers report as a single string.
        """
        width = tables.validate_report_width(width)
        return tables.multipliers_report(
            self.evaluate_kkt_conditions(x_norm, tol),
            self.get_constraint_data(x_norm, tol),
            self.problem.variable_names,
            self.problem.get_bounds_normalized(),
            width=width,
        )

    @property
    def func_count_tot(self):
        return self._evaluation.func_count_tot

    @func_count_tot.setter
    def func_count_tot(self, value):
        self._evaluation.func_count_tot = value

    @property
    def func_count(self):
        return self._evaluation.func_count

    @func_count.setter
    def func_count(self, value):
        self._evaluation.func_count = value

    @property
    def grad_count(self):
        return self._evaluation.grad_count

    @grad_count.setter
    def grad_count(self, value):
        self._evaluation.grad_count = value

    @property
    def derivative_func_count(self):
        return self._evaluation.derivative_func_count

    @derivative_func_count.setter
    def derivative_func_count(self, value):
        self._evaluation.derivative_func_count = value

    @property
    def diagnostic_func_count(self):
        return self._evaluation.diagnostic_func_count

    @diagnostic_func_count.setter
    def diagnostic_func_count(self, value):
        self._evaluation.diagnostic_func_count = value

    @property
    def diagnostic_grad_count(self):
        return self._evaluation.diagnostic_grad_count

    @diagnostic_grad_count.setter
    def diagnostic_grad_count(self, value):
        self._evaluation.diagnostic_grad_count = value

    @property
    def failed_grad_count(self):
        return self._evaluation.failed_grad_count

    @failed_grad_count.setter
    def failed_grad_count(self, value):
        self._evaluation.failed_grad_count = value

    @property
    def cache(self):
        return self._evaluation.cache

    @cache.setter
    def cache(self, value):
        self._evaluation.cache = value

    @property
    def _kkt_cache(self):
        return self._evaluation._kkt_cache

    @_kkt_cache.setter
    def _kkt_cache(self, value):
        self._evaluation._kkt_cache = value

    @property
    def kkt_count(self):
        return self._evaluation.kkt_count

    @kkt_count.setter
    def kkt_count(self, value):
        self._evaluation.kkt_count = value

    @property
    def kkt_time(self):
        return self._evaluation.kkt_time

    @kkt_time.setter
    def kkt_time(self, value):
        self._evaluation.kkt_time = value

    def _plot_view(self):
        """Supply detached observations and presentation settings to the renderer."""
        view = getattr(self, "_plot_state", None)
        if view is None:
            view = self._plot_state = plots.PlotView()
        view.convergence_history = copy.deepcopy(self.convergence_history)
        view.x_final = self.x_final
        view.update_on = self.update_on
        view.problem_name = type(self.problem).__name__
        view.start_datetime = getattr(self, "start_datetime", "")
        view.library = self.library
        view.plot_scale_objective = self.plot_scale_objective
        view.plot_scale_constraints = self.plot_scale_constraints
        view.plot_improvement_only = self.plot_improvement_only
        view.has_constraints = self._has_constraint_plot()
        return view

    def _sync_plot_handles(self):
        """Retain the existing public Matplotlib handle attributes."""
        for name in ["fig", "ax_1", "ax_2", "obj_line_1", "obj_line_2"]:
            if hasattr(self._plot_state, name):
                setattr(self, name, getattr(self._plot_state, name))

    def _plot_history(self, savefile, filename, output_dir, showfig):
        fig = plots.plot_optimization_history(
            self._plot_view(),
            savefile=savefile,
            filename=filename,
            output_dir=output_dir,
            showfig=showfig,
        )
        self._sync_plot_handles()
        return fig
