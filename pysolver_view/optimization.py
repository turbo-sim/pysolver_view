import os
import time
import copy
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import OptimizeResult
from datetime import datetime
from abc import ABC, abstractmethod
from matplotlib.ticker import MaxNLocator

from . import numerical_differentiation
from . import optimization_wrappers as _opt
from .custom_optimization import minimize_pysolver
from .diagnostics.kkt import assess_kkt
from .solvers.slsqp import DomainError
from .pysolver_utilities import savefig_in_formats, validate_keys

# Define valid libraries and their corresponding methods
OPTIMIZATION_LIBRARIES = {
    "pysolver": minimize_pysolver,
    "scipy": _opt.minimize_scipy,
    "pygmo": _opt.minimize_pygmo,
    "pygmo_nlopt": _opt.minimize_nlopt,
}

VALID_LIBRARIES_AND_METHODS = {
    "pysolver": ["slsqp"],
    "scipy": _opt.SCIPY_SOLVERS,
    "pygmo": _opt.PYGMO_SOLVERS,
    "pygmo_nlopt": _opt.NLOPT_SOLVERS,
}


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
        self.result = None
        self.backend_result = None
        self.x_final_norm = None
        self.iteration_count = None
        self.iteration_history = []
        self.derivative_func_count = 0
        self.diagnostic_func_count = 0
        self.diagnostic_grad_count = 0
        self.failed_grad_count = 0
        self.callback_function_call_count = 0
        self._in_callback = False
        self._custom_initial_pending = self.library == "pysolver"
        self._custom_kkt = None
        self._kkt_cache = None
        self.kkt_count = 0
        self.kkt_time = 0.0
        # Initialize variables for convergence report
        self.f_final = None
        self.x_final = None
        self.x_last_norm = None
        self.grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
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
        self.cache = {
            "x": None,
            "f": None,
            "c_eq": None,
            "c_ineq": None,
            "x_jac": None,
            "f_jac": None,
            "c_eq_jac": None,
            "c_ineq_jac": None,
            "fitness": None,
            "gradient": None,
            "constraint_data": None,
            "has_constraint_data": False,
        }

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
        problem = _PygmoProblem(self)
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
        self.result = OptimizeResult(
            x=self.x_final.copy(),
            x_solver=self.x_final_norm.copy(),
            fun=self.f_final,
            fitness=final_values.copy(),
            optimality=final_snapshot["optimality"],
            constraint_violation=final_snapshot["constraint_violation"],
            success=self.success,
            solver_success=solution.get("solver_success", solution.success),
            status=solution.get("status"),
            message=self.message,
            nit=self.iteration_count,
            nfev=self.func_count_tot,
            njev=self.grad_count,
            derivative_func_count=self.derivative_func_count,
            diagnostic_func_count=self.diagnostic_func_count,
            diagnostic_grad_count=self.diagnostic_grad_count,
            failed_grad_count=self.failed_grad_count,
            func_count=self.func_count,
            func_count_total=self.func_count_tot,
            iteration_count=self.iteration_count,
            elapsed_time=self.elapsed_time,
            kkt_enabled=self.check_kkt,
            kkt_passed=None,
            kkt_evaluations=self.kkt_count,
            kkt_time=self.kkt_time,
        )
        if self.library == "pysolver":
            for name in ("feasible", "stationary", "events", "effective_options"):
                self.result[name] = copy.deepcopy(solution.get(name))
            assessment = self._public_driver_kkt(solution.get("kkt"))
            self.result.kkt = assessment
            self.result.kkt_passed = (
                bool(assessment and assessment["passed"]) if self.check_kkt else None
            )
            self.result.kkt_evaluations = solution.get("kkt_evaluations", 0)
            self.result.kkt_time = solution.get("kkt_time", 0.0)
            self.result.optimality = (
                solution.optimality if self.check_kkt else float("nan")
            )
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
        point = np.asarray(x_norm).copy()
        if (
            use_cache
            and self.cache["x"] is not None
            and np.array_equal(point, self.cache["x"])
        ):
            return self.cache["fitness"].copy()
        self.func_count_tot += 1
        if derivative:
            self.derivative_func_count += 1
        elif not diagnostic:
            self.func_count += 1
        if diagnostic:
            self.diagnostic_func_count += 1
        values = np.atleast_1d(
            np.asarray(self.problem.fitness_normalized_input(point.copy()))
        ).copy()
        if values.shape != (1 + self.N_eq + self.N_ineq,):
            raise ValueError(
                "Fitness must contain the objective and declared constraints."
            )
        if cache_result:
            self._kkt_cache = None
            self.cache.update(
                x=point,
                f=values[0],
                c_eq=values[1 : 1 + self.N_eq].copy(),
                c_ineq=values[1 + self.N_eq :].copy(),
                fitness=values.copy(),
                constraint_data=copy.deepcopy(
                    getattr(self.problem, "constraint_data", None)
                ),
                has_constraint_data=hasattr(self.problem, "constraint_data"),
            )
        return values

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
        point = np.asarray(x_norm, dtype=float).copy()
        if self.cache["x_jac"] is not None and np.array_equal(
            point, self.cache["x_jac"]
        ):
            return self.cache["gradient"].copy()
        self.grad_count += 1
        if diagnostic:
            self.diagnostic_grad_count += 1
        try:
            # The unperturbed base is an ordinary model point even when SciPy
            # requests its Jacobian first. Only perturbed samples are probes.
            base = (
                self._evaluate_fitness(point, diagnostic=True)
                if diagnostic
                else self.fitness(point)
            )
            if hasattr(self.problem, "gradient"):
                grad = self.problem.gradient_normalized_input(point.copy())
            else:

                def evaluate_probe(probe):
                    """Count derivative model work without disturbing the base snapshot."""
                    if np.array_equal(probe, point):
                        return base.copy()
                    try:
                        return self._evaluate_fitness(
                            probe,
                            derivative=True,
                            diagnostic=diagnostic,
                            use_cache=False,
                            cache_result=False,
                        )
                    except DomainError as error:
                        if self.library != "pysolver":
                            raise
                        changed = np.flatnonzero(probe != point)
                        raise DomainError(
                            "Derivative probe failed; probes are not retried: "
                            + str(error),
                            metadata={
                                "probe": np.real(probe).tolist(),
                                "column": (
                                    int(changed[0]) if changed.size == 1 else None
                                ),
                                "cause": error.to_dict(),
                            },
                        ) from error

                if self.library == "pysolver":
                    methods = {
                        "forward_finite_differences": "2-point",
                        "central_finite_differences": "3-point",
                        "complex_step": "cs",
                    }
                    bounds = self.problem.get_bounds_normalized()
                    grad = numerical_differentiation.approx_derivative(
                        evaluate_probe,
                        point,
                        f0=base,
                        method=methods.get(
                            self.derivative_method, self.derivative_method
                        ),
                        abs_step=(
                            self.derivative_abs_step
                            if self.derivative_abs_step is not None
                            else self.options.get("finite_diff_step", 1e-7)
                        ),
                        bounds=(-np.inf, np.inf) if bounds is None else bounds,
                    )
                else:
                    grad = numerical_differentiation.approx_gradient(
                        evaluate_probe,
                        point,
                        f0=base,
                        method=self.derivative_method,
                        abs_step=self.derivative_abs_step,
                    )
            grad = (
                np.asarray(grad).reshape(1 + self.N_eq + self.N_ineq, point.size).copy()
            )
        except Exception:
            self.failed_grad_count += 1
            raise
        self._kkt_cache = None
        self.cache.update(
            x_jac=point,
            f_jac=grad[0].copy(),
            c_eq_jac=grad[1 : 1 + self.N_eq].copy(),
            c_ineq_jac=grad[1 + self.N_eq :].copy(),
            gradient=grad.copy(),
        )
        return grad

    def _handle_output(
        self, text: str, savefile: bool = False, filename: str = None, to_console=False
    ):
        """
        Unified output handler: print, log, and optionally save to a file.

        Parameters
        ----------
        text : str
            The content to emit. Can be single or multiline.
        savefile : bool, optional
            If True, saves `text` to the given filename. Default is False.
        filename : str or None, optional
            Path to file where output should be saved. Required if savefile is True.
        """
        # Print to logger
        if self.logger:
            for line in text.splitlines():
                self.logger.info(line)

        # Print to screen if display is enabled and no logger used
        # if self.display and not self.logger:
        if self.display and to_console:
            print(text)

        # Save to file if requested
        if savefile:
            if not filename:
                raise ValueError("Filename must be provided if savefile is True.")
            with open(filename, "w") as f:
                f.write(text)

    def _write_header(self):
        """
        Print a formatted header for the optimization report.

        This internal method is used to display a consistent header format at the
        beginning of the optimization process. The header includes columns for function
        evaluations, gradient evaluations, objective function value, constraint violations,
        and norm of the steps.
        """

        # Define header text
        initial_message = (
            f" Starting optimization process for {type(self.problem).__name__}\n"
            f" Optimization algorithm employed: {self.method}"
        )
        self.header = f" {'Grad-eval':>13}{'Func-eval':>13}{'Func-value':>16}{'Infeasibility':>18}{'Norm of step':>18}{'Optimality':>18} "
        separator = "-" * len(self.header)
        lines_to_output = [
            separator,
            initial_message,
            separator,
            self.header,
            separator,
        ]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line, to_console=True)

        # Store text in memory
        self.convergence_report.extend(lines_to_output)

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
        if assessment is None:
            return None
        public = copy.deepcopy(assessment)
        if public.get("multipliers") is not None:
            public["multipliers"]["equality"] *= -1
        if public.get("complementarity_products") is not None:
            public["complementarity_products"]["inequality"] *= -1
        public["coordinates"] = "solver"
        public["convention"] = "h=0, g<=0; L=f+lambda*h+mu*g-zL*(x-l)+zU*(x-u)"
        return public

    def _record_custom_iteration(self, entry):
        """Record accepted custom-driver points and notify cancellable callbacks.

        The driver invokes this only at the initial or accepted point. Function
        and gradient observation streams may include trials, but user callbacks
        for this backend always receive accepted points. Raising StopIteration
        or returning False stops at that point with a structured failed result.
        """
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
        if self.callback_functions:
            self.callback_function_call_count += 1
            self._in_callback = True
            try:
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
        for name in self.convergence_history:
            self.convergence_history[name].append(copy.deepcopy(record[name]))
        status = (
            f" {record['grad_count']:13d}{record['func_count']:13d}"
            f"{record['objective_value']:+16.3e}{record['constraint_violation']:+18.3e}"
            f"{record['norm_step']:+18.3e}{record['optimality']:+18.3e} "
        )
        if record["event_kind"] == "final":
            status += " Final solution"
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
        """
        Print a formatted footer for the optimization report.

        This method displays the final optimization result, including the
        exit message, success status, objective function value, and decision variables.

        Notes
        -----
        The footer's structure is intended to match the header's style,
        providing a consistent look to the optimization report.
        """
        # Define footer text
        separator = "-" * len(self.header)
        exit_message = f" Exit message: {self.message}"
        success_status = f" Success: {self.success}"
        time_message = f" Solution time: {self.elapsed_time:.3f} seconds"
        lines_to_output = [separator, success_status, exit_message, time_message]
        lines_to_output += [
            f" Final objective: {self.result.fun:+.6e}",
            f" Final constraint violation: {self.result.constraint_violation:+.6e}",
            (
                f" Final optimality (cached derivatives): {self.result.optimality:+.6e}"
                if self.check_kkt
                else " Final optimality: not assessed (KKT disabled)."
            ),
            f" Total model calls: {self.result.nfev}; "
            f"gradient evaluations: {self.result.njev}",
        ]
        if not self.check_kkt:
            lines_to_output.append(
                " Automatic KKT checks: disabled; optimality not assessed."
            )
        lines_to_output += [separator]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line, to_console=True)

        # Store text in memory
        self.convergence_report.extend(lines_to_output)

    def _has_constraint_plot(self):
        """Show feasibility whenever nonlinear constraints or finite bounds exist."""
        bounds = self.problem.get_bounds_normalized()
        has_bounds = bounds is not None and np.any(np.isfinite(np.asarray(bounds)))
        return self.N_eq > 0 or self.N_ineq > 0 or has_bounds

    def _plot_convergence_callback(self, x, iter, initialize=False, showfig=True):
        """
        Callback function to dynamically update the convergence progress plot.

        This method initializes a matplotlib plot on the first iteration and updates
        the data for each subsequent iteration. The plot showcases the evolution of
        the objective function and the constraint violation with respect to the
        number of iterations.

        The left y-axis depicts the objective function values, while the right y-axis
        showcases the constraint violation values. The x-axis represents the number
        of iterations. Both lines are updated and redrawn dynamically as iterations progress.

        Note:
            This is an internal method, meant to be called within the optimization process.
        """

        # Initialize figure before first iteration
        if initialize:
            self.fig, self.ax_1 = plt.subplots()
            (self.obj_line_1,) = self.ax_1.plot(
                [], [], color="#0072BD", marker="o", label="Objective function"
            )
            self.ax_1.set_xlabel(
                {
                    "function": "Function evaluations (excluding derivative/diagnostic calls)",
                    "gradient": "Gradient evaluations",
                    "iteration": (
                        "Accepted SLSQP iterations"
                        if self.library == "pysolver"
                        else "Native SLSQP major iterations"
                    ),
                }[self.update_on]
            )
            self.ax_1.set_ylabel("Objective function")
            self.ax_1.set_yscale(self.plot_scale_objective)
            self.ax_1.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # Integer ticks
            self.ax_1.grid(False)
            if self._has_constraint_plot():
                self.ax_2 = self.ax_1.twinx()
                self.ax_2.grid(False)
                self.ax_2.set_ylabel("Constraint violation")
                self.ax_2.set_yscale(self.plot_scale_constraints)
                (self.obj_line_2,) = self.ax_2.plot(
                    [], [], color="#D95319", marker="o", label="Constraint violation"
                )
                lines = [self.obj_line_1, self.obj_line_2]
                labels = [l.get_label() for l in lines]
                self.ax_2.legend(lines, labels, loc="upper right")
            else:
                self.ax_1.legend(loc="upper right")

            self.fig.tight_layout(pad=1)

        # Update plot data with current values
        axis_field = {
            "function": "func_count",
            "gradient": "grad_count",
            "iteration": "iteration",
        }[self.update_on]
        iteration = self.convergence_history[axis_field]
        objective_function = np.asarray(
            self.convergence_history["objective_value"]
        ).copy()
        constraint_violation = self.convergence_history["constraint_violation"]

        # Derive a display curve without changing the raw observations.
        if self.plot_improvement_only:
            objective_function = np.minimum.accumulate(objective_function)

        # Update graphic objects data
        self.obj_line_1.set_xdata(iteration)
        self.obj_line_1.set_ydata(objective_function)
        if self._has_constraint_plot():
            self.obj_line_2.set_xdata(iteration)
            self.obj_line_2.set_ydata(constraint_violation)

        # Adjust the plot limits
        self.ax_1.relim()
        self.ax_1.autoscale_view()
        if self._has_constraint_plot():
            self.ax_2.relim()
            self.ax_2.autoscale_view()

        # Redraw the plot
        if showfig:
            plt.draw()
            plt.pause(0.01)  # small pause to allow for update

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
        if self.x_final is None:
            raise ValueError(
                "This method can only be used after invoking the 'solve()' method."
            )

        # Initialize and optionally show plot
        self._plot_convergence_callback([], [], initialize=True, showfig=showfig)

        # Save if requested
        if savefile:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                basename = f"{type(self.problem).__name__}_{self.start_datetime}"
                filename = f"convergence_history_{basename}"
            savefig_in_formats(
                self.fig, os.path.join(output_dir, filename), formats=[".png", ".svg"]
            )

        # Close the figure if it should not be displayed
        if not showfig:
            plt.close(self.fig)

        return self.fig

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
    ):
        """
            Generate and print or save a complete optimization report with customizable content.

            This method assembles a detailed summary of the optimization process and results,
            allowing fine-grained control over which components to include. It supports
            outputting the report to the console or saving it to a file.

            The report may include the following sections:
            - Convergence history: number of function and gradient evaluations, objective function value, maximum constraint violation, and step norm. The report also includes The method provides a detailed report on an exit message, success status, and execution time
            - Number of function evaluations
            - Number of gradient evaluations
            - Objective function value
            - Maximum constraint violation
            - Two-norm of the update step


        Objective value, constraint violation, and step norm over iterations.
            - Design variables: Final values, shown in physical units and (if applicable) normalized space.
            - Constraints: Numerical values and satisfaction status of all constraints.
            - KKT conditions: Checks of the Karush-Kuhn-Tucker optimality conditions.
            - Lagrange multipliers: Values of multipliers for equality, inequality, and bound constraints.

            This method is intended to be called after the `solve()` method has been completed.


            Parameters
            ----------
            include_convergence_history : bool
            include_design_variables_normalized : bool
            include_design_variables_physical : bool
            include_constraints : bool
            include_kkt : bool
            include_multipliers : bool
            savefile : bool
            filename : str or None
            tol : float or None  (tolerance for constraints/KKT)
            output_dir : str      (directory for saving file)
        """

        # Define the vector of independent variables
        x_phys = x if x is not None else self.x_final
        x_norm = self.problem.scale_physical_to_normalized(x_phys)

        # Use a loose tolerance if not given as argument
        tol = 5 * self.options["tolerance"] if tol is None else tol

        # Collect each section’s text and soin into a single string
        r = []
        if include_kkt_conditions:
            r.append(self.make_kkt_optimality_report(x_norm, tol))
        if include_design_variables:
            if self.problem.problem_scale is None:
                r.append(self.make_variables_report(x_norm, normalized=False))
            else:
                r.append(self.make_variables_report(x_norm, normalized=True))
                r.append(self.make_variables_report(x_phys, normalized=False))
        if include_constraints:
            r.append(self.make_constraint_report(x_norm, tol))
        if include_multipliers:
            r.append(self.make_lagrange_multipliers_report(x_norm, tol))
        full_text = "\n".join(r)

        # Decide on filename if saving
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

        # Explicit reports honor their own output request, independently of
        # the option that controls live convergence printing.
        self._handle_output(
            full_text, savefile=savefile, filename=filepath, to_console=False
        )
        if to_console:
            print(full_text)

    # def make_optimization_report(self, x_phys, tol=1e-6):
    #     """
    #     Print or save a complete optimization report, including variables and constraints.
    #     """
    #     report = []
    #     report.append(self.make_variables_report(x_phys, normalized=True))
    #     report.append(self.make_variables_report(x_phys, normalized=False))
    #     report.append(self.make_constraint_report(x_phys, tol=tol))
    #     return "\n".join(report)

    def make_variables_report(self, x_norm, normalized=True):
        """
        Generate design variable report as a string.

        Parameters
        ----------
        x_norm : array-like
            Normalized design variables (input to the solver).
        normalized : bool, optional
            Whether to report in normalized or physical values. Default is True.

        Returns
        -------
        str
            Formatted string report.
        """
        x_norm = np.asarray(x_norm)
        if normalized:
            values = x_norm
            bounds = self.problem.get_bounds_normalized()
        else:
            values = x_norm
            bounds = self.problem.get_bounds()

        lb_raw, ub_raw = (
            (np.full(values.size, -np.inf), np.full(values.size, np.inf))
            if bounds is None
            else bounds
        )
        names = self.problem.variable_names or [
            f"design_variable_{i}" for i in range(len(values))
        ]
        scale_type = "normalized" if normalized else "physical"
        max_name_width = 38

        lines = []
        lines.append("")
        lines.append("-" * 80)
        header = f"{' Optimization variables report (' + scale_type + ' values)':<80}"
        lines.append(header)
        lines.append("-" * 80)
        lines.append(f" {'Variable name':<39}{'Lower':>13}{'Value':>13}{'Upper':>13}")
        lines.append("-" * 80)

        for key, lb, val, ub in zip(names, lb_raw, values, ub_raw):
            if len(key) > max_name_width:
                key = "..." + key[-(max_name_width - 3) :]

            if normalized:
                lines.append(f" {key:<39}{lb:>13.4f}{val:>13.4f}{ub:>13.4f}")
            else:
                lines.append(f" {key:<39}{lb:>13.3e}{val:>13.3e}{ub:>13.3e}")

        lines.append("-" * 80)
        return "\n".join(lines)

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
        # Metadata must belong to the same model evaluation as the requested values.
        self._evaluate_fitness(x_norm, diagnostic=True)
        data = copy.deepcopy(self.cache["constraint_data"])
        if self.cache["has_constraint_data"]:
            if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
                raise TypeError("Expected self.constraint_data to be a list of dicts")
            required = {"name", "type", "target", "value"}
            allowed = required | {
                "satisfied",
                "normalized_mismatch",
                "mismatch",
                "normalize",
            }
            equality_index, inequality_index = 0, 0
            for entry in data:
                validate_keys(entry, required, allowed)
                if entry["type"] == "=":
                    if equality_index >= self.N_eq:
                        raise ValueError("Too many named equality constraints.")
                    mismatch = self.cache["c_eq"][equality_index]
                    entry["satisfied"] = bool(abs(mismatch) <= tol)
                    equality_index += 1
                elif entry["type"] in {"<", ">"}:
                    if inequality_index >= self.N_ineq:
                        raise ValueError("Too many named inequality constraints.")
                    mismatch = self.cache["c_ineq"][inequality_index]
                    entry["satisfied"] = bool(mismatch <= tol)
                    inequality_index += 1
                else:
                    raise ValueError("Named constraint type must be '=', '<', or '>'.")
            if (equality_index, inequality_index) != (self.N_eq, self.N_ineq):
                raise ValueError(
                    "Named constraint counts must match the fitness vector."
                )
            return data

        # Otherwise evaluate constraints from scratch
        x_norm = np.asarray(x_norm)
        fitness = self._evaluate_fitness(x_norm, diagnostic=True)
        c_eq = fitness[1 : 1 + self.N_eq]
        c_ineq = fitness[1 + self.N_eq :]
        data = []

        # Equality constraints
        for i, val in enumerate(c_eq):
            data.append(
                {
                    "name": f"equality_constraint_{i}",
                    "type": "=",
                    "target": 0.0,
                    "value": val,
                    "satisfied": abs(val) <= tol,
                }
            )

        # Inequality constraints
        for i, val in enumerate(c_ineq):
            data.append(
                {
                    "name": f"inequality_constraint_{i}",
                    "type": "<",
                    "target": 0.0,
                    "value": val,
                    "satisfied": val <= tol,
                }
            )

        return data

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
        point = np.asarray(x_norm, dtype=float)
        if not (
            np.array_equal(point, self.cache["x"])
            and np.array_equal(point, self.cache["x_jac"])
        ):
            raise ValueError(
                "KKT assessment requires values and derivatives at the same point."
            )
        if self._kkt_cache is not None and self._kkt_cache[0] == tol:
            return copy.deepcopy(self._kkt_cache[1])
        bounds = self.problem.get_bounds_normalized()
        lower, upper = (
            (np.full(point.size, -np.inf), np.full(point.size, np.inf))
            if bounds is None
            else map(np.asarray, bounds)
        )
        started_check = time.perf_counter()
        assessment = assess_kkt(
            point,
            self.cache["f_jac"],
            self.cache["c_eq"],
            self.cache["c_eq_jac"],
            self.cache["c_ineq"],
            self.cache["c_ineq_jac"],
            lower,
            upper,
            feasibility_tol=tol,
            optimality_tol=tol,
        )
        self.kkt_time += time.perf_counter() - started_check
        self.kkt_count += 1
        assessment["coordinates"] = "solver"
        assessment["physical_x"] = self.problem.scale_normalized_to_physical(
            point
        ).copy()
        assessment["problem_scale"] = self.problem.problem_scale
        assessment["derivative_source"] = (
            "analytic" if hasattr(self.problem, "gradient") else self.derivative_method
        )
        inequalities = np.r_[self.cache["c_ineq"], lower - point, point - upper]
        finite = np.r_[
            np.ones(self.N_ineq, dtype=bool), np.isfinite(lower), np.isfinite(upper)
        ]
        if assessment["available"]:
            multipliers = assessment["multipliers"]
            duals = np.r_[
                multipliers["inequality"], multipliers["lower"], multipliers["upper"]
            ]
            signed_dual = float(np.min(duals[finite])) if np.any(finite) else 0.0
            selected = {i: float(duals[i]) for i in np.flatnonzero(finite)}
            products = (duals[finite] * inequalities[finite]).tolist()
            equality_multipliers = multipliers["equality"].copy()
        else:
            signed_dual = float("nan")
            selected = {i: float("nan") for i in np.flatnonzero(finite)}
            products = []
            equality_multipliers = np.full(self.N_eq, np.nan)
        first = assessment["stationarity"]
        first = float("nan") if first is None else first
        complementarity = assessment["complementarity"]
        complementarity = float("nan") if complementarity is None else complementarity
        equality_error = float(np.max(np.abs(self.cache["c_eq"]), initial=0))
        inequality_error = float(np.max(np.maximum(inequalities[finite], 0), initial=0))
        result = {
            **assessment,
            "equality_violation": self.cache["c_eq"].copy(),
            "inequality_violation": inequalities,
            "multipliers_eq": equality_multipliers,
            "multipliers_ineq": selected,
            "slack_products": products,
            "first_order_optimality": first,
            "feasibility_equality": equality_error,
            "feasibility_inequality": inequality_error,
            "feasibility_dual": signed_dual,
            "complementary_slackness": complementarity,
            "first_order_optimality_ok": bool(first <= tol),
            "feasibility_equality_ok": bool(equality_error <= tol),
            "feasibility_inequality_ok": bool(inequality_error <= tol),
            "feasibility_dual_ok": bool(signed_dual >= -tol),
            "complementary_slackness_ok": bool(complementarity <= tol),
        }
        self._kkt_cache = tol, copy.deepcopy(result)
        return result

    def make_kkt_optimality_report(self, x_norm, tol):
        r"""
        Generate a detailed KKT condition satisfaction report (80-character width).

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

        Returns
        -------
        str
            A formatted report string summarizing KKT condition satisfaction.
        """
        # Compute KKT conditions
        kkt_data = self.evaluate_kkt_conditions(x_norm, tol)

        # Build report lines
        sep = "-" * 80
        lines = [
            "",
            sep,
            f" {'Karush-Kuhn-Tucker (KKT) conditions':<36}{'Value':>16}{'Target':>16}{'Ok?':>10}",
            sep,
        ]

        entries = [
            (
                "First order optimality",
                kkt_data["first_order_optimality"],
                f"<= {tol:+.3e}",
                kkt_data["first_order_optimality_ok"],
            ),
            (
                "Equality feasibility",
                kkt_data["feasibility_equality"],
                f"<= {tol:+.3e}",
                kkt_data["feasibility_equality_ok"],
            ),
            (
                "Inequality feasibility",
                kkt_data["feasibility_inequality"],
                f"<= {tol:+.3e}",
                kkt_data["feasibility_inequality_ok"],
            ),
            (
                "Dual feasibility",
                kkt_data["feasibility_dual"],
                f">= {-tol:+.3e}",
                kkt_data["feasibility_dual_ok"],
            ),
            (
                "Complementary slackness",
                kkt_data["complementary_slackness"],
                f"<= {tol:+.3e}",
                kkt_data["complementary_slackness_ok"],
            ),
        ]

        for name, val, target, ok in entries:
            status = "yes" if ok else " no"
            lines.append(f" {name:<36}{val:>+16.3e}{target:>16}{status:>10}")

        lines.append(" Coordinates: solver; stationarity norm: infinity.")
        lines.append(
            " Multipliers: independently estimated; first-order conditions only."
        )
        if not kkt_data["available"]:
            lines.append(" Assessment unavailable: " + kkt_data.get("message", ""))
        lines.append(sep)
        return "\n".join(lines)

    # def make_constraint_report(self, x_norm, tol):
    #     """
    #     Generate a compact constraint report that includes:
    #     - Constraint value and target
    #     - Satisfaction status
    #     - Lagrange multiplier (if active), otherwise 'inactive'

    #     This merges value and multiplier reports for better readability.

    #     Parameters
    #     ----------
    #     x_norm : array-like
    #         Normalized design variable vector.
    #     tol : float
    #         Tolerance used for satisfaction and KKT activeness checks.

    #     Returns
    #     -------
    #     str
    #         A formatted string with the constraint summary.
    #     """
    #     max_name_width = 35
    #     constraint_data = self.get_constraint_data(x_norm, tol)
    #     kkt_data = self.evaluate_kkt_conditions(x_norm, tol)
    #     active_multipliers = kkt_data["multipliers_ineq"]
    #     multipliers_eq = kkt_data["multipliers_eq"]

    #     sep = "-" * 80
    #     lines = [
    #         "",
    #         sep,
    #         " Constraint summary report",
    #         sep,
    #         f" {'Constraint name':<{max_name_width}}{'Value':>12}{'Target':>13}{'Ok?':>6}{'Multiplier':>12}",
    #         sep,
    #     ]

    #     eq_count = 0
    #     ineq_count = 0
    #     var_names = self.problem.variable_names
    #     n_vars = len(var_names)
    #     n_ineq = self.problem.get_nic()

    #     for i, entry in enumerate(constraint_data):
    #         name = entry.get("name", "")
    #         ctype = entry.get("type", "")
    #         value = entry.get("value", 0.0)
    #         target = entry.get("target", 0.0)
    #         satisfied = "yes" if entry.get("satisfied", False) else " no"

    #         if len(name) > max_name_width:
    #             name = "..." + name[-(max_name_width - 3):]

    #         # Determine the target symbol
    #         symbol = "=" if ctype == "=" else "<"
    #         target_str = f"{symbol} {target:+.3e}"
    #         value_str = f"{value:+.3e}"

    #         # Multiplier
    #         if ctype == "=":
    #             multiplier = f"{multipliers_eq[eq_count]:+.3e}"
    #             eq_count += 1
    #         elif ctype == "<":
    #             # Inequality constraint or bounds
    #             if ineq_count in active_multipliers:
    #                 multiplier = f"{active_multipliers[ineq_count]:+.3e}"
    #             else:
    #                 multiplier = "inactive"
    #             ineq_count += 1
    #         else:
    #             multiplier = "---"

    #         lines.append(f" {name:<{max_name_width}}{value_str:>12}{target_str:>13}{satisfied:>6}{multiplier:>12}")

    #     lines.append(sep)
    #     return "\n".join(lines)

    # def make_constraint_report(self, x_norm, tol):
    #     """
    #     Generate a compact constraint report that includes:
    #     - Constraint value and target
    #     - Satisfaction status
    #     - Lagrange multiplier (if active), otherwise 'inactive'

    #     Supports equality, less-than, and greater-than constraints, including bounds.

    #     Parameters
    #     ----------
    #     x_norm : array-like
    #         Normalized design variable vector.
    #     tol : float
    #         Tolerance used for constraint satisfaction and activeness checks.

    #     Returns
    #     -------
    #     str
    #         A formatted string with the constraint summary.
    #     """
    #     max_name_width = 35
    #     var_names = self.problem.variable_names
    #     n_vars = len(var_names)

    #     constraint_data = self.get_constraint_data(x_norm, tol)
    #     kkt_data = self.evaluate_kkt_conditions(x_norm, tol)
    #     active_multipliers = kkt_data["multipliers_ineq"]
    #     multipliers_eq = kkt_data["multipliers_eq"]

    #     # Append bound constraints with natural inequality direction
    #     lb, ub = self.problem.get_bounds_normalized()
    #     bounds_lower, bounds_upper = [], []
    #     for i, (name, xi, lb_i, ub_i) in enumerate(zip(var_names, x_norm, lb, ub)):
    #         if lb_i is not None:
    #             bounds_lower.append({
    #                 "name": name,
    #                 "type": ">",
    #                 "value": xi,
    #                 "target": lb_i,
    #                 "satisfied": (xi - lb_i >= -tol)  # xi ≥ lb_i
    #             })
    #         if ub_i is not None:
    #             bounds_upper.append({
    #                 "name": name,
    #                 "type": "<",
    #                 "value": xi,
    #                 "target": ub_i,
    #                 "satisfied": (ub_i - xi >= -tol)  # xi ≤ ub_i
    #             })

    #     # Then concatenate in order:
    #     constraint_data = bounds_lower + bounds_upper + constraint_data
    #     print(constraint_data)

    #     # Build the report
    #     sep = "-" * 80
    #     lines = [
    #         "",
    #         sep,
    #         f" {'Constraint summary report':<{max_name_width}}{'Value':>12}{'Target':>13}{'Ok?':>6}{'Multiplier':>12}",
    #         sep,
    #     ]

    #     eq_count = 0
    #     ineq_count = 0
    #     for entry in constraint_data:
    #         name = entry.get("name", "")
    #         ctype = entry.get("type", "")
    #         value = entry.get("value", 0.0)
    #         target = entry.get("target", 0.0)
    #         satisfied = "yes" if entry.get("satisfied", False) else " no"

    #         if len(name) > max_name_width:
    #             name = "..." + name[-(max_name_width - 3):]

    #         # Set the symbol for the target constraint
    #         if ctype == "=":
    #             symbol = "="
    #         elif ctype == "<":
    #             symbol = "<"
    #         elif ctype == ">":
    #             symbol = ">"
    #         else:
    #             symbol = "?"

    #         value_str = f"{value:+.3e}"
    #         target_str = f"{symbol} {target:+.3e}"

    #         # Handle multipliers
    #         if ctype == "=":
    #             multiplier = f"{multipliers_eq[eq_count]:+.3e}"
    #             eq_count += 1
    #         elif ctype in {"<", ">"}:
    #             if ineq_count in active_multipliers:
    #                 multiplier = f"{active_multipliers[ineq_count]:+.3e}"
    #             else:
    #                 multiplier = "inactive"
    #             ineq_count += 1
    #         else:
    #             multiplier = "---"

    #         lines.append(f" {name:<{max_name_width}}{value_str:>12}{target_str:>13}{satisfied:>6}{multiplier:>12}")

    #     lines.append(sep)
    #     return "\n".join(lines)

    def make_constraint_report(self, x_norm, tol):
        """
        generate a formatted constraint report at x_phys, using get_constraint_data to build/validate the entries.
        """
        max_name_width = 48
        constraint_data = self.get_constraint_data(x_norm, tol)

        # from .pysolver_utilities import print_dict
        # print("Data dictionary")
        # for item in constraint_data:
        #     print()
        #     print_dict(item)

        lines = []
        lines = []
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{' Optimization constraints report':<80}")
        lines.append("-" * 80)
        lines.append(f"{' Constraint name':<49}{'Value':>11}{'Target':>13}{'Ok?':>6}")
        lines.append("-" * 80)

        for entry in constraint_data:
            name = entry.get("name", "")
            ctype = entry.get("type", "")
            target = entry.get("target", 0.0)
            value = entry.get("value", 0.0)
            satisfied = "yes" if entry.get("satisfied", False) else "no"

            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3) :]

            symbol_target = f"{ctype} {target:+.2e}"
            lines.append(f" {name:<48}{value:>+11.2e}{symbol_target:>13}{satisfied:>6}")

        lines.append("-" * 80)
        return "\n".join(lines)

    def make_lagrange_multipliers_report(self, x_norm, tol):
        """
        Generate a report of all Lagrange multipliers:
        - Equalities: always included
        - Inequalities and finite bounds: all independently estimated multipliers.
        - Unavailable estimates: NaN with an explicit reason.

        Returns
        -------
        str
            The formatted multipliers report as a single string.
        """
        # Compute KKT data with given tolerance
        kkt_data = self.evaluate_kkt_conditions(x_norm, tol)
        constraint_data = self.get_constraint_data(x_norm, tol)
        eq_data = [c for c in constraint_data if c["type"] == "="]
        ineq_data = [c for c in constraint_data if c["type"] in {"<", ">"}]

        # Print Lagrange multipliers in tabular format
        sep = "-" * 80
        max_name_width = 48
        lines = [
            "",
            sep,
            " Lagrange multipliers report (estimated, solver coordinates)",
            sep,
            f" {'Multiplier for constraint':<{max_name_width}}{'Type':>15}{'Value':>15}",
            sep,
        ]

        # 1. Equality constraints
        for i, entry in enumerate(eq_data):
            name = entry["name"]
            ctype = "equality"
            lam = kkt_data["multipliers_eq"][i]
            valstr = f"{lam:+.3e}"
            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3) :]
            lines.append(f" {name:<{max_name_width}}{ctype:>15}{valstr:>15}")

        # 2. Inequality + bound constraints
        var_names = self.problem.variable_names
        n_vars = len(var_names)
        n_ineq = len(ineq_data)
        total_ineq = n_ineq + 2 * n_vars
        active = set(kkt_data["multipliers_ineq"].keys())

        for j in range(total_ineq):
            if j < n_ineq:
                name = ineq_data[j]["name"]
                ctype = "inequality"
            elif j < n_ineq + n_vars:
                idx = j - n_ineq
                name = var_names[idx]
                ctype = "lower bound"
            else:
                idx = j - n_ineq - n_vars
                name = var_names[idx]
                ctype = "upper bound"

            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3) :]

            if j >= n_ineq:
                bound = self.problem.get_bounds_normalized()
                index = j - n_ineq if j < n_ineq + n_vars else j - n_ineq - n_vars
                side = 0 if j < n_ineq + n_vars else 1
                if bound is None or not np.isfinite(bound[side][index]):
                    continue
            if j in active:
                valstr = f"{kkt_data['multipliers_ineq'][j]:+.3e}"
            else:
                valstr = "inactive"

            lines.append(f" {name:<{max_name_width}}{ctype:>15}{valstr:>15}")

        if not kkt_data["available"]:
            lines.append(" Assessment unavailable: " + kkt_data.get("message", ""))
        lines.append(sep)
        return "\n".join(lines)


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.

    Derived optimization problem objects must implement the following methods:

    - `fitness`: Evaluate the objective function and constraints for a given set of decision variables.
    - `get_bounds`: Get the bounds for each decision variable.
    - `get_neq`: Return the number of equality constraints associated with the problem.
    - `get_nineq`: Return the number of inequality constraints associated with the problem.

    Additionally, specific problem classes can define the `gradient` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    fitness(x)
        Evaluate the objective function and constraints for a given set of decision variables.
    get_bounds()
        Get the bounds for each decision variable.
    get_neq()
        Return the number of equality constraints associated with the problem.
    get_nineq()
        Return the number of inequality constraints associated with the problem.

    Parameters
    ----------
    problem_scale : float, optional
        Scaling factor for normalization. Default is 1.0.
    variable_names : list of str, optional
        Names of the decision variables. Used for reporting and debugging.

    """

    @abstractmethod
    def fitness(self, x):
        """
        Evaluate the objective function and constraints for given decision variables.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).

        Returns
        -------
        array_like
            Vector containing the objective function, equality constraints, and inequality constraints.
        """
        pass

    @abstractmethod
    def get_bounds(self):
        """
        Get the bounds for each decision variable (Pygmo format)

        Returns
        -------
        bounds : tuple of lists
            A tuple of two items where the first item is the list of lower bounds and the second
            item of the list of upper bounds for the vector of decision variables. For example,
            ([-2 -1], [2, 1]) indicates that the first decision variable has bounds between
            -2 and 2, and the second has bounds between -1 and 1.
        """
        pass

    @abstractmethod
    def get_nec(self):
        """
        Return the number of equality constraints associated with the problem.

        Returns
        -------
        neq : int
            Number of equality constraints.
        """
        pass

    @abstractmethod
    def get_nic(self):
        """
        Return the number of inequality constraints associated with the problem.

        Returns
        -------
        nineq : int
            Number of inequality constraints.
        """
        pass

    def get_bounds_normalized(self):
        """
        Return normalized bounds in [0, problem_scale]. If no scaling is applied (i.e., problem_scale is None),
        the physical bounds are returned instead.

        Returns
        -------
        tuple of lists
            Tuple (lb, ub) in normalized space or the original physical bounds if problem_scale is None.
        """
        if self.problem_scale is None:
            return self.get_bounds()

        n = len(self.get_bounds()[0])
        return [0.0] * n, [self.problem_scale] * n

    def scale_physical_to_normalized(self, x_phys):
        """
        Convert physical design variable values to normalized values in the range [0, problem_scale].
        If self.problem_scale is None, no scaling is applied.

        The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.
        It automatically handles the case of fixed variables (i.e., upper bound = lower bound) by returning 0.0.

        Parameters
        ----------
        x_phys : array-like
            Physical values of the decision variables.

        Returns
        -------
        np.ndarray
            Normalized values in the range [0, problem_scale] if scaling is applied,
            otherwise the original physical values.
        """
        x_phys = np.asarray(x_phys)
        if self.problem_scale is None:
            return x_phys
        lb, ub = self.get_bounds()
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        x_scaled = []
        for xi, lbi, ubi in zip(x_phys, lb, ub):
            if ubi == lbi:
                x_scaled.append(0.0)
            else:
                x_scaled.append(self.problem_scale * (xi - lbi) / (ubi - lbi))

        return np.array(x_scaled)

    def scale_normalized_to_physical(self, x_norm):
        """
        Convert normalized values in the range [0, problem_scale] back to physical variable values.
        If self.problem_scale is None, no scaling is applied.

        The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.

        Parameters
        ----------
        x_norm : array-like
            Normalized values of the decision variables.

        Returns
        -------
        np.ndarray
            Physical values corresponding to the normalized input, or the original values if no scaling.
        """
        x_norm = np.asarray(x_norm)
        if self.problem_scale is None:
            return x_norm
        lb, ub = self.get_bounds()
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        return lb + (ub - lb) * (x_norm / self.problem_scale)

    def fitness_normalized_input(self, x_norm):
        """
        Evaluate fitness starting from normalized input.

        Parameters
        ----------
        x_norm : array-like
            Normalized input vector.

        Returns
        -------
        array_like
            Output of `fitness` evaluated on the corresponding physical vector.
        """
        x_phys = self.scale_normalized_to_physical(x_norm)
        return self.fitness(x_phys)

    def gradient_normalized_input(self, x_norm):
        """
        Compute the gradient of the objective and constraints with respect to normalized variables.

        This function applies the chain rule to convert the gradient computed in the physical space
        to the corresponding gradient in the normalized space. If `problem_scale` is set to `None`,
        the problem is considered unscaled and the gradient is returned directly.

        Parameters
        ----------
        x_norm : array-like
            Normalized input vector (typically in [0, problem_scale]).

        Returns
        -------
        np.ndarray
            Gradient with respect to the normalized variables. The shape is:
            - (n,) for scalar objective or flat constraint vectors.
            - (m, n) for vector-valued constraints with m rows and n design variables.

        Notes
        -----
        The chain rule is applied as follows:

            x_phys = lb + (ub - lb) * (x_norm / problem_scale)
            ∇f(x_norm) = ∇f(x_phys) * d(x_phys)/d(x_norm)

        where:

            d(x_phys)/d(x_norm) = (ub - lb) / problem_scale    [elementwise]

        This correction ensures that exact user-defined gradients are consistent
        with the scaling applied during optimization.
        """
        x_phys = self.scale_normalized_to_physical(x_norm)
        grad_phys = self.gradient(x_phys)

        if self.problem_scale is None:
            return grad_phys

        lb, ub = self.get_bounds()
        lb, ub = np.asarray(lb), np.asarray(ub)
        scaling_factor = (ub - lb) / self.problem_scale

        # Apply scaling using broadcasting to support both 1D and 2D gradients
        return grad_phys * scaling_factor

    def clip_to_bounds(self, x_physical, logger=None):
        """
        Clip physical variable values to lie within specified bounds.

        Parameters
        ----------
        x_physical : array-like
            Input vector in physical space.
        logger : logging.Logger or None, optional
            Logger for outputting warnings. If None, warnings are printed to standard output.

        Returns
        -------
        np.ndarray
            Clipped vector.
        """
        lb, ub = self.get_bounds()
        x_physical = np.atleast_1d(x_physical)
        x_clipped = np.clip(x_physical, lb, ub)

        for i, (orig, new, low, high) in enumerate(zip(x_physical, x_clipped, lb, ub)):
            if orig != new:
                msg = (
                    "-" * 80 + "\n"
                    f"Warning: optimization variable out of bounds\n"
                    f"  Name       : {self.variable_names[i]}\n"
                    f"  Original   : {orig:.3e}\n"
                    f"  Bounds     : [{low:.3e}, {high:.3e}]\n"
                    f"  Clipped to : {new:.3e}\n" + "-" * 80
                )
                if logger:
                    for line in msg.splitlines():
                        logger.warning(line)
                else:
                    print(msg)

        return x_clipped


def count_constraints(var):
    """
    Retrieve the number of constraints based on the provided input.

    This function returns the count of constraints based on the nature of the
    input:

    - `None` returns 0
    - Scalar values return 1
    - Array-like structures return their length

    Parameters
    ----------
    var : None, scalar, or array-like (list, tuple, np.ndarray)
        The input representing the constraint(s). This can be `None`, a scalar value,
        or an array-like structure containing multiple constraints.

    Returns
    -------
    int
        The number of constraints:

        - 0 for `None`
        - 1 for scalar values
        - Length of the array-like for array-like inputs

    Examples
    --------
    >>> count_constraints(None)
    0

    >>> count_constraints(5.0)
    1

    >>> count_constraints([1.0, 2.0, 3.0])
    3
    """
    # If constraint is None
    if var is None:
        return 0
    # If constraint is a scalar (assuming it's numeric)
    elif np.isscalar(var):
        return 1
    # If constraint is array-like
    else:
        return len(var)


def combine_objective_and_constraints(f, c_eq=None, c_ineq=None):
    """
    Combine an objective function with its associated equality and inequality constraints.

    This function takes in an objective function value, a set of equality constraints,
    and a set of inequality constraints. It then returns a combined Numpy array of
    these values. The constraints can be given as a list, tuple, numpy array, or as
    individual values.

    Parameters
    ----------
    f : float
        The value of the objective function.
    c_eq : float, list, tuple, np.ndarray, or None
        The equality constraint(s). This can be a single value or a collection of values.
        If `None`, no equality constraints will be added.
    c_ineq : float, list, tuple, np.ndarray, or None
        The inequality constraint(s). This can be a single value or a collection of values.
        If `None`, no inequality constraints will be added.

    Returns
    -------
    np.ndarray
        A numpy array consisting of the objective function value followed by equality and
        inequality constraints.

    Examples
    --------
    >>> combine_objective_and_constraints(1.0, [0.5, 0.6], [0.7, 0.8])
    array([1. , 0.5, 0.6, 0.7, 0.8])

    >>> combine_objective_and_constraints(1.0, 0.5, 0.7)
    array([1. , 0.5, 0.7])
    """

    # Validate objective function value
    if isinstance(f, (list, tuple, np.ndarray)):
        if len(f) != 1:
            raise ValueError(
                "Objective function value 'f' must be a scalar or single-element array."
            )
        f = f[0]  # Unwrap the single element to ensure it's treated as a scalar

    # Add objective function
    combined_list = [f]

    # Add equality constraints
    if c_eq is not None:
        if isinstance(c_eq, (list, tuple, np.ndarray)):
            combined_list.extend(c_eq)
        else:
            combined_list.append(c_eq)

    # Add inequality constraints
    if c_ineq is not None:
        if isinstance(c_ineq, (list, tuple, np.ndarray)):
            combined_list.extend(c_ineq)
        else:
            combined_list.append(c_ineq)

    return np.array(combined_list)


class _PygmoProblem:
    """
    A wrapper class for optimization problems to be compatible with Pygmo's need for deep-copiable problems.
    This class uses anonymous functions (lambda) to prevent issues with deep copying complex objects,
    (like Coolprop's AbstractState objects) which are not deep-copiable.
    """

    def __init__(self, wrapped_problem):
        # Pygmo requires a flattened Jacobian for gradients, unlike SciPy's two-dimensional array.
        self.iteration_callback = lambda x: wrapped_problem._record_iteration(x)
        self.accepted_callback = lambda entry: wrapped_problem._record_custom_iteration(
            entry
        )
        self.fitness = lambda x: wrapped_problem.fitness(x)
        self.gradient = lambda x: wrapped_problem.gradient(x).flatten()

        # Directly link bounds and constraint counts from the original problem.
        self.get_bounds = lambda: wrapped_problem.problem.get_bounds_normalized()
        self.get_nec = lambda: wrapped_problem.problem.get_nec()
        self.get_nic = lambda: wrapped_problem.problem.get_nic()

        # If the original problem defines Hessians, provide them as well.
        if hasattr(wrapped_problem.problem, "hessians"):
            self.hessians = lambda x: wrapped_problem.problem.hessians(x)

        # Define anonymous functions for objective and constraints with their Jacobians.
        self.f = lambda x: wrapped_problem.fitness(x)[0]
        self.c_eq = lambda x: wrapped_problem.fitness(x)[1 : 1 + self.get_nec()]
        self.c_ineq = lambda x: wrapped_problem.fitness(x)[1 + self.get_nec() :]

        self.f_jac = lambda x: wrapped_problem.gradient(x)[0, :]
        self.c_eq_jac = lambda x: wrapped_problem.gradient(x)[1 : 1 + self.get_nec(), :]
        self.c_ineq_jac = lambda x: wrapped_problem.gradient(x)[1 + self.get_nec() :, :]
