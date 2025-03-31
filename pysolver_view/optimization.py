import os
import time
import copy
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from abc import ABC, abstractmethod
from matplotlib.ticker import MaxNLocator

from . import numerical_differentiation
from . import optimization_wrappers as _opt
from .pysolver_utilities import savefig_in_formats, validate_keys


# Define valid libraries and their corresponding methods
OPTIMIZATION_LIBRARIES = {
    "scipy": _opt.minimize_scipy,
    "pygmo": _opt.minimize_pygmo,
    "pygmo_nlopt": _opt.minimize_nlopt,
}

VALID_LIBRARIES_AND_METHODS = {
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
        Specifies if the convergence report should be updated based on new function evaluations or gradient evaluations (default is 'gradient', alternative is 'function').
    callback_functions : list of callable or callable, optional
        Optional list of callback functions to pass to the solver.
    plot_improvement_only : bool, optional
        If True, plots only display iterations that improve the objective function value (useful for gradient-free optimizers) (default is False).

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
    ):
        # Initialize class variables
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

        # Validate library and method
        self._validate_library_and_method()

        # Define options dictionary
        self.options = {"tolerance": tolerance, "max_iterations": max_iterations}
        self.options = self.options | extra_options

        # Set scaling for the optimization problem (define at the solver, rather than problem level)
        self.problem.problem_scale = problem_scale

        # Auto-generate design variable names if not provided
        if not hasattr(self.problem, "variable_names"):
            n_vars = len(self.problem.get_bounds()[0])
            self.problem.variable_names = [f"design_variable_{i}" for i in range(n_vars)]

        # Check for logger validity
        if self.logger is not None:
            if not isinstance(self.logger, logging.Logger):
                raise ValueError(
                    "The provided logger is not a valid logging.Logger instance."
                )

        # Check for valid display_on value
        self.update_on = update_on
        if update_on not in ["function", "gradient"]:
            raise ValueError(
                "Invalid value for 'update_on'. It should be either 'function' or 'gradient'."
            )

        # Rename number of constraints
        self.N_eq = self.problem.get_nec()
        self.N_ineq = self.problem.get_nic()

        # Initialize variables for convergence report
        self.f_final = None
        self.x_final = None
        self.x_last_norm = None
        self.grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.success = None
        self.message = None
        self.solution_report = []
        self.elapsed_time = None
        self.include_solution_in_footer = False
        self.convergence_history = {
            "x": [],
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
        """
        Solve the optimization problem using the specified library and solver.

        This method initializes the optimization process, manages the flow of the optimization,
        and handles the results, utilizing the solver from a specified library such as scipy or pygmo.

        Parameters
        ----------
        x0 : array-like, optional
            Initial guess for the solution of the optimization problem.

        Returns
        -------
        x_final : array-like
            An array with the optimal vector of design variables

        """
        # Initialize convergence plot
        if self.plot:
            self._plot_convergence_callback([], [], initialize=True)

        # Get start datetime
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Start timing with high-resolution timer
        start_time = time.perf_counter()

        # Normalize the problem as the very first thing
        x0 = self.problem.clip_to_bounds(x0, verbose=True)
        x0 = self.problem.scale_physical_to_normalized(x0)

        # Print report header
        self._write_header()

        # Define new problem with anonymous methods (avoid problems when Pygmo creates a deep copy)
        problem = _PygmoProblem(self)

        # Fetch the solver function
        lib_wrapper = OPTIMIZATION_LIBRARIES[self.library]

        # Print initial guess evaluation when using gradient
        if self.update_on == "gradient":
            self._print_convergence_progress(x0)

        # Solv ethe problem
        solution = lib_wrapper(problem, x0, self.method, self.options)

        # Retrieve last solution (also works for gradient-free solvers when updating on gradient)
        x_last = self.x_last_norm if self.x_last_norm is not None else self.cache["x"]

        # Save solution
        self.x_final = copy.deepcopy(x_last)
        self.f_final = copy.deepcopy(self.cache["f"])
        self.success, self.message = solution

        # Calculate elapsed time
        self.elapsed_time = time.perf_counter() - start_time

        # Print report footer
        self._print_convergence_progress(self.x_final)
        self._write_footer()

        # Save variable in physical scale
        self.x_final = self.problem.scale_normalized_to_physical(self.x_final)

        return self.x_final

    def fitness(self, x_norm, called_from_grad=False):
        """
        Evaluates the optimization problem values at a given point x.

        This method queries the `fitness` method of the OptimizationProblem class to
        compute the objective function value and constraint values. It first checks the cache
        to avoid redundant evaluations. If no matching cached result exists, it proceeds to
        evaluate the objective function and constraints.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).
        called_from_grad : bool, optional
            Flag used to indicate if the method is called during gradient evaluation.
            This helps in preventing redundant increments in evaluation counts during
            finite-differences gradient calculations. Default is False.

        Returns
        -------
        fitness : numpy.ndarray
            A 1D array containing the objective function, equality constraints, and inequality constraints at `x`.

        """
        # If x hasn't changed, use cached values
        if np.array_equal(x_norm, self.cache["x"]):
            return self.cache["fitness"]

        # Increase total counter (includes finite differences)
        self.func_count_tot += 1

        # Evaluate objective function and constraints at once
        fitness = self.problem.fitness_normalized_input(x_norm)

        # Does not include finite differences
        if not called_from_grad:
            # Update cached variabled
            self.cache.update(
                {
                    "x": x_norm.copy(),  # Needed for finite differences
                    "f": fitness[0],
                    "c_eq": fitness[1 : 1 + self.N_eq],
                    "c_ineq": fitness[1 + self.N_eq :],
                    "fitness": fitness,
                }
            )

            # Increase minor iteration counter (line search)
            self.func_count += 1

            # Update progress report
            # print(self.problem.make_design_variables_report(x))
            if self.update_on == "function":
                self._print_convergence_progress(x_norm)

        return fitness

    def gradient(self, x_norm):
        """
        Evaluates the Jacobian matrix of the optimization problem at the given point x.

        This method utilizes the `gradient` method of the OptimizationProblem class if implemented.
        If the `gradient` method is not implemented, the Jacobian is approximated using forward finite differences.

        To prevent redundant calculations, cached results are checked first. If a matching
        cached result is found, it is returned; otherwise, a fresh calculation is performed.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).

        Returns
        -------
        numpy.ndarray
            A 2D array representing the Jacobian matrix of the optimization problem at `x`.
            The Jacobian matrix includes:
            - Gradient of the objective function
            - Jacobian of equality constraints
            - Jacobian of inequality constraints
        """

        # If x hasn't changed, use cached values
        if np.array_equal(x_norm, self.cache["x_jac"]):
            return self.cache["gradient"]

        # Use problem gradient method if it exists
        if hasattr(self.problem, "gradient"):
            grad = self.problem.gradient_normalized_input(x_norm)
        else:
            # Fall back to finite differences
            fun = lambda x: self.fitness(x, called_from_grad=True)
            grad = numerical_differentiation.approx_gradient(
                fun,
                x_norm,
                f0=fun(x_norm),
                method=self.derivative_method,
                abs_step=self.derivative_abs_step,
            )

        # Reshape gradient for unconstrained problems
        grad = np.atleast_2d(grad)

        # Update cache
        self.cache.update(
            {
                "x_jac": x_norm.copy(),
                "f_jac": grad[0, :],
                "c_eq_jac": grad[1 : 1 + self.N_eq, :],
                "c_ineq_jac": grad[1 + self.N_eq :, :],
                "gradient": grad,
            }
        )

        # Update progress report
        self.grad_count += 1
        if self.update_on == "gradient":
            self._print_convergence_progress(x_norm)

        return grad

    def _handle_output(self, message):
        """
        Handles output by printing to the screen or logging it.

        Parameters
        ----------
        message : str
            The message to output.
        log_only : bool, optional
            If True, only logs the message (ignores `self.display`). Default is False.
        """
        if self.logger:
            for line in message.splitlines():
                self.logger.info(line)
                
        if self.display and not self.logger:
            print(message)

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
        self.header = f" {'Grad-eval':>13}{'Func-eval':>13}{'Func-value':>16}{'Infeasibility':>18}{'Norm of step':>18} "
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
            self._handle_output(line)

        # Store text in memory
        self.solution_report.extend(lines_to_output)

    def _print_convergence_progress(self, x):
        """
        Print the current optimization status and update convergence history.

        This method captures and prints the following metrics:
        - Number of gradient evaluations
        - Number of function evaluations
        - Objective function value
        - Maximum constraint violation
        - Norm of the update step

        The method also updates the stored convergence history for potential future analysis.

        Parameters
        ----------
        x : array-like
            The current solution (i.e., vector of independent variable values)

        Notes
        -----
        The norm of the update step is calculated as the two-norm of the difference
        between the current and the last independent variables. Constraints violation is
        computed as the infinity norm of the active constraints.
        """

        # Ensure fitness is computed at least once before printing
        if self.cache["fitness"] is None:
            self.fitness(x)

        # Compute the norm of the last step
        norm_step = np.linalg.norm(x - self.x_last_norm) if self.x_last_norm is not None else 0
        self.x_last_norm = x.copy()

        # Compute the maximun constraint violation
        c_eq = self.cache["c_eq"]
        c_ineq = self.cache["c_ineq"]
        violation_all = np.concatenate((c_eq, np.maximum(c_ineq, 0)))
        violation_max = np.max(np.abs(violation_all)) if len(violation_all) > 0 else 0.0

        # Store convergence status
        self.convergence_history["x"].append(self.x_last_norm)
        self.convergence_history["grad_count"].append(self.grad_count)
        self.convergence_history["func_count"].append(self.func_count)
        self.convergence_history["func_count_total"].append(self.func_count_tot)
        self.convergence_history["objective_value"].append(self.cache["f"])
        self.convergence_history["constraint_violation"].append(violation_max)
        self.convergence_history["norm_step"].append(norm_step)

        # Current convergence message
        status = f" {self.grad_count:13d}{self.func_count:13d}{self.cache['f']:+16.3e}{violation_max:+18.3e}{norm_step:+18.3e} "
        self._handle_output(status)

        # Store text in memory
        self.solution_report.append(status)

        # Refresh the plot with current values
        if self.plot:
            self._plot_convergence_callback([], [])

        # Evaluate callback functions
        if self.callback_functions:
            self.callback_function_call_count += 1
            for func in self.callback_functions:
                func(x, self.callback_function_call_count)

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
        exit_message = f"Exit message: {self.message}"
        success_status = f"Success: {self.success}"
        time_message = f"Solution time: {self.elapsed_time:.3f} seconds"
        solution_header = "Solution:"
        solution_objective = f"   f  = {self.f_final:+6e}"
        solution_vars = [f"   x{i} = {x:+6e}" for i, x in enumerate(self.x_final)]
        lines_to_output = [separator, success_status, exit_message, time_message]
        if self.include_solution_in_footer:
            lines_to_output += [solution_header]
            lines_to_output += solution_objective
            lines_to_output += solution_vars
        lines_to_output += [separator, ""]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line)

        # Store text in memory
        self.solution_report.extend(lines_to_output)

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
            self.ax_1.set_xlabel("Number of iterations")
            self.ax_1.set_ylabel("Objective function")
            self.ax_1.set_yscale(self.plot_scale_objective)
            self.ax_1.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # Integer ticks
            self.ax_1.grid(False)
            if self.N_eq > 0 or self.N_ineq > 0:
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
        iteration = (
            self.convergence_history["func_count"]
            if self.update_on == "function"
            else self.convergence_history["grad_count"]
        )
        objective_function = self.convergence_history["objective_value"]
        constraint_violation = self.convergence_history["constraint_violation"]

        # Iterate through the objective_function values to create the new series
        if self.plot_improvement_only:
            for i in range(1, len(objective_function)):
                if objective_function[i] > objective_function[i - 1]:
                    objective_function[i] = objective_function[i - 1]

        # Update graphic objects data
        self.obj_line_1.set_xdata(iteration)
        self.obj_line_1.set_ydata(objective_function)
        if self.N_eq > 0 or self.N_ineq > 0:
            self.obj_line_2.set_xdata(iteration)
            self.obj_line_2.set_ydata(constraint_violation)

        # Adjust the plot limits
        self.ax_1.relim()
        self.ax_1.autoscale_view()
        if self.N_eq > 0 or self.N_ineq > 0:
            self.ax_2.relim()
            self.ax_2.autoscale_view()

        # Redraw the plot
        if showfig:
            plt.draw()
            plt.pause(0.01)  # small pause to allow for update

    def plot_convergence_history(
        self, savefile=False, filename=None, output_dir="output", showfig=True,
    ):
        """
        Plot the convergence history of the problem.

        This method plots the optimization progress against the number of iterations:
            - Objective function value (left y-axis)
            - Maximum constraint violation (right y-axis)

        The constraint violation is only displayed if the problem has nonlinear constraints

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
        # TODO: streamline code, it is too long and not clear
        if self.x_final is not None:
            self._plot_convergence_callback([], [], initialize=True, showfig=showfig)
        else:
            raise ValueError(
                "This method can only be used after invoking the 'solve()' method."
            )

        if savefile:
            # Create output directory if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Give a name to the file if it is not specified
            if filename is None:
                filename = (
                    f"convergence_{type(self.problem).__name__}_{self.start_datetime}"
                )

            # Save plots
            fullfile = os.path.join(output_dir, filename)
            savefig_in_formats(self.fig, fullfile, formats=[".png", ".svg"])

        # Optionally close the figure so it is not shown
        if not showfig:
            plt.close(self.fig)

        return self.fig

    def print_convergence_history(
        self, savefile=False, filename=None, output_dir="output"
    ):
        """
        Print the convergence history of the problem.

        The convergence history includes:
            - Number of function evaluations
            - Number of gradient evaluations
            - Objective function value
            - Maximum constraint violation
            - Two-norm of the update step

        The method provides a detailed report on:
            - Exit message
            - Success status
            - Execution time

        This method should be called only after the optimization problem has been solved, as it relies on data generated by the solving process.

        Parameters
        ----------
        savefile : bool, optional
            If True, the convergence history will be saved to a file, otherwise printed to standard output. Default is False.
        filename : str, optional
            The name of the file to save the convergence history. If not specified, the filename is automatically generated
            using the problem name and the start datetime. The file extension is not required.
        output_dir : str, optional
            The directory where the plot file will be saved if savefile is True. Default is "output".

        Raises
        ------
        ValueError
            If this method is called before the problem has been solved.

        """
        # TODO: streamline code, it is too long and not clear
        if self.x_final is not None:
            if savefile:
                # Create output directory if it does not exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Give a name to the file if it is not specified
                if filename is None:
                    filename = f"convergence_{type(self.problem).__name__}_{self.start_datetime}.txt"

                # Write report to file
                fullfile = os.path.join(output_dir, filename)
                with open(fullfile, "w") as f:
                    f.write("\n".join(self.solution_report))

            else:
                for line in self.solution_report:
                    print(line)

        else:
            raise ValueError(
                "This method can only be used after invoking the 'solve()' method."
            )

    def print_optimization_report(self, savefile=False, filename=None, output_dir="output"):
        """
        Print or save a complete optimization report, including variables and constraints.

        This method generates a report with:
            - Design variable values (normalized and/or physical)
            - Constraint values and satisfaction status

        This method should be called only after the optimization problem has been solved.

        Parameters
        ----------
        savefile : bool, optional
            If True, the report is written to a file. Otherwise, it is printed. Default is False.
        filename : str, optional
            Output filename. If not specified, a default name is generated based on the problem type and datetime.
        output_dir : str, optional
            Directory where the report file will be saved if `savefile=True`. Default is "output".

        Raises
        ------
        ValueError
            If this method is called before the optimization problem has been solved.
        """
        # TODO: streamline code, it is too long and not clear
        if self.x_final is None:
            raise ValueError("This method can only be used after invoking the 'solve()' method.")
            
        full_report = self.problem.make_optimization_report(self.x_final, tol=2*self.options["tolerance"])

        if savefile:
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if filename is None:
                filename = f"optimization_report_{type(self.problem).__name__}_{self.start_datetime}.txt"

            fullfile = os.path.join(output_dir, filename)

            with open(fullfile, "w") as f:
                f.write(full_report)
        else:
            print(full_report)


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


    def clip_to_bounds(self, x_physical, verbose=False):
        """
        Clip normalized values to lie within specified normalized bounds.

        Parameters
        ----------
        x : array-like
            Input vector.
        bounds : tuple of lists
            Tuple (lb, ub) of lower and upper bounds.
        verbose : bool, optional
            If True, prints warnings for clipped values.

        Returns
        -------
        np.ndarray
            Clipped vector.
        """
        lb, ub = self.get_bounds()
        x_physical = np.asarray(x_physical)
        x_clipped = np.clip(x_physical, lb, ub)

        if verbose:
            for i, (orig, new, low, high) in enumerate(zip(x_physical, x_clipped, lb, ub)):
                if orig != new:
                    print(
                        f"\nWarning: optimization variable out of bounds\n"
                        f"  Name       : {self.variable_names[i]}\n"
                        f"  Original   : {orig:.3e}\n"
                        f"  Bounds     : [{low:.3e}, {high:.3e}]\n"
                        f"  Clipped to : {new:.3e}\n"
                    )

        return x_clipped


    def make_optimization_report(self, x_phys, tol=1e-6):
        """
        Print or save a complete optimization report, including variables and constraints.
        """
        report = []
        report.append(self.make_design_variables_report(x_phys, normalized=True))
        report.append(self.make_design_variables_report(x_phys, normalized=False))
        report.append(self.make_constraint_report(x_phys, tol=tol))
        return "\n".join(report)
    

    def make_design_variables_report(self, x_phys, normalized=True):
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
        x_phys = np.asarray(x_phys)
        x_norm = self.scale_physical_to_normalized(x_phys)

        if normalized:
            values = x_norm
            lb_raw, ub_raw = self.get_bounds_normalized()
        else:
            values = x_phys
            lb_raw, ub_raw = self.get_bounds()

        names = self.variable_names or [f"design_variable_{i}" for i in range(len(values))]
        scale_type = "normalized" if normalized else "physical"

        lines = []
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{'Optimization variables report (' + scale_type + ' values)':<80}")
        lines.append("-" * 80)
        lines.append(f"{'Variable name':<35}{'Lower':>15}{'Value':>15}{'Upper':>15}")
        lines.append("-" * 80)

        for key, lb, val, ub in zip(names, lb_raw, values, ub_raw):
            if normalized:
                lines.append(f"{key:<35}{lb:>15.4f}{val:>15.4f}{ub:>15.4f}")
            else:
                lines.append(f"{key:<35}{lb:>15.3e}{val:>15.3e}{ub:>15.3e}")

        lines.append("-" * 80)
        return "\n".join(lines)


    def make_constraint_report(self, x_phys, tol=1e-6):
        """
        Generate a constraint report as a string by evaluating constraints at the given point.

        This method uses `self.constraint_data` if available, otherwise it evaluates the fitness function to get the constraints information.

        By default:
        - Equality constraints are named "equality_constraint_{i}" with target 0 and type "=".
        - Inequality constraints are named "inequality_constraint_{i}" with target 0 and type "<".
        - Tolerance `tol` is used to determine satisfaction.

        Parameters
        ----------
        x_phys : array-like
            Physical design variable vector.
        tol : float, optional
            Tolerance used to determine whether constraints are satisfied (default is 1e-6).

        Returns
        -------
        str
            A formatted string report of constraints and satisfaction status.

        Raises
        ------
        TypeError
            If `self.constraint_data` exists but is not a valid list of dictionaries.
        """
        max_name_width = 50

        # Use cached constraint data if available and valid
        if hasattr(self, "constraint_data"):
            if not isinstance(self.constraint_data, list) or not all(isinstance(d, dict) for d in self.constraint_data):
                raise TypeError("Expected self.constraint_data to be a list of dictionaries.")

            required_keys = {"name", "type", "target", "value", "satisfied"}
            allowed_keys = required_keys | {"normalized_mismatch", "mismatch", "normalize"}
            for i, entry in enumerate(self.constraint_data):
                validate_keys(entry, required_keys, allowed_keys)

            constraint_data = self.constraint_data

        else:
            # Evaluate constraints from scratch
            x_phys = np.asarray(x_phys)
            result = self.fitness(x_phys)
            n_eq = self.get_nec()
            n_ineq = self.get_nic()
            c_eq = result[1:1 + n_eq]
            c_ineq = result[1 + n_eq:]

            # Build default report structure
            constraint_data = []

            for i, val in enumerate(c_eq):
                print(val, tol)
                constraint_data.append({
                    "name": f"equality_constraint_{i}",
                    "type": "=",
                    "target": 0.0,
                    "value": val,
                    "satisfied": np.abs(val) < tol,
                })

            for i, val in enumerate(c_ineq):
                print(val, tol)
                constraint_data.append({
                    "name": f"inequality_constraint_{i}",
                    "type": "<",
                    "target": 0.0,
                    "value": val,
                    "satisfied": val <= tol,
                })

        # Format report lines
        lines = []
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{'Optimization constraints report':<80}")
        lines.append("-" * 80)
        lines.append(f"{'Constraint name':<52}{'Value':>10}{'Target':>12}{'Ok?':>6}")
        lines.append("-" * 80)

        for entry in constraint_data:
            name = entry.get("name", "")
            ctype = entry.get("type", "")
            target = entry.get("target", 0.0)
            value = entry.get("value", 0.0)
            satisfied = "yes" if entry.get("satisfied", False) else "no"

            if len(name) > max_name_width:
                name = "..." + name[-(max_name_width - 3):]

            symbol_target = f"{ctype} {target:.3f}"
            lines.append(f"{name:<52}{value:>10.3f}{symbol_target:>12}{satisfied:>6}")

        lines.append("-" * 80)
        return "\n".join(lines)


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
