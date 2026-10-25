import os
import time
import copy
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from abc import ABC, abstractmethod
from matplotlib.ticker import MultipleLocator
from scipy.optimize import root

from . import numerical_differentiation
from .pysolver_utilities import savefig_in_formats

SOLVER_OPTIONS = ["hybr", "lm"]


class NonlinearSystemSolver:
    r"""
    Solver class for nonlinear systems of equations.

    The solver is designed to handle system of nonlinear equations of the form:

    .. math::

        F(x) = 0

    where :math:`F: \mathbb{R}^n \rightarrow \mathbb{R}^n` is a vector-valued
    function of the vector :math:`x`.

    The class interfaces with the `root` method from `scipy.optimize` to solve the equations and provides a structured framework for initialization, solution monitoring, and post-processing.

    Parameters
    ----------
    problem : NonlinearSystemProblem
        An instance of a problem defining the system of equations to be solved.
    method : str, optional
        Method to be used by scipy's root for solving the nonlinear system. Available solvers are:

        - :code:`hybr`:  Uses MINPACK's 'hybrd' method, which is is a modification of Powell's hybrid method.
        - :code:`lm`: The Levenberg-Marquardt method, which blends the steepest descent and the Gauss-Newton methods.

        The choice between :code:`hybr` and :code:`lm` largely depends on the specifics of the problem at hand.
        The :code:`hybr` usually requires less gradient evaluations and it is often faster when analytic gradients are not available.
        It is advisable to experiment with both methods to determine the most appropriate choice for a given problem.

        Defaults to 'hybr'.
    tol : float, optional
        Tolerance for the solver termination. Defaults to 1e-9.
    max_iter : integer, optional
        Maximum number of function evaluations for the solver termination. Defaults to 100.
    options : dict, optional
        Additional options to be passed to scipy's root.
    derivative_method : str, optional
        Finite difference method to be used when the problem Jacobian is not provided. Defaults to '2-point'
    derivative_abs_step : float, optional
        Finite difference absolute step size to be used when the problem Jacobian is not provided. Defaults to 1e-6
    print_convergence : bool, optional
        If True, displays the convergence progress. Defaults to True.
    plot_convergence : bool, optional
        If True, plots the convergence progress. Defaults to False.
    logger : logging.Logger, optional
        Logger object to which logging messages will be directed. Defaults to None.
    update_on : str, optional
        Specifies if the convergence report should be updated on a new function evaluations ("function") or on gradient evaluations ("gradient"). Defaults to "function".
    callback_func : callable, optional
        Observation callback receiving a detached ``(x, observation_count)``.
        The count describes recorded function/Jacobian observations, not native
        solver iterations. Numerical derivative probes do not invoke it.

    Attributes
    ----------
    backend_result : scipy.optimize.OptimizeResult or None
        Detached native result, including SciPy's original counters and status.
    result : scipy.optimize.OptimizeResult or None
        Final point and residual, package-level call counts, native success,
        and the residual-based acceptance decision. The existing additional
        residual check uses ``10 * tolerance``, reported as ``residual_tolerance``.
    convergence_history : dict
        Point-consistent function/Jacobian observations and optional final
        result record. ``norm_step`` is displacement between observed points,
        not necessarily an accepted solver step.
    iteration_count : None
        Native LM/hybrid do not expose accepted-iteration counts. Their
        ``iteration_history`` is therefore empty.


    Methods
    -------
    solve(x0)
        Solve the system of nonlinear equations using the specified initial guess `x0`.
    residual(x)
        Evaluate the vector of residuals of the at a given point `x`.
    gradient(x)
        Evaluate the Jacobian of the system at a given point `x`.
    print_convergence_history()
        Print the convergence history of the problem.
    plot_convergence_history()
        Plot the convergence history.

    """

    def __init__(
        self,
        problem,
        method="hybr",
        tolerance=1e-6,
        max_iterations=100,
        options={},
        derivative_method="2-point",
        derivative_abs_step=1e-6,
        print_convergence=True,
        plot_convergence=False,
        plot_scale="log",
        logger=None,
        update_on="function",
        callback_func=None,
    ):
        # Initialize class variables
        self.problem = problem
        self.display = print_convergence
        self.plot = plot_convergence
        self.plot_scale = plot_scale
        self.logger = logger
        self.method = method
        self.tol = tolerance
        self.maxiter = max_iterations
        self.options = copy.deepcopy(options) if options else {}
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
        self.callback_func = callback_func
        self.callback_func_call_count = 0

        # Define the maximun number of iterations
        if method == "hybr":
            self.options["maxfev"] = self.maxiter
        elif method == "lm":
            self.options["maxiter"] = self.maxiter
        else:
            raise ValueError(
                f"Invalid solver. Available options: {', '.join(SOLVER_OPTIONS)}"
            )

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

        self.include_solution_in_footer = False
        self._reset_run_state()
        if self.plot:
            self._plot_callback(initialize=True)

    def _reset_run_state(self):
        """Start a run with empty results, counters, snapshots, and observations."""
        self.f_final = None
        self.x_final = None
        self.x_last = None
        self.residuals_last = None
        self._residual_x = None
        self._residual_value = None
        self.grad_count = 0
        self.failed_grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.derivative_func_count = 0
        self.callback_func_call_count = 0
        self.iteration_count = None
        self.iteration_history = []
        self.success = None
        self.message = None
        self.backend_result = None
        self.result = None
        self.solution_report = []
        self.elapsed_time = None
        self.convergence_history = {
            "x": [],
            "residual": [],
            "event_kind": [],
            "iteration": [],
            "grad_count": [],
            "func_count": [],
            "func_count_total": [],
            "derivative_func_count": [],
            "norm_residual": [],
            "norm_step": [],
        }

    def solve(self, x0):
        """Return the native solver's final point and retain detailed results.

        ``backend_result`` is a detached native SciPy result. ``result`` adds
        package-level evaluation accounting and residual-based acceptance.
        Native LM/hybrid iteration counts are unavailable; evaluations and
        Jacobian requests are never relabeled as iterations. Reusing this
        solver starts a fresh history and resets its per-run counters.
        """
        self._reset_run_state()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time = time.perf_counter()
        self._write_header()
        solution = root(
            self.residual,
            np.array(x0, dtype=float, copy=True),
            jac=self.gradient,
            method=self.method,
            tol=self.tol,
            options=self.options.copy(),
        )

        # The backend's returned point and residual are authoritative, even
        # when its most recent evaluation was an unsuccessful trial.
        self.backend_result = copy.deepcopy(solution)
        self.x_final = np.array(solution.x, copy=True)
        self.f_final = np.atleast_1d(np.array(solution.fun, copy=True))
        self.success = bool(solution.success)
        self.message = str(solution.message)
        residual_norm = float(np.linalg.norm(self.f_final))
        residual_tolerance = 10 * self.tol
        if self.success and (
            not np.isfinite(residual_norm) or residual_norm > residual_tolerance
        ):
            self.success = False
            self.message = (
                "The equation solver returned a success exit flag.\n"
                "However, the two-norm of the final residual does not satisfy "
                f"the acceptance tolerance ({residual_norm:0.2e}; "
                f"target {residual_tolerance:0.2e})."
            )

        self.residuals_last = self.f_final.copy()
        self._residual_x = self.x_final.copy()
        self._residual_value = self.f_final.copy()
        history = self.convergence_history
        if not history["x"] or not (
            np.array_equal(history["x"][-1], self.x_final)
            and np.array_equal(history["residual"][-1], self.f_final)
        ):
            self._print_convergence_progress(
                self.x_final, self.f_final, event_kind="final", notify=False
            )
        self.elapsed_time = time.perf_counter() - start_time
        self.result = copy.deepcopy(solution)
        self.result.update(
            x=self.x_final.copy(),
            fun=self.f_final.copy(),
            success=self.success,
            solver_success=bool(solution.success),
            message=self.message,
            residual_norm=residual_norm,
            residual_tolerance=residual_tolerance,
            nfev=self.func_count_tot,
            njev=self.grad_count,
            nit=None,
            iteration_count=None,
            func_count=self.func_count,
            func_count_total=self.func_count_tot,
            derivative_func_count=self.derivative_func_count,
            failed_grad_count=self.failed_grad_count,
            elapsed_time=self.elapsed_time,
        )
        self._write_footer()
        return self.x_final.copy()

    def residual(self, x, called_from_jac=False):
        """Evaluate residuals, counting derivative probes without recording them.

        Every actual model evaluation increments ``func_count_tot``. Calls
        made for numerical derivatives also increment ``derivative_func_count``
        but never change ordinary observation state or invoke user callbacks.
        Inputs, returned values, and stored snapshots do not share writable
        arrays with a model or callback.
        """
        point = np.array(x, copy=True)
        self.func_count_tot += 1
        if called_from_jac:
            self.derivative_func_count += 1
        else:
            self.func_count += 1
        value = np.atleast_1d(np.array(self.problem.residual(point.copy()), copy=True))
        if not called_from_jac:
            self.residuals_last = value.copy()
            self._residual_x = point.copy()
            self._residual_value = value.copy()
            if self.update_on == "function":
                self._print_convergence_progress(point, value, event_kind="function")
        return value

    def gradient(self, x):
        """Evaluate one Jacobian request at a point-consistent residual base.

        Numerical differences use residual values at the requested point,
        never the most recent probe or an unrelated observation. A request
        increments ``grad_count`` once, regardless of how many probes it uses;
        failed requests also increment ``failed_grad_count``.
        """
        point = np.array(x, dtype=float, copy=True)
        self.grad_count += 1
        try:
            if self._residual_x is not None and np.array_equal(point, self._residual_x):
                base = self._residual_value.copy()
            else:
                base = self.residual(point, called_from_jac=True)
                self._residual_x = point.copy()
                self._residual_value = base.copy()

            if hasattr(self.problem, "gradient"):
                jacobian = self.problem.gradient(point.copy())
            else:

                def evaluate_probe(probe):
                    """Reuse the known base, evaluating only actual new probes."""
                    if np.array_equal(probe, point):
                        return base.copy()
                    return self.residual(probe, called_from_jac=True)

                jacobian = numerical_differentiation.approx_gradient(
                    evaluate_probe,
                    point,
                    f0=base,
                    method=self.derivative_method,
                    abs_step=self.derivative_abs_step,
                )
            jacobian = np.asarray(jacobian).reshape(base.size, point.size).copy()
        except Exception:
            self.failed_grad_count += 1
            raise

        if self.update_on == "gradient":
            self._print_convergence_progress(point, base, event_kind="gradient")
        return jacobian

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
        Print a formatted header for the root finding report.

        This internal method is used to display a consistent header format at the
        beginning of the root finding process. The header includes columns for residual evaluations,
        gradient evaluations, norm of the residuals, and norm of the steps.
        """
        # Define header text
        initial_message = (
            f" Solve system of equations for {type(self.problem).__name__}\n"
            f" Root-finding algorithm employed: {self.method}"
        )
        self.header = f" {'Func-eval':>15}{'Grad-eval':>15}{'Norm of residual':>24}{'Point displacement':>24} "
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

    def _print_convergence_progress(
        self, x, residual=None, *, event_kind=None, notify=True
    ):
        """Record and render a supplied point/residual snapshot without evaluation.

        This history contains observations, not native accepted iterations.
        ``norm_step`` measures displacement between recorded points. The legacy
        callback receives ``(x, observation_count)`` with a detached point.
        Final-result records do not generate an extra callback.
        """
        point = np.array(x, copy=True)
        if residual is None:
            if self._residual_x is None or not np.array_equal(point, self._residual_x):
                raise ValueError(
                    "A progress record requires residuals at its exact point."
                )
            residual = self._residual_value
        value = np.array(residual, copy=True)
        norm_step = (
            float(np.linalg.norm(point - self.x_last))
            if self.x_last is not None
            else 0.0
        )
        record = {
            "x": point.copy(),
            "residual": value.copy(),
            "event_kind": self.update_on if event_kind is None else event_kind,
            "iteration": None,
            "grad_count": self.grad_count,
            "func_count": self.func_count,
            "func_count_total": self.func_count_tot,
            "derivative_func_count": self.derivative_func_count,
            "norm_residual": float(np.linalg.norm(value)),
            "norm_step": norm_step,
        }
        self.x_last = point.copy()
        for key, entry in record.items():
            self.convergence_history[key].append(entry)
        self._render_progress(record)
        if self.plot:
            self._plot_callback()
        if notify and self.callback_func is not None:
            self.callback_func_call_count += 1
            self.callback_func(point.copy(), self.callback_func_call_count)

    def _render_progress(self, record):
        """Format an existing observation without changing model or solver state."""
        status = (
            f" {record['func_count']:15d}{record['grad_count']:15d}"
            f"{record['norm_residual']:24.6e}{record['norm_step']:24.6e} "
        )
        self._handle_output(status)
        self.solution_report.append(status)

    def _write_footer(self):
        """
        Print a formatted footer for the root finding report.

        This method displays the final root finding result, including the
        exit message, success status, and decision variables.

        Notes
        -----
        The footer's structure is intended to match the header's style,
        providing a consistent look to the root finding report.
        """
        # Define footer text
        separator = "-" * len(self.header)
        exit_message = f"Exit message: {self.message}"
        success_status = f"Success: {self.success}"
        time_message = f"Solution time: {self.elapsed_time:.3f} seconds"
        solution_header = "Solution:"
        solution_vars = [f"   x{i} = {x:+6e}" for i, x in enumerate(self.x_final)]
        lines_to_output = [separator, success_status, exit_message, time_message]
        if self.include_solution_in_footer:
            lines_to_output += [solution_header]
            lines_to_output += solution_vars
        lines_to_output += [separator, ""]

        # Print or log content
        for line in lines_to_output:
            self._handle_output(line)

        # Store text in memory
        self.solution_report.extend(lines_to_output)

    def _plot_callback(self, initialize=False, showfig=True):
        """
        Callback function to dynamically update the convergence progress plot.

        Initialize or update a plot of residual observations against their
        function/Jacobian evaluation counts. Rendering never evaluates the
        model or modifies the recorded data.

        Note:
            This is an internal method, meant to be called within the optimization process.
        """

        # Initialize figure before first iteration
        if initialize:
            self.fig, self.ax = plt.subplots()
            (self.obj_line,) = self.ax.plot([], [], color="#D95319", marker="o")
            self.ax.set_xlabel(
                "Function evaluations (excluding derivative probes)"
                if self.update_on == "function"
                else "Jacobian evaluations"
            )
            self.ax.set_ylabel("Two-norm of residual vector")
            self.ax.set_yscale(self.plot_scale)
            # self.ax.xaxis.set_major_locator(MultipleLocator(1))
            # self.ax.legend(loc="upper right")
            self.fig.tight_layout(pad=1)

        # Update plot data with current values
        iteration = (
            self.convergence_history["func_count"]
            if self.update_on == "function"
            else self.convergence_history["grad_count"]
        )
        residual = self.convergence_history["norm_residual"]
        self.obj_line.set_xdata(iteration)
        self.obj_line.set_ydata(residual)

        # Adjust the plot limits
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the plot
        if showfig:
            plt.draw()
            plt.pause(0.01)  # allow interactive figure updates

    def print_convergence_history(
        self, savefile=False, filename=None, output_dir="output"
    ):
        """
        Print the convergence history of the problem.

        The convergence history includes:
            - Number of function evaluations
            - Number of gradient evaluations
            - Two-norm of the residual vector
            - Two-norm of the solution step

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

    def plot_convergence_history(
        self, savefile=False, filename=None, output_dir="output", *, showfig=True
    ):
        """
        Plot the residual two-norm against function or Jacobian evaluations.

        Numerical derivative probes are excluded from the function-observation
        axis and remain included in total model-call counters. Native accepted
        iteration counts are not available from SciPy's LM/hybrid interface.

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
        if self.x_final is not None:
            self._plot_callback(initialize=True, showfig=showfig)
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

        if not showfig:
            plt.close(self.fig)
        return self.fig


class NonlinearSystemProblem(ABC):
    """
    Abstract base class for root-finding problems.

    Derived root-finding problem objects must implement the following method:

    - `residual`: Evaluate the system of equations for a given set of decision variables.

    Additionally, specific problem classes can define the `gradient` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    residual(x)
        Evaluate the system of equations for a given set of decision variables.

    Examples
    --------
    Here's an example of how to derive from `RootFindingProblem`::

        class MyRootFindingProblem(RootFindingProblem):
            def residual(self, x):
                # Implement evaluation logic here
                pass
    """

    @abstractmethod
    def residual(self, x):
        """
        Evaluate the system of equations for given decision variables.

        Parameters
        ----------
        x : array-like
            Vector of decision variables.

        Returns
        -------
        array_like
            Vector containing the values of the system of equations for the given decision variables.
        """
        pass
