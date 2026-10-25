from ..reporting.text import emit
from ..results import root_result
from ..reporting import plots, text
from ..history import append_observation
import os
import time
import copy
import logging
import numpy as np

from datetime import datetime

from .evaluation import RootEvaluation
from .backends.custom import resolve_root_options
from .backends.registry import solve_backend
from .methods import normalize_method, SOLVER_OPTIONS as SOLVER_OPTIONS

from .problems.base import NonlinearSystemProblem as NonlinearSystemProblem


class NonlinearSystemSolver:
    r"""Solve nonlinear equations with native SciPy or controlled custom methods.

    Parameters
    ----------
    problem : NonlinearSystemProblem
        Model with ``residual(x)`` and optional analytic ``gradient(x)``.
    method : str, optional
        Default ``hybr``. SciPy supports ``lm``/``hybr`` and their full aliases
        ``levenberg_marquardt``/``powell_hybrid``. The custom backend additionally
        supports ``newton_raphson``/``newton`` and ``gauss_newton``/``gn``.
    tolerance : float, optional
        Default 1e-6. SciPy's existing residual acceptance is 10*tolerance.
        Custom options use this as residual_tol unless explicitly overridden.
    max_iterations : int, optional
        Default 100. Sets native maxfev (hybr) or maxiter (lm). For custom
        solvers, fills the trial-attempt budget unless explicitly overridden;
        accepted iterations, trials, and residual evaluations are distinct.
    options : dict or root-options dataclass, optional
        Native SciPy options, or StepControlOptions/NewtonOptions for custom
        solvers. Explicit custom dictionary fields override constructor defaults;
        a matching dataclass supplies every option. See result.effective_options.
    derivative_method : str, optional
        ``2-point``, ``3-point``, or ``cs`` for a missing analytic Jacobian.
        The selected backend owns differentiation, avoiding nested probes.
    derivative_abs_step : float or array, optional
        Physical probe distance, default 1e-6. Custom options may override this
        or select scaled probes using finite_diff_step instead.
    print_convergence, plot_convergence : bool, optional
        Enable console and live figure output; defaults True and False.
    plot_scale : str, optional
        Residual axis scale; default ``log``.
    logger : logging.Logger, optional
        Destination for convergence messages.
    update_on : str, optional
        ``function`` (default), ``gradient``, or custom-only ``iteration``.
        Function observations can include rejected trials; iteration observations
        contain the initial state and accepted moves only. Probes never add rows.
    callback_func : callable, optional
        Receives detached ``(x, callback_count)``. Native SciPy calls it for
        selected observations. Custom solvers call it at the initial state and
        accepted moves, independently of update_on; False or StopIteration stops.
    library : str, keyword-only
        ``scipy`` (default) or ``pysolver``. Custom roots are independent Python
        algorithms using SciPy linear algebra, not native MINPACK wrappers.

    Attributes
    ----------
    backend_result, result : scipy.optimize.OptimizeResult or None
        Detached backend result and enriched public result. Final coordinates
        come from the backend. Custom success requires the residual target;
        result.fun is None when the initial residual is undefined.
    convergence_history : dict
        Selected observations, with scalar values even in summary history mode.
        A displacement between function observations is not an accepted step.
    iteration_history : list
        Custom initial/accepted records. Native LM/hybrid expose no iterations,
        so their list is empty and iteration_count is None.
    iteration_count : int or None
        Accepted custom moves, excluding the initial record and derivative probes.

    Notes
    -----
    solve(x0) continues to return a solution vector. Custom history_level selects
    full coordinate storage, scalar summary records, or no stored histories.
    Model-domain recovery uses pysolver_view.DomainError; unrelated programming
    exceptions propagate. A new solve resets all per-run counters and histories.
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
        *,
        library="scipy",
    ):
        # Initialize class variables
        self.problem = problem
        self.display = print_convergence
        self.plot = plot_convergence
        self.plot_scale = plot_scale
        self.logger = logger
        if library not in ("scipy", "pysolver"):
            raise ValueError("library must be 'scipy' or 'pysolver'.")
        self.library = library
        self.requested_method = method
        canonical = normalize_method(method)
        if library == "scipy":
            native_methods = {"levenberg_marquardt": "lm", "powell_hybrid": "hybr"}
            if canonical not in native_methods:
                raise ValueError(
                    "SciPy root supports only lm/hybr and their author-name aliases here."
                )
            self.method = native_methods[canonical]
        else:
            self.method = canonical
        self.tol = tolerance
        self.maxiter = max_iterations
        self.options = (
            resolve_root_options(
                canonical,
                options,
                tolerance,
                max_iterations,
                derivative_method,
                derivative_abs_step,
            )
            if library == "pysolver"
            else copy.deepcopy(options) if options else {}
        )
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
        self.callback_func = callback_func
        self.callback_func_call_count = 0

        # Preserve the native constructor's evaluation-budget mapping.
        if library == "scipy":
            self.options["maxfev" if self.method == "hybr" else "maxiter"] = (
                self.maxiter
            )

        # Check for logger validity
        if self.logger is not None:
            if not isinstance(self.logger, logging.Logger):
                raise ValueError(
                    "The provided logger is not a valid logging.Logger instance."
                )

        # Check for valid display_on value
        self.update_on = update_on
        allowed_updates = (
            ("function", "gradient", "iteration")
            if library == "pysolver"
            else ("function", "gradient")
        )
        if update_on not in allowed_updates:
            raise ValueError(
                f"Invalid update_on for {library}; choose from {allowed_updates}."
            )

        self.include_solution_in_footer = False
        self._reset_run_state()
        if self.plot:
            self._plot_callback(initialize=True)

    def _reset_run_state(self):
        """Start a run with empty results, counters, snapshots, and observations."""
        self._evaluation = RootEvaluation(
            self.problem,
            derivative_method=self.derivative_method,
            derivative_abs_step=self.derivative_abs_step,
            update_on=self.update_on,
            observe=self._print_convergence_progress,
        )
        self.f_final = None
        self.x_final = None
        self.x_last = None
        self.callback_func_call_count = 0
        self.iteration_count = 0 if self.library == "pysolver" else None
        self.iteration_history = []
        self.success = None
        self.message = None
        self.backend_result = None
        self.result = None
        self.solution_report = []
        self.elapsed_time = None
        self._last_record_x = self._last_record_residual = None
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
        """Return the selected backend's final point and retain detailed results.

        ``backend_result`` is a detached backend result. ``result`` adds
        package-level evaluation accounting and residual-based acceptance.
        Native LM/hybrid iteration counts are unavailable; evaluations and
        Jacobian requests are never relabeled as iterations. Reusing this
        solver starts a fresh history and resets its per-run counters.
        """
        self._reset_run_state()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time = time.perf_counter()
        self._write_header()
        solution = solve_backend(self, np.array(x0, dtype=float, copy=True))

        # The backend's returned point and residual are authoritative, even
        # when its most recent evaluation was an unsuccessful trial.
        self.backend_result = copy.deepcopy(solution)
        self.x_final = np.array(solution.x, copy=True)
        self.f_final = (
            None
            if solution.fun is None
            else np.atleast_1d(np.array(solution.fun, copy=True))
        )
        self.success = bool(solution.success)
        self.message = str(solution.message)
        residual_norm = (
            float(np.linalg.norm(self.f_final))
            if self.f_final is not None
            else float("inf")
        )
        residual_tolerance = (
            self.options.residual_tol if self.library == "pysolver" else 10 * self.tol
        )
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

        self.residuals_last = None if self.f_final is None else self.f_final.copy()
        self._residual_x = None if self.f_final is None else self.x_final.copy()
        self._residual_value = None if self.f_final is None else self.f_final.copy()
        if (
            self.f_final is not None
            and not (
                np.array_equal(self._last_record_x, self.x_final)
                and np.array_equal(self._last_record_residual, self.f_final)
            )
            and not (
                self.library == "pysolver" and self.options.history_level == "none"
            )
        ):
            self._print_convergence_progress(
                self.x_final,
                self.f_final,
                event_kind="final",
                notify=False,
                iteration=self.iteration_count,
            )
        self.elapsed_time = time.perf_counter() - start_time
        self.result = root_result(self, solution, residual_norm, residual_tolerance)
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
        return self._evaluation.residual(x, called_from_jac=called_from_jac)

    def gradient(self, x):
        """Evaluate one Jacobian request at a point-consistent residual base.

        Numerical differences use residual values at the requested point,
        never the most recent probe or an unrelated observation. A request
        increments ``grad_count`` once, regardless of how many probes it uses;
        failed requests also increment ``failed_grad_count``.
        """
        return self._evaluation.gradient(x)

    def _handle_output(self, message):
        """Emit prepared text using the existing display/logging policy."""
        return emit(
            message, logger=self.logger, to_console=self.display and not self.logger
        )

    def _write_header(self):
        """Render and store the prepared header."""
        self.header, lines = text.root_header(
            type(self.problem).__name__, self.method, self.library
        )
        for line in lines:
            self._handle_output(line)
        self.solution_report.extend(lines)

    def _print_convergence_progress(
        self, x, residual=None, *, event_kind=None, notify=True, iteration=None
    ):
        """Record and render a supplied point/residual snapshot without evaluation.

        This history contains the selected observation stream. Custom iteration
        rows represent accepted moves; native root rows remain observations.
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
            "iteration": iteration,
            "grad_count": self.grad_count,
            "func_count": self.func_count,
            "func_count_total": self.func_count_tot,
            "derivative_func_count": self.derivative_func_count,
            "norm_residual": float(np.linalg.norm(value)),
            "norm_step": norm_step,
        }
        self.x_last = point.copy()
        self._last_record_x = point.copy()
        self._last_record_residual = value.copy()
        level = self.options.history_level if self.library == "pysolver" else "full"
        append_observation(
            self.convergence_history, record, level=level, coordinates=("x", "residual")
        )
        self._render_progress(record)
        if self.plot:
            self._plot_callback()
        if notify and self.callback_func is not None:
            self.callback_func_call_count += 1
            self.callback_func(point.copy(), self.callback_func_call_count)

    def _render_progress(self, record):
        """Format an existing observation without changing model or solver state."""
        status = text.root_progress(record)
        self._handle_output(status)
        self.solution_report.append(status)

    def _write_footer(self):
        """Render and store the prepared footer."""
        lines = text.root_footer(
            self.header,
            self.message,
            self.success,
            self.elapsed_time,
            self.x_final,
            self.include_solution_in_footer,
        )
        for line in lines:
            self._handle_output(line)
        self.solution_report.extend(lines)

    def _plot_callback(self, initialize=False, showfig=True):
        plots.update_root_plot(self._plot_view(), initialize, showfig)
        self._sync_plot_handles()

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
        return self._plot_history(savefile, filename, output_dir, showfig)

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
    def failed_grad_count(self):
        return self._evaluation.failed_grad_count

    @failed_grad_count.setter
    def failed_grad_count(self, value):
        self._evaluation.failed_grad_count = value

    @property
    def residuals_last(self):
        return self._evaluation.residuals_last

    @residuals_last.setter
    def residuals_last(self, value):
        self._evaluation.residuals_last = value

    @property
    def _residual_x(self):
        return self._evaluation._residual_x

    @_residual_x.setter
    def _residual_x(self, value):
        self._evaluation._residual_x = value

    @property
    def _residual_value(self):
        return self._evaluation._residual_value

    @_residual_value.setter
    def _residual_value(self, value):
        self._evaluation._residual_value = value

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
        view.plot_scale = self.plot_scale
        return view

    def _sync_plot_handles(self):
        """Retain the existing public Matplotlib handle attributes."""
        for name in ["fig", "ax", "obj_line"]:
            if hasattr(self._plot_state, name):
                setattr(self, name, getattr(self._plot_state, name))

    def _plot_history(self, savefile, filename, output_dir, showfig):
        fig = plots.plot_root_history(
            self._plot_view(),
            savefile=savefile,
            filename=filename,
            output_dir=output_dir,
            showfig=showfig,
        )
        self._sync_plot_handles()
        return fig
