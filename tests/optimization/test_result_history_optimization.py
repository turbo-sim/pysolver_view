"""Behavioral regressions for optimizer results, observations, and probe accounting."""

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pysolver_view.optimization.backends import scipy as optimization_wrappers
from pysolver_view.optimization import OptimizationProblem, OptimizationSolver


class QuadraticProblem(OptimizationProblem):
    """A two-variable quadratic with observable model calls and a known minimum."""

    def __init__(self):
        """Record every physical model evaluation for independent call accounting."""
        self.calls = []

    def fitness(self, x):
        """Evaluate a polynomial that also supports complex-step differentiation."""
        self.calls.append(np.array(x, copy=True))
        return np.array([(x[0] - 1) ** 2 + 2 * (x[1] + 2) ** 2])

    def get_bounds(self):
        """Return a wide finite box containing all test points."""
        return [-10.0, -10.0], [10.0, 10.0]

    def get_nec(self):
        """Declare no equality constraints."""
        return 0

    def get_nic(self):
        """Declare no inequality constraints."""
        return 0


class AnalyticQuadratic(QuadraticProblem):
    """The same problem with an analytic objective gradient."""

    def gradient(self, x):
        """Return the exact physical objective Jacobian."""
        return np.array([[2 * (x[0] - 1), 4 * (x[1] + 2)]])


class ConstrainedQuadratic(AnalyticQuadratic):
    """A quadratic subject to x[0] <= 0, for snapshot consistency checks."""

    def fitness(self, x):
        """Return the objective and inequality from one physical evaluation."""
        self.calls.append(np.array(x, copy=True))
        return np.array([x[0] ** 2 + x[1] ** 2, x[0]])

    def gradient(self, x):
        """Return objective and constraint Jacobian rows at the supplied point."""
        return np.array([[2 * x[0], 2 * x[1]], [1.0, 0.0]])

    def get_nic(self):
        """Declare the one nonpositive inequality."""
        return 1


def make_solver(problem=None, **options):
    """Build a quiet SciPy SLSQP facade with no scaling for readable assertions."""
    configured = dict(
        library="scipy",
        method="slsqp",
        print_convergence=False,
        plot_convergence=False,
        tolerance=1e-9,
        max_iterations=50,
    )
    configured.update(options)
    return OptimizationSolver(problem or AnalyticQuadratic(), **configured)


def test_trust_constr_function_history_includes_gradient_first_base_points():
    """Gradient-first model requests must not hide the function trajectory."""
    solver = make_solver(method="trust-constr", update_on="function")
    solver.solve(np.array([3.0, 4.0]))
    assert solver.success
    assert solver.func_count > 2
    assert solver.convergence_history["event_kind"].count("function") > 2
    assert solver.derivative_func_count == 0
    for x, value in zip(
        solver.convergence_history["x"], solver.convergence_history["objective_value"]
    ):
        assert value == pytest.approx((x[0] - 1) ** 2 + 2 * (x[1] + 2) ** 2)


def test_nelder_mead_respects_iteration_budget():
    """The common iteration budget must reach the native simplex solver."""
    solver = make_solver(method="nelder-mead", max_iterations=2, update_on="function")
    solver.solve(np.array([3.0, 4.0]))
    assert not solver.success
    assert solver.backend_result.nit == 2
    assert solver.grad_count == 0


@pytest.fixture(autouse=True)
def close_test_figures():
    """Release Matplotlib figures after each headless test."""
    yield
    plt.close("all")


def test_native_result_is_authoritative_after_a_later_trial(monkeypatch):
    """A final evaluated trial must not replace the native returned solution."""
    target = np.array([1.0, -2.0])

    def return_solution_after_trial(fun, x0, **kwargs):
        """Model a backend that evaluates a trial after finding its returned point."""
        fun(target)
        fun(np.array([7.0, 8.0]))
        return OptimizeResult(
            x=target.copy(),
            fun=0.0,
            success=True,
            status=0,
            message="Known minimum",
            nit=1,
            nfev=2,
            njev=0,
        )

    monkeypatch.setattr(optimization_wrappers, "minimize", return_solution_after_trial)
    solver = make_solver(update_on="function")
    returned = solver.solve(np.array([0.0, 0.0]))

    np.testing.assert_array_equal(returned, target)
    np.testing.assert_array_equal(solver.result.x, target)
    assert solver.f_final == pytest.approx(0.0)
    assert solver.result.status == 0
    assert solver.convergence_history["event_kind"][-1] == "final"
    np.testing.assert_array_equal(solver.convergence_history["x"][-1], target)
    assert any("Final solution" in line for line in solver.convergence_report)


def test_unchanged_final_point_does_not_add_progress_row(monkeypatch):
    """A late gradient updates final diagnostics without inventing a new step."""
    target = np.array([1.0, -2.0])

    def return_last_evaluated_point(fun, x0, **kwargs):
        """Finish with a Jacobian after the last function observation."""
        fun(target)
        kwargs["jac"](target)
        return OptimizeResult(
            x=target.copy(),
            fun=0.0,
            success=True,
            status=0,
            message="Known minimum",
            nit=1,
        )

    monkeypatch.setattr(optimization_wrappers, "minimize", return_last_evaluated_point)
    solver = make_solver(update_on="function")
    solver.solve(np.array([0.0, 0.0]))

    assert solver.convergence_history["event_kind"] == ["initial", "function"]
    assert np.isnan(solver.convergence_history["optimality"][-1])
    assert solver.result.optimality == pytest.approx(0.0)
    assert solver.result.constraint_violation == pytest.approx(0.0)
    assert solver.result.njev == 1
    assert any("Final optimality" in line for line in solver.convergence_report)


def test_initial_progress_request_records_one_snapshot():
    """KKT's derivative request must not recursively duplicate the initial row."""
    solver = make_solver()
    point = np.array([2.0, -1.0])
    solver._print_convergence_progress(point)

    assert len(solver.convergence_history["x"]) == 1
    np.testing.assert_array_equal(solver.convergence_history["x"][0], point)


def test_gradient_observation_refreshes_all_values_at_its_point():
    """Coordinates, objective, and inequality violation must describe one point."""
    solver = make_solver(ConstrainedQuadratic())
    solver.fitness(np.array([2.0, 0.0]))
    point = np.array([-1.0, 0.0])
    solver.gradient(point)

    history = solver.convergence_history
    np.testing.assert_array_equal(history["x"][-1], point)
    assert history["objective_value"][-1] == pytest.approx(1.0)
    assert history["constraint_violation"][-1] == pytest.approx(0.0)


def test_kkt_assessment_does_not_generate_progress_events():
    """An explicit diagnostic call may evaluate derivatives but must not report."""
    callbacks = []

    def record_callback(x, count):
        """Capture any unexpected progress events generated by diagnostics."""
        callbacks.append((x.copy(), count))

    solver = make_solver(callback_functions=record_callback)
    solver.evaluate_kkt_conditions(np.array([2.0, -1.0]), 1e-6)

    assert solver.convergence_history["x"] == []
    assert callbacks == []


def test_improvement_plot_keeps_raw_objective_observations():
    """A derived best-so-far curve must not overwrite actual objective values."""
    solver = make_solver(plot_improvement_only=True)
    for point in (np.array([2.0, -2.0]), np.array([3.0, -2.0]), np.array([1.5, -2.0])):
        solver._print_convergence_progress(point)
    before = deepcopy(solver.convergence_history)

    solver._plot_convergence_callback([], [], initialize=True, showfig=False)

    assert solver.convergence_history["objective_value"] == before["objective_value"]
    np.testing.assert_array_equal(solver.convergence_history["x"], before["x"])
    expected_display = np.minimum.accumulate(before["objective_value"])
    np.testing.assert_array_equal(solver.obj_line_1.get_ydata(), expected_display)


@pytest.mark.parametrize(
    "method, step", [("2-point", 1e-6), ("3-point", 1e-4), ("cs", 1e-20)]
)
def test_derivative_probes_are_evaluations_not_progress_points(method, step):
    """FD and complex probes contribute calls while only the base may be reported."""
    problem = QuadraticProblem()
    callbacks = []

    def record_callback(x, count):
        """Keep callback coordinates for detecting accidental probe notifications."""
        callbacks.append(x.copy())

    solver = make_solver(
        problem,
        derivative_method=method,
        derivative_abs_step=step,
        callback_functions=record_callback,
    )
    point = np.array([2.0, -1.0])
    solver.fitness(point)
    derivative = solver.gradient(point)

    np.testing.assert_allclose(derivative, [[2.0, 4.0]], atol=5e-6, rtol=0)
    assert solver.grad_count == 1
    assert solver.func_count_tot == len(problem.calls)
    assert len(problem.calls) > 1
    assert len(solver.convergence_history["x"]) == 1
    assert len(callbacks) == 1
    for observed in callbacks + solver.convergence_history["x"]:
        np.testing.assert_array_equal(observed, point)
    assert solver.iteration_count in (None, 0)
    assert solver.iteration_history == []
    assert solver.derivative_func_count > 0
    assert solver.derivative_func_count <= solver.func_count_tot


def test_callback_mutation_cannot_change_caller_or_history():
    """A callback receives detached coordinates rather than the solver input."""

    def mutate_callback(x, count):
        """Attempt a deliberately hostile in-place edit of callback data."""
        x[:] = 99.0

    solver = make_solver(callback_functions=mutate_callback)
    point = np.array([2.0, -1.0])
    solver._print_convergence_progress(point)

    np.testing.assert_array_equal(point, [2.0, -1.0])
    for stored in solver.convergence_history["x"]:
        np.testing.assert_array_equal(stored, [2.0, -1.0])


@pytest.mark.parametrize("update_on", ["function", "gradient"])
def test_real_slsqp_converges_and_history_matches_recorded_points(update_on):
    """A real constrained solve provides honest values and feasible final output."""
    solver = make_solver(ConstrainedQuadratic(), update_on=update_on)
    returned = solver.solve(np.array([-3.0, 4.0]))

    assert solver.success
    np.testing.assert_allclose(returned, [0.0, 0.0], atol=2e-5)
    for point, objective, violation in zip(
        solver.convergence_history["x"],
        solver.convergence_history["objective_value"],
        solver.convergence_history["constraint_violation"],
    ):
        assert objective == pytest.approx(float(point @ point))
        assert violation == pytest.approx(max(float(point[0]), 0.0))


def test_display_and_plotting_do_not_change_a_real_slsqp_solve(monkeypatch):
    """Passive presentation toggles cannot change the solution or observations."""

    def no_pause(interval):
        """Avoid GUI waiting while exercising the actual plotting callback."""
        return None

    monkeypatch.setattr(plt, "pause", no_pause)
    quiet = make_solver()
    displayed = make_solver(print_convergence=True, plot_convergence=True)
    start = np.array([3.0, 4.0])
    quiet.solve(start)
    displayed.solve(start)

    assert quiet.success and displayed.success
    np.testing.assert_allclose(quiet.x_final, displayed.x_final, atol=1e-12)
    np.testing.assert_allclose(
        quiet.convergence_history["objective_value"],
        displayed.convergence_history["objective_value"],
        atol=1e-12,
    )


@pytest.mark.parametrize("method, step", [("2-point", 1e-6), ("cs", 1e-20)])
def test_only_backend_iteration_callback_advances_iterations(monkeypatch, method, step):
    """Model calls and full Jacobian requests never impersonate native iterations."""
    target = np.array([1.0, -2.0])

    def one_iteration_with_extra_evaluations(fun, x0, **kwargs):
        """Expose one explicit backend iteration amid function and derivative work."""
        fun(x0)
        kwargs["jac"](x0)
        assert solver.iteration_count == 0
        assert solver.iteration_history == []
        fun(target)
        kwargs["callback"](target.copy())
        assert solver.iteration_count == 1
        fun(np.array([4.0, 5.0]))
        kwargs["jac"](np.array([4.0, 5.0]))
        assert solver.iteration_count == 1
        return OptimizeResult(
            x=target.copy(),
            fun=0.0,
            success=True,
            status=0,
            message="One explicit iteration",
            nit=1,
            nfev=3,
            njev=2,
        )

    monkeypatch.setattr(
        optimization_wrappers, "minimize", one_iteration_with_extra_evaluations
    )
    solver = make_solver(
        QuadraticProblem(),
        derivative_method=method,
        derivative_abs_step=step,
    )
    solver.solve(np.array([2.0, -1.0]))

    assert solver.iteration_count == 1
    assert len(solver.iteration_history) == 1
    np.testing.assert_array_equal(solver.iteration_history[0]["x"], target)
    assert solver.derivative_func_count > 0
    assert solver.func_count_tot > solver.iteration_count


def test_scaled_result_keeps_physical_and_solver_coordinates_separate():
    """The facade returns physical coordinates while preserving normalized backend data."""
    solver = make_solver(problem_scale=2.0)
    returned = solver.solve(np.array([4.0, 5.0]))

    assert solver.success
    np.testing.assert_allclose(returned, [1.0, -2.0], atol=1e-5)
    np.testing.assert_array_equal(solver.result.x, returned)
    expected_solver_point = solver.problem.scale_physical_to_normalized(returned)
    np.testing.assert_allclose(solver.result.x_solver, expected_solver_point)
    np.testing.assert_allclose(solver.backend_result.x, expected_solver_point)


def test_reusing_solver_starts_a_fresh_run():
    """A second solve resets history, counters, and callback numbering."""
    callbacks = []

    def record_callback(x, count):
        """Collect callback numbering across two independently initialized runs."""
        callbacks.append(count)

    solver = make_solver(QuadraticProblem(), callback_functions=record_callback)
    solver.solve(np.array([3.0, 4.0]))
    previous_result = solver.result
    previous_history = solver.convergence_history
    first_callback_count = len(callbacks)
    second_start = np.array([1.0, -2.0])
    solver.solve(second_start)

    reference = make_solver(QuadraticProblem())
    reference.solve(second_start)
    assert solver.success and reference.success
    assert solver.result is not previous_result
    assert solver.convergence_history is not previous_history
    assert callbacks[first_callback_count] == 1
    assert solver.func_count_tot == reference.func_count_tot
    assert solver.grad_count == reference.grad_count
    assert solver.derivative_func_count == reference.derivative_func_count
    assert solver.iteration_count == reference.iteration_count
    np.testing.assert_array_equal(
        solver.convergence_history["x"], reference.convergence_history["x"]
    )


def test_iteration_observations_use_native_events_and_honest_labels():
    """Iteration-mode history uses native callbacks without promising acceptance."""
    solver = make_solver(QuadraticProblem(), update_on="iteration")
    solver.solve(np.array([3.0, 4.0]))

    assert solver.success
    assert solver.iteration_count == len(solver.iteration_history)
    assert solver.iteration_count == solver.backend_result.nit
    kinds = solver.convergence_history["event_kind"]
    assert kinds[0] == "initial"
    assert kinds[1:] == ["iteration"] * solver.iteration_count
    assert all(record["accepted"] is None for record in solver.iteration_history)
    assert all(
        record["event_kind"] == "iteration" for record in solver.iteration_history
    )
    solver.plot_convergence_history(showfig=False)
    assert solver.ax_1.get_xlabel() == "Native SLSQP major iterations"
    np.testing.assert_array_equal(
        solver.obj_line_1.get_xdata(), solver.convergence_history["iteration"]
    )


@pytest.mark.parametrize("update_on", ["function", "gradient", "iteration"])
def test_postsolve_diagnostics_and_reports_preserve_history(update_on):
    """Evaluating reports at another point cannot append solver progress."""
    callbacks = []

    def record_callback(x, count):
        """Detect observer notifications accidentally emitted by postprocessing."""
        callbacks.append(count)

    solver = make_solver(
        ConstrainedQuadratic(), update_on=update_on, callback_functions=record_callback
    )
    solver.solve(np.array([-3.0, 4.0]))
    before = deepcopy(solver.convergence_history)
    iteration_before = deepcopy(solver.iteration_history)
    callback_count = len(callbacks)
    final_point = solver.x_final.copy()
    final_result_point = solver.result.x.copy()
    other_point = np.array([-1.0, 2.0])

    solver.evaluate_kkt_conditions(other_point, 1e-6)
    solver.print_optimization_report(x=np.array([-2.0, 3.0]), to_console=False)
    solver.print_optimization_report(
        include_kkt_conditions=True, include_multipliers=True, to_console=False
    )
    solver.print_convergence_history(to_console=False)

    assert len(callbacks) == callback_count
    for key in before:
        if key == "event_kind":
            assert solver.convergence_history[key] == before[key]
        else:
            np.testing.assert_equal(solver.convergence_history[key], before[key])
    np.testing.assert_equal(solver.iteration_history, iteration_before)
    np.testing.assert_array_equal(solver.x_final, final_point)
    np.testing.assert_array_equal(solver.result.x, final_result_point)


def test_adjacent_points_are_distinct_even_with_legacy_cache_tolerance():
    """Approximate cache settings must not merge representably different points."""
    problem = AnalyticQuadratic()
    solver = make_solver(problem, tolerance_check_cache=1.0)
    first = np.array([2.0, -2.0])
    adjacent = first.copy()
    adjacent[0] = np.nextafter(adjacent[0], np.inf)

    first_value = solver.fitness(first)
    first_gradient = solver.gradient(first)
    second_value = solver.fitness(adjacent)
    second_gradient = solver.gradient(adjacent)

    assert solver.func_count_tot == 2
    assert solver.grad_count == 2
    assert len(problem.calls) == 2
    assert not np.array_equal(first_value, second_value)
    assert not np.array_equal(first_gradient, second_gradient)
    np.testing.assert_array_equal(solver.cache["x"], adjacent)
    np.testing.assert_array_equal(solver.cache["x_jac"], adjacent)


def test_partial_derivative_failure_is_counted_without_progress():
    """A failed second probe records one failed Jacobian request, never an iteration."""

    class ProbeFailure(QuadraticProblem):
        """Allow the first derivative probe and reject the second probe."""

        def fitness(self, x):
            """Record a real attempted evaluation before raising the model error."""
            value = super().fitness(x)
            if x[1] > -1.0:
                raise RuntimeError("Second derivative probe failed")
            return value

    problem = ProbeFailure()
    solver = make_solver(problem, derivative_abs_step=1e-6)
    with pytest.raises(RuntimeError, match="Second derivative probe failed"):
        solver.gradient(np.array([2.0, -1.0]))

    assert solver.grad_count == 1
    assert solver.failed_grad_count == 1
    assert solver.func_count_tot == len(problem.calls) == 3
    assert solver.derivative_func_count == 2
    assert solver.func_count == 1
    assert solver.iteration_count in (None, 0)
    assert solver.iteration_history == []
    assert solver.convergence_history["x"] == []


def test_constraint_metadata_uses_requested_point_not_last_derivative_probe():
    """Dynamic physical report metadata must follow the cached base, not probe state."""

    class MetadataProblem(QuadraticProblem):
        """Update application-owned constraint metadata on every model evaluation."""

        def fitness(self, x):
            """Return a linear constraint and matching named diagnostic metadata."""
            objective = super().fitness(x)[0]
            constraint = x[0] + 2 * x[1]
            self.constraint_data = [
                {
                    "name": "probe_sensitive_constraint",
                    "type": "<",
                    "target": 0.0,
                    "value": constraint,
                    "satisfied": bool(np.real(constraint) <= 0),
                }
            ]
            return np.array([objective, constraint])

        def get_nic(self):
            """Declare one linear inequality used by the metadata report."""
            return 1

    solver = make_solver(MetadataProblem(), derivative_abs_step=1e-3)
    base = np.array([0.5, -0.25])
    solver.fitness(base)
    solver.gradient(base)
    history_before = deepcopy(solver.convergence_history)

    base_data = solver.get_constraint_data(base, 1e-9)
    assert base_data[0]["name"] == "probe_sensitive_constraint"
    assert base_data[0]["value"] == pytest.approx(0.0)
    assert base_data[0]["satisfied"]
    base_data[0]["value"] = 123.0
    assert solver.get_constraint_data(base, 1e-9)[0]["value"] == pytest.approx(0.0)

    different_point = np.array([0.5, 0.25])
    report = solver.make_constraint_report(different_point, 1e-9)
    assert "probe_sensitive_constraint" in report
    assert "+1.00e+00" in report
    other_data = solver.get_constraint_data(different_point, 1e-9)
    assert other_data[0]["value"] == pytest.approx(1.0)
    assert not other_data[0]["satisfied"]
    assert solver.get_constraint_data(base, 1e-9)[0]["value"] == pytest.approx(0.0)
    for key in history_before:
        if key == "event_kind":
            assert solver.convergence_history[key] == history_before[key]
        else:
            np.testing.assert_equal(
                solver.convergence_history[key], history_before[key]
            )
