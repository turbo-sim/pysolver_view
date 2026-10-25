"""Regressions for authoritative root results and truthful evaluation histories."""

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import pysolver_view.nonlinear_system as nonlinear_system
from pysolver_view.roots.backends import scipy as scipy_backend


def nonlinear_residual(x):
    """Evaluate a coupled polynomial system without changing any counters."""
    return np.array([x[0] ** 2 + x[1] - 3.0, x[0] + x[1] ** 2 - 5.0])


class CountedProblem(nonlinear_system.NonlinearSystemProblem):
    """Count actual model calls, including numerical derivative probes."""

    def __init__(self):
        """Start with an empty record of points evaluated by the model."""
        self.calls = []

    def residual(self, x):
        """Record a detached input and return the coupled system residual."""
        self.calls.append(np.array(x, copy=True))
        return nonlinear_residual(x)


def make_solver(problem=None, **kwargs):
    """Create a quiet native root solver suitable for direct regression checks."""
    return nonlinear_system.NonlinearSystemSolver(
        CountedProblem() if problem is None else problem,
        print_convergence=False,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Close figures created by plotting regressions after each test."""
    yield
    plt.close("all")


@pytest.mark.parametrize("native_success", [True, False])
def test_backend_point_wins_over_last_evaluated_trial(monkeypatch, native_success):
    """A final evaluated trial must never replace the backend's returned point."""
    final_x = np.array([1.0, 2.0])

    def fake_root(fun, x0, **kwargs):
        """Leave a rejected point in the observation state before returning."""
        fun(np.asarray(x0))
        fun(np.array([9.0, 8.0]))
        return OptimizeResult(
            x=final_x.copy(),
            fun=nonlinear_residual(final_x),
            success=native_success,
            status=1 if native_success else 5,
            message="controlled backend result",
            nfev=2,
        )

    monkeypatch.setattr(scipy_backend, "root", fake_root)
    solver = make_solver()
    returned = solver.solve([0.0, 0.0])
    np.testing.assert_array_equal(returned, final_x)
    np.testing.assert_array_equal(solver.f_final, [0.0, 0.0])
    np.testing.assert_array_equal(solver.result.x, final_x)
    np.testing.assert_array_equal(solver.result.fun, [0.0, 0.0])
    assert solver.backend_result.success == native_success
    assert solver.success == native_success
    assert solver.func_count_tot == 2
    assert solver.iteration_count is None
    assert solver.iteration_history == []


def test_gradient_uses_residual_at_requested_base():
    """A numerical Jacobian must not reuse a residual from another point."""
    solver = make_solver(derivative_abs_step=1e-6)
    solver.residual(np.array([2.0, 1.0]))
    jacobian = solver.gradient(np.array([3.0, 2.0]))
    np.testing.assert_allclose(jacobian, [[6.0, 1.0], [1.0, 4.0]], atol=2e-6)


@pytest.mark.parametrize("method", ["2-point", "3-point", "cs"])
def test_derivative_probes_do_not_change_history_or_callback_state(method):
    """Probes are function calls, not observations or solver iterations."""
    observed = []

    def callback(x, count):
        """Keep independent records of the ordinary observation callbacks."""
        observed.append((x.copy(), count))

    problem = CountedProblem()
    solver = make_solver(
        problem,
        derivative_method=method,
        derivative_abs_step=1e-5,
        callback_func=callback,
    )
    base = np.array([2.0, 1.0])
    solver.residual(base)
    before = deepcopy(solver.convergence_history)
    residual_before = solver.residuals_last.copy()
    callback_count = len(observed)
    solver.gradient(base)
    np.testing.assert_array_equal(solver.residuals_last, residual_before)
    assert len(observed) == callback_count
    assert len(solver.convergence_history["norm_residual"]) == len(
        before["norm_residual"]
    )
    assert solver.grad_count == 1
    assert solver.func_count == 1
    assert solver.func_count_tot == len(problem.calls)
    assert solver.derivative_func_count == len(problem.calls) - 1
    assert solver.iteration_count is None
    assert solver.iteration_history == []


@pytest.mark.parametrize("method", ["2-point", "3-point", "cs"])
def test_gradient_history_records_its_base_not_the_final_probe(method):
    """Derivative observations pair their base point with that point's residual."""
    solver = make_solver(
        update_on="gradient",
        derivative_method=method,
        derivative_abs_step=1e-3,
    )
    base = np.array([2.0, 1.0])
    solver.gradient(base)
    history = solver.convergence_history
    assert len(history["norm_residual"]) == 1
    np.testing.assert_allclose(
        history["norm_residual"],
        [np.linalg.norm(nonlinear_residual(base))],
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(history["x"][0], base)
    np.testing.assert_array_equal(history["residual"][0], nonlinear_residual(base))
    assert history["event_kind"] == ["gradient"]
    assert history["iteration"] == [None]


def test_callback_cannot_modify_the_backend_input():
    """Legacy observation callbacks receive independent copies of the point."""

    def mutate_callback(x, count):
        """Try to overwrite the point provided to a user observer."""
        x[:] = 1234.0

    solver = make_solver(callback_func=mutate_callback)
    point = np.array([2.0, 1.0])
    expected = nonlinear_residual(point)
    value = solver.residual(point)
    np.testing.assert_array_equal(point, [2.0, 1.0])
    np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("method", ["hybr", "lm"])
@pytest.mark.parametrize("update_on", ["function", "gradient"])
def test_real_root_history_and_results_are_point_consistent(method, update_on):
    """Native solves expose correct final values and self-consistent observations."""
    solver = make_solver(method=method, update_on=update_on, tolerance=1e-9)
    returned = solver.solve(np.array([1.1, 1.9]))
    assert solver.success
    np.testing.assert_allclose(returned, [1.0, 2.0], atol=1e-7)
    np.testing.assert_array_equal(returned, solver.backend_result.x)
    np.testing.assert_array_equal(solver.f_final, solver.result.fun)
    for point, residual, norm in zip(
        solver.convergence_history["x"],
        solver.convergence_history["residual"],
        solver.convergence_history["norm_residual"],
    ):
        np.testing.assert_allclose(residual, nonlinear_residual(point), atol=1e-14)
        assert norm == np.linalg.norm(residual)
    assert solver.iteration_count is None
    assert all(value is None for value in solver.convergence_history["iteration"])


@pytest.mark.parametrize("update_on", ["function", "gradient"])
def test_plot_uses_evaluation_labels_and_preserves_history(update_on):
    """Drawing a history does not mislabel evaluations as solver iterations."""
    solver = make_solver(update_on=update_on)
    solver.solve(np.array([1.1, 1.9]))
    before = deepcopy(solver.convergence_history)
    total_calls = solver.func_count_tot
    figure = solver.plot_convergence_history(showfig=False)
    assert "iteration" not in figure.axes[0].get_xlabel().lower()
    assert "evaluation" in figure.axes[0].get_xlabel().lower()
    for key in before:
        for old, new in zip(before[key], solver.convergence_history[key]):
            np.testing.assert_equal(new, old)
        assert len(before[key]) == len(solver.convergence_history[key])
    assert solver.func_count_tot == total_calls


def test_repeated_solve_resets_run_history_and_counters():
    """Reusing a solver starts a new run rather than appending old observations."""
    solver = make_solver()
    solver.solve(np.array([1.1, 1.9]))
    first_calls = solver.func_count_tot
    first_rows = len(solver.convergence_history["norm_residual"])
    solver.solve(np.array([1.1, 1.9]))
    assert solver.func_count_tot == first_calls
    assert len(solver.convergence_history["norm_residual"]) == first_rows


@pytest.mark.parametrize("derivative_method", ["2-point", "cs"])
def test_live_plotting_does_not_change_numerical_work(monkeypatch, derivative_method):
    """Live rendering must not add model calls or alter a native root solve."""

    def no_pause(seconds):
        """Skip GUI waiting while still exercising the actual plotting path."""

    monkeypatch.setattr(plt, "pause", no_pause)
    quiet = make_solver(derivative_method=derivative_method)
    plotted = make_solver(derivative_method=derivative_method, plot_convergence=True)
    quiet.solve([1.1, 1.9])
    plotted.solve([1.1, 1.9])
    np.testing.assert_array_equal(plotted.x_final, quiet.x_final)
    assert plotted.func_count_tot == quiet.func_count_tot
    assert plotted.grad_count == quiet.grad_count
    np.testing.assert_array_equal(
        plotted.convergence_history["norm_residual"],
        quiet.convergence_history["norm_residual"],
    )


def test_failed_derivative_request_counts_work_without_recording_probe(monkeypatch):
    """A failed probe remains counted work and never becomes an observation."""
    solver = make_solver()
    base = np.array([2.0, 1.0])
    solver.residual(base)

    def failing_probe(x):
        """Represent an unexpected model error during numerical differentiation."""
        raise RuntimeError("probe failed")

    monkeypatch.setattr(solver.problem, "residual", failing_probe)
    with pytest.raises(RuntimeError, match="probe failed"):
        solver.gradient(base)
    assert solver.grad_count == 1
    assert solver.failed_grad_count == 1
    assert solver.func_count_tot == 2
    assert solver.derivative_func_count == 1
    assert len(solver.convergence_history["x"]) == 1
    np.testing.assert_array_equal(solver.residuals_last, nonlinear_residual(base))
