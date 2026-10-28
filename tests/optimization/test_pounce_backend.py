"""Independent checks for the optional Pounce NLP adapter and observations."""

from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pysolver_view import OptimizationProblem, OptimizationSolver, RosenbrockProblem
from pysolver_view.diagnostics import kkt
from pysolver_view.optimization.problems.coupled_rosenbrock import (
    CoupledRosenbrockProblem,
)


class MixedQuadratic(OptimizationProblem):
    """A convex NLP with known equality, inequality, and bound multipliers."""

    def __init__(self):
        """Record model calls independently of the frontend's counters."""
        self.calls = []

    @staticmethod
    def values(x):
        """Pack the objective, one equality, and one nonpositive inequality."""
        return np.array(
            [
                (x[0] - 2) ** 2 + x[1] ** 2 + (x[2] + 1) ** 2,
                x[0] + x[1] - 3,
                x[0] - 1,
            ]
        )

    def fitness(self, x):
        """Count a physical model evaluation without discarding complex probes."""
        self.calls.append(x.copy())
        return self.values(x)

    def get_bounds(self):
        """Make the third variable's lower bound active at the solution."""
        return [-10, -10, 0], [10, 10, 10]

    def get_nec(self):
        """Declare one equality."""
        return 1

    def get_nic(self):
        """Declare one nonpositive inequality."""
        return 1


class AnalyticMixedQuadratic(MixedQuadratic):
    """Supply exact physical derivatives for sign and coordinate checks."""

    def gradient(self, x):
        """Return objective, equality, and inequality derivative rows."""
        return np.array(
            [
                [2 * (x[0] - 2), 2 * x[1], 2 * (x[2] + 1)],
                [1, 1, 0],
                [1, 0, 0],
            ]
        )


@pytest.fixture
def requires_pounce():
    """Skip numerical Pounce checks when its optional extra is not installed."""
    pytest.importorskip("pounce")


def make_solver(problem=None, method="ipm", **kwargs):
    """Configure a quiet solve with independent iteration observations."""
    configured = dict(
        library="pounce",
        method=method,
        tolerance=1e-8,
        max_iterations=200,
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
    )
    configured.update(kwargs)
    return OptimizationSolver(problem or AnalyticMixedQuadratic(), **configured)


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
@pytest.mark.parametrize("scale", [None, 1.0])
def test_known_mixed_optimum_respects_public_signs_and_scaling(method, scale):
    """Recover the physical optimum and reconstruct its public Lagrangian."""
    problem = AnalyticMixedQuadratic()
    solver = make_solver(problem, method=method, problem_scale=scale)
    returned = solver.solve(np.array([0.5, 2.5, 0.5]))

    assert solver.success, solver.message
    np.testing.assert_allclose(returned, [1, 2, 0], atol=3e-6, rtol=0)
    np.testing.assert_array_equal(returned, solver.result.x)
    assert solver.result.nfev == len(problem.calls)
    assert solver.result.derivative_func_count == 0
    np.testing.assert_allclose(solver.result.fitness, problem.values(returned))
    assert solver.result.constraint_violation < 1e-6
    assert solver.result.kkt_passed

    assessment = solver.evaluate_kkt_conditions(solver.x_final_norm, 1e-5)
    assert assessment["passed"]
    multipliers = assessment["multipliers"]
    assert multipliers["equality"][0] == pytest.approx(-4, abs=3e-5)
    assert multipliers["inequality"][0] == pytest.approx(6, abs=3e-5)
    factors = np.ones(3) if scale is None else np.array([20, 20, 10])
    stationarity = problem.gradient(returned)[0] * factors
    stationarity += np.array([1, 1, 0]) * factors * multipliers["equality"][0]
    stationarity += np.array([1, 0, 0]) * factors * multipliers["inequality"][0]
    stationarity -= multipliers["lower"]
    stationarity += multipliers["upper"]
    assert np.linalg.norm(stationarity, np.inf) < 1e-5
    native = solver.result.native_multipliers
    assert solver.result.native_multiplier_coordinates == "solver"
    native_stationarity = problem.gradient(returned)[0] * factors
    native_stationarity += np.array([1, 1, 0]) * factors * native["equality"][0]
    native_stationarity += np.array([1, 0, 0]) * factors * native["inequality"][0]
    native_stationarity -= native["lower"]
    native_stationarity += native["upper"]
    assert np.linalg.norm(native_stationarity, np.inf) < 3e-5


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_numerical_derivatives_count_probes_without_inventing_iterations(method):
    """Finite-difference model work stays separate from native iteration rows."""
    problem = MixedQuadratic()
    solver = make_solver(problem, method=method, derivative_method="3-point")
    solver.solve(np.array([0.5, 2.5, 0.5]))

    assert solver.success, solver.message
    np.testing.assert_allclose(solver.x_final, [1, 2, 0], atol=5e-6, rtol=0)
    assert solver.result.nfev == len(problem.calls)
    assert solver.result.derivative_func_count > 0
    assert solver.result.njev > 0
    assert len(solver.iteration_history) <= solver.iteration_count + 1
    for row in solver.iteration_history:
        physical_x = problem.scale_normalized_to_physical(row["x"])
        assert row["objective_value"] == pytest.approx(problem.values(physical_x)[0])


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_disabled_kkt_checks_do_not_run_hidden_multiplier_estimation(
    method, monkeypatch
):
    """Native dual metrics do not activate independent KKT work when disabled."""

    def forbidden(*args, **kwargs):
        """Fail if a solve or report invokes the optional multiplier fit."""
        raise AssertionError("Unexpected independent KKT assessment")

    monkeypatch.setattr(kkt, "linprog", forbidden)
    solver = make_solver(method=method, check_kkt=False)
    solver.solve(np.array([0.5, 2.5, 0.5]))
    assert solver.success, solver.message
    assert solver.result.kkt_evaluations == 0
    assert solver.result.kkt_passed is None
    assert solver.result.kkt_time == 0
    assert np.all(np.isnan(solver.convergence_history["optimality"]))
    assert np.isnan(solver.result.optimality)


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_native_observations_are_point_consistent_and_detached(method):
    """Cached objective points remain explicitly distinct from accepted iterates."""
    problem = AnalyticMixedQuadratic()
    callback_records = []
    solver = make_solver(
        problem, method=method, backend_callback=callback_records.append
    )
    solver.solve(np.array([0.5, 2.5, 0.5]))
    assert solver.success, solver.message
    assert callback_records
    assert solver.iteration_history
    assert all(row["accepted"] is None for row in solver.iteration_history)
    for row in solver.iteration_history:
        assert row["event_kind"] == "native_iteration"
        assert row["coordinate_source"] == "last_objective_evaluation"
        assert "native_metrics" in row
        assert row["objective_value"] == pytest.approx(problem.values(row["x"])[0])
    saved_history = deepcopy(solver.iteration_history)
    saved_point = solver.result.x.copy()
    record = next(row for row in callback_records if row.get("x") is not None)
    record["x"][:] = 1e9
    native_record = next(
        row for row in solver.backend_result.history if row.get("x") is not None
    )
    native_record["x"][:] = -1e9
    np.testing.assert_array_equal(solver.result.x, saved_point)
    for before, after in zip(saved_history, solver.iteration_history):
        np.testing.assert_array_equal(after["x"], before["x"])
        assert after["native_metrics"] == before["native_metrics"]


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_common_iteration_budget_reaches_the_selected_pounce_algorithm(method):
    """A small budget truncates a nonlinear solve rather than using backend defaults."""
    solver = make_solver(RosenbrockProblem(4), method=method, max_iterations=1)
    solver.solve(np.array([-1.2, 1.0, -1.2, 1.0]))
    assert not solver.success
    assert solver.iteration_count <= 1
    assert solver.backend_result.nit <= 1


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
@pytest.mark.parametrize("stop_kind", ["return_false", "stop_iteration"])
def test_callback_cancellation_returns_a_structured_unsuccessful_result(
    method, stop_kind
):
    """Both documented cancellation forms stop a native solve at its callback point."""
    observed = []

    def cancel(x, count):
        """Stop once the solver has begun iterating, retaining detached coordinates."""
        observed.append(x.copy())
        if count >= 2:
            if stop_kind == "stop_iteration":
                raise StopIteration("Requested cancellation")
            return False
        return None

    solver = make_solver(method=method, callback_functions=cancel, check_kkt=False)
    returned = solver.solve(np.array([0.5, 2.5, 0.5]))
    assert observed
    assert not solver.success
    assert not solver.result.solver_success
    np.testing.assert_allclose(returned, observed[-1], atol=1e-7, rtol=0)


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_unrelated_callback_errors_propagate_without_becoming_native_failure(method):
    """A broken consumer callback retains its original exception identity."""
    error = RuntimeError("Consumer callback failed")

    def broken_callback(x, count):
        """Raise after initial frontend notification to exercise the native boundary."""
        if count >= 2:
            raise error

    solver = make_solver(method=method, callback_functions=broken_callback)
    with pytest.raises(RuntimeError, match="Consumer callback failed") as caught:
        solver.solve(np.array([0.5, 2.5, 0.5]))
    assert caught.value is error


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_unrelated_model_errors_propagate_through_native_callbacks(method):
    """Native evaluation-failure handling must not swallow application bugs."""
    error = RuntimeError("Analytic model gradient failed")

    class BrokenGradient(AnalyticMixedQuadratic):
        """Raise inside an evaluation callback reached by the native solver."""

        def gradient(self, x):
            """Expose an application error instead of an invalid numeric derivative."""
            raise error

    solver = make_solver(BrokenGradient(), method=method)
    with pytest.raises(RuntimeError, match="Analytic model gradient failed") as caught:
        solver.solve(np.array([0.5, 2.5, 0.5]))
    assert caught.value is error


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
@pytest.mark.parametrize("source", ["gradient", "callback"])
def test_keyboard_interrupt_propagates_through_native_callback_boundary(method, source):
    """Preserve user interruption even when Rust catches Python callback failures."""
    interruption = KeyboardInterrupt("Interrupt native optimization")

    class InterruptingGradient(AnalyticMixedQuadratic):
        """Interrupt from a model callback reached during the native solve."""

        def gradient(self, x):
            """Raise the original interruption without returning a derivative."""
            raise interruption

    def interrupt_callback(x, count):
        """Wait for a native notification after the initial frontend observation."""
        if count >= 2:
            raise interruption

    solver = make_solver(
        InterruptingGradient() if source == "gradient" else AnalyticMixedQuadratic(),
        method=method,
        callback_functions=interrupt_callback if source == "callback" else None,
        check_kkt=False,
    )
    with pytest.raises(
        KeyboardInterrupt, match="Interrupt native optimization"
    ) as caught:
        solver.solve(np.array([0.5, 2.5, 0.5]))
    assert caught.value is interruption


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_unsupported_initial_step_cap_is_rejected_explicitly(method):
    """Do not silently ignore a SLSQP-only control or map it to a different knob."""
    with pytest.raises(ValueError, match="first_step_fraction"):
        solver = make_solver(method=method, extra_options={"first_step_fraction": 0.2})
        solver.solve(np.array([0.5, 2.5, 0.5]))


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize(
    "method, options",
    [
        ("ipm", {"hessian_approximation": "exact"}),
        ("sqp", {"sqp_hessian": "exact"}),
    ],
)
def test_unavailable_exact_lagrangian_hessian_request_is_rejected(method, options):
    """A first-derivative adapter must not silently downgrade exact curvature."""
    with pytest.raises(ValueError, match="[Hh]essian|exact"):
        solver = make_solver(method=method, extra_options=options)
        solver.solve(np.array([0.5, 2.5, 0.5]))


def test_missing_pounce_extra_is_lazy_and_actionable():
    """Ordinary imports work without Pounce; selecting its backend explains installation."""
    script = """
import importlib.abc
import sys

class NoPounce(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pounce':
            raise ModuleNotFoundError('Pounce deliberately unavailable')

sys.meta_path.insert(0, NoPounce())
import pysolver_view as psv
from pysolver_view.optimization.backends.registry import VALID_LIBRARIES_AND_METHODS
assert 'pounce' not in sys.modules
assert set(VALID_LIBRARIES_AND_METHODS['pounce']) == {'ipm', 'sqp'}
problem = psv.RosenbrockProblem(2)
native = psv.OptimizationSolver(problem, library='scipy', print_convergence=False)
assert native.solve([1.0, 1.0]).shape == (2,)
assert 'pounce' not in sys.modules
try:
    psv.OptimizationSolver(problem, library='pounce', method='ipm',
                          print_convergence=False).solve([0.0, 0.0])
except ImportError as error:
    assert 'pysolver_view[pounce]' in str(error), str(error)
else:
    raise AssertionError('Selecting an unavailable optional backend must fail clearly')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("blocks", [3, 12])
def test_comparison_problem_derivatives_match_complex_step(blocks):
    """Check every objective and constraint row independently of solver behavior."""
    problem = CoupledRosenbrockProblem(blocks=blocks)
    point = problem.initial
    numerical = np.empty((1 + problem.get_nec() + problem.get_nic(), point.size))
    for column in range(point.size):
        probe = point.astype(complex)
        probe[column] += 1e-30j
        numerical[:, column] = np.imag(problem.fitness(probe)) / 1e-30
    np.testing.assert_allclose(
        problem.gradient(point), numerical, rtol=1e-12, atol=1e-12
    )


def test_native_scaling_and_restoration_metrics_do_not_invent_model_points():
    """Keep internal objectives separate and omit ambiguous restoration coordinates."""
    from pysolver_view.optimization.backends.pounce import _PounceCallbacks

    calls = []
    observations = []

    def fitness(x):
        """Provide an original-unit objective with independently counted work."""
        calls.append(x.copy())
        return np.array([1000 * x[0] ** 2, x[0] - 1])

    problem = SimpleNamespace(fitness=fitness, native_callback=observations.append)
    callbacks = _PounceCallbacks(
        problem, dimension=1, equality_count=0, inequality_count=1
    )
    callbacks.objective(np.array([2.0]))
    assert callbacks.intermediate(0, 0, 4.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0)
    assert callbacks.intermediate(1, 1, 99.0, 0.3, 0.4, 0.2, 1.0, 0.0, 1.0, 1.0, 2)

    assert len(calls) == 1
    ordinary, restoration = observations
    assert ordinary["fitness"][0] == pytest.approx(4000)
    assert ordinary["native_metrics"]["obj_value"] == pytest.approx(4)
    assert ordinary["accepted"] is None
    assert restoration["event_kind"] == "restoration"
    assert restoration["x"] is None
    assert restoration["fitness"] is None
    assert restoration["native_metrics"]["obj_value"] == pytest.approx(99)
    ordinary["x"][:] = -123
    np.testing.assert_array_equal(callbacks.history[0]["x"], [2.0])


def test_cached_native_point_does_not_assess_kkt_with_another_points_values():
    """A later trial may move shared values without replacing the native point."""
    problem = AnalyticMixedQuadratic()
    solver = make_solver(problem)
    native_point = np.array([0.5, 2.5, 0.5])
    later_trial = np.array([2.0, 0.0, 1.0])
    solver._evaluate_gradient(native_point)
    native_values = solver.fitness(native_point).copy()
    solver.fitness(later_trial)
    np.testing.assert_array_equal(solver.cache["x"], later_trial)
    np.testing.assert_array_equal(solver.cache["x_jac"], native_point)
    calls_before = len(problem.calls)

    solver._record_pounce_iteration(
        {
            "iteration": 1,
            "x": native_point,
            "fitness": native_values,
            "coordinate_source": "last_objective_evaluation",
            "native_metrics": {},
        }
    )

    assert len(problem.calls) == calls_before
    snapshot = solver.iteration_history[-1]
    np.testing.assert_array_equal(snapshot["x"], native_point)
    assert snapshot["objective_value"] == pytest.approx(native_values[0])
    assert snapshot["constraint_violation"] == pytest.approx(0)
    assert np.isnan(snapshot["optimality"])


@pytest.mark.usefixtures("requires_pounce")
@pytest.mark.parametrize("method", ["ipm", "sqp"])
@pytest.mark.parametrize("update_on", ["gradient", "function"])
def test_native_step_sizes_use_consecutive_native_points(method, update_on):
    """Trial observation streams must not reset displacement in native history."""
    solver = make_solver(method=method, update_on=update_on, check_kkt=False)
    solver.solve(np.array([0.5, 2.5, 0.5]))
    assert solver.success, solver.message
    history = solver.iteration_history
    assert len(history) > 1
    expected_steps = [
        np.linalg.norm(current["x"] - previous["x"])
        for previous, current in zip(history, history[1:])
    ]
    assert any(step > 0 for step in expected_steps)
    np.testing.assert_allclose(
        [row["norm_step"] for row in history[1:]],
        expected_steps,
        atol=1e-14,
        rtol=1e-14,
    )
