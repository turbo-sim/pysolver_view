"""Independent checks for KKT stopping and consistent near-bound evaluations."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pysolver_view import DomainError, OptimizationProblem, OptimizationSolver
from pysolver_view.optimization.backends.pounce import _PounceCallbacks
from pysolver_view.optimization.evaluation import BackendProblem


class StrictBoxQuadratic(OptimizationProblem):
    """A smooth objective whose model forbids all real box excursions."""

    def __init__(self, target=2.0):
        """Record every actual evaluation, including numerical derivative probes."""
        self.target = target
        self.calls = []

    def fitness(self, x):
        """Reject outside-domain points and return the scalar objective."""
        self.calls.append(np.asarray(x).copy())
        assert np.all(np.real(x) >= 0) and np.all(np.real(x) <= 1)
        return np.array([(x[0] - self.target) ** 2])

    def get_bounds(self):
        """Declare a unit box in physical coordinates."""
        return [0.0], [1.0]

    def get_nec(self):
        """Declare no equality constraints."""
        return 0

    def get_nic(self):
        """Declare no nonlinear inequalities."""
        return 0


def solver_for(problem, method="ipm", **kwargs):
    """Create a quiet solver with a stricter native tolerance than the user gate."""
    settings = dict(
        library="pounce",
        method=method,
        print_convergence=False,
        update_on="iteration",
        tolerance=1e-12,
        max_iterations=100,
    )
    settings.update(kwargs)
    return OptimizationSolver(problem, **settings)


@pytest.mark.parametrize("method", ["2-point", "3-point", "cs"])
def test_small_bound_excursions_reconcile_values_and_all_derivative_probes(method):
    """All callback families use one feasible base; actual probes remain inside."""
    problem = StrictBoxQuadratic()
    solver = solver_for(
        problem,
        derivative_method=method,
        derivative_abs_step=1e-6 if method != "cs" else 1e-20,
        check_kkt=False,
    )
    callbacks = _PounceCallbacks(
        BackendProblem(solver),
        1,
        0,
        0,
        bounds=(np.zeros(1), np.ones(1)),
        bound_tolerance=1e-8,
    )
    native = np.array([1.0 + 2.2e-9])
    assert callbacks.objective(native) == pytest.approx(1.0)
    np.testing.assert_allclose(callbacks.gradient(native), [-2.0], atol=3e-6)
    assert callbacks.constraints(native).size == 0
    assert callbacks.jacobian(native).size == 0
    assert callbacks.intermediate(0, 0, 1.0, 0, 2, 0.1, 0, 0, 0, 0, 0)
    row = callbacks.history[0]
    np.testing.assert_array_equal(row["x"], [1.0])
    np.testing.assert_array_equal(row["native_x"], native)
    assert callbacks.maximum_projection == pytest.approx(2.2e-9)
    assert solver.derivative_func_count > 0
    assert len(problem.calls) == solver.func_count_tot
    assert all(np.all(np.real(point) <= 1) for point in problem.calls)


@pytest.mark.parametrize("point", [[-1e-4], [1.0001]])
def test_larger_bound_violation_is_not_silently_projected(point):
    """An appreciably invalid trial remains an explicit model-domain failure."""
    problem = StrictBoxQuadratic()
    solver = solver_for(problem)
    callbacks = _PounceCallbacks(
        BackendProblem(solver), 1, 0, 0, bounds=(np.zeros(1), np.ones(1))
    )
    with pytest.raises(DomainError, match="bound_tolerance"):
        callbacks.objective(np.asarray(point))
    assert problem.calls == []


@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_real_native_solver_stops_on_full_user_kkt_criteria(method):
    """Engineering termination succeeds while preserving native user-stop status."""
    pytest.importorskip("pounce")
    solver = solver_for(
        StrictBoxQuadratic(),
        method,
        extra_options=dict(stop_on_kkt=True, feasibility_tol=1e-6, optimality_tol=1e-4),
    )
    solver.solve([0.2])
    assert solver.success, solver.message
    assert solver.result.kkt_passed
    assert solver.result.stopped_on_kkt
    assert solver.result.termination == "kkt_converged"
    assert solver.result.status == solver.backend_result.info["status"] == 5
    assert not solver.result.solver_success
    assert solver.result.kkt["feasibility_tol"] == 1e-6
    assert solver.result.kkt["optimality_tol"] == 1e-4
    assert solver.result.kkt["kkt_ratio"] <= 1
    assert solver.result.nfev == len(solver.problem.calls)


@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_user_cancellation_is_not_promoted_even_when_initial_point_is_optimal(method):
    """Explicit cancellation wins over simultaneous engineering convergence."""
    pytest.importorskip("pounce")
    solver = solver_for(
        StrictBoxQuadratic(target=0.25),
        method,
        callback_functions=lambda x, count: False,
        extra_options=dict(stop_on_kkt=True, feasibility_tol=1e-6, optimality_tol=1e-4),
    )
    solver.solve([0.25])
    assert not solver.success
    assert not solver.result.stopped_on_kkt
    assert solver.result.termination == "cancelled"


def fake_native(monkeypatch, final_point, status=5):
    """Supply a native callback point different from an authoritative final point."""

    class NativeProblem:
        """Minimal native protocol needed to test the final-point certificate."""

        def __init__(self, **kwargs):
            """Retain user callbacks without invoking any mathematical work."""
            self.callbacks = kwargs["problem_obj"]

        def add_option(self, *args):
            """Accept options while leaving the controlled test algorithm fixed."""

        def solve(self, x0):
            """Offer a converged observation but return the requested final point."""
            self.callbacks.objective(np.array([0.25]))
            continuing = self.callbacks.intermediate(
                0, 1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 1.0, 0
            )
            assert continuing is False
            return np.array([final_point]), dict(
                status=status,
                status_msg="User_Requested_Stop",
                obj_val=0.0,
                iter_count=1,
                mult_g=np.empty(0),
                mult_x_L=np.zeros(1),
                mult_x_U=np.zeros(1),
            )

    monkeypatch.setitem(sys.modules, "pounce", SimpleNamespace(Problem=NativeProblem))


@pytest.mark.parametrize("final_point", [0.8, 1.0 + 2e-9])
def test_returned_point_is_rechecked_after_a_converged_observation(
    monkeypatch, final_point
):
    """A stale cached observation cannot certify a different nonoptimal result."""
    fake_native(monkeypatch, final_point=final_point)
    solver = solver_for(
        StrictBoxQuadratic(target=0.25),
        extra_options=dict(stop_on_kkt=True, optimality_tol=1e-4),
    )
    solver.solve([0.1])
    assert solver.result.kkt_stop_requested
    assert not solver.success
    assert not solver.result.stopped_on_kkt
    assert solver.result.termination == "kkt_check_failed"
    np.testing.assert_array_equal(solver.result.x, [min(final_point, 1.0)])
    np.testing.assert_array_equal(solver.result.native_x_solver, [final_point])
    assert solver.result.bound_projection["final_correction"] == pytest.approx(
        max(final_point - 1.0, 0.0)
    )
    assert not solver.result.kkt_passed


@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_native_budget_stops_before_user_criteria_pass(method):
    """Feasibility alone cannot trigger KKT success or extend a native budget."""
    pytest.importorskip("pounce")
    solver = solver_for(
        StrictBoxQuadratic(target=0.75),
        method,
        max_iterations=1,
        extra_options=dict(stop_on_kkt=True, optimality_tol=1e-14),
    )
    solver.solve([0.2])
    assert solver.result.feasible
    assert not solver.result.kkt_passed
    assert not solver.result.kkt_stop_requested
    assert not solver.success
    assert solver.result.termination == "native_failure"
    assert solver.result.nit <= 1


def test_native_termination_does_not_force_additional_iterations(monkeypatch):
    """An earlier native failure ends the run even if user KKT has not passed."""
    fake_native(monkeypatch, final_point=0.8, status=-2)
    solver = solver_for(
        StrictBoxQuadratic(target=0.25),
        extra_options=dict(stop_on_kkt=True, optimality_tol=1e-4),
    )
    solver.solve([0.1])
    assert solver.result.status == -2
    assert not solver.success
    assert solver.result.termination == "native_failure"


@pytest.mark.parametrize(
    "options",
    [
        dict(stop_on_kkt=True),
        dict(stop_on_kkt="true"),
        dict(feasibility_tol=0),
        dict(optimality_tol=-1),
        dict(bound_tolerance=-1),
    ],
)
def test_control_validation_rejects_ambiguous_or_invalid_settings(options):
    """Automatic engineering stopping cannot silently enable disabled KKT work."""
    with pytest.raises((ValueError, TypeError)):
        solver_for(StrictBoxQuadratic(), check_kkt=False, extra_options=options)
