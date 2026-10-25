"""Independent high-level checks for signs, scaling, accounting, and domain recovery."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pysolver_view import OptimizationProblem, OptimizationSolver
from pysolver_view import optimization
from pysolver_view.solvers import DomainError


class MixedConstraints(OptimizationProblem):
    """A convex problem with known equality, inequality, and lower-bound multipliers."""

    def __init__(self):
        """Count model attempts independently of facade counters."""
        self.calls = []

    def fitness(self, x):
        """Pack a quadratic objective, equality, and a nonpositive inequality."""
        self.calls.append(x.copy())
        return np.array(
            [(x[0] - 2) ** 2 + x[1] ** 2 + (x[2] + 1) ** 2, x[0] + x[1] - 3, x[0] - 1]
        )

    def get_bounds(self):
        """Leave the first two coordinates wide and make the third bound active."""
        return [-10, -10, 0], [10, 10, 10]

    def get_nec(self):
        """Declare one equality."""
        return 1

    def get_nic(self):
        """Declare one nonpositive inequality."""
        return 1


class AnalyticMixedConstraints(MixedConstraints):
    """Supply exact physical derivatives for checking scaling and sign adapters."""

    def gradient(self, x):
        """Return objective, equality, and inequality derivative rows."""
        return np.array(
            [[2 * (x[0] - 2), 2 * x[1], 2 * (x[2] + 1)], [1, 1, 0], [1, 0, 0]]
        )


def make_solver(problem, **kwargs):
    """Use strict but realistic tolerances for small analytic benchmark problems."""
    options = dict(
        library="pysolver",
        method="slsqp",
        tolerance=1e-10,
        max_iterations=200,
        print_convergence=False,
        plot_convergence=False,
        update_on="iteration",
        extra_options={
            "optimality_tol": 1e-5,
            "feasibility_tol": 1e-7,
            "first_step_fraction": 1.0,
        },
    )
    options.update(kwargs)
    return OptimizationSolver(problem, **options)


@pytest.mark.parametrize("problem_type", [MixedConstraints, AnalyticMixedConstraints])
@pytest.mark.parametrize("scale", [None, 1.0, 10.0])
def test_mixed_constraints_have_correct_public_signs_and_scaled_bounds(
    problem_type, scale
):
    """Recover a known convex optimum and independently reconstruct its Lagrangian."""
    problem = problem_type()
    solver = make_solver(problem, problem_scale=scale)
    solver.solve(np.array([0.5, 2.5, 0.5]))
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.x_final, [1, 2, 0], atol=2e-6)
    multipliers = solver.result.kkt["multipliers"]
    assert multipliers["equality"][0] == pytest.approx(-4, abs=1e-5)
    assert multipliers["inequality"][0] == pytest.approx(6, abs=1e-5)
    factors = np.ones(3) if scale is None else np.array([20, 20, 10]) / scale
    assert multipliers["lower"][2] == pytest.approx(2 * factors[2], abs=2e-5)
    exact = np.array([-2, 4, 2]) * factors
    exact += np.array([1, 1, 0]) * factors * multipliers["equality"][0]
    exact += np.array([1, 0, 0]) * factors * multipliers["inequality"][0]
    exact -= multipliers["lower"]
    exact += multipliers["upper"]
    assert np.linalg.norm(exact, np.inf) < 5e-5
    assert solver.result.nfev == len(problem.calls)
    assert all(record["accepted"] for record in solver.iteration_history)
    np.testing.assert_array_equal(
        solver.result.fitness, problem.fitness(solver.x_final)
    )


@pytest.mark.parametrize("method", ["2-point", "3-point", "cs"])
def test_numerical_derivatives_keep_bound_probes_out_of_iteration_history(method):
    """Derivative schemes solve a bound-active model without publishing probe points."""
    problem = MixedConstraints()
    solver = make_solver(problem, derivative_method=method)
    solver.solve(np.array([0.5, 2.5, 0.0]))
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.x_final, [1, 2, 0], atol=2e-6)
    assert solver.derivative_func_count > 0
    assert solver.result.nfev == len(problem.calls)
    for x in problem.calls:
        assert np.all(np.real(x) >= np.array(problem.get_bounds()[0]))
        assert np.all(np.real(x) <= np.array(problem.get_bounds()[1]))
    assert solver.callback_function_call_count == 0
    assert len(solver.iteration_history) == len(solver.backend_result.history) - 1


class FragileCoupledProblem(OptimizationProblem):
    """Two coupled physical quantities whose square-root model requires x[0]>0."""

    def __init__(self, always_invalid=False, reject_probe=False):
        """Keep independent attempt records and optional failure modes."""
        self.calls = []
        self.always_invalid = always_invalid
        self.reject_probe = reject_probe

    def fitness(self, x):
        """Evaluate a coupled least-squares objective only inside its true domain."""
        self.calls.append(x.copy())
        if self.always_invalid or x[0] <= 0 or (self.reject_probe and x[1] > 0.9):
            raise DomainError("Physical state outside square-root domain")
        a = np.sqrt(x[0]) - 0.2
        b = x[1] - 0.4
        return np.array([a * a + b * b + (a + b) ** 2])

    def gradient(self, x):
        """Differentiate the coupled model analytically at a valid state."""
        a = np.sqrt(x[0]) - 0.2
        b = x[1] - 0.4
        return np.array([[(2 * a + b) / np.sqrt(x[0]), 2 * a + 4 * b]])

    def get_bounds(self):
        """Use a box wider than the physical domain so recovery is exercised."""
        return [-2, -2], [2, 2]

    def get_nec(self):
        """Declare no equalities."""
        return 0

    def get_nic(self):
        """Declare no inequalities."""
        return 0


@pytest.mark.parametrize("update_on", ["function", "gradient", "iteration"])
def test_domain_recovery_keeps_callbacks_at_accepted_points(update_on):
    """Rejected physical states never reach accepted callbacks or final results."""
    problem = FragileCoupledProblem()
    observed = []
    solver = make_solver(
        problem,
        update_on=update_on,
        callback_functions=lambda x, n: observed.append(x.copy()),
    )
    solver.solve([0.36, 0.9])
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.x_final, [0.04, 0.4], atol=2e-6)
    assert solver.backend_result.domain_rejections > 0
    assert len(observed) == len(solver.backend_result.history)
    for x, record in zip(observed, solver.backend_result.history):
        np.testing.assert_array_equal(x, record["x"])
        assert x[0] > 0
    assert solver.result.nfev == len(problem.calls)


def test_invalid_initial_point_returns_controlled_failure_without_reevaluation():
    """An invalid initial state is reported once and never forced through final reporting."""
    problem = FragileCoupledProblem(always_invalid=True)
    solver = make_solver(problem)
    solver.solve([0.36, 0.9])
    assert not solver.success
    assert solver.result.status == 10
    assert solver.result.nfev == len(problem.calls) == 1
    assert not solver.result.kkt_passed
    assert solver.convergence_history["x"] == []


@pytest.mark.parametrize("how", ["false", "exception"])
def test_accepted_callback_can_cancel_without_losing_last_point(how):
    """Both documented cancellation forms return the accepted callback point."""
    observed = []

    def callback(x, count):
        """Cancel at the first post-initial accepted state."""
        observed.append(x.copy())
        if count == 2:
            if how == "exception":
                raise StopIteration
            return False

    solver = make_solver(FragileCoupledProblem(), callback_functions=callback)
    solver.solve([0.36, 0.9])
    assert not solver.success
    assert solver.result.status == 13
    np.testing.assert_array_equal(solver.x_final_norm, observed[-1])
    assert len(observed) == 2


def test_tiny_controlled_step_does_not_turn_native_success_into_kkt_success():
    """A small native step cannot override a failed independent KKT assessment."""
    solver = make_solver(
        FragileCoupledProblem(),
        extra_options={
            "first_step_fraction": 1e-12,
            "max_step_fraction": 1e-12,
            "optimality_tol": 1e-7,
            "ftol": 1e-6,
        },
    )
    solver.solve([0.36, 0.9])
    assert solver.result.solver_success
    assert not solver.result.success
    assert not solver.result.kkt_passed
    assert solver.result.optimality > 1e-7


def test_failed_probe_is_counted_and_never_reported_as_iteration():
    """A numerical derivative domain failure consumes calls but no accepted step."""

    class ProbeProblem(OptimizationProblem):
        """Delegate only model and constraint declarations to the fragile model."""

        def __init__(self):
            """Create the independently counted physical model."""
            self.model = FragileCoupledProblem(reject_probe=True)

        def fitness(self, x):
            """Allow the base and first probe, then reject the second probe."""
            return self.model.fitness(x)

        def get_bounds(self):
            """Return the physical box."""
            return self.model.get_bounds()

        def get_nec(self):
            """Declare no equalities."""
            return 0

        def get_nic(self):
            """Declare no inequalities."""
            return 0

    problem = ProbeProblem()
    solver = make_solver(problem, derivative_abs_step=1e-6)
    solver.solve([0.36, 0.9])
    assert not solver.success
    assert solver.result.status == 12
    assert solver.failed_grad_count == 1
    assert solver.derivative_func_count == 2
    assert solver.result.nfev == len(problem.model.calls) == 3
    assert solver.iteration_history == []
    assert solver.convergence_history["event_kind"] == ["initial"]
    assert solver.backend_result.derivative_rejections == 1
    event = solver.backend_result.events[-1]
    assert event["kind"] == "derivative_rejection"
    assert event["column"] == 1
    np.testing.assert_array_equal(event["x"], problem.model.calls[-1])
    np.testing.assert_array_equal(event["failure"]["probe"], problem.model.calls[-1])


def test_returned_fitness_is_authoritative_after_a_different_cached_trial(monkeypatch):
    """A backend's final response cannot be replaced by its last trial response."""
    target = np.array([1.0, 2.0, 0.0])
    problem = AnalyticMixedConstraints()

    def backend(wrapped, x0, method, options):
        """Return a valid solution after evaluating a worse, distinct trial."""
        final = wrapped.fitness(target)
        wrapped.fitness(np.array([3.0, 4.0, 1.0]))
        return OptimizeResult(
            x=target,
            fitness=final,
            fun=final[0],
            success=True,
            solver_success=True,
            status=0,
            message="known minimum",
            nit=1,
            optimality=0.0,
            kkt=None,
        )

    monkeypatch.setitem(optimization.OPTIMIZATION_LIBRARIES, "pysolver", backend)
    solver = make_solver(problem)
    solver.solve([0.5, 2.5, 0.5])
    np.testing.assert_array_equal(solver.x_final, target)
    np.testing.assert_array_equal(solver.result.fitness, [6.0, 0.0, 0.0])
    assert solver.result.fun == 6.0
    assert solver.func_count_tot == len(problem.calls) == 2
    assert solver.diagnostic_func_count == 0


def test_disabled_domain_recovery_returns_last_accepted_point_and_its_kkt():
    """An undefined trial does not become the solution or corrupt its certificate."""
    problem = FragileCoupledProblem()
    solver = make_solver(
        problem,
        extra_options={
            "first_step_fraction": 1.0,
            "domain_backtracking": False,
            "optimality_tol": 1e-5,
        },
    )
    initial = np.array([0.36, 0.9])
    solver.solve(initial)
    assert not solver.success
    assert solver.result.status == 11
    np.testing.assert_array_equal(solver.x_final, initial)
    assert solver.backend_result.domain_rejections == 1
    assert solver.result.nfev == len(problem.calls) == 2
    assert solver.result.kkt["available"]
    multipliers = solver.result.kkt["multipliers"]
    stationarity = (
        problem.gradient(initial)[0] - multipliers["lower"] + multipliers["upper"]
    )
    complementarity = max(
        np.max(np.abs(multipliers["lower"] * (initial + 2))),
        np.max(np.abs(multipliers["upper"] * (2 - initial))),
    )
    assert solver.result.optimality == pytest.approx(
        max(np.linalg.norm(stationarity, np.inf), complementarity)
    )


def test_backend_callback_exposes_complete_raw_accepted_records():
    """Detailed normalized records preserve backend signs and exclude probes."""
    problem = MixedConstraints()
    observed = []
    solver = make_solver(problem, problem_scale=1.0, backend_callback=observed.append)
    solver.solve([0.5, 2.5, 0.5])
    assert solver.success, solver.message
    assert observed[0]["iteration"] == 0
    assert len(observed) == len(solver.backend_result.history)
    assert solver.derivative_func_count > 0
    assert len(observed) < len(problem.calls)
    assert solver.callback_function_call_count == 0
    for record, native in zip(observed, solver.backend_result.history):
        assert {
            "iteration",
            "fitness",
            "kkt",
            "step_norm",
            "objective_change",
            "domain_rejections",
            "derivative_rejections",
            "nfev",
            "njev",
        } <= record.keys()
        np.testing.assert_array_equal(record["x"], native["x"])
        np.testing.assert_array_equal(record["fitness"], native["fitness"])
        x = problem.scale_normalized_to_physical(record["x"])
        assert record["fitness"][1] == pytest.approx(x[0] + x[1] - 3)
        assert record["fitness"][2] == pytest.approx(1 - x[0])
    raw_multipliers = observed[-1]["kkt"]["multipliers"]
    public_multipliers = solver.result.kkt["multipliers"]
    np.testing.assert_allclose(
        raw_multipliers["equality"], -public_multipliers["equality"]
    )
    np.testing.assert_allclose(
        raw_multipliers["inequality"], public_multipliers["inequality"]
    )


def test_backend_callback_mutation_does_not_change_solver_or_legacy_callbacks(capsys):
    """Mutating nested diagnostic arrays cannot steer a solve or another observer."""
    original_records, legacy_points = [], []

    def mutate(record):
        """Modify all mutable payload layers after keeping a clean observation."""
        original_records.append(deepcopy(record))
        record["x"][:] = 999
        record["fitness"][:] = 999
        if record["kkt"] is not None and record["kkt"]["multipliers"] is not None:
            record["kkt"]["multipliers"]["equality"][:] = 999
        record.clear()

    solver = make_solver(
        AnalyticMixedConstraints(),
        backend_callback=mutate,
        callback_functions=lambda x, count: legacy_points.append((x, count)),
        print_convergence=True,
    )
    solver.solve([0.5, 2.5, 0.5])
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.x_final, [1, 2, 0], atol=2e-6)
    assert len(legacy_points) == len(original_records)
    assert solver.callback_function_call_count == len(legacy_points)
    for expected_count, ((point, count), record, native) in enumerate(
        zip(legacy_points, original_records, solver.backend_result.history), start=1
    ):
        assert count == expected_count
        np.testing.assert_array_equal(point, record["x"])
        np.testing.assert_array_equal(point, native["x"])
    assert solver.result.kkt["multipliers"]["equality"][0] == pytest.approx(-4)
    output = capsys.readouterr().out
    assert output.count("Starting optimization process") == 1
    assert output.count("Success:") == 1
    assert "Starting pysolver SLSQP" not in output


@pytest.mark.parametrize("how", ["false", "exception"])
def test_backend_callback_cancellation_preserves_point_and_solver_reuse(how):
    """Both cancellation forms stop at an accepted point and reset on a new solve."""
    observed = []
    cancel = True

    def callback(record):
        """Cancel only the first solve at its first post-initial accepted point."""
        observed.append(record)
        if cancel and len(observed) == 2:
            if how == "exception":
                raise StopIteration
            return False

    solver = make_solver(FragileCoupledProblem(), backend_callback=callback)
    solver.solve([0.36, 0.9])
    assert not solver.success
    assert solver.result.status == 13
    assert len(observed) == 2
    np.testing.assert_array_equal(solver.x_final_norm, observed[-1]["x"])
    cancel = False
    observed.clear()
    solver.solve([0.64, 0.8])
    assert solver.success, solver.message
    assert observed[0]["iteration"] == 0
    np.testing.assert_array_equal(observed[0]["x"], [0.64, 0.8])
    assert len(observed) == len(solver.backend_result.history)
    assert solver.callback_function_call_count == 0


def test_backend_callback_keeps_recovery_counters_without_rejected_states():
    """Physical trial failures update accepted diagnostics without becoming points."""
    observed = []
    solver = make_solver(FragileCoupledProblem(), backend_callback=observed.append)
    solver.solve([0.36, 0.9])
    assert solver.success, solver.message
    assert solver.backend_result.domain_rejections > 0
    assert max(record["domain_rejections"] for record in observed) > 0
    assert len(observed) == len(solver.backend_result.history)
    assert all(record["x"][0] > 0 for record in observed)


def test_backend_callback_rejects_unsupported_configuration_and_propagates_errors():
    """Callback misuse fails early and unrelated observer errors remain visible."""
    with pytest.raises(TypeError, match="backend_callback"):
        make_solver(MixedConstraints(), backend_callback=2)
    with pytest.raises(ValueError, match="library='pysolver'"):
        make_solver(
            MixedConstraints(), library="scipy", backend_callback=lambda _: None
        )

    def fail(record):
        """An observer programming error is not a cancellation request."""
        raise RuntimeError("observer failed")

    solver = make_solver(MixedConstraints(), backend_callback=fail)
    with pytest.raises(RuntimeError, match="observer failed"):
        solver.solve([0.5, 2.5, 0.5])
