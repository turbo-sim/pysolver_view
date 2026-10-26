"""Independent analytical regressions for shared KKT facade and text reporting."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pysolver_view.diagnostics import kkt
from pysolver_view.optimization import OptimizationProblem, OptimizationSolver


class AffineProblem(OptimizationProblem):
    """Expose an affine model with independently specified gradients and names."""

    def __init__(
        self,
        gradient,
        *,
        lower,
        upper,
        jh=(),
        h_offset=(),
        jg=(),
        g_offset=(),
        names=None,
    ):
        """Store coefficients in physical coordinates with public g<=0 signs."""
        self.objective_gradient = np.asarray(gradient, dtype=float)
        n = self.objective_gradient.size
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.jh = np.asarray(jh, dtype=float).reshape((-1, n))
        self.jg = np.asarray(jg, dtype=float).reshape((-1, n))
        self.h_offset, self.g_offset = np.asarray(h_offset), np.asarray(g_offset)
        self.variable_names = names or [f"physical_{i}" for i in range(n)]

    def fitness(self, x):
        """Compute values and named metadata from the same physical model point."""
        h = self.jh @ x + self.h_offset
        g = self.jg @ x + self.g_offset
        self.constraint_data = [
            dict(name=f"balance_{i}", type="=", target=0.0, value=value)
            for i, value in enumerate(h)
        ] + [
            dict(name=f"capacity_{i}", type="<", target=0.0, value=value)
            for i, value in enumerate(g)
        ]
        return np.r_[self.objective_gradient @ x, h, g]

    def gradient(self, x):
        """Return the exact stacked objective/equality/inequality Jacobian."""
        return np.vstack((self.objective_gradient, self.jh, self.jg))

    def get_bounds(self):
        """Return explicitly declared physical variable bounds."""
        return self.lower, self.upper

    def get_nec(self):
        """Return the number of equality rows."""
        return self.jh.shape[0]

    def get_nic(self):
        """Return the number of nonpositive inequality rows."""
        return self.jg.shape[0]


def make_solver(problem, **options):
    """Create a quiet SciPy facade for pointwise diagnostic tests."""
    return OptimizationSolver(
        problem,
        library="scipy",
        method="slsqp",
        print_convergence=False,
        plot_convergence=False,
        **options,
    )


def named_problem():
    """Create an analytically solved equality/inequality model at (1, 2)."""
    return AffineProblem(
        [-4, 2],
        lower=[-2, -4],
        upper=[6, 16],
        jh=[[1, 1]],
        h_offset=[-3],
        jg=[[1, 0]],
        g_offset=[-1],
        names=["hot_stream", "cold_stream"],
    )


def test_bound_only_infeasibility_is_visible_in_legacy_report():
    """An infeasible bound fails the report even with no user inequalities."""
    solver = make_solver(AffineProblem([0, 0], lower=[-1, -1], upper=[1, 1]))
    result = solver.evaluate_kkt_conditions([2, 0], 1e-6)
    assert result["first_order_optimality_ok"]
    assert result["feasibility_inequality"] == 1
    assert not result["feasibility_inequality_ok"]
    assert not result["passed"]
    report = solver.make_kkt_optimality_report([2, 0], 1e-6)
    line = next(
        line for line in report.splitlines() if "Inequality feasibility" in line
    )
    assert "+1.000e+00" in line and line.rstrip().endswith("no")


@pytest.mark.parametrize("gradient", [-1.0, 1.0])
def test_fixed_bound_has_valid_nonnegative_legacy_multipliers(gradient):
    """The old unrestricted least-squares false failure is removed for a fixed variable."""
    solver = make_solver(AffineProblem([gradient, 0], lower=[0, -1], upper=[0, 1]))
    result = solver.evaluate_kkt_conditions([0, 0], 1e-8)
    assert result["passed"] and result["feasibility_dual_ok"]
    assert all(mu >= 0 for mu in result["multipliers_ineq"].values())
    assert result["multipliers_ineq"][0] - result["multipliers_ineq"][
        2
    ] == pytest.approx(gradient)


def test_opposing_user_inequalities_do_not_require_unique_duals():
    """The facade accepts a valid nonnegative set for opposing active normals."""
    solver = make_solver(
        AffineProblem(
            [1, 0], lower=[-1, -1], upper=[1, 1], jg=[[1, 0], [-1, 0]], g_offset=[0, 0]
        )
    )
    result = solver.evaluate_kkt_conditions([0, 0], 1e-8)
    assert result["passed"]
    mu = result["multipliers"]["inequality"]
    assert np.all(mu >= 0)
    assert 1 + mu[0] - mu[1] == pytest.approx(0)


@pytest.mark.parametrize("scale", [None, 1.0, 10.0, 100.0])
def test_public_signs_and_variable_scaling_keep_named_constraint_duals(scale):
    """Exact stationarity and physical coordinates survive variable normalization."""
    solver = make_solver(named_problem(), problem_scale=scale)
    point = solver.problem.scale_physical_to_normalized(np.array([1.0, 2.0]))
    result = solver.evaluate_kkt_conditions(point, 1e-8)
    assert result["passed"]
    assert result["coordinates"] == "solver"
    assert result["problem_scale"] == scale
    np.testing.assert_allclose(result["physical_x"], [1, 2])
    np.testing.assert_allclose(result["multipliers_eq"], [-2], atol=1e-8)
    assert result["multipliers_ineq"][0] == pytest.approx(6)
    report = solver.make_lagrange_multipliers_report(point, 1e-8)
    assert "balance_0" in report and "capacity_0" in report
    assert "hot_stream" in report and "cold_stream" in report
    assert "-2.000e+00" in report and "+6.000e+00" in report


def test_bound_duals_are_explicitly_in_solver_coordinate_units():
    """Bound duals transform with dx_physical/dx_solver rather than staying physical."""
    solver = make_solver(
        AffineProblem([3], lower=[2], upper=[10], names=["flow"]), problem_scale=4
    )
    result = solver.evaluate_kkt_conditions([0], 1e-8)
    assert result["passed"]
    assert result["multipliers"]["lower"][0] == pytest.approx(6)
    assert result["multipliers"]["upper"][0] == pytest.approx(0)
    np.testing.assert_array_equal(result["physical_x"], [2])


def test_unavailable_fit_has_no_false_report_certificate(monkeypatch):
    """Unavailable numerical fitting remains explicit and cannot print passing optimality."""
    monkeypatch.setattr(
        kkt,
        "linprog",
        lambda *a, **kw: OptimizeResult(
            success=False, message="Synthetic unavailable diagnostic"
        ),
    )
    solver = make_solver(named_problem())
    result = solver.evaluate_kkt_conditions([1, 2], 1e-8)
    assert not result["available"] and not result["passed"]
    assert not result["first_order_optimality_ok"]
    assert not result["feasibility_dual_ok"]
    report = solver.make_kkt_optimality_report([1, 2], 1e-8)
    assert "Assessment unavailable" in report
    assert "Synthetic unavailable diagnostic" in " ".join(report.split())
    line = next(
        line for line in report.splitlines() if "First order optimality" in line
    )
    assert line.rstrip().endswith("no")
    multipliers = solver.make_lagrange_multipliers_report([1, 2], 1e-8)
    assert "nan" in multipliers


@pytest.mark.parametrize("update_on", ["function", "gradient", "iteration"])
def test_diagnostics_and_named_reports_do_not_publish_history(update_on):
    """Extra diagnostic work never becomes solver iterations or callback events."""
    callbacks = []

    def record(x, count):
        """Remember observable callbacks independently of the stored history."""
        callbacks.append((x.copy(), count))

    solver = make_solver(
        named_problem(), update_on=update_on, callback_functions=record
    )
    solver._print_convergence_progress(np.array([0.0, 3.0]), event_kind="initial")
    before = deepcopy(solver.convergence_history)
    callback_count = len(callbacks)
    for _ in range(2):
        solver.evaluate_kkt_conditions([1, 2], 1e-8)
        solver.make_kkt_optimality_report([1, 2], 1e-8)
        solver.make_lagrange_multipliers_report([1, 2], 1e-8)
        solver.make_constraint_report([1, 2], 1e-8)
    np.testing.assert_equal(solver.convergence_history, before)
    assert len(callbacks) == callback_count
    assert solver.iteration_history == []
    assert solver.kkt_count == 1


def test_kkt_cache_tracks_point_and_tolerance_and_returns_copies():
    """Cached assessments cannot be stale or corrupted through returned dictionaries."""
    solver = make_solver(named_problem())
    first = solver.evaluate_kkt_conditions([1, 2], 1e-8)
    first["multipliers"]["equality"][:] = 123
    again = solver.evaluate_kkt_conditions([1, 2], 1e-8)
    np.testing.assert_allclose(again["multipliers"]["equality"], [-2])
    assert solver.kkt_count == 1
    other = solver.evaluate_kkt_conditions([2, 2], 1e-8)
    assert not other["passed"]
    assert other["feasibility_equality"] == 1
    assert other["feasibility_inequality"] == 1
    assert solver.kkt_count == 2
    solver.evaluate_kkt_conditions([2, 2], 1e-6)
    assert solver.kkt_count == 3


def test_unbounded_multiplier_slots_are_not_printed_as_real_constraints():
    """Absent bounds retain internal zero slots but do not clutter named reports."""
    solver = make_solver(
        AffineProblem(
            [0, 0], lower=[-np.inf, -1], upper=[np.inf, 1], names=["free", "bounded"]
        )
    )
    result = solver.evaluate_kkt_conditions([0, 0], 1e-8)
    assert set(result["multipliers_ineq"]) == {1, 3}
    report = solver.make_lagrange_multipliers_report([0, 0], 1e-8)
    assert "free" not in report
    assert report.count("bounded") == 2
