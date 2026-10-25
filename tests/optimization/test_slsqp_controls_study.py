"""Check independent study mathematics and meaningful step-control behaviors."""

import numpy as np
import pytest

from pysolver_view.optimization.problems import StudyProblem
from pysolver_view.diagnostics.slsqp_audit import reconstruct, fresh_jacobian
from demos.demo_slsqp_controls.run_sensitivity import configurations, run_case


@pytest.mark.parametrize(
    "case", ["rosenbrock", "constrained_rosenbrock", "square_root", "inconsistent"]
)
def test_analytic_jacobian_matches_independent_samples(case):
    """Verify every study objective/constraint derivative before comparing solvers."""
    problem = StudyProblem(case)
    np.testing.assert_allclose(
        fresh_jacobian(problem, problem.initial, 1e-5),
        problem.gradient(problem.initial),
        atol=1e-6,
        rtol=1e-7,
    )
    assert problem.calls == []


def test_known_constrained_solution_and_multiplier():
    """Derive the boundary optimum from its cubic and reconstruct exact KKT."""
    roots = np.roots([400.0, 600.0, -198.0, -202.0])
    candidates = [r.real for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1]
    assert len(candidates) == 1
    a = candidates[0]
    point = np.array([a, 1 - a])
    problem = StudyProblem("constrained_rosenbrock")
    mu = -problem.gradient(point)[0, 1]
    assert mu > 0
    metrics = reconstruct(
        problem,
        point,
        dict(inequality=np.array([mu]), lower=np.zeros(2), upper=np.zeros(2)),
    )
    assert metrics["optimality"] < 1e-10
    assert metrics["primal_feasibility"] < 1e-12
    wrong = reconstruct(
        problem,
        point,
        dict(inequality=np.array([-mu]), lower=np.zeros(2), upper=np.zeros(2)),
    )
    assert wrong["dual_feasibility"] > 0.1


def test_first_fraction_reduces_actual_trial_without_faking_kkt():
    """Check evaluated coordinates, not just the configured fraction label."""
    profiles = {p["name"]: p for p in configurations()}
    full = run_case("rosenbrock", profiles["first_1"])
    short = run_case("rosenbrock", profiles["first_0.1"])
    assert full["first_trial_distance"] > 200
    assert short["first_trial_distance"] == pytest.approx(
        0.1 * full["first_trial_distance"]
    )
    assert short["kkt_passed"] and short["audit"]["passed"]


def test_domain_recovery_and_native_failure_remain_visible():
    """Undefined native trials fail visibly; recovery reaches the known 2D minimum."""
    profiles = {p["name"]: p for p in configurations()}
    native = run_case("square_root", profiles["scipy"])
    recovered = run_case(
        "square_root",
        dict(
            name="recovery_check",
            library="pysolver",
            group="test",
            controls=dict(
                first_step_fraction=1.0,
                max_step_fraction=1.0,
                step_growth_factor=None,
                domain_backtracking=True,
                backtrack_factor=0.5,
                max_backtracks=20,
            ),
        ),
    )
    assert not native["success"] and native["invalid_trials"] == 1
    assert native["kkt_ratio"] is None
    assert recovered["success"] and recovered["kkt_passed"]
    np.testing.assert_allclose(recovered["x"], [0.04, 0.4], atol=2e-6)
    assert recovered["invalid_trials"] > 0
