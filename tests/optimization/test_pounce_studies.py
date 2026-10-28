"""Verify native/wrapped NLP fidelity and actual first-step shortening."""

import numpy as np
import pytest

from pysolver_view.optimization.problems import CoupledRosenbrockProblem
from development.pounce_solvers.run_native_solver_comparison import (
    run_custom,
    run_native,
)
from development.pounce_solvers.run_wrapper_solver_comparison import (
    compare_native_reference,
    run_wrapped,
)


@pytest.mark.parametrize("method", ["ipm", "sqp"])
def test_moderate_study_wrapper_preserves_native_solution_and_observations(method):
    """Solve the actual study NLP and check original KKT plus interface fidelity."""
    pytest.importorskip("pounce")
    problem = CoupledRosenbrockProblem()
    native = run_native(problem, method, 1e-8, 300)
    wrapped = run_wrapped(problem, method, 1e-8, 300)
    assert native["success"] and native["audit"]["kkt_passed"]
    assert wrapped["success"] and wrapped["audit"]["kkt_passed"]
    assert compare_native_reference(wrapped, native)["passed"]
    assert all(row["accepted"] is None for row in wrapped["history"])
    assert all(
        row["x"] is None or row["coordinate_verified"] for row in wrapped["history"]
    )


def test_study_first_fraction_shortens_actual_trial_then_restores_full_cap():
    """Verify evaluated trial coordinates and independently accepted final points."""
    problem = CoupledRosenbrockProblem(blocks=3)
    full = run_custom(problem, 1.0, 1e-12, 300)
    shortened = run_custom(problem, 0.20, 1e-12, 300)
    assert full["success"] and full["audit"]["kkt_passed"]
    assert shortened["success"] and shortened["audit"]["kkt_passed"]
    assert not [event for event in full["events"] if event["kind"] == "step_cap"]
    caps = [event for event in shortened["events"] if event["kind"] == "step_cap"]
    assert len(caps) == 1
    cap = caps[0]
    assert cap["iteration"] == 1
    assert cap["fraction"] == pytest.approx(0.20)
    np.testing.assert_allclose(
        cap["x"] - problem.initial,
        0.20 * (cap["original_x"] - problem.initial),
        atol=1e-14,
        rtol=1e-13,
    )
    assert shortened["options"]["step_growth_factor"] is None
