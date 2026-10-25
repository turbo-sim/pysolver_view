"""Regression checks for optional automatic KKT assessment and honest acceptance."""

import numpy as np
import pytest
from pysolver_view import OptimizationSolver
from pysolver_view.diagnostics import kkt
from demos.demo_kkt_conditions.demo_named_kkt import NamedAllocationProblem


@pytest.mark.parametrize("library", ["scipy", "pysolver"])
def test_disabled_checks_do_not_run_multiplier_estimation(library, monkeypatch):
    """Disable automatic assessments while retaining an explicit on-demand audit."""

    def forbidden(*args, **kwargs):
        """Fail if solve or its reports silently invoke the LP estimator."""
        raise AssertionError("Unexpected KKT estimation")

    with monkeypatch.context() as patch:
        patch.setattr(kkt, "linprog", forbidden)
        solver = OptimizationSolver(
            NamedAllocationProblem(),
            library=library,
            check_kkt=False,
            print_convergence=False,
        )
        solver.solve(np.array([0.5, 1.5, 0.6]))
    assert solver.success
    assert solver.result.kkt_time == 0
    assert solver.result.kkt_evaluations == 0
    assert solver.result.kkt_passed is None
    assert np.all(np.isnan(solver.convergence_history["optimality"]))
    history_length = len(solver.convergence_history["x"])
    assessment = solver.evaluate_kkt_conditions(solver.x_final_norm, 1e-5)
    assert assessment["passed"]
    assert len(solver.convergence_history["x"]) == history_length


@pytest.mark.parametrize("check", [False, True])
def test_tiny_step_acceptance_explicitly_depends_on_kkt_policy(check):
    """Native small-step success is unverified when the KKT gate is disabled."""
    solver = OptimizationSolver(
        NamedAllocationProblem(),
        library="pysolver",
        check_kkt=check,
        print_convergence=False,
        tolerance=1e-5,
        extra_options={"first_step_fraction": 1e-12, "max_step_fraction": 1e-12},
    )
    solver.solve(np.array([0.5, 1.5, 0.6]))
    assert solver.result.solver_success
    assert solver.success is (not check)
    assert solver.result.kkt_passed == (False if check else None)
    assert (solver.result.kkt_evaluations > 0) == check


def test_check_option_requires_a_boolean():
    """Avoid treating strings such as 'false' as truthy activation requests."""
    with pytest.raises(TypeError, match="check_kkt"):
        OptimizationSolver(NamedAllocationProblem(), check_kkt="false")
    with pytest.raises(ValueError, match="directly"):
        OptimizationSolver(NamedAllocationProblem(), extra_options={"check_kkt": False})
