"""Validate named model adapters and reports against an analytic constrained solve."""

import numpy as np
import pytest

import pysolver_view as psv
from pysolver_view.diagnostics import kkt
from demos.demo_kkt_conditions.demo_named_kkt import (
    NamedAllocationProblem,
    check_known_solution,
)


def test_dictionary_input_uses_declared_order_and_rejects_missing_names():
    """Application dictionary order must not change the numerical variable order."""
    problem = NamedAllocationProblem()
    design = {"reserve": 0.4, "stream_b": 0.8, "stream_a": 1.2}
    packed = problem.pack(design)
    np.testing.assert_array_equal(packed, [1.2, 0.8, 0.4])
    assert problem.unpack(packed) == design
    with pytest.raises(ValueError, match="exactly"):
        problem.pack({"stream_a": 1.2, "stream_b": 0.8})
    with pytest.raises(ValueError, match="exactly"):
        problem.pack({**design, "unexpected": 0.0})


@pytest.mark.parametrize("library", ["scipy", "pysolver"])
@pytest.mark.parametrize("scale", [None, 1.0])
def test_named_reports_match_independent_solution_and_multipliers(library, scale):
    """Both adapters preserve names and the known equality/inequality/bound signs."""
    problem = NamedAllocationProblem()
    solver = psv.OptimizationSolver(
        problem,
        library=library,
        method="slsqp",
        tolerance=1e-10,
        max_iterations=200,
        print_convergence=False,
        plot_convergence=False,
        problem_scale=scale,
    )
    solver.solve(problem.pack({"stream_a": 0.5, "stream_b": 1.5, "reserve": 0.6}))
    check_known_solution(solver)
    point = solver.x_final_norm
    variables = solver.make_variables_report(solver.x_final, normalized=False)
    constraints = solver.make_constraint_report(point, tol=1e-6)
    multipliers = solver.make_lagrange_multipliers_report(point, tol=1e-6)
    for name in problem.variable_names:
        assert name in variables
        assert name in multipliers
    for name in ["total_delivery", "stream_a_capacity", "stream_b_capacity"]:
        assert name in constraints
        assert name in multipliers
    assert "design_variable_" not in variables + multipliers
    assert "equality_constraint_" not in constraints + multipliers
    assert "inequality_constraint_" not in constraints + multipliers


def test_named_constraint_report_uses_requested_tolerance():
    """Physical display metadata must not override the requested residual tolerance."""
    solver = psv.OptimizationSolver(
        NamedAllocationProblem(),
        library="scipy",
        method="slsqp",
        print_convergence=False,
        plot_convergence=False,
    )
    # Model metadata uses 1e-6, but the requested report uses a stricter tolerance.
    point = np.array([1.2 + 1e-8, 0.8 - 1e-8, 0.0])
    entries = solver.get_constraint_data(point, tol=1e-10)
    by_name = {entry["name"]: entry for entry in entries}
    assert by_name["total_delivery"]["satisfied"]
    assert not by_name["stream_a_capacity"]["satisfied"]
    assert by_name["stream_b_capacity"]["satisfied"]


@pytest.mark.parametrize("to_console", [False, True])
def test_explicit_report_output_is_independent_of_live_progress(to_console, capsys):
    """A quiet solve can explicitly print named KKT and multiplier reports."""
    solver = psv.OptimizationSolver(
        NamedAllocationProblem(),
        library="scipy",
        method="slsqp",
        print_convergence=False,
        plot_convergence=False,
    )
    solver.solve(np.array([0.5, 1.5, 0.6]))
    assert capsys.readouterr().out == ""
    solver.print_optimization_report(
        include_kkt_conditions=True,
        include_multipliers=True,
        to_console=to_console,
    )
    output = capsys.readouterr().out
    if to_console:
        assert "stream_a" in output
        assert "total_delivery" in output
        assert "Karush-Kuhn-Tucker (KKT)" in output
        assert "Lagrange multipliers" in output
    else:
        assert output == ""


def test_custom_history_reuses_the_driver_multiplier_fit(monkeypatch):
    """Accepted-point reporting must reuse rather than repeat the driver's LP."""
    calls = []
    original_linprog = kkt.linprog

    def count_fit(*args, **kwargs):
        """Count actual multiplier fits across the driver and report facade."""
        calls.append(1)
        return original_linprog(*args, **kwargs)

    monkeypatch.setattr(kkt, "linprog", count_fit)
    solver = psv.OptimizationSolver(
        NamedAllocationProblem(),
        library="pysolver",
        method="slsqp",
        tolerance=1e-9,
        max_iterations=200,
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
    )
    solver.solve(np.array([0.5, 1.5, 0.6]))
    assert solver.success
    assert len(calls) == len(solver.backend_result.history)


@pytest.mark.parametrize("stop", ["initial_callback", "iteration_limit"])
def test_public_driver_complementarity_products_use_nonpositive_inequalities(stop):
    """Nonzero products reveal sign errors hidden by a converged zero residual."""

    def stop_initial(point, count):
        """Cancel at the initial accepted point to retain nonzero KKT residuals."""
        return False

    problem = NamedAllocationProblem()
    solver = psv.OptimizationSolver(
        problem,
        library="pysolver",
        method="slsqp",
        tolerance=1e-9,
        max_iterations=1 if stop == "iteration_limit" else 200,
        callback_functions=stop_initial if stop == "initial_callback" else None,
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
    )
    solver.solve(np.array([0.5, 1.5, 0.6]))
    assert not solver.success
    assert solver.result.status == (13 if stop == "initial_callback" else 9)
    public = solver.result.kkt
    inequalities = problem.fitness(solver.x_final)[2:]
    expected = public["multipliers"]["inequality"] * inequalities
    assert np.max(np.abs(expected)) > 1e-3
    np.testing.assert_allclose(
        public["complementarity_products"]["inequality"], expected, atol=1e-12
    )
    for entry in solver.iteration_history:
        products = entry["kkt"]["complementarity_products"]["inequality"]
        multipliers = entry["kkt"]["multipliers"]["inequality"]
        expected = multipliers * problem.fitness(entry["x"])[2:]
        np.testing.assert_allclose(products, expected, atol=1e-12)
