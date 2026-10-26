"""Audit the heat demo's shared equations, root acceptance and recorded histories."""

import json
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from demos.demo_eqn_solving import demo_radiative_heat_collocation as original
from demos.demo_eqn_solving import demo_radiative_heat_slsqp as demo


@pytest.fixture(scope="module")
def comparison():
    """Solve a fixed 33-node model and its independent adaptive BVP reference."""
    problem = original.RadiativeHeatProblem(degree=32)
    reference = original.solve_reference(problem)
    runs = demo.solve_comparison(problem, print_convergence=False)
    return problem, reference, runs


def _stored_state(runs):
    """Serialize detached backend histories and final points for mutation checks."""
    return json.dumps(
        demo.json_value(
            [
                dict(
                    history=run["solver"].backend_result.history,
                    x=run["solver"].result.x,
                    nfev=run["solver"].result.nfev,
                    njev=run["solver"].result.njev,
                )
                for run in runs
            ]
        ),
        sort_keys=True,
        allow_nan=False,
    )


def test_original_model_and_common_start_with_requested_step_policies(comparison):
    """Compare the prescribed solver controls without replacing the physical model."""
    problem, _, runs = comparison
    assert demo.RadiativeHeatProblem is original.RadiativeHeatProblem
    assert demo.solve_reference is original.solve_reference
    assert demo.NUM_POINTS == original.NUM_POINTS
    assert len(runs) == 8
    assert [run["fraction"] for run in runs] == [1.0, 0.3, 0.2, 0.1] * 2
    initial = problem.initial_guess(0.3)
    assert runs[0]["solver"].problem is problem
    assert runs[0]["solver"].result.effective_options["backtracking"] is None
    for run in runs:
        solver = run["solver"]
        np.testing.assert_array_equal(solver.backend_result.history[0]["x"], initial)
        if run["method"] == "newton":
            assert solver.problem is problem
            assert solver.result.effective_options["backtracking"] is None
            first = solver.backend_result.history[1]["x"]
            direction = np.linalg.solve(
                problem.gradient(initial), -problem.residual(initial)
            )
            np.testing.assert_allclose(
                first - initial, run["fraction"] * direction, atol=1e-12
            )
        if run["method"] == "slsqp":
            options = solver.result.effective_options
            assert options["first_step_fraction"] == run["fraction"]
            assert options["max_step_fraction"] == 1.0
            assert options["step_growth_factor"] == 1.5
            assert options["domain_backtracking"] is True
            assert solver.problem.root_problem is problem
            assert solver.problem.get_nec() == initial.size
            assert solver.problem.get_nic() == 0
            assert solver.problem.problem_scale is None
            np.testing.assert_array_equal(
                solver.problem.fitness(initial)[1:], problem.residual(initial)
            )
    assert not hasattr(problem, "problem_scale")


def test_all_final_roots_match_original_equations_and_independent_fields(comparison):
    """Require small equation errors and physical agreement away from mesh nodes."""
    problem, reference, runs = comparison
    positions = np.linspace(0.0, 1.0, 237)
    expected_temperature, expected_slope = reference.sol(positions)
    for run in runs:
        result = run["solver"].result
        assert result.success, result.message
        assert np.linalg.norm(problem.residual(result.x)) <= 1e-8
        temperature, flux = problem.interpolate(result.x, positions)
        assert np.min(temperature) > 0
        np.testing.assert_allclose(temperature, expected_temperature, atol=1e-7, rtol=0)
        np.testing.assert_allclose(
            flux, -expected_slope / problem.heating, atol=1e-7, rtol=0
        )


def test_accepted_history_preserves_norms_counts_steps_and_detachment(
    comparison, monkeypatch
):
    """Extract cached diagnostics without hidden work or relabeling backend events."""
    problem, _, runs = comparison
    for run in runs:
        source = run["solver"].backend_result.history
        expected_norms = [np.linalg.norm(problem.residual(row["x"])) for row in source]
        with monkeypatch.context() as patcher:
            patcher.setattr(
                problem, "residual", Mock(side_effect=AssertionError("probe"))
            )
            patcher.setattr(
                problem, "gradient", Mock(side_effect=AssertionError("probe"))
            )
            records = demo.accepted_history(run)
        assert [row["accepted_move"] for row in records] == list(range(len(source)))
        np.testing.assert_allclose(
            [row["residual_norm"] for row in records],
            expected_norms,
            rtol=1e-12,
            atol=0,
        )
        for index, (record, original_record) in enumerate(zip(records, source)):
            assert record["backend_iteration"] == original_record["iteration"]
            assert record["nfev"] == original_record["nfev"]
            assert record["njev"] == original_record["njev"]
            np.testing.assert_array_equal(record["x"], original_record["x"])
            expected_step = (
                0.0
                if index == 0
                else np.linalg.norm(record["x"] - records[index - 1]["x"])
            )
            assert record["step_norm"] == pytest.approx(expected_step, abs=1e-14)
        assert np.all(np.diff([row["nfev"] for row in records]) >= 0)
        assert records[-1]["nfev"] <= run["solver"].result.nfev
        original_point = source[0]["x"].copy()
        records[0]["x"][:] = -99.0
        np.testing.assert_array_equal(source[0]["x"], original_point)


def test_postsolve_audits_are_explicit_and_plots_leave_histories_unchanged(
    comparison, monkeypatch
):
    """Count fresh final checks separately while keeping plot rendering passive."""
    problem, reference, runs = comparison
    before = _stored_state(runs)
    with monkeypatch.context() as patcher:
        residual = Mock(wraps=problem.residual)
        patcher.setattr(problem, "residual", residual)
        rows = demo.summarize_runs(problem, runs, reference)
        assert residual.call_count == len(runs)
        assert all(row["root_found"] for row in rows)
        for run, row in zip(runs, rows):
            assert row["residual_calls"] == run["solver"].result.nfev
            assert row["residual_norm"] <= row["residual_tolerance"]
            assert row["max_residual"] <= row["residual_norm"]
            assert row["accepted_moves"] == len(demo.accepted_history(run)) - 1
        patcher.setattr(problem, "residual", Mock(side_effect=AssertionError("probe")))
        patcher.setattr(problem, "gradient", Mock(side_effect=AssertionError("probe")))
        plt.switch_backend("Agg")
        figures = demo.plot_comparison(problem, runs, rows, reference, 0.3, 1e-8)
        try:
            for axis in figures[0].axes:
                curves = axis.lines[:16:2]
                points = axis.lines[1:16:2]
                assert len(curves) == 8
                for index, (curve, marker) in enumerate(zip(curves, points)):
                    assert curve.get_linestyle() == ("-" if index < 4 else "--")
                    assert marker.get_marker() == ("o" if index < 4 else "s")
                    np.testing.assert_array_equal(
                        curve.get_color(), curves[index % 4].get_color()
                    )
            for figure in figures:
                figure.canvas.draw()
        finally:
            for figure in figures:
                plt.close(figure)
    assert _stored_state(runs) == before


def test_iteration_limited_zero_objective_is_not_reported_as_a_root(comparison):
    """A stationary zero objective cannot disguise unsatisfied heat equations."""
    problem, reference, _ = comparison
    runs = demo.solve_comparison(problem, max_iterations=1, print_convergence=False)
    rows = demo.summarize_runs(problem, runs, reference)
    assert len(rows) == 8
    for run, row in zip(runs, rows):
        assert not row["root_found"]
        assert row["residual_norm"] > row["residual_tolerance"]
        assert not row["solver_success"]
        if run["method"] == "slsqp":
            assert run["solver"].result.fun == 0.0
