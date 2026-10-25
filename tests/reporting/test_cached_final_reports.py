"""Final custom-optimizer reports render the solved certificate without model work."""

from copy import deepcopy

import numpy as np
import pytest

from pysolver_view import DomainError, OptimizationProblem, OptimizationSolver
from pysolver_view.diagnostics import kkt


class NamedMixedProblem(OptimizationProblem):
    """A convex physical model with known signed and bound multipliers."""

    def __init__(self):
        """Keep independent model-call counts and descriptive physical names."""
        self.calls = 0
        self.variable_names = ["hot_stream", "cold_stream", "reserve"]

    def fitness(self, x):
        """Return a quadratic objective, delivery equality, and capacity limit."""
        self.calls += 1
        self.constraint_data = [
            dict(name="delivery_balance", type="=", target=3.0, value=x[0] + x[1]),
            dict(name="hot_capacity", type="<", target=1.0, value=x[0]),
        ]
        return np.array(
            [(x[0] - 2) ** 2 + x[1] ** 2 + (x[2] + 1) ** 2, x[0] + x[1] - 3, x[0] - 1]
        )

    def gradient(self, x):
        """Provide exact physical Jacobian rows for objective and constraints."""
        return np.array(
            [[2 * (x[0] - 2), 2 * x[1], 2 * (x[2] + 1)], [1, 1, 0], [1, 0, 0]]
        )

    def get_bounds(self):
        """Make the reserve's lower bound active at the known optimum."""
        return [-10, -10, 0], [10, 10, 10]

    def get_nec(self):
        """Declare the delivery equality."""
        return 1

    def get_nic(self):
        """Declare the nonpositive capacity residual."""
        return 1


def solved_named_problem(**options):
    """Solve with deliberately distinct feasibility and optimality tolerances."""
    solver = OptimizationSolver(
        NamedMixedProblem(),
        library="pysolver",
        print_convergence=False,
        tolerance=1e-10,
        max_iterations=100,
        extra_options={
            "first_step_fraction": 1.0,
            "feasibility_tol": 1e-6,
            "optimality_tol": 1e-4,
        },
        **options,
    )
    solver.solve([0.5, 2.5, 0.5])
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.result.x, [1.0, 2.0, 0.0], atol=2e-6)
    return solver


def print_full_report(solver, **options):
    """Request all existing report sections through the public frontend API."""
    solver.print_optimization_report(
        include_design_variables=True,
        include_constraints=True,
        include_kkt_conditions=True,
        include_multipliers=True,
        **options,
    )


def forbid_diagnostic_work(monkeypatch, solver):
    """Reject model, derivative, and multiplier work during passive reporting."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Final reporting must use the solved certificate")

    monkeypatch.setattr(solver.problem, "fitness", forbidden)
    monkeypatch.setattr(solver.problem, "gradient", forbidden, raising=False)
    monkeypatch.setattr(kkt, "linprog", forbidden)


def test_final_full_report_is_passive_and_keeps_public_signed_names(
    monkeypatch, capsys, tmp_path
):
    """Render original physical metadata and public multipliers without new work."""
    solver = solved_named_problem()
    np.testing.assert_allclose(solver.result.kkt["multipliers"]["equality"], [-4.0])
    before = deepcopy(solver.convergence_history)
    native_before = deepcopy(solver.backend_result.history)
    result_before = deepcopy(solver.result)
    counters = solver.func_count_tot, solver.grad_count, solver.kkt_count
    calls = solver.problem.calls
    forbid_diagnostic_work(monkeypatch, solver)

    print_full_report(
        solver, savefile=True, filename="optimization_report.txt", output_dir=tmp_path
    )
    report = capsys.readouterr().out
    assert report == (tmp_path / "optimization_report.txt").read_text() + "\n"
    for section in (
        "Optimization variables report",
        "Optimization constraints report",
        "Karush-Kuhn-Tucker (KKT) conditions",
        "Lagrange multipliers report",
    ):
        assert section in report
    for name in (
        "hot_stream",
        "cold_stream",
        "reserve",
        "delivery_balance",
        "hot_capacity",
    ):
        assert name in report
    equality_line = next(
        line
        for line in report.splitlines()
        if "delivery_balance" in line and "equality" in line
    )
    assert "-4.000e+00" in equality_line
    assert solver.problem.calls == calls
    assert (solver.func_count_tot, solver.grad_count, solver.kkt_count) == counters
    np.testing.assert_equal(solver.convergence_history, before)
    np.testing.assert_equal(solver.backend_result.history, native_before)
    np.testing.assert_equal(solver.result, result_before)


def test_final_report_preserves_separate_acceptance_tolerances(capsys):
    """An accepted residual between both tolerances must not be relabeled failed."""
    solver = solved_named_problem()
    assessment = solver.result.kkt
    assessment.update(
        stationarity=3e-6,
        complementarity=2e-6,
        optimality=3e-6,
        equality_feasibility=5e-7,
        inequality_feasibility=0.0,
        bound_feasibility=0.0,
        primal_feasibility=5e-7,
        kkt_residual=3e-6,
        kkt_ratio=0.5,
        passed=True,
    )
    print_full_report(solver)
    report = capsys.readouterr().out
    for name, value, target in (
        ("First order optimality", "+3.000e-06", "+1.000e-04"),
        ("Stationarity", "+3.000e-06", "+1.000e-04"),
        ("Equality feasibility", "+5.000e-07", "+1.000e-06"),
        ("Complementary slackness", "+2.000e-06", "+1.000e-04"),
    ):
        line = next(line for line in report.splitlines() if name in line)
        assert value in line and target in line
        assert line.rstrip().endswith("yes")


def test_final_report_metadata_survives_later_diagnostic_points(monkeypatch, capsys):
    """Reports of the solved design cannot borrow metadata from a later audit."""
    solver = solved_named_problem()
    final = solver.result.x.copy()
    other = solver.problem.scale_physical_to_normalized(final + [0.3, 0.4, 0.0])
    solver.evaluate_kkt_conditions(other, 1e-6)
    solver.problem.constraint_data[0]["name"] = "mutated_later_metadata"
    solver.problem.constraint_data[0]["value"] = 123.0
    solver.problem.variable_names = ["mutated_name"] * 3
    solver.problem.problem_scale = 10.0
    forbid_diagnostic_work(monkeypatch, solver)

    print_full_report(solver)
    report = capsys.readouterr().out
    assert "mutated_later_metadata" not in report
    assert "mutated_name" not in report
    assert "hot_stream" in report and "normalized values" not in report
    equality = next(
        line
        for line in report.splitlines()
        if "delivery_balance" in line and "equality" not in line
    )
    assert "+3.00e+00" in equality
    assert "+1.23e+02" not in report
    np.testing.assert_array_equal(solver.result.x, final)


@pytest.mark.parametrize("failure", ["disabled", "probe"])
def test_default_final_report_does_not_retry_unavailable_diagnostics(
    failure, monkeypatch, capsys
):
    """Disabled KKT and controlled derivative failures retain honest final reports."""
    if failure == "disabled":
        solver = solved_named_problem(check_kkt=False)
    else:

        class ProbeFailure(OptimizationProblem):
            """Reject the first finite-difference probe beyond its initial point."""

            def __init__(self):
                """Retain the named model's call counter and physical ordering."""
                self.calls = 0
                self.variable_names = ["hot_stream", "cold_stream", "reserve"]

            def fitness(self, x):
                """Permit the initial design but reject its first derivative probe."""
                if x[0] > 0.5:
                    raise DomainError(
                        "The derivative sample is outside the model domain"
                    )
                return NamedMixedProblem.fitness(self, x)

            def get_bounds(self):
                """Use the original physical bounds without analytic derivatives."""
                return [-10, -10, 0], [10, 10, 10]

            def get_nec(self):
                """Declare the delivery equality."""
                return 1

            def get_nic(self):
                """Declare the capacity inequality."""
                return 1

        problem = ProbeFailure()
        solver = OptimizationSolver(
            problem, library="pysolver", print_convergence=False
        )
        solver.solve([0.5, 2.5, 0.5])
        assert not solver.success and solver.result.status == 12
    counters = solver.func_count_tot, solver.grad_count, solver.kkt_count
    forbid_diagnostic_work(monkeypatch, solver)

    print_full_report(solver)
    report = capsys.readouterr().out.lower()
    assert "unavailable" in report or "not assessed" in report or "disabled" in report
    assert (solver.func_count_tot, solver.grad_count, solver.kkt_count) == counters


def test_explicit_point_and_tolerance_reports_remain_diagnostic():
    """Explicit audit requests still compute their requested certificate."""
    solver = solved_named_problem(check_kkt=False)
    assert solver.kkt_count == 0
    print_full_report(solver, tol=1e-6, to_console=False)
    assert solver.kkt_count == 1
    before = solver.func_count_tot, solver.kkt_count
    print_full_report(
        solver, x=solver.result.x + [0.3, 0.4, 0.0], tol=1e-6, to_console=False
    )
    assert solver.func_count_tot > before[0]
    assert solver.kkt_count > before[1]
    assert solver.result.kkt_passed is None
