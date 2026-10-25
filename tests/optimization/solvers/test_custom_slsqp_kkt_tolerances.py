"""Exercise production SLSQP acceptance against independent analytic residuals.

Native ``ftol`` can accept a point before an independently requested feasibility
or first-order tolerance is met. These small problems deliberately expose that
case, including the inclusive pass threshold and finite-difference accuracy.
"""

import re

import numpy as np
import pytest

from pysolver_view.solvers.slsqp import Evaluation, SLSQPOptions, minimize_slsqp


def assert_report_consistent(result, *, feasibility_tol, optimality_tol):
    """Check the production final result and every accepted history record."""
    assert result.diagnostics_available
    assert result.kkt["available"]
    assert result.feasible == (result.max_constraint_violation <= feasibility_tol)
    assert result.success == (
        result.solver_success
        and result.feasible
        and result.optimality <= optimality_tol
    )
    assert result.kkt["passed"] == (
        result.feasible and result.optimality <= optimality_tol
    )
    assert result.kkt_ratio == pytest.approx(
        max(
            result.max_constraint_violation / feasibility_tol,
            result.optimality / optimality_tol,
        )
    )
    assert result.kkt["primal_feasibility"] == result.max_constraint_violation
    assert result.history[-1]["kkt_ratio"] == result.kkt_ratio
    assert result.history[-1]["optimality"] == result.optimality

    previous = None
    for record in result.history:
        check = record["kkt"]
        assert record["diagnostics_available"]
        assert check["feasibility_tol"] == feasibility_tol
        assert check["optimality_tol"] == optimality_tol
        assert record["optimality"] == max(
            check["stationarity"],
            check["dual_feasibility"],
            check["complementarity"],
        )
        assert record["kkt_ratio"] == pytest.approx(
            max(
                check["primal_feasibility"] / feasibility_tol,
                check["optimality"] / optimality_tol,
            )
        )
        assert check["passed"] == (record["kkt_ratio"] <= 1.0)
        expected_step = (
            0.0 if previous is None else np.linalg.norm(record["x"] - previous)
        )
        assert record["step_norm"] == pytest.approx(expected_step)
        previous = record["x"]


@pytest.mark.parametrize("kind", ["equality", "inequality"])
@pytest.mark.parametrize("ftol", [1e-1, 1e-3, 1e-7])
def test_native_success_is_independently_gated_by_feasibility(kind, ftol):
    # Newton's first step from 1.1 does not solve x**2 == 1 exactly. The
    # constant objective means stationarity is zero even while infeasible.
    def evaluate(x):
        constraint = x[0] ** 2 - 1.0
        return Evaluation(
            0.0,
            equalities=[constraint] if kind == "equality" else [],
            inequalities=[-constraint] if kind == "inequality" else [],
        )

    def jacobian(x):
        derivative = 2.0 * x[0]
        return np.array([[0.0], [derivative if kind == "equality" else -derivative]])

    def solve(tolerance):
        return minimize_slsqp(
            evaluate,
            [1.1],
            jac=jacobian,
            options=SLSQPOptions(
                ftol=ftol, feasibility_tol=tolerance, optimality_tol=1e-12
            ),
        )

    reference = solve(1.0)
    assert reference.status == 0
    independently_computed = abs(reference.x[0] ** 2 - 1.0)
    assert independently_computed > 0.0
    assert reference.max_constraint_violation == independently_computed
    assert reference.optimality == 0.0

    for multiplier, expected_success in [(0.5, False), (1.0, True), (2.0, True)]:
        tolerance = multiplier * independently_computed
        result = solve(tolerance)
        assert result.status == 0
        assert result.solver_success
        assert bool(result.success) is expected_success
        assert result.feasible is expected_success
        np.testing.assert_array_equal(result.x, reference.x)
        assert_report_consistent(
            result, feasibility_tol=tolerance, optimality_tol=1e-12
        )
        if not expected_success:
            assert "constraint violation" in result.message
        for record in result.history:
            assert record["kkt"]["primal_feasibility"] == pytest.approx(
                abs(record["x"][0] ** 2 - 1.0)
            )


@pytest.mark.parametrize("initial", [-0.7, 1.0, 2.0])
@pytest.mark.parametrize("ftol", [1e-1, 1e-3, 1e-7])
def test_native_success_is_independently_gated_by_optimality(initial, ftol):
    def solve(tolerance):
        return minimize_slsqp(
            lambda x: Evaluation(x[0] ** 4),
            [initial],
            jac=lambda x: np.array([[4.0 * x[0] ** 3]]),
            options=SLSQPOptions(
                ftol=ftol,
                feasibility_tol=1e-12,
                optimality_tol=tolerance,
                # Keep the trajectory independent of the default quarter cap,
                # which lands exactly at the quartic optimum from x=1.
                first_step_fraction=1.0,
                step_growth_factor=None,
            ),
        )

    reference = solve(100.0)
    assert reference.status == 0
    independently_computed = abs(4.0 * reference.x[0] ** 3)
    assert independently_computed > 0.0
    assert reference.optimality == independently_computed
    assert reference.max_constraint_violation == 0.0

    for multiplier, expected_success in [(0.5, False), (1.0, True), (2.0, True)]:
        tolerance = multiplier * independently_computed
        result = solve(tolerance)
        assert result.status == 0
        assert result.solver_success
        assert bool(result.success) is expected_success
        assert result.feasible
        np.testing.assert_array_equal(result.x, reference.x)
        np.testing.assert_array_equal(
            [record["x"] for record in result.history],
            [record["x"] for record in reference.history],
        )
        assert_report_consistent(
            result, feasibility_tol=1e-12, optimality_tol=tolerance
        )
        if not expected_success:
            assert "first-order residual" in result.message
        for record in result.history:
            assert record["optimality"] == abs(4.0 * record["x"][0] ** 3)


@pytest.mark.parametrize("initial", [[1.5, -0.5, -0.5, -0.5, 1.5], [0, 1, 1, 1, -1]])
@pytest.mark.parametrize("equality_sign", [-1.0, 1.0])
@pytest.mark.parametrize("tolerance", [1e-4, 1e-7, 1e-10])
def test_mixed_constraints_recover_correct_signed_multipliers(
    initial, equality_sign, tolerance
):
    # Each active normal has a separate coordinate, making the multipliers
    # unique: lower=1, upper=1, inequality=2, equality=+/-3.
    def evaluate(x):
        return Evaluation(
            x[0] - x[1] + 2.0 * x[2] + equality_sign * 3.0 * x[3] + (x[4] - 0.3) ** 2,
            equalities=[x[3] - 0.4],
            inequalities=[x[2] - 0.25, 2.0 - x[4]],
        )

    def jacobian(x):
        return np.array(
            [
                [1.0, -1.0, 2.0, equality_sign * 3.0, 2.0 * (x[4] - 0.3)],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )

    result = minimize_slsqp(
        evaluate,
        initial,
        bounds=[(0.0, 2.0), (-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        jac=jacobian,
        options=SLSQPOptions(
            ftol=1e-12, feasibility_tol=tolerance, optimality_tol=tolerance
        ),
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [0.0, 1.0, 0.25, 0.4, 0.3], atol=1e-10, rtol=0)
    multipliers = result.kkt_multipliers
    np.testing.assert_allclose(multipliers["lower"], [1, 0, 0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(multipliers["upper"], [0, 1, 0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(multipliers["inequality"], [2, 0], atol=1e-10)
    np.testing.assert_allclose(multipliers["equality"], [equality_sign * 3], atol=1e-10)
    assert_report_consistent(
        result, feasibility_tol=tolerance, optimality_tol=tolerance
    )


@pytest.mark.parametrize("difference_step", [1e-4, 1e-6, 1e-8])
def test_strict_optimality_exposes_one_sided_difference_accuracy(difference_step):
    # The exact derivative is zero at x=0.4, but the forward difference of
    # (x-0.4)**2 equals the perturbation. Report that actual numerical gradient;
    # do not silently cap the KKT residual to pass a strict user tolerance.
    for multiplier, expected_success in [(0.5, False), (2.0, True)]:
        tolerance = multiplier * difference_step
        result = minimize_slsqp(
            lambda x: Evaluation((x[0] - 0.4) ** 2),
            [0.4],
            options=SLSQPOptions(
                ftol=1e-4,
                finite_diff_step=difference_step,
                feasibility_tol=1e-12,
                optimality_tol=tolerance,
            ),
        )
        assert result.status == 0
        assert result.solver_success
        assert bool(result.success) is expected_success
        np.testing.assert_array_equal(result.x, [0.4])
        assert result.optimality == pytest.approx(difference_step, rel=1e-6)
        assert_report_consistent(
            result, feasibility_tol=1e-12, optimality_tol=tolerance
        )


@pytest.mark.parametrize("kind", ["feasibility", "optimality"])
@pytest.mark.parametrize("separation", ["small", "one_ulp"])
def test_failure_message_distinguishes_nearby_residual_and_tolerance(kind, separation):
    if kind == "feasibility":
        evaluate = lambda x: Evaluation(0.0, equalities=[x[0] ** 2 - 1.0])
        jacobian = lambda x: np.array([[0.0], [2.0 * x[0]]])
        initial = [1.1]
    else:
        # Reproduce the near-threshold residual found in the turbine study.
        slope = 1.0033539e-6
        evaluate = lambda x: Evaluation(slope * x[0])
        jacobian = lambda x: np.array([[slope]])
        initial = [0.0]

    def solve(tolerance):
        options = {f"{kind}_tol": tolerance, "ftol": 0.1}
        return minimize_slsqp(evaluate, initial, jac=jacobian, options=options)

    reference = solve(1.0)
    residual = (
        reference.max_constraint_violation
        if kind == "feasibility"
        else reference.optimality
    )
    tolerance = (
        float(np.nextafter(residual, 0.0))
        if separation == "one_ulp"
        else residual / 1.0033539
    )
    result = solve(tolerance)
    assert result.solver_success
    assert not result.success
    np.testing.assert_array_equal(result.x, reference.x)
    phrase = "constraint violation" if kind == "feasibility" else "first-order residual"
    match = re.search(
        rf"{phrase} ([\d.e+-]+) exceeds (?:tolerance )?([\d.e+-]+?)(?:\. | \(|\.$)",
        result.message,
    )
    assert match is not None, result.message
    displayed_residual, displayed_tolerance = map(float, match.groups())
    assert displayed_residual > displayed_tolerance
    if separation == "one_ulp":
        assert displayed_residual == residual
        assert displayed_tolerance == tolerance
