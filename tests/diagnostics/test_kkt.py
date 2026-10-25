"""Analytic KKT diagnostics ported from Turbodash, plus public-sign checks.

The model convention is h(x) == 0 and g(x) >= 0.  The Lagrangian is
f - lambda*h - mu*g - mu_lower*(x-lower) - mu_upper*(upper-x).
Inequality and bound multipliers must therefore be nonnegative.  These
tests include nonoptimal points where unrestricted least squares would
incorrectly report a zero Lagrangian gradient.
"""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pysolver_view.diagnostics import kkt
from pysolver_view.diagnostics.kkt import assess_kkt_nonnegative as assess_kkt


def assess(
    x,
    gradient,
    *,
    equalities=(),
    equality_jacobian=None,
    inequalities=(),
    inequality_jacobian=None,
    lower=None,
    upper=None,
    **tolerances,
):
    """Keep the analytical examples readable while supplying full arrays."""
    point = np.asarray(x, dtype=float)
    n = point.size
    eq = np.asarray(equalities, dtype=float)
    iq = np.asarray(inequalities, dtype=float)
    return assess_kkt(
        point,
        np.asarray(gradient, dtype=float),
        eq,
        (
            np.empty((0, n))
            if equality_jacobian is None
            else np.asarray(equality_jacobian, dtype=float)
        ),
        iq,
        (
            np.empty((0, n))
            if inequality_jacobian is None
            else np.asarray(inequality_jacobian, dtype=float)
        ),
        np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float),
        np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float),
        **tolerances,
    )


def assert_residuals(check, *, stationarity, primal=0.0, complementarity=0.0):
    """Compare independently known analytical residuals with the reported values."""
    assert check["available"]
    assert check["stationarity"] == pytest.approx(stationarity, abs=1e-10)
    assert check["primal_feasibility"] == pytest.approx(primal, abs=1e-10)
    assert check["dual_feasibility"] == pytest.approx(0.0, abs=1e-12)
    assert check["complementarity"] == pytest.approx(complementarity, abs=1e-10)
    assert check["optimality"] == pytest.approx(
        max(stationarity, complementarity), abs=1e-10
    )
    assert check["kkt_residual"] == pytest.approx(
        max(stationarity, primal, complementarity), abs=1e-10
    )


def test_unconstrained_residual_is_infinity_norm_of_actual_gradient():
    """Unconstrained residual is infinity norm of actual gradient."""
    check = assess([1.0, -2.0], [3.0, -4.0])
    assert_residuals(check, stationarity=4.0)
    assert not check["passed"]
    assert check["kkt_ratio"] == pytest.approx(4.0 / 1e-4)
    np.testing.assert_array_equal(check["multipliers"]["lower"], [0.0, 0.0])
    np.testing.assert_array_equal(check["multipliers"]["upper"], [0.0, 0.0])
    assert len(check["multipliers"]["equality"]) == 0
    assert len(check["multipliers"]["inequality"]) == 0


def test_stationary_interior_point_satisfies_all_kkt_conditions():
    """Stationary interior point satisfies all kkt conditions."""
    check = assess([0.2, 0.3], [0.0, 0.0], lower=[-1.0, -1.0], upper=[1.0, 1.0])
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert check["kkt_ratio"] == 0.0


@pytest.mark.parametrize("gradient", [-2.0, 2.0])
def test_equality_multiplier_is_free_to_have_either_sign(gradient):
    """Equality multiplier is free to have either sign."""
    check = assess([0.0], [gradient], equalities=[0.0], equality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert check["multipliers"]["equality"][0] == pytest.approx(gradient)


@pytest.mark.parametrize(
    "gradient, lower, upper, multiplier_kind",
    [(2.0, [0.0], [np.inf], "lower"), (-2.0, [-np.inf], [0.0], "upper")],
)
def test_active_bound_can_balance_gradient_with_nonnegative_multiplier(
    gradient, lower, upper, multiplier_kind
):
    """Active bound can balance gradient with nonnegative multiplier."""
    check = assess([0.0], [gradient], lower=lower, upper=upper)
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert check["multipliers"][multiplier_kind][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "gradient, lower, upper",
    [(-2.0, [0.0], [np.inf]), (2.0, [-np.inf], [0.0])],
)
def test_wrong_signed_bound_multiplier_cannot_fake_stationarity(gradient, lower, upper):
    """Wrong signed bound multiplier cannot fake stationarity."""
    check = assess([0.0], [gradient], lower=lower, upper=upper)
    assert_residuals(check, stationarity=2.0)
    assert not check["passed"]
    for kind in ("lower", "upper"):
        assert check["multipliers"][kind][0] == pytest.approx(0.0)


@pytest.mark.parametrize("gradient, expected", [(2.0, 0.0), (-2.0, 2.0)])
def test_greater_than_zero_inequality_uses_correct_multiplier_sign(gradient, expected):
    """Greater than zero inequality uses correct multiplier sign."""
    check = assess([0.0], [gradient], inequalities=[0.0], inequality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=expected)
    assert check["passed"] == (expected == 0.0)
    assert check["multipliers"]["inequality"][0] == pytest.approx(max(gradient, 0.0))


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1.0, 1e8, 1e12])
def test_positive_constraint_rescaling_preserves_active_optimality(scale):
    """Positive constraint rescaling preserves active optimality."""
    check = assess([0.0], [2.0], inequalities=[0.0], inequality_jacobian=[[scale]])
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert scale * check["multipliers"]["inequality"][0] == pytest.approx(2.0)


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1.0, 1e8, 1e12])
def test_scaled_inactive_constraint_cannot_absorb_gradient_without_complementarity(
    scale,
):
    # At x=1 for f=x^2 and g=scale*x, the minimax fit is scale*mu=1:
    # stationarity |2-scale*mu| = 1 and complementarity |scale*mu| = 1.
    # An activity threshold or unconstrained multiplier fit could hide this.
    """Scaled inactive constraint cannot absorb gradient without complementarity."""
    check = assess([1.0], [2.0], inequalities=[scale], inequality_jacobian=[[scale]])
    assert_residuals(check, stationarity=1.0, complementarity=1.0)
    assert not check["passed"]
    assert scale * check["multipliers"]["inequality"][0] == pytest.approx(1.0)


def test_inactive_bound_has_its_own_complementarity_penalty():
    # min x at x=1 with x>=0: the point is feasible, but not stationary.
    """Inactive bound has its own complementarity penalty."""
    check = assess([1.0], [2.0], lower=[0.0])
    assert_residuals(check, stationarity=1.0, complementarity=1.0)
    assert not check["passed"]
    assert check["multipliers"]["lower"][0] == pytest.approx(1.0)


def test_equality_feasibility_is_checked_even_with_exact_stationarity():
    """Equality feasibility is checked even with exact stationarity."""
    check = assess([1.0], [2.0], equalities=[1.0], equality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=0.0, primal=1.0)
    assert not check["passed"]
    assert check["kkt_ratio"] == pytest.approx(1.0 / 1e-6)


def test_inequality_feasibility_is_checked_even_with_zero_gradient():
    """Inequality feasibility is checked even with zero gradient."""
    check = assess([-0.1], [0.0], inequalities=[-0.1], inequality_jacobian=[[1.0]])
    assert_residuals(check, stationarity=0.0, primal=0.1)
    assert not check["passed"]


@pytest.mark.parametrize(
    "x, lower, upper", [(-0.1, [0.0], [np.inf]), (0.1, [-np.inf], [0.0])]
)
def test_bound_feasibility_is_checked_without_user_inequalities(x, lower, upper):
    """Bound feasibility is checked without user inequalities."""
    check = assess([x], [0.0], lower=lower, upper=upper)
    assert_residuals(check, stationarity=0.0, primal=0.1)
    assert not check["passed"]


def test_redundant_equalities_do_not_require_unique_multipliers():
    """Redundant equalities do not require unique multipliers."""
    check = assess(
        [0.0], [2.0], equalities=[0.0, 0.0], equality_jacobian=[[1.0], [2.0]]
    )
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    assert np.dot([1.0, 2.0], check["multipliers"]["equality"]) == pytest.approx(2.0)


def test_opposite_active_inequality_normals_use_nonnegative_multipliers():
    """Opposite active inequality normals use nonnegative multipliers."""
    check = assess(
        [0.0], [2.0], inequalities=[0.0, 0.0], inequality_jacobian=[[1.0], [-1.0]]
    )
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    multipliers = check["multipliers"]["inequality"]
    assert np.all(multipliers >= 0.0)
    assert multipliers[0] - multipliers[1] == pytest.approx(2.0)


def test_zero_constraint_gradient_does_not_invent_a_kkt_certificate():
    # min x subject to x^3>=0 has its minimum at 0 but fails the usual
    # constraint qualification: no finite multiplier makes grad L vanish.
    """Zero constraint gradient does not invent a kkt certificate."""
    check = assess([0.0], [1.0], inequalities=[0.0], inequality_jacobian=[[0.0]])
    assert_residuals(check, stationarity=1.0)
    assert not check["passed"]
    assert check["multipliers"]["inequality"][0] == 0.0


def test_constant_constraints_and_unbounded_variables_keep_full_multiplier_arrays():
    """Constant constraints and unbounded variables keep full multiplier arrays."""
    check = assess(
        [0.0, 2.0],
        [1.0, 0.0],
        equalities=[0.0],
        equality_jacobian=[[0.0, 0.0]],
        inequalities=[0.0, 3.0],
        inequality_jacobian=[[0.0, 0.0], [0.0, 0.0]],
        lower=[0.0, -np.inf],
        upper=[np.inf, np.inf],
    )
    assert_residuals(check, stationarity=0.0)
    assert check["passed"]
    np.testing.assert_array_equal(check["multipliers"]["equality"], [0.0])
    np.testing.assert_array_equal(check["multipliers"]["inequality"], [0.0, 0.0])
    np.testing.assert_allclose(check["multipliers"]["lower"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_array_equal(check["multipliers"]["upper"], [0.0, 0.0])


def test_feasibility_and_optimality_tolerances_are_applied_separately():
    """Feasibility and optimality tolerances are applied separately."""
    for feasibility_tol, passed, ratio in [(1e-4, True, 0.2), (1e-6, False, 20.0)]:
        check = assess(
            [2e-5],
            [0.0],
            equalities=[2e-5],
            equality_jacobian=[[1.0]],
            feasibility_tol=feasibility_tol,
            optimality_tol=1e-12,
        )
        assert check["passed"] == passed
        assert check["kkt_ratio"] == pytest.approx(ratio)

    for optimality_tol, passed, ratio in [(1e-4, True, 0.2), (1e-6, False, 20.0)]:
        check = assess(
            [0.0], [2e-5], feasibility_tol=1e-12, optimality_tol=optimality_tol
        )
        assert check["passed"] == passed
        assert check["kkt_ratio"] == pytest.approx(ratio)


def test_minimax_fit_does_not_hide_a_large_component_in_an_average():
    """Minimax fit does not hide a large component in an average."""
    check = assess(
        [1.0, 0.0],
        [2.0, 1.0],
        inequalities=[1.0],
        inequality_jacobian=[[1.0, 0.0]],
    )
    assert_residuals(check, stationarity=1.0, complementarity=1.0)
    assert not check["passed"]


def test_small_residuals_are_not_rounded_away_by_lp_solver_tolerance():
    """Small residuals are not rounded away by lp solver tolerance."""
    check = assess(
        [1.0],
        [2e-12],
        inequalities=[1.0],
        inequality_jacobian=[[1.0]],
        optimality_tol=1e-14,
    )
    assert check["available"]
    assert not check["passed"]
    assert check["stationarity"] == pytest.approx(1e-12, rel=1e-10, abs=0)
    assert check["complementarity"] == pytest.approx(1e-12, rel=1e-10, abs=0)
    assert check["kkt_ratio"] == pytest.approx(100.0)


def test_unavailable_multiplier_estimate_is_explicitly_not_a_certificate(monkeypatch):
    """Unavailable multiplier estimate is explicitly not a certificate."""
    monkeypatch.setattr(
        kkt,
        "linprog",
        lambda *args, **kwargs: OptimizeResult(
            success=False, message="Diagnostic failure"
        ),
    )
    check = assess([0.0], [1.0], lower=[0.0])
    assert not check["available"]
    assert not check["passed"]
    assert check["optimality"] is None
    assert check["kkt_ratio"] is None
    assert "Diagnostic failure" in check["message"]


def test_nonfinite_derivatives_are_rejected_instead_of_reported_as_optimal():
    """Nonfinite derivatives are rejected instead of reported as optimal."""
    with pytest.raises(ValueError, match="finite"):
        assess([0.0], [np.nan])


def test_reported_metrics_reconstruct_from_the_returned_multipliers():
    """Reported metrics reconstruct from the returned multipliers."""
    x = np.array([0.2, 0.7])
    gradient = np.array([3.0, -2.0])
    equality_values = np.array([-0.1])
    equality_jacobian = np.array([[1.0, 1.0]])
    inequality_values = np.array([0.3, -0.2])
    inequality_jacobian = np.array([[2.0, -1.0], [-1.0, 0.5]])
    lower, upper = np.zeros(2), np.ones(2)
    check = assess(
        x,
        gradient,
        equalities=equality_values,
        equality_jacobian=equality_jacobian,
        inequalities=inequality_values,
        inequality_jacobian=inequality_jacobian,
        lower=lower,
        upper=upper,
    )
    m = check["multipliers"]
    lagrangian_gradient = (
        gradient
        - equality_jacobian.T @ m["equality"]
        - inequality_jacobian.T @ m["inequality"]
        - m["lower"]
        + m["upper"]
    )
    products = np.r_[
        m["inequality"] * inequality_values,
        m["lower"] * (x - lower),
        m["upper"] * (upper - x),
    ]
    assert check["available"]
    assert check["stationarity"] == pytest.approx(
        np.linalg.norm(lagrangian_gradient, ord=np.inf)
    )
    assert check["complementarity"] == pytest.approx(np.max(np.abs(products)))
    assert check["primal_feasibility"] == pytest.approx(0.2)
    assert check["dual_feasibility"] == pytest.approx(
        max(0.0, -np.min(np.r_[m["inequality"], m["lower"], m["upper"]]))
    )
    assert check["optimality"] == pytest.approx(
        max(check["stationarity"], check["complementarity"], check["dual_feasibility"])
    )
    assert check["kkt_residual"] == pytest.approx(
        max(check["optimality"], check["primal_feasibility"])
    )


def public_example():
    """Return a two-dimensional active equality/inequality example with known duals."""
    # At (1, 2), grad f=(-4, 2), h=x+y-3=0, g=x-1=0.
    # Stationarity is (-4,2) + lambda*(1,1) + mu*(1,0)=0,
    # so lambda=-2 and mu=6 in the public plus-sign convention.
    return dict(
        x=np.array([1.0, 2.0]),
        gradient=np.array([-4.0, 2.0]),
        equality_values=np.array([0.0]),
        equality_jacobian=np.array([[1.0, 1.0]]),
        inequality_values=np.array([0.0]),
        inequality_jacobian=np.array([[1.0, 0.0]]),
        lower=np.full(2, -np.inf),
        upper=np.full(2, np.inf),
    )


def test_public_signs_and_reconstruction_are_independent_of_compatibility_adapter():
    """Public equality/inequality multipliers satisfy the documented Lagrangian."""
    args = public_example()
    result = kkt.assess_kkt(**args)
    assert result["passed"]
    assert result["multiplier_source"] == "estimated"
    np.testing.assert_allclose(result["multipliers"]["equality"], [-2])
    np.testing.assert_allclose(result["multipliers"]["inequality"], [6])
    m = result["multipliers"]
    actual_gradient = (
        args["gradient"]
        + args["equality_jacobian"].T @ m["equality"]
        + args["inequality_jacobian"].T @ m["inequality"]
    )
    np.testing.assert_allclose(actual_gradient, [0, 0], atol=1e-12)
    supplied = kkt.evaluate_kkt_residuals(
        **args, multipliers=m, multiplier_source="native_complete"
    )
    assert supplied["passed"]
    assert supplied["multiplier_source"] == "native_complete"
    assert supplied["method"] == "supplied_multipliers"


def test_explicit_wrong_signed_multipliers_fail_dual_feasibility():
    """Supplied negative multipliers cannot be hidden by zero stationarity."""
    args = public_example()
    args["gradient"] = np.array([4.0, -2.0])
    m = dict(equality=[2.0], inequality=[-6.0], lower=[0.0, 0.0], upper=[0.0, 0.0])
    result = kkt.evaluate_kkt_residuals(**args, multipliers=m)
    assert result["stationarity"] == 0
    assert result["dual_feasibility"] == 6
    assert not result["passed"]


def test_supplied_multipliers_do_not_invoke_estimation(monkeypatch):
    """Residual reconstruction is usable independently of the linear-program solver."""

    def forbidden_fit(*args, **kwargs):
        """Fail if a supplied-multiplier diagnostic incorrectly estimates duals."""
        raise AssertionError("Unexpected multiplier estimation")

    monkeypatch.setattr(kkt, "linprog", forbidden_fit)
    result = kkt.evaluate_kkt_residuals(
        **public_example(),
        multipliers=dict(
            equality=[-2.0], inequality=[6.0], lower=[0.0, 0.0], upper=[0.0, 0.0]
        ),
    )
    assert result["passed"]


@pytest.mark.parametrize("missing", ["equality", "inequality", "lower", "upper"])
def test_incomplete_native_multipliers_are_not_silently_certified(missing):
    """Missing bound duals require an explicit completion policy outside the core."""
    m = dict(equality=[-2.0], inequality=[6.0], lower=[0.0, 0.0], upper=[0.0, 0.0])
    del m[missing]
    with pytest.raises(ValueError, match="complete set"):
        kkt.evaluate_kkt_residuals(**public_example(), multipliers=m)


def test_nonzero_multiplier_for_absent_bound_is_rejected():
    """A fictitious unbounded-variable normal must not cancel objective gradients."""
    m = dict(equality=[-2.0], inequality=[6.0], lower=[1.0, 0.0], upper=[0.0, 0.0])
    with pytest.raises(ValueError, match="absent bounds"):
        kkt.evaluate_kkt_residuals(**public_example(), multipliers=m)


@pytest.mark.parametrize("gradient", [-3.0, 3.0])
def test_fixed_variable_uses_sign_feasible_bound_multipliers(gradient):
    """Opposing normals of a fixed bound admit nonnegative, nonunique duals."""
    result = kkt.assess_kkt(
        [0], [gradient], [], np.empty((0, 1)), [], np.empty((0, 1)), [0], [0]
    )
    m = result["multipliers"]
    assert result["passed"]
    assert m["lower"][0] >= 0 and m["upper"][0] >= 0
    assert gradient - m["lower"][0] + m["upper"][0] == pytest.approx(0)


def test_public_bound_only_infeasibility_is_separate_from_stationarity():
    """Zero gradient does not excuse an infeasible bound without user inequalities."""
    result = kkt.assess_kkt(
        [2, 0], [0, 0], [], np.empty((0, 2)), [], np.empty((0, 2)), [-1, -1], [1, 1]
    )
    assert result["stationarity"] == 0
    assert result["inequality_feasibility"] == 0
    assert result["bound_feasibility"] == 1
    assert result["primal_feasibility"] == 1
    assert not result["feasible"] and not result["passed"]


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_public_positive_constraint_scaling_transforms_multipliers(scale):
    """Positive constraint scaling changes dual magnitudes but preserves stationarity."""
    args = public_example()
    args["equality_jacobian"] *= scale
    args["inequality_jacobian"] *= scale
    result = kkt.assess_kkt(**args)
    assert result["passed"]
    assert result["multipliers"]["equality"][0] * scale == pytest.approx(-2)
    assert result["multipliers"]["inequality"][0] * scale == pytest.approx(6)


def test_coordinate_scaling_uses_transformed_gradients_and_supplied_units():
    """A linear variable transformation preserves a known exact KKT point."""
    args = public_example()
    scale = np.array([1e-3, 1e3])
    args["x"] /= scale
    args["gradient"] *= scale
    args["equality_jacobian"] *= scale
    args["inequality_jacobian"] *= scale
    result = kkt.assess_kkt(**args)
    assert result["passed"]
    np.testing.assert_allclose(result["multipliers"]["equality"], [-2], rtol=1e-7)
    np.testing.assert_allclose(result["multipliers"]["inequality"], [6], rtol=1e-7)


def test_assessment_does_not_mutate_inputs_or_alias_returned_point():
    """Diagnostics are detached records and cannot change their model inputs."""
    args = public_example()
    original = {key: value.copy() for key, value in args.items()}
    result = kkt.assess_kkt(**args)
    result["x"][:] = 100
    result["multipliers"]["equality"][:] = 100
    for key in args:
        np.testing.assert_array_equal(args[key], original[key])
    np.testing.assert_array_equal(result["point"], original["x"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("gradient", [np.nan, 1]),
        ("gradient", [1]),
        ("x", [[1, 2]]),
        ("equality_jacobian", [[1]]),
        ("inequality_values", [[0]]),
        ("lower", [np.inf, 0]),
        ("upper", [-np.inf, 0]),
        ("lower", [np.nan, 0]),
        ("lower", [10, 10]),
        ("gradient", [1 + 1j, 2]),
        ("feasibility_tol", 0),
        ("optimality_tol", np.inf),
        ("optimality_tol", [1e-6, 1e-6]),
    ],
)
def test_invalid_public_inputs_are_rejected(key, value):
    """Malformed or nonfinite mathematical inputs cannot produce a certificate."""
    args = public_example()
    if key == "lower" and value == [10, 10]:
        args["upper"] = np.array([1, 2])
    args[key] = value
    with pytest.raises(ValueError):
        kkt.assess_kkt(**args)


def test_nonfinite_fit_returns_unavailable_with_point_and_reason(monkeypatch):
    """Even a nominally successful LP cannot certify nonfinite recovered duals."""
    monkeypatch.setattr(
        kkt,
        "linprog",
        lambda *a, **kw: OptimizeResult(
            success=True, x=np.array([np.inf, np.inf, 0.0]), message="bad fit"
        ),
    )
    result = kkt.assess_kkt(**public_example())
    assert not result["available"] and not result["passed"]
    assert result["stationarity"] is None
    assert result["multipliers"] is None
    assert result["reason"]
    np.testing.assert_array_equal(result["point"], [1, 2])


@pytest.mark.parametrize("dimension", [3, 8, 20])
def test_multidimensional_known_kkt_system_with_mixed_active_constraints(dimension):
    """Recover a valid certificate for independently constructed multidimensional KKT data."""
    random = np.random.default_rng(527)
    x = random.uniform(0.1, 0.9, dimension)
    x[0], x[1] = 0, 1
    jh = random.normal(size=(2, dimension))
    jg = random.normal(size=(4, dimension))
    equality_duals = np.array([-1.0, 3.0])
    inequality_duals = np.array([2.0, 1.0, 0.0, 0.0])
    lower_duals, upper_duals = np.zeros(dimension), np.zeros(dimension)
    lower_duals[0], upper_duals[1] = 0.5, 0.7
    gradient = (
        -jh.T @ equality_duals - jg.T @ inequality_duals + lower_duals - upper_duals
    )
    result = kkt.assess_kkt(
        x,
        gradient,
        np.zeros(2),
        jh,
        [0, 0, -1, -2],
        jg,
        np.zeros(dimension),
        np.ones(dimension),
        optimality_tol=1e-9,
    )
    assert result["passed"]
    fitted = result["multipliers"]
    actual_stationarity = (
        gradient
        + jh.T @ fitted["equality"]
        + jg.T @ fitted["inequality"]
        - fitted["lower"]
        + fitted["upper"]
    )
    np.testing.assert_allclose(actual_stationarity, np.zeros(dimension), atol=1e-9)
    np.testing.assert_allclose(fitted["inequality"][2:], [0, 0], atol=1e-9)
    assert (
        np.min(np.r_[fitted["inequality"], fitted["lower"], fitted["upper"]]) >= -1e-12
    )
