"""Verify Jacobian equilibration without weakening physical root-solver contracts."""

import numpy as np
import pytest

import pysolver_view as psv
from demos.demo_eqn_solving.demo_radiative_heat_collocation import RadiativeHeatProblem
from pysolver_view.solvers.root.scaling import JacobianScaling
from pysolver_view.solvers.root.trial_step import prepare_equilibrated_trial

METHODS = ("levenberg_marquardt", "powell_hybrid", "newton_raphson", "gauss_newton")


@pytest.mark.parametrize("option_class", [psv.StepControlOptions, psv.NewtonOptions])
def test_jacobian_scaling_is_enabled_by_default_and_requires_a_boolean(option_class):
    """The default is explicit and strings, integers, or null cannot enable it silently."""
    assert option_class().jacobian_scaling is True
    assert option_class(jacobian_scaling=False).jacobian_scaling is False
    for invalid in ("false", 0, 1, None):
        with pytest.raises((TypeError, ValueError), match="jacobian_scaling"):
            option_class(jacobian_scaling=invalid)


@pytest.mark.parametrize("method", ["newton_raphson", "gauss_newton"])
def test_full_newton_family_step_preserves_the_physical_linear_solution(method):
    """Equilibration must not change the unique full-rank Newton/Gauss-Newton step."""
    initial = np.array([1.5, 2.0])

    def residual(point):
        """Give coupled equations whose derivative columns have very different units."""
        a, b = point
        return np.array([1000 * (a * a + b - 7), 0.01 * (a + b * b - 11)])

    def jacobian(point):
        """Differentiate the coupled equations analytically."""
        a, b = point
        return np.array([[2000 * a, 1000], [0.01, 0.02 * b]])

    expected = np.linalg.solve(jacobian(initial), -residual(initial))
    for enabled in (False, True):
        result = psv.solve_root(
            residual,
            initial,
            jacobian,
            method=method,
            options=dict(
                jacobian_scaling=enabled,
                x_scale=[0.1, 100.0],
                backtracking=None,
                max_iterations=1,
            ),
        )
        assert len(result.trials) == 1
        np.testing.assert_allclose(
            result.trials[0]["x"] - initial,
            expected,
            atol=2e-9,
            rtol=2e-9,
        )
        assert result.trials[0]["accepted"]


@pytest.mark.parametrize("method", METHODS)
def test_physical_trial_limits_remain_independent_of_jacobian_equilibration(method):
    """Audit actual model inputs against the user scale, including rejected trials."""
    physical_scale = np.array([0.2, 4.0])
    matrix = np.array([[500.0, 1.0], [1.0, 0.01]])
    target = np.array([0.3, 0.25])
    initial = np.array([-0.1, -0.5])
    for enabled in (False, True):
        calls = []

        def residual(point):
            """Record physical calls independently of any diagnostic history."""
            calls.append(point.copy())
            return matrix @ (point - target)

        def jacobian(point):
            """Return the constant, anisotropic Jacobian."""
            return matrix

        options = dict(
            jacobian_scaling=enabled,
            x_scale=physical_scale,
            residual_tol=1e-9,
            max_iterations=500,
        )
        if method in METHODS[:2]:
            options.update(initial_radius=0.05, max_radius=0.25)
            limit_key = "radius"
        else:
            options["backtracking"] = dict(
                step_limits=dict(initial_limit=0.05, max_limit=0.25)
            )
            limit_key = "step_limit"
        result = psv.solve_root(
            residual, initial, jacobian, method=method, options=options
        )
        assert result.success, result.message
        np.testing.assert_allclose(result.x, target, atol=1e-6, rtol=0)
        assert result.nfev == len(calls) == len(result.trials) + 1
        for call, trial in zip(calls[1:], result.trials):
            np.testing.assert_array_equal(call, trial["x"])
            actual = np.linalg.norm((call - trial["x_base"]) / physical_scale)
            assert actual <= trial[limit_key] * (1 + 2e-14)
            assert actual <= 0.25 * (1 + 2e-14)
            np.testing.assert_allclose(actual, trial["scaled_step_norm"], atol=2e-14)
        first = np.linalg.norm((calls[1] - calls[0]) / physical_scale)
        assert first <= 0.05 * (1 + 2e-14)


@pytest.mark.parametrize("probe_kind", ["scaled", "absolute", "complex"])
def test_derivative_probe_geometry_and_accounting_do_not_follow_numerical_scaling(
    probe_kind,
):
    """The physical difference distances belong to x_scale/abs_step, never equilibration."""
    initial = np.array([0.4, 1.2])
    physical_scale = np.array([0.2, 4.0])
    expected_steps = (
        np.array([3e-7, 2e-6]) if probe_kind == "absolute" else 1e-6 * physical_scale
    )
    if probe_kind == "complex":
        expected_steps = np.array([1e-20, 2e-20])
    for method in ("levenberg_marquardt", "newton_raphson"):
        for enabled in (False, True):
            calls = []

            def residual(point):
                """Preserve complex parts and expose each finite-difference model call."""
                calls.append(point.copy())
                return np.array([500 * point[0] ** 2 + point[1] - 2, point[1] ** 2 - 3])

            options = dict(
                jacobian_scaling=enabled,
                x_scale=physical_scale,
                finite_diff_step=1e-6,
                max_nfev=3,
            )
            if probe_kind in ("absolute", "complex"):
                options["derivative_abs_step"] = expected_steps
            if probe_kind == "complex":
                options["derivative_method"] = "cs"
            result = psv.solve_root(residual, initial, method=method, options=options)
            assert result.nfev == len(calls) == 3
            assert result.derivative_nfev == 2
            assert result.nit == result.ntrials == 0
            for column, point in enumerate(calls[1:]):
                expected = np.zeros(
                    2, dtype=complex if probe_kind == "complex" else float
                )
                expected[column] = expected_steps[column] * (
                    1j if probe_kind == "complex" else 1
                )
                if probe_kind == "complex":
                    np.testing.assert_array_equal(point.real, initial)
                    np.testing.assert_allclose(
                        point.imag, expected.imag, atol=0, rtol=1e-14
                    )
                else:
                    np.testing.assert_allclose(
                        point - initial, expected, atol=2e-16, rtol=1e-8
                    )


@pytest.mark.parametrize(
    "method", ["levenberg_marquardt", "powell_hybrid", "gauss_newton"]
)
def test_an_initially_zero_jacobian_column_can_become_informative(method):
    """Zero initial column norms must not freeze a variable or create infinite scales."""

    def residual(point):
        """The second variable first becomes identifiable after the first one moves."""
        a, b = point
        return np.array([a - 1.0, a * b - 2.0])

    def jacobian(point):
        """Include the exactly zero second column at the initial point."""
        a, b = point
        return np.array([[1.0, 0.0], [b, a]])

    result = psv.solve_root(
        residual,
        [0.0, 0.0],
        jacobian,
        method=method,
        options=dict(jacobian_scaling=True, max_iterations=300, residual_tol=1e-9),
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-8)
    assert all(np.isfinite(row["scaled_step_norm"]) for row in result.trials)


@pytest.mark.parametrize("method", METHODS)
def test_zero_jacobian_at_nonzero_residual_never_reports_root_success(method):
    """An entirely zero Jacobian is unavailable progress, not a reason to return NaNs."""

    def residual(point):
        """Return a constant system with no root."""
        return np.array([1.0, -2.0])

    def jacobian(point):
        """Return zero derivative columns whose equilibration must stay finite."""
        return np.zeros((2, 2))

    result = psv.solve_root(
        residual,
        [0.2, 0.4],
        jacobian,
        method=method,
        options=dict(jacobian_scaling=True),
    )
    assert not result.success
    assert result.nit == 0
    assert np.isfinite(result.residual_norm)
    np.testing.assert_array_equal(result.x, [0.2, 0.4])


def test_public_default_lm_repairs_the_medium_heat_case_without_reweighting_equations():
    """The original 100-node failure must be fixed with the same residual and physical cap."""
    problem = RadiativeHeatProblem(degree=99)
    outcomes = {}
    for enabled in (False, True):
        solver = psv.NonlinearSystemSolver(
            problem,
            library="pysolver",
            method="lm",
            tolerance=1e-8,
            max_iterations=300,
            print_convergence=False,
            update_on="iteration",
            options=dict(jacobian_scaling=enabled, initial_radius=0.1),
        )
        solver.solve(problem.initial_guess())
        outcomes[enabled] = solver
        assert solver.result.effective_options["jacobian_scaling"] is enabled
    assert not outcomes[False].success
    scaled = outcomes[True]
    assert scaled.success, scaled.message
    assert np.linalg.norm(problem.residual(scaled.x_final)) <= 1e-8
    temperature, _ = problem.unpack(scaled.x_final)
    np.testing.assert_allclose(temperature[[0, -1]], [1.0, 1.0], atol=1e-8)
    assert temperature.min() >= 1.0 - 1e-8
    accepted = [row["residual_norm"] for row in scaled.iteration_history]
    assert np.all(np.diff(accepted) <= 1e-12)
    for trial in scaled.result.trials:
        assert np.linalg.norm(trial["x"] - trial["x_base"]) <= trial["radius"] * (
            1 + 2e-14
        )


def test_public_default_hybrid_solves_the_medium_heat_case_with_physical_protection():
    """Hybrid's Broyden updates must remain consistent after numerical equilibration."""
    problem = RadiativeHeatProblem(degree=99)
    solver = psv.NonlinearSystemSolver(
        problem,
        library="pysolver",
        method="hybr",
        tolerance=1e-8,
        max_iterations=300,
        print_convergence=False,
        update_on="iteration",
        options=dict(initial_radius=0.1),
    )
    solver.solve(problem.initial_guess())
    assert solver.options.jacobian_scaling is True
    assert solver.success, solver.message
    assert np.linalg.norm(problem.residual(solver.x_final)) <= 1e-8
    first = solver.result.trials[0]
    assert np.linalg.norm(first["x"] - first["x_base"]) <= 0.1 * (1 + 2e-14)
    assert np.all(
        np.diff([row["residual_norm"] for row in solver.iteration_history]) <= 1e-12
    )


def test_running_column_norms_preserve_established_scales_and_activate_zero_columns():
    """Smaller derivatives cannot expand an established metric; new columns stay usable."""
    matrix = np.array([[3.0, 6.0, 0.0], [4.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    scaling = JacobianScaling(3, enabled=True)
    first = scaling.update(matrix).copy()
    np.testing.assert_allclose(
        np.linalg.norm(matrix[:, :2] * first[:2], axis=0),
        np.full(2, np.sqrt(50.0)),
        rtol=1e-14,
    )
    assert first[2] == 1.0
    np.testing.assert_array_equal(scaling.update(matrix / 10), first)
    changed = matrix.copy()
    changed[:, 0] *= 4
    changed[2, 2] = 100
    updated = scaling.update(changed).copy()
    assert updated[0] < first[0]
    assert updated[1] == first[1]
    assert 0 < updated[2] < 1
    np.testing.assert_array_equal(scaling.update(matrix), updated)
    snapshot = scaling.diagnostics()
    snapshot["jacobian_scale"][:] = -1
    assert np.all(scaling.factors > 0)


def test_column_norms_do_not_overflow_or_underflow_for_finite_extreme_entries():
    """Logarithmic norms must handle columns whose direct squared sums are unrepresentable."""
    matrix = np.tile([1e308, 1e-308, 0.0], (4, 1))
    scaling = JacobianScaling(3, enabled=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        factors = scaling.update(matrix)
    assert np.all(np.isfinite(factors)) and np.all(factors > 0)
    assert factors[0] < 1 < factors[1]
    assert factors[2] == 1
    np.testing.assert_allclose(factors[0] * factors[1], 1.0, rtol=1e-14)


def test_equilibration_is_invariant_to_uniform_equation_units():
    """Changing the unit of every residual equally must not change numerical column ratios."""
    matrices = [
        np.array([[3.0, 6.0, 0.0], [4.0, 8.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[30.0, 3.0, 0.0], [40.0, 4.0, 0.0], [0.0, 0.0, 20.0]]),
    ]
    reference = JacobianScaling(3, enabled=True)
    expected = [reference.update(matrix).copy() for matrix in matrices]
    for multiplier in (1e-150, 1e150):
        scaling = JacobianScaling(3, enabled=True)
        for matrix, factors in zip(matrices, expected):
            np.testing.assert_allclose(
                scaling.update(multiplier * matrix),
                factors,
                rtol=2e-13,
                atol=0,
            )


@pytest.mark.parametrize("offset", [np.zeros(3), np.array([1e8, -1e5, 1e4])])
def test_rounded_equilibrated_trials_satisfy_both_metrics(offset):
    """Separate numerical and physical norms must use the displacement actually evaluated."""
    physical_scale = np.array([0.01, 10.0, 2.0])
    numerical_factors = np.array([1e-4, 1e4, 0.5])
    radius = 0.1
    for direction in (np.array([0.03, 0.04, 0.02]), np.array([0.1, 0.0, 0.0])):
        original = direction.copy()
        trial, step, scaled, physical_norm, numerical, numerical_norm = (
            prepare_equilibrated_trial(
                offset, physical_scale, numerical_factors, direction, radius
            )
        )
        np.testing.assert_array_equal(direction, original)
        np.testing.assert_array_equal(step, trial - offset)
        np.testing.assert_allclose(scaled, step / physical_scale, rtol=0, atol=0)
        np.testing.assert_allclose(
            numerical, scaled / numerical_factors, rtol=0, atol=0
        )
        assert physical_norm <= radius * (1 + 2e-14)
        assert numerical_norm <= radius * (1 + 2e-14)
        np.testing.assert_allclose(physical_norm, np.linalg.norm(scaled), rtol=1e-14)
        np.testing.assert_allclose(
            numerical_norm, np.linalg.norm(numerical), rtol=1e-14
        )


@pytest.mark.parametrize("method", ["levenberg_marquardt", "newton_raphson"])
def test_solver_reuse_and_history_storage_do_not_retain_or_change_scaling(method):
    """A new solve rebuilds its metric and storing histories never changes the algorithm."""

    class ChangingEquations(psv.NonlinearSystemProblem):
        """Switchable column anisotropy exposes a stale metric across repeated solves."""

        def __init__(self):
            """Start with much stronger sensitivity to the first variable."""
            self.matrix = np.diag([1000.0, 1.0])

        def residual(self, point):
            """Use a square linear model with a known, fixed root."""
            return self.matrix @ (point - [0.3, 0.7])

        def gradient(self, point):
            """Return the current sensitivity matrix without residual work."""
            return self.matrix

    problem = ChangingEquations()
    solver = psv.NonlinearSystemSolver(
        problem,
        library="pysolver",
        method=method,
        tolerance=1e-10,
        max_iterations=300,
        print_convergence=False,
        update_on="iteration",
    )
    solver.solve([0.0, 0.0])
    assert solver.success
    old_metric = solver.result.jacobian_scale.copy()
    problem.matrix = np.diag([1.0, 1000.0])
    solver.solve([0.0, 0.0])
    assert solver.success
    assert not np.allclose(solver.result.jacobian_scale, old_metric)
    reference = solver.result
    for level in ("full", "summary", "none"):
        result = psv.solve_root(
            problem.residual,
            [0.0, 0.0],
            problem.gradient,
            method=method,
            options=dict(history_level=level, residual_tol=1e-10, max_iterations=300),
        )
        assert result.success
        np.testing.assert_array_equal(result.x, reference.x)
        np.testing.assert_array_equal(result.jacobian_scale, reference.jacobian_scale)
        assert result.jacobian_scale_updates == reference.jacobian_scale_updates
        assert (result.nfev, result.njev, result.nit) == (
            reference.nfev,
            reference.njev,
            reference.nit,
        )
