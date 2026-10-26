"""Check equality adaptation independently of optimizer stopping flags."""

from types import SimpleNamespace

import numpy as np
import pytest

import pysolver_view as psv
from pysolver_view.optimization.problems import as_optimization_problem


class Equations(psv.NonlinearSystemProblem):
    """Two nonlinear equations with independent physical-call records."""

    def __init__(self):
        """Keep original metadata to detect optimization-state leakage."""
        self.calls = []
        self.variable_names = ["a", "b"]
        self.problem_scale = "model-owned state"

    def residual(self, x):
        """Preserve complex perturbations while seeking the positive root (1, 2)."""
        self.calls.append(x.copy())
        return np.array([x[0] ** 2 - 1, x[1] ** 2 - 4])


class AnalyticEquations(Equations):
    """Provide an exact residual Jacobian without hidden residual evaluations."""

    def __init__(self):
        """Track independent derivative work."""
        super().__init__()
        self.jacobian_calls = 0

    def gradient(self, x):
        """Differentiate each equation in original physical coordinates."""
        self.jacobian_calls += 1
        return np.diag(2 * x)


def make_solver(adapted, library, **overrides):
    """Use one explicit feasibility target for native and controlled SLSQP."""
    options = dict(
        library=library,
        method="slsqp",
        tolerance=1e-10,
        max_iterations=100,
        print_convergence=False,
        plot_convergence=False,
        update_on="iteration",
        extra_options=(
            {"first_step_fraction": 1.0, "feasibility_tol": 1e-9}
            if library == "pysolver"
            else {}
        ),
    )
    options.update(overrides)
    return psv.OptimizationSolver(adapted, **options)


@pytest.mark.parametrize("library", ["scipy", "pysolver"])
def test_analytic_solve_preserves_model_metadata_and_counts(library):
    """A root adapter changes neither physical equations nor their call accounting."""
    model = AnalyticEquations()
    adapted = as_optimization_problem(model, n_variables=2, bounds=([0, 0], [3, 4]))
    assert model.calls == [] and model.jacobian_calls == 0
    assert psv.as_optimization_problem is as_optimization_problem
    solver = make_solver(adapted, library, problem_scale=2.0)
    assert model.problem_scale == "model-owned state"
    assert model.variable_names == ["a", "b"]
    assert adapted.variable_names is not model.variable_names
    solver.solve([0.8, 1.7])
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.result.x, [1, 2], atol=1e-9)
    assert solver.result.fun == 0
    assert np.linalg.norm(solver.result.fitness[1:]) < 1e-9
    assert solver.result.nfev == len(model.calls)
    assert solver.result.njev == model.jacobian_calls
    assert solver.result.derivative_func_count == 0
    assert model.problem_scale == "model-owned state"
    assert model.variable_names == ["a", "b"]


@pytest.mark.parametrize("library", ["scipy", "pysolver"])
@pytest.mark.parametrize("method", ["2-point", "cs"])
def test_numerical_fallback_preserves_probes_and_physical_counts(library, method):
    """Absent analytic derivatives use the frontend's counted numerical probes."""
    model = Equations()
    adapted = as_optimization_problem(model, n_variables=2)
    assert not hasattr(adapted, "gradient")
    solver = make_solver(adapted, library, derivative_method=method)
    solver.solve([0.8, 1.7])
    assert solver.success, solver.message
    np.testing.assert_allclose(solver.result.x, [1, 2], atol=1e-8)
    assert solver.result.nfev == len(model.calls)
    assert solver.result.derivative_func_count > 0
    if method == "cs":
        assert any(np.any(np.imag(point) != 0) for point in model.calls)


def test_rectangular_mapping_is_exact_and_returned_arrays_are_detached():
    """Preserve all equations and Jacobian rows without reducing dependent rows."""
    matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    residual_buffer = np.array([2.0, 3.0, 5.0])
    model = SimpleNamespace(
        residual=lambda x: residual_buffer, gradient=lambda x: matrix
    )
    lower, upper = np.array([-4.0, -5.0]), np.array([4.0, 5.0])
    adapted = as_optimization_problem(
        model, n_variables=2, n_equations=3, bounds=(lower, upper)
    )
    assert adapted.get_nec() == 3 and adapted.get_nic() == 0
    np.testing.assert_array_equal(adapted.fitness([1, 2]), [0, 2, 3, 5])
    np.testing.assert_array_equal(
        adapted.gradient([1, 2]), np.vstack((np.zeros(2), matrix))
    )
    adapted.fitness([1, 2])[:] = 17
    adapted.gradient([1, 2])[:] = 18
    lower[:] = -99
    returned_lower, returned_upper = adapted.get_bounds()
    returned_lower[:] = -88
    returned_upper[:] = 88
    np.testing.assert_array_equal(residual_buffer, [2, 3, 5])
    np.testing.assert_array_equal(matrix, [[1, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(adapted.get_bounds(), [[-4, -5], [4, 5]])


def test_underdetermined_system_keeps_each_equality_separate():
    """An optimizer can choose a feasible point when the adapter has free directions."""
    model = SimpleNamespace(
        residual=lambda x: x[0] + x[1] - 3,
        gradient=lambda x: np.array([1.0, 1.0]),
    )
    adapted = as_optimization_problem(model, n_variables=2, n_equations=1)
    solver = make_solver(adapted, "pysolver")
    solver.solve([0, 0])
    assert solver.success, solver.message
    assert abs(model.residual(solver.result.x)) < 1e-9
    assert adapted.gradient([1, 2]).shape == (2, 2)


def test_scalar_equation_and_jacobian_preserve_complex_values():
    """The one-variable convention accepts scalars without discarding probes."""
    model = SimpleNamespace(
        residual=lambda x: x[0] ** 2 - 1, gradient=lambda x: 2 * x[0]
    )
    adapted = as_optimization_problem(model, n_variables=1, bounds=(0, np.inf))
    point = np.array([1 + 1e-30j])
    values = adapted.fitness(point)
    assert values.shape == (2,) and values[0] == 0
    assert values[1].imag / 1e-30 == pytest.approx(2)
    np.testing.assert_array_equal(adapted.gradient(point), [[0], [2 + 2e-30j]])


def test_zero_objective_does_not_turn_inconsistent_equations_into_success():
    """Feasibility still fails when a stationary residual has no real root."""
    model = SimpleNamespace(
        residual=lambda x: x[0] ** 2 + 1, gradient=lambda x: 2 * x[0]
    )
    solver = make_solver(as_optimization_problem(model, n_variables=1), "pysolver")
    solver.solve([0.0])
    assert solver.result.fun == 0
    assert not solver.success
    assert not solver.result.feasible
    assert solver.result.fitness[1] >= 1


def test_domain_errors_and_input_ownership_are_preserved():
    """Domain failures retain their identity and model mutation cannot overwrite x."""
    failure = psv.DomainError("outside model domain")

    def residual(x):
        """Mutate a received buffer before signalling an undefined state."""
        x[:] = -7
        raise failure

    adapted = as_optimization_problem(SimpleNamespace(residual=residual), n_variables=2)
    point = np.array([1.0, 2.0])
    with pytest.raises(psv.DomainError) as caught:
        adapted.fitness(point)
    assert caught.value is failure
    np.testing.assert_array_equal(point, [1, 2])


@pytest.mark.parametrize(
    "name,value",
    [
        ("n_variables", 0),
        ("n_variables", True),
        ("n_variables", 1.5),
        ("n_equations", -1),
        ("n_equations", False),
    ],
)
def test_invalid_dimensions_fail_before_evaluating(name, value):
    """Declared geometry must be usable before any external model work happens."""
    model = AnalyticEquations()
    kwargs = dict(n_variables=2, n_equations=2)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        as_optimization_problem(model, **kwargs)
    assert model.calls == []


@pytest.mark.parametrize(
    "bounds",
    [
        ([0], [1]),
        ([0, np.nan], [1, 2]),
        ([2, 0], [1, 2]),
        ([np.inf, 0], [np.inf, 1]),
        ([-np.inf, 0], [-np.inf, 1]),
        ([0j, 0], [1, 2]),
    ],
)
def test_invalid_bounds_are_rejected(bounds):
    """Wrong dimensions, undefined values and empty infinite intervals fail early."""
    with pytest.raises(ValueError, match="bounds"):
        as_optimization_problem(Equations(), n_variables=2, bounds=bounds)


def test_wrong_evaluation_shapes_fail_instead_of_changing_equation_count():
    """Silent reshape must not turn a model error into different solver geometry."""
    model = SimpleNamespace(
        residual=lambda x: np.ones((2, 1)), gradient=lambda x: np.ones(4)
    )
    adapted = as_optimization_problem(model, n_variables=2)
    with pytest.raises(ValueError, match="residual must have shape"):
        adapted.fitness([1, 2])
    with pytest.raises(ValueError, match="gradient must have shape"):
        adapted.gradient([1, 2])
    with pytest.raises(ValueError, match="x must have shape"):
        adapted.fitness([1])


def test_noncallable_optional_gradient_uses_numerical_fallback():
    """Optional placeholders do not falsely advertise an analytic Jacobian."""
    model = SimpleNamespace(residual=lambda x: x - 1, gradient=None)
    adapted = as_optimization_problem(model, n_variables=1)
    assert not hasattr(adapted, "gradient")
    with pytest.raises(TypeError, match="callable residual"):
        as_optimization_problem(SimpleNamespace(residual=None), n_variables=1)
