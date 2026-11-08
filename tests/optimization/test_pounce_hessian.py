"""Exact component Hessians: coordinate scaling, public signs and native solves."""

import numpy as np
import pytest

from pysolver_view import OptimizationProblem, OptimizationSolver, RosenbrockProblem
from pysolver_view.optimization.evaluation import BackendProblem
from pysolver_view.optimization.backends.pounce import _PounceHessianCallbacks


class CurvedDisk(OptimizationProblem):
    """Known optimum with one linear equality and one curved inequality."""

    def __init__(self, packed=False, sparse=False):
        self.packed = packed
        self.calls = []
        if sparse:
            self.hessian_sparsity = lambda: (np.array([0, 1]), np.array([0, 1]))

    def fitness(self, x):
        return np.array([(x[0] - 2) ** 2 + x[1] ** 2, x[1] - 0.5, x @ x - 1])

    def gradient(self, x):
        return np.array([[2 * (x[0] - 2), 2 * x[1]], [0.0, 1.0], [2 * x[0], 2 * x[1]]])

    def hessians(self, x):
        self.calls.append(x.copy())
        matrices = np.array([2 * np.eye(2), np.zeros((2, 2)), 2 * np.eye(2)])
        if hasattr(self, "hessian_sparsity"):
            rows, cols = self.hessian_sparsity()
            return matrices[:, rows, cols]
        return (
            matrices[:, np.tril_indices(2)[0], np.tril_indices(2)[1]]
            if self.packed
            else matrices
        )

    def get_bounds(self):
        return np.array([-2.0, -3.0]), np.array([4.0, 5.0])

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("scale", [None, 2.0])
def test_lagrangian_hessian_scaling_signs_and_cache(packed, sparse, scale):
    problem = CurvedDisk(packed, sparse)
    solver = OptimizationSolver(
        problem,
        library="pounce",
        method="ipm",
        problem_scale=scale,
        print_convergence=False,
        check_kkt=False,
    )
    callbacks = _PounceHessianCallbacks(BackendProblem(solver), 2, 1, 1)
    physical = np.array([0.4, 0.5])
    point = problem.scale_physical_to_normalized(physical)
    multipliers = np.array([-0.7, 1.3])
    rows, cols = callbacks.hessianstructure()
    factor = np.ones(2) if scale is None else np.array([3.0, 4.0])
    expected = (2.5 + 1.3) * 2 * np.diag(factor**2)
    got = callbacks.hessian(point, multipliers, 2.5)
    np.testing.assert_allclose(got, expected[rows, cols])
    # New Lagrangian weights at the same point reuse component Hessians only.
    callbacks.hessian(point, np.zeros(2), 0.3)
    assert len(problem.calls) == 1 and callbacks.hessian_calls == 2
    np.testing.assert_allclose(problem.calls[0], physical)
    assert solver._evaluation.hess_count == 1
    assert solver._evaluation.func_count_tot == 0 and solver._evaluation.grad_count == 0


@pytest.mark.parametrize("method", ["ipm", "sqp"])
@pytest.mark.parametrize("scale", [None, 1.0])
def test_native_exact_hessian_with_nonlinear_constraint(method, scale):
    pytest.importorskip("pounce")
    problem = CurvedDisk(sparse=True)
    solver = OptimizationSolver(
        problem,
        library="pounce",
        method=method,
        problem_scale=scale,
        print_convergence=False,
        check_kkt=False,
        tolerance=1e-8,
        max_iterations=100,
    )
    x = solver.solve(np.array([0.4, 0.5]))
    assert solver.success, solver.message
    np.testing.assert_allclose(x, [np.sqrt(0.75), 0.5], atol=3e-6)
    assert solver.result.nhev == len(problem.calls) > 0
    assert solver.result.hessian_time > 0 and solver.result.failed_hessian_count == 0
    assert solver.result.kkt_evaluations == 0
    assert solver.backend_result.hessian_nnz == 2


@pytest.mark.parametrize(
    "method,options",
    [
        ("ipm", {"hessian_approximation": "limited-memory"}),
        ("sqp", {"sqp_hessian": "damped-bfgs"}),
    ],
)
def test_explicit_limited_memory_does_not_evaluate_available_hessians(method, options):
    pytest.importorskip("pounce")
    problem = CurvedDisk()
    solver = OptimizationSolver(
        problem,
        library="pounce",
        method=method,
        check_kkt=False,
        print_convergence=False,
        extra_options=options,
    )
    solver.solve(np.array([0.4, 0.5]))
    assert solver.success, solver.message
    assert solver.result.nhev == 0 and problem.calls == []


def test_curved_equality_and_inequality_multiplier_weights_match_differences():
    """Check both public constraint signs and zero objective weight independently."""
    problem = CurvedDisk()
    problem.fitness = lambda x: np.array(
        [(x[0] - 2) ** 2 + x[1] ** 2, x[0] * x[1] - 0.1, x @ x - 1.0]
    )
    problem.gradient = lambda x: np.array(
        [[2 * (x[0] - 2), 2 * x[1]], [x[1], x[0]], [2 * x[0], 2 * x[1]]]
    )
    problem.hessians = lambda x: np.array(
        [2 * np.eye(2), [[0.0, 1.0], [1.0, 0.0]], 2 * np.eye(2)]
    )
    solver = OptimizationSolver(
        problem,
        library="pounce",
        method="ipm",
        problem_scale=2.0,
        print_convergence=False,
        check_kkt=False,
    )
    callback = _PounceHessianCallbacks(BackendProblem(solver), 2, 1, 1)
    point = problem.scale_physical_to_normalized(np.array([0.4, 0.5]))
    multipliers = np.array([-0.7, 1.3])
    direction = np.array([0.3, -0.4])
    rows, cols = callback.hessianstructure()
    for factor in (0.0, 2.5):
        values = callback.hessian(point, multipliers, factor)
        matrix = np.zeros((2, 2))
        matrix[rows, cols] = values
        matrix[cols, rows] = values

        def gradient(z):
            physical = problem.scale_normalized_to_physical(z)
            return (
                np.r_[factor, multipliers]
                @ problem.gradient(physical)
                * np.array([3.0, 4.0])
            )

        h = 1e-6
        np.testing.assert_allclose(
            matrix @ direction,
            (gradient(point + h * direction) - gradient(point - h * direction))
            / (2 * h),
            rtol=1e-8,
        )


@pytest.mark.parametrize(
    "bad_pattern",
    [
        (np.array([0]), np.array([1])),
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        (np.array([0, 0]), np.array([0, 0])),
        (np.array([2]), np.array([0])),
    ],
)
def test_invalid_hessian_pattern_fails_before_native_solve(bad_pattern):
    problem = CurvedDisk()
    problem.hessian_sparsity = lambda: bad_pattern
    solver = OptimizationSolver(
        problem, library="pounce", method="ipm", print_convergence=False
    )
    with pytest.raises(ValueError, match="[Hh]essian"):
        solver._evaluation.hessian_structure(2)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.zeros((2, 3)),
        np.full((3, 3), np.nan),
        np.array([[[1.0, 2.0], [0.0, 1.0]]] * 3),
    ],
)
def test_malformed_component_hessians_are_rejected(bad_value):
    problem = CurvedDisk()
    problem.hessians = lambda x: bad_value
    solver = OptimizationSolver(
        problem, library="pounce", method="ipm", print_convergence=False
    )
    with pytest.raises(ValueError):
        solver._evaluation.evaluate_hessians(np.array([0.4, 0.5]))
    assert solver._evaluation.failed_hess_count == 1


def test_hessian_callback_preserves_original_application_error():
    pytest.importorskip("pounce")
    problem = CurvedDisk()
    error = RuntimeError("Hessian application failure")

    def broken(x):
        raise error

    problem.hessians = broken
    solver = OptimizationSolver(
        problem,
        library="pounce",
        method="ipm",
        print_convergence=False,
        check_kkt=False,
    )
    with pytest.raises(RuntimeError, match="Hessian application failure") as caught:
        solver.solve(np.array([0.4, 0.5]))
    assert caught.value is error


def test_existing_packed_rosenbrock_hessians_work_without_adapter_changes():
    pytest.importorskip("pounce")
    solver = OptimizationSolver(
        RosenbrockProblem(2),
        library="pounce",
        method="ipm",
        print_convergence=False,
        check_kkt=False,
        tolerance=1e-8,
        max_iterations=100,
    )
    x = solver.solve(np.array([-1.2, 1.0]))
    assert solver.success, solver.message
    np.testing.assert_allclose(x, [1.0, 1.0], atol=2e-5)
    assert solver.result.nhev > 0
