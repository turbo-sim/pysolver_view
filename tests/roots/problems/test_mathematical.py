"""Installed mathematical models, derivatives and reference metadata."""

import unittest
import numpy as np
from pysolver_view.roots.problems import (
    NonlinearSystemProblem,
    BrownAlmostLinearProblem,
    get_problems,
)


class BenchmarkProblemTests(unittest.TestCase):
    """Check that standard problems implement the PSV interface and supply correct
    analytic derivatives.
    """

    def test_benchmarks_are_coupled_multidimensional_psv_problem_instances(self):
        """Every standard benchmark must expose finite coupled residuals/Jacobians
        through NonlinearSystemProblem.
        """
        for problem in get_problems():
            with self.subTest(problem=problem.name):
                self.assertIsInstance(problem, NonlinearSystemProblem)
                self.assertGreaterEqual(problem.n, 2)
                self.assertEqual(problem.x0.shape, (problem.n,))
                self.assertEqual(problem.x_scale.shape, (problem.n,))
                self.assertTrue(np.all(problem.x_scale > 0))
                point = problem.x0.copy()
                residual = problem.residual(point)
                jacobian = problem.gradient(point)
                self.assertEqual(residual.shape, (problem.n,))
                self.assertEqual(jacobian.shape, (problem.n, problem.n))
                self.assertTrue(np.all(np.isfinite(residual)))
                self.assertTrue(np.all(np.isfinite(jacobian)))
                self.assertTrue(np.any(jacobian - np.diag(np.diag(jacobian))))
                np.testing.assert_array_equal(point, problem.x0)

    def test_benchmark_analytic_jacobians_match_independent_central_differences(self):
        """Check each mathematical Jacobian in scaled coordinates against independent
        central differences.
        """
        for problem in get_problems():
            perturbation = 0.13 * problem.x_scale * np.sin(np.arange(problem.n) + 1)
            for point in (problem.x0, problem.x0 + perturbation):
                with self.subTest(problem=problem.name, point=point.tolist()):
                    # Compare in dimensionless coordinates so the check remains
                    # meaningful for Powell's variables with dissimilar scales.
                    delta = 1e-5
                    numerical = np.empty((problem.n, problem.n))
                    for column in range(problem.n):
                        change = np.zeros(problem.n)
                        change[column] = delta * problem.x_scale[column]
                        numerical[:, column] = (
                            problem.residual(point + change)
                            - problem.residual(point - change)
                        ) / (2 * delta)
                    np.testing.assert_allclose(
                        problem.gradient(point) * problem.x_scale,
                        numerical,
                        rtol=2e-7,
                        atol=2e-7,
                    )

    def test_supplied_reference_solutions_are_roots(self):
        """Every advertised reference solution must satisfy its equations to the stated
        verification accuracy.
        """
        checked = 0
        for problem in get_problems():
            if problem.solution is not None:
                with self.subTest(problem=problem.name):
                    self.assertEqual(problem.solution.shape, (problem.n,))
                    self.assertLessEqual(
                        np.linalg.norm(problem.residual(problem.solution)), 1e-10
                    )
                    checked += 1
        self.assertGreater(checked, 0)

    def test_brown_product_jacobian_is_valid_when_a_variable_is_zero(self):
        """The Brown product derivative must remain valid at zero components without
        dividing by them.
        """
        problem = BrownAlmostLinearProblem(n=5)
        point = np.array([0.0, 1.2, 0.8, 1.1, 0.9])
        derivative = problem.gradient(point)[-1]
        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertAlmostEqual(derivative[0], np.prod(point[1:]))
        np.testing.assert_array_equal(derivative[1:], np.zeros(4))
