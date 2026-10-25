"""Independent checks of the new dense MINPACK equation workload."""

import unittest

import numpy as np
from scipy.optimize import root

from pysolver_view.nonlinear_system import NonlinearSystemProblem
from ..src.problems import DiscreteIntegralEquationProblem


class DiscreteIntegralEquationTests(unittest.TestCase):
    """Validate the dense integral-equation workload independently of the custom
    solvers.
    """

    def test_dense_formula_agrees_with_minpack_split_sums(self):
        """The matrix-kernel residual and standard start must agree with the independent
        split-sum equations.
        """
        problem = DiscreteIntegralEquationProblem(n=8)
        self.assertIsInstance(problem, NonlinearSystemProblem)
        x = problem.x0 + np.linspace(-0.1, 0.2, problem.n)
        values = (x + problem.t + 1) ** 3
        expected = []
        for i, ti in enumerate(problem.t):
            left = sum(problem.t[j] * values[j] for j in range(i + 1))
            right = sum((1 - problem.t[j]) * values[j] for j in range(i + 1, problem.n))
            expected.append(x[i] + problem.h / 2 * ((1 - ti) * left + ti * right))
        np.testing.assert_allclose(
            problem.residual(x), expected, rtol=1e-14, atol=1e-16
        )
        np.testing.assert_allclose(problem.x0, problem.t * (problem.t - 1))

    def test_analytic_dense_jacobian_matches_complex_step(self):
        """Every dense derivative entry must match independent complex-step
        differentiation.
        """
        problem = DiscreteIntegralEquationProblem(n=12)
        x = problem.x0 + np.linspace(-0.07, 0.13, problem.n)
        numerical = np.column_stack(
            [
                np.imag(problem.residual(x.astype(complex) + 1e-30j * basis)) / 1e-30
                for basis in np.eye(problem.n)
            ]
        )
        analytic = problem.gradient(x)
        self.assertTrue(np.all(analytic != 0))
        np.testing.assert_allclose(analytic, numerical, rtol=3e-15, atol=1e-16)

    def test_standard_100_variable_start_reaches_a_root(self):
        """Both native SciPy methods must solve the standard 100-variable
        integral-equation case.
        """
        problem = DiscreteIntegralEquationProblem(n=100)
        for method in ("lm", "hybr"):
            with self.subTest(method=method):
                result = root(
                    problem.residual, problem.x0, jac=problem.gradient, method=method
                )
                self.assertLess(np.linalg.norm(problem.residual(result.x)), 1e-8)


if __name__ == "__main__":
    unittest.main()
