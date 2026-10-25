"""Behavioral contracts for the four named solver entry points and public dispatch."""

import unittest

import numpy as np

from pysolver_view.roots.solvers import (
    solve_gauss_newton,
    solve_levenberg_marquardt,
    solve_newton_raphson,
    solve_powell_hybrid,
    solve_root,
)
from pysolver_view.roots.problems import BroydenTridiagonalProblem


class SolverDispatchTests(unittest.TestCase):
    """Check that direct methods preserve numerical behavior and method contracts."""

    def test_direct_methods_match_dispatched_results_and_complete_histories(self):
        """All four functions reproduce dispatch, including model reuse and trials."""
        problem = BroydenTridiagonalProblem(n=10)
        for method, solve in (
            ("levenberg_marquardt", solve_levenberg_marquardt),
            ("powell_hybrid", solve_powell_hybrid),
            ("newton_raphson", solve_newton_raphson),
            ("gauss_newton", solve_gauss_newton),
        ):
            with self.subTest(method=method):
                direct = solve(problem.residual, problem.x0, problem.gradient)
                dispatched = solve_root(
                    problem.residual, problem.x0, problem.gradient, method=method
                )
                self.assertTrue(direct.success, direct.message)
                self.assertEqual(direct.method, method)
                self.assertEqual(dispatched.requested_method, method)
                dispatched.pop("requested_method")
                dispatched.pop("method_name")
                np.testing.assert_equal(dict(direct), dict(dispatched))
                self.assertLessEqual(np.linalg.norm(problem.residual(direct.x)), 1e-8)
                if method == "powell_hybrid":
                    self.assertGreater(direct.nupdates, 0)
                    self.assertLess(direct.njev, direct.nit)
                else:
                    self.assertEqual(direct.nupdates, 0)

    def test_named_newton_methods_keep_distinct_shape_and_rank_requirements(self):
        """Strict Newton rejects singular/rectangular models that Gauss–Newton solves."""
        matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
        target = np.array([1.0, 2.0])
        residual = lambda x: matrix @ x - target
        jacobian = lambda x: matrix.copy()
        strict = solve_newton_raphson(residual, [0.0, 0.0], jacobian)
        least_squares = solve_gauss_newton(residual, [0.0, 0.0], jacobian)
        self.assertFalse(strict.success)
        self.assertEqual(strict.status, 7)
        self.assertEqual(strict.nit, 0)
        self.assertTrue(least_squares.success, least_squares.message)
        np.testing.assert_allclose(least_squares.x, [0.5, 0.5], atol=1e-12)

        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = np.array([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "square"):
            solve_newton_raphson(residual, [0.0, 0.0], jacobian)
        rectangular = solve_gauss_newton(residual, [0.0, 0.0], jacobian)
        self.assertTrue(rectangular.success, rectangular.message)
        np.testing.assert_allclose(rectangular.x, [1.0, 2.0], atol=1e-12)
