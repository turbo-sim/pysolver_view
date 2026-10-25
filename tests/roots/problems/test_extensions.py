"""Packaged rectangular and positive-domain model contracts."""

import unittest
import numpy as np
from pysolver_view.roots.problems import (
    NonlinearSystemProblem,
    CoupledOverdeterminedRosenbrockProblem,
    LogarithmicDomainProblem,
)


class ExtensionProblemTests(unittest.TestCase):
    """Check multidimensional controls against independent derivatives and known roots."""

    def test_rectangular_derivative_and_known_residual_floor(self):
        """Dense rectangular derivatives match complex step and the constant row remains one."""
        for inconsistent in (False, True):
            problem = CoupledOverdeterminedRosenbrockProblem(
                6, inconsistent=inconsistent
            )
            self.assertIsInstance(problem, NonlinearSystemProblem)
            point = np.linspace(0.7, 1.3, problem.n)
            numerical = np.column_stack(
                [
                    problem.residual(point.astype(complex) + 1e-30j * direction).imag
                    / 1e-30
                    for direction in np.eye(problem.n)
                ]
            )
            np.testing.assert_allclose(problem.gradient(point), numerical, atol=1e-13)
            self.assertEqual(problem.residual(point).size, problem.m)
            self.assertAlmostEqual(
                np.linalg.norm(problem.residual(np.ones(problem.n))),
                float(inconsistent),
            )

    def test_logarithmic_derivative_and_undefined_full_newton_trial(self):
        """The coupled domain control has a known root and an independently invalid full trial."""
        problem = LogarithmicDomainProblem(6)
        self.assertIsInstance(problem, NonlinearSystemProblem)
        point = np.linspace(0.5, 2, problem.n)
        difference = np.column_stack(
            [
                (
                    problem.residual(point + 1e-6 * direction)
                    - problem.residual(point - 1e-6 * direction)
                )
                / 2e-6
                for direction in np.eye(problem.n)
            ]
        )
        np.testing.assert_allclose(
            problem.gradient(point), difference, rtol=1e-8, atol=1e-9
        )
        np.testing.assert_array_equal(
            problem.residual(np.ones(problem.n)), np.zeros(problem.n)
        )
        trial = problem.x0 - np.linalg.solve(
            problem.gradient(problem.x0), problem.residual(problem.x0)
        )
        self.assertTrue(np.all(trial < 0))
