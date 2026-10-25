"""Independent problem and reporting checks for the Newton-family comparisons."""

import unittest

import numpy as np

from pysolver_view.nonlinear_system import NonlinearSystemProblem
from ..run_convergence_benchmark import (
    audit_custom_steps,
    display_label,
    run_custom_case,
    summarize_case,
)
from ..src.problems import (
    CoupledOverdeterminedRosenbrockProblem,
    ExtendedRosenbrockProblem,
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


class ConvergencePolicyTests(unittest.TestCase):
    """Ensure convergence studies distinguish policy failures, unsupported cases and aliases."""

    def test_domain_recovery_and_full_step_failure_remain_visible(self):
        """Uncapped backtracking recovers, controlled steps avoid invalid trials, full steps fail."""
        problem = LogarithmicDomainProblem(4)
        for method in ("newton_raphson", "gauss_newton"):
            controlled = run_custom_case(problem, method, "controlled")
            recovered = run_custom_case(problem, method, "backtracking")
            full = run_custom_case(problem, method, "full")
            for run in (controlled, recovered, full):
                audit_custom_steps(problem, run)
            self.assertTrue(controlled["success"])
            self.assertTrue(recovered["success"])
            self.assertFalse(full["success"])
            self.assertEqual(
                sum(not item["valid"] for item in controlled["evaluations"]), 0
            )
            self.assertGreater(
                sum(not item["valid"] for item in recovered["evaluations"]), 0
            )
            self.assertGreater(
                sum(not item["valid"] for item in full["evaluations"]), 0
            )

    def test_full_step_audit_permits_nonmonotone_history(self):
        """A committed Newton residual increase is recorded and not rejected by protected-only audits."""
        problem = ExtendedRosenbrockProblem(2)
        run = run_custom_case(problem, "newton_raphson", "full")
        audit_custom_steps(problem, run)
        self.assertTrue(run["success"])
        self.assertGreater(
            summarize_case(problem, run)["residual_increases_at_committed_points"], 0
        )

    def test_rectangular_newton_is_explicitly_unsupported(self):
        """An incompatible dimension is retained without attempting a solve or counting a failure."""
        problem = CoupledOverdeterminedRosenbrockProblem(4)
        run = run_custom_case(problem, "newton_raphson", "controlled")
        row = summarize_case(problem, run)
        self.assertFalse(row["supported"])
        self.assertEqual(row["nfev"], 0)
        self.assertIn("Newton–Raphson", display_label(run))
        self.assertIn("unsupported", display_label(run))


if __name__ == "__main__":
    unittest.main()
