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
from pysolver_view.roots.problems import (
    CoupledOverdeterminedRosenbrockProblem,
    ExtendedRosenbrockProblem,
    LogarithmicDomainProblem,
)





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
        self.assertIn("Newton-Raphson", display_label(run))
        self.assertIn("unsupported", display_label(run))


if __name__ == "__main__":
    unittest.main()
