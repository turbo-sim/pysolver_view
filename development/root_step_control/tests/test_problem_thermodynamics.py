"""Validate the physical timing workload independently of the timing harness."""

import unittest

import numpy as np
from scipy.optimize import root

from ..src.problems.thermodynamics import CP, HeatedGasDuctProblem
from pysolver_view.solvers.root.solver import DomainError, solve_root


@unittest.skipIf(CP is None, "CoolProp is optional for the timing experiment")
class HeatedGasDuctTests(unittest.TestCase):
    """Check thermodynamic conservation, analytic derivatives, state reset and grid
    refinement.
    """

    def test_refined_mesh_preserves_total_load_and_physical_regime(self):
        """Refining to 50 cells must preserve total loading and yield a physical
        low-speed solution.
        """
        original = HeatedGasDuctProblem(n_cells=10)
        refined = HeatedGasDuctProblem(
            n_cells=50, total_heat_per_mass=1e5, total_loss_coefficient=80
        )
        self.assertEqual(np.sum(original.heat_per_mass), np.sum(refined.heat_per_mass))
        self.assertEqual(
            original.n_cells * original.loss_coefficient,
            refined.n_cells * refined.loss_coefficient,
        )
        self.assertEqual(refined.n, 100)
        result = root(refined.residual, refined.x0, jac=refined.gradient, method="hybr")
        self.assertLess(np.linalg.norm(refined.residual(result.x)), 1e-8)
        physical = refined.solution_diagnostics(result.x)
        self.assertTrue(physical["temperature_increases"])
        self.assertTrue(physical["pressure_decreases"])
        self.assertLess(physical["maximum_velocity_m_per_s"], 30)

    def test_analytic_scaled_jacobian_matches_central_differences(self):
        """Normalized thermodynamic derivatives must agree with scaled
        central-difference checks.
        """
        problem = HeatedGasDuctProblem(n_cells=4)
        for offset in (np.zeros(problem.n), np.tile([30.0, -1e5], problem.n_cells)):
            x = problem.x0 + offset
            analytic = problem.gradient(x) * problem.x_scale
            numerical = np.empty_like(analytic)
            for j in range(problem.n):
                increment = 1e-5 * problem.x_scale[j]
                xp, xm = x.copy(), x.copy()
                xp[j] += increment
                xm[j] -= increment
                numerical[:, j] = (problem.residual(xp) - problem.residual(xm)) / 2e-5
            np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-8)

    def test_each_call_updates_all_requested_states(self):
        """Every residual/Jacobian call must refresh all cells and recover state after
        an invalid request.
        """
        problem = HeatedGasDuctProblem(n_cells=4)
        first = problem.residual(problem.x0)
        self.assertEqual(problem.property_updates, problem.n_cells)
        problem.residual(problem.x0 + np.tile([20.0, -2e4], problem.n_cells))
        problem.gradient(problem.x0)
        repeated = problem.residual(problem.x0)
        self.assertEqual(problem.property_updates, 4 * problem.n_cells)
        np.testing.assert_array_equal(first, repeated)
        invalid = problem.x0.copy()
        invalid[1] = -1.0
        with self.assertRaises(DomainError):
            problem.residual(invalid)
        np.testing.assert_array_equal(first, problem.residual(problem.x0))

    def test_all_four_solvers_reach_a_physical_conservation_solution(self):
        """Native and custom LM/hybrid solutions must satisfy conservation and the
        expected temperature/pressure trends.
        """
        for family in ("scipy", "custom"):
            for method in ("lm", "hybr"):
                with self.subTest(family=family, method=method):
                    problem = HeatedGasDuctProblem(n_cells=10)
                    if family == "scipy":
                        options = dict(
                            factor=100.0, diag=1 / problem.x_scale, xtol=1e-12
                        )
                        if method == "lm":
                            options.update(ftol=1e-12, gtol=1e-12)
                        result = root(
                            problem.residual,
                            problem.x0,
                            jac=problem.gradient,
                            method=method,
                            options=options,
                        )
                    else:
                        result = solve_root(
                            problem.residual,
                            problem.x0,
                            problem.gradient,
                            method=method,
                            options=dict(
                                initial_radius=100.0,
                                max_radius=1e6,
                                x_scale=problem.x_scale,
                                residual_tol=1e-8,
                            ),
                        )
                    self.assertLessEqual(
                        np.linalg.norm(problem.residual(result.x)), 1e-8
                    )
                    diagnostics = problem.solution_diagnostics(result.x)
                    self.assertTrue(diagnostics["temperature_increases"])
                    self.assertTrue(diagnostics["pressure_decreases"])
                    self.assertLess(diagnostics["max_energy_error_J_per_kg"], 1e-4)
                    self.assertLess(diagnostics["max_pressure_error_Pa"], 2e-4)
                    self.assertLess(abs(diagnostics["total_heat_error_J_per_kg"]), 1e-4)
                    self.assertLess(diagnostics["maximum_velocity_m_per_s"], 30.0)


if __name__ == "__main__":
    unittest.main()
