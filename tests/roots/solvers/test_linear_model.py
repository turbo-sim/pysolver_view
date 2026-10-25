"""Numerical properties of QR updates and constrained LM/dogleg subproblems."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.linalg import norm
from scipy.optimize import brentq

from pysolver_view.roots.solvers.linear_model import LinearModel


class LinearModelTests(unittest.TestCase):
    """Compare subproblem steps and updated factors against independent numerical
    references.
    """

    def test_qr_gn_matches_least_squares_without_an_svd(self):
        """Healthy square and overdetermined QR models must match independent least
        squares without computing SVD.
        """
        rng = np.random.default_rng(71)
        for m, n in ((2, 2), (15, 10), (30, 30)):
            with self.subTest(m=m, n=n):
                a, f = rng.normal(size=(m, n)), rng.normal(size=m)
                model = LinearModel(a, f)
                reference = np.linalg.lstsq(a, -f, rcond=None)[0]
                np.testing.assert_allclose(model.gauss_newton, reference, rtol=1e-12)
                for method in ("lm", "hybr"):
                    step = model.step(method, 2 * norm(reference))
                    np.testing.assert_allclose(step, reference, rtol=1e-12)
                self.assertEqual(model.stats["svd_factorizations"], 0)
                self.assertEqual(model.stats["qr_factorizations"], 1)
                self.assertEqual(model.stats["triangular_solves"], 1)

    def test_rank_deficient_and_ill_conditioned_minimum_norm_fallback(self):
        """Singular and ill-conditioned models must use SVD and match the numerical-rank
        minimum-norm reference.
        """
        for a in (
            np.zeros((5, 3)),
            np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12.0]]),
            np.diag([1.0, 1e-10, 1e-17]),
        ):
            with self.subTest(a=a):
                f = np.arange(1, a.shape[0] + 1, dtype=float)
                model = LinearModel(a, f)
                expected = np.linalg.lstsq(a, -f, rcond=None)[0]
                np.testing.assert_allclose(model.gauss_newton, expected, rtol=1e-12)
                self.assertEqual(model.stats["svd_factorizations"], 1)
                self.assertEqual(model.stats["triangular_solves"], 0)

    def test_lm_matches_independent_secular_root_and_kkt_conditions(self):
        """Randomized LM steps must match an independent damping-root solve and satisfy
        feasibility and stationarity.
        """
        rng = np.random.default_rng(142)
        for m, n in ((5, 5), (16, 10)):
            for condition in (1, 1e4, 1e10):
                q, _ = np.linalg.qr(rng.normal(size=(m, n)))
                v, _ = np.linalg.qr(rng.normal(size=(n, n)))
                a = (q * np.geomspace(1, 1 / condition, n)) @ v.T
                f = rng.normal(size=m)
                u, s, vh = np.linalg.svd(a, full_matrices=False)
                weights, d = s * (u.T @ f), s * s
                gn = np.linalg.lstsq(a, -f, rcond=None)[0]
                model = LinearModel(a, f)
                for fraction in (0.001, 0.2, 0.99):
                    with self.subTest(m=m, n=n, condition=condition, fraction=fraction):
                        radius = fraction * norm(gn)
                        upper = norm(weights) / radius
                        damping = brentq(
                            lambda value: norm(weights / (d + value)) - radius,
                            0,
                            upper,
                            xtol=np.nextafter(0.0, 1.0),
                            rtol=1e-14,
                        )
                        expected = -(vh.T @ (weights / (d + damping)))
                        step = model.step("lm", radius)
                        self.assertLessEqual(norm(step), radius * (1 + 3e-16))
                        self.assertGreaterEqual(norm(step), radius * (1 - 2e-10))
                        np.testing.assert_allclose(step, expected, rtol=2e-7, atol=1e-9)
                        stationarity = a.T @ (f + a @ step) + damping * step
                        self.assertLess(
                            norm(stationarity), 3e-7 * max(1, norm(model.g))
                        )
                self.assertEqual(model.stats["svd_factorizations"], 1)

    def test_dogleg_matches_boundary_geometry_and_cauchy_decrease(self):
        """Dogleg steps must lie on the correct segment/boundary and improve at least as
        much as the clipped Cauchy step.
        """
        rng = np.random.default_rng(221)
        for _ in range(30):
            a, f = rng.normal(size=(8, 5)), rng.normal(size=8)
            model = LinearModel(a, f)
            gn = model.gauss_newton
            g = a.T @ f
            cauchy = -g * (g @ g) / ((a @ g) @ (a @ g))
            for radius in (
                0.5 * norm(cauchy),
                (norm(cauchy) + norm(gn)) / 2,
                norm(gn) * (1 - 1e-12),
            ):
                step = model.step("hybr", radius)
                self.assertLessEqual(norm(step), radius * (1 + 3e-16))
                self.assertAlmostEqual(norm(step) / radius, 1.0, places=12)
                clipped_cauchy = cauchy * min(1.0, radius / norm(cauchy))
                self.assertLessEqual(
                    norm(f + a @ step), norm(f + a @ clipped_cauchy) * (1 + 1e-14)
                )
                if radius > norm(cauchy):
                    displacement = step - cauchy
                    ray = gn - cauchy
                    fraction = (displacement @ ray) / (ray @ ray)
                    self.assertGreaterEqual(fraction, 0)
                    self.assertLessEqual(fraction, 1)
                    np.testing.assert_allclose(displacement, fraction * ray, atol=1e-13)
            self.assertEqual(model.stats["svd_factorizations"], 0)

    def test_updated_qr_keeps_permutation_and_residual_anchor(self):
        """Rank-one QR updates must preserve the permuted factorization, orthogonality
        and current residual anchor.
        """
        rng = np.random.default_rng(503)
        for m, n in ((9, 9), (15, 9)):
            a = rng.normal(size=(m, n)) * np.arange(1, n + 1)
            f = rng.normal(size=m)
            stats = {}
            model = LinearModel(a, f, stats=stats, refresh_interval=7)
            self.assertFalse(np.array_equal(model.perm, np.arange(n)))
            for iteration in range(30):
                correction = rng.normal(size=m) * 0.001
                direction = rng.normal(size=n)
                # Alternate accepted/rejected anchors; update both kinds.
                if iteration % 2 == 0:
                    f = rng.normal(size=m)
                a += np.outer(correction, direction)
                self.assertTrue(model.update(correction, direction, f))
                np.testing.assert_allclose(model.a, a, rtol=1e-14, atol=1e-14)
                np.testing.assert_allclose(
                    model.q @ model.r, a[:, model.perm], atol=2e-13
                )
                np.testing.assert_allclose(model.q.T @ model.q, np.eye(n), atol=5e-14)
                np.testing.assert_allclose(model.g, a.T @ f, atol=2e-13)
                expected = np.linalg.lstsq(a, -f, rcond=None)[0]
                np.testing.assert_allclose(model.gauss_newton, expected, atol=2e-12)
            self.assertEqual(stats["qr_factorizations"], 5)
            self.assertEqual(stats["qr_updates"], 26)
            self.assertEqual(stats["factorizations_refreshed"], 4)
            self.assertEqual(stats["svd_factorizations"], 0)

    def test_update_invalidates_svd_and_gn_at_new_residual(self):
        """Changing the residual anchor must invalidate cached steps and SVD projections
        before the next solve.
        """
        a = np.diag([2.0, 3.0, 4.0])
        model = LinearModel(a, np.ones(3))
        model.step("lm", 0.01)
        self.assertEqual(model.stats["svd_factorizations"], 1)
        self.assertTrue(model.update(np.zeros(3), np.ones(3), np.full(3, 2.0)))
        np.testing.assert_allclose(model.gauss_newton, -2 / np.diag(a))
        model.step("lm", 0.01)
        self.assertEqual(model.stats["svd_factorizations"], 2)

    def test_condition_guard_detects_bad_off_diagonal_after_update(self):
        """The condition guard must detect an unhealthy updated factor even when its
        diagonal stays unchanged.
        """
        model = LinearModel(np.eye(3), np.ones(3))
        # All diagonal entries remain one, but conditioning becomes poor.
        self.assertFalse(model.update(np.array([1e10, 0, 0]), [0, 1, 0], np.ones(3)))

    def test_gradient_overflow_and_nonfinite_updates_fail_without_warnings(self):
        """Unusable updates and overflowing initial gradients must fail cleanly under
        warnings-as-errors.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = LinearModel(np.eye(2), np.ones(2))
            self.assertFalse(model.update([1e300, 1e300], [1, 1], [1e100, 1e100]))
            self.assertFalse(model.update([np.inf, 0], [1, 1], np.ones(2)))
            self.assertFalse(model.update([0, 0], [1, 1], [0, np.nan]))
            with self.assertRaises(ValueError):
                LinearModel(np.full((2, 2), 1e300), np.full(2, 1e100))

    def test_missing_condition_estimator_preserves_svd_fallback(self):
        """Without TRCON, the model must retain a correct conservative SVD solve."""
        with patch("pysolver_view.roots.solvers.linear_model._TRCON", None):
            model = LinearModel(np.eye(4), np.arange(4.0))
            np.testing.assert_array_equal(model.gauss_newton, -np.arange(4.0))
            self.assertEqual(model.stats["svd_factorizations"], 1)

    def test_dogleg_large_coordinates_and_tiny_radii_remain_feasible(self):
        """Extreme model scales and tiny radii must still produce finite feasible dogleg
        steps.
        """
        for a, f, radius in (
            (np.diag([1.0, 1e-14]), np.ones(2), 1e7),
            (np.eye(3), np.ones(3), 1e-200),
            (np.eye(2) * 1e150, np.ones(2) * 1e100, 1e-60),
        ):
            model = LinearModel(a, f)
            step = model.step("hybr", radius)
            self.assertTrue(np.all(np.isfinite(step)))
            self.assertLessEqual(norm(step), radius * (1 + 3e-16))
            self.assertGreater(norm(step), radius * (1 - 1e-12))


if __name__ == "__main__":
    unittest.main()
