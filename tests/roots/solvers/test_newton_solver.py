"""Independent behavioral checks for full and protected Newton-family methods."""

import unittest
import warnings

import numpy as np

from pysolver_view.roots.solvers.linear_model import LinearModel
from pysolver_view.roots.solvers.solver import (
    BacktrackingOptions,
    DomainError,
    METHOD_DISPLAY_NAMES,
    NewtonOptions,
    StepLimitOptions,
    normalize_method,
    solve_root,
)

METHODS = ("newton_raphson", "gauss_newton")


def rosenbrock(x):
    """Return the coupled Rosenbrock system whose initial Newton step increases merit."""
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def rosenbrock_jacobian(x):
    """Return the independent analytic Jacobian of the Rosenbrock fixture."""
    return np.array([[-20 * x[0], 10], [-1, 0]])


def logarithmic(x):
    """Return coupled logarithmic equations defined only for positive coordinates."""
    if np.any(x <= 0):
        raise DomainError("Positive states are required.")
    return np.array([[1.0, 0.2], [0.3, 1.0]]) @ np.log(x)


def logarithmic_jacobian(x):
    """Differentiate the coupled logarithmic fixture analytically."""
    return np.array([[1.0, 0.2], [0.3, 1.0]]) / x


class NewtonSolverTests(unittest.TestCase):
    """Verify direction, policy, counting and recovery contracts from actual calls."""

    def test_author_names_are_canonical_in_registry_and_default_result(self):
        """Public names and returned metadata use authors, including normalized aliases."""
        self.assertEqual(
            set(METHOD_DISPLAY_NAMES),
            {"levenberg_marquardt", "powell_hybrid", "newton_raphson", "gauss_newton"},
        )
        result = solve_root(lambda x: x - [2, 3], [0, 0], lambda x: np.eye(2))
        self.assertTrue(result.success)
        self.assertEqual(result.method, "levenberg_marquardt")
        self.assertEqual(result.requested_method, "levenberg_marquardt")
        self.assertEqual(result.method_name, "Levenberg–Marquardt")
        self.assertEqual(normalize_method(" LM "), "levenberg_marquardt")
        self.assertEqual(normalize_method(" Powell_Hybrid "), "powell_hybrid")

    def test_full_linear_step_matches_independent_square_solve(self):
        """Both full-step directions solve a nonsymmetric square system in one move."""
        matrix = np.array([[2.0, 1.0], [0.5, 3.0]])
        target = np.array([3.0, -2.0])
        for method in METHODS:
            result = solve_root(
                lambda x: matrix @ x - target,
                [0.0, 0.0],
                lambda x: matrix,
                method=method,
                options={"backtracking": None},
            )
            self.assertTrue(result.success, result.message)
            np.testing.assert_allclose(result.x, np.linalg.solve(matrix, target))
            self.assertEqual(
                (result.nit, result.ntrials, result.nfev, result.njev), (1, 1, 2, 1)
            )
            self.assertEqual(result.trials[0]["step_fraction"], 1.0)
            self.assertIsNone(result.step_limit)
            self.assertIsNone(result.radius)
            self.assertEqual(result.nupdates, 0)

    def test_full_rosenbrock_commits_the_residual_increase(self):
        """Full steps must retain the known increasing first Rosenbrock iterate."""
        for method in METHODS:
            result = solve_root(
                rosenbrock,
                [-1.2, 1],
                rosenbrock_jacobian,
                method=method,
                options=NewtonOptions(backtracking=None),
            )
            self.assertTrue(result.success, result.message)
            np.testing.assert_allclose(result.history[1]["x"], [1.0, -3.84], atol=1e-13)
            self.assertGreater(
                result.history[1]["residual_norm"], result.history[0]["residual_norm"]
            )
            self.assertLess(result.trials[0]["actual_reduction"], 0)
            self.assertTrue(result.trials[0]["accepted"])

    def test_uncapped_armijo_uses_fixed_jacobian_during_retries(self):
        """Accepted Armijo steps satisfy an independent merit calculation and reuse factors."""
        for method in METHODS:
            result = solve_root(
                rosenbrock,
                [-1.2, 1],
                rosenbrock_jacobian,
                method=method,
                options={"backtracking": {"step_limits": None}},
            )
            self.assertTrue(result.success, result.message)
            self.assertGreater(result.nrejected, 0)
            self.assertEqual(result.njev, result.nit)
            self.assertEqual(result.linear_algebra["qr_factorizations"], result.njev)
            for trial in result.trials:
                base = trial["x_base"]
                displacement = trial["x"] - base
                slope = rosenbrock(base) @ (rosenbrock_jacobian(base) @ displacement)
                reduction = 0.5 * (
                    np.dot(rosenbrock(base), rosenbrock(base))
                    - np.dot(rosenbrock(trial["x"]), rosenbrock(trial["x"]))
                )
                self.assertIsNone(trial["step_limit"])
                if trial["accepted"]:
                    self.assertGreater(reduction, 0)
                    self.assertGreaterEqual(reduction, -1e-4 * slope - 1e-12)

    def test_zero_backtracks_is_not_full_step_mode(self):
        """Disabling retries still rejects the increasing initial Rosenbrock trial."""
        for method in METHODS:
            result = solve_root(
                rosenbrock,
                [-1.2, 1],
                rosenbrock_jacobian,
                method=method,
                options={"backtracking": {"max_backtracks": 0, "step_limits": None}},
            )
            self.assertFalse(result.success)
            self.assertEqual(
                (result.nit, result.ntrials, result.nbacktracks), (0, 1, 0)
            )
            np.testing.assert_array_equal(result.x, [-1.2, 1])
            self.assertIn("Backtracking exhausted", result.message)

    def test_actual_scaled_limits_and_growth(self):
        """Model calls honor the initial and persistent envelopes with anisotropic scales."""
        scale = np.array([0.2, 4.0])
        calls = []

        def residual(x):
            """Record actual model requests for an anisotropically scaled linear problem."""
            calls.append(x.copy())
            return x / scale - [2, 3]

        result = solve_root(
            residual,
            [0, 0],
            lambda x: np.diag(1 / scale),
            method="newton_raphson",
            options={
                "x_scale": scale,
                "backtracking": {
                    "step_limits": {
                        "initial_limit": 0.1,
                        "max_limit": 0.3,
                        "growth_factor": 1.5,
                    }
                },
            },
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(calls), 1 + result.ntrials)
        self.assertLessEqual(
            np.linalg.norm((calls[1] - calls[0]) / scale), 0.1 * (1 + 2e-14)
        )
        for point, trial in zip(calls[1:], result.trials):
            np.testing.assert_array_equal(point, trial["x"])
            actual = np.linalg.norm((point - trial["x_base"]) / scale)
            self.assertLessEqual(actual, trial["step_limit"] * (1 + 2e-14))
            self.assertLessEqual(trial["step_limit"], 0.3)
        self.assertLessEqual(result.max_step_limit_ratio, 1 + 2e-14)

    def test_domain_backtracking_and_carry_forward(self):
        """Undefined trials recover without new derivatives and constrain the next envelope."""
        for method in METHODS:
            result = solve_root(
                logarithmic,
                [10, 8],
                logarithmic_jacobian,
                method=method,
                options={
                    "backtracking": {
                        "step_limits": {
                            "initial_limit": 100.0,
                            "max_limit": 100.0,
                            "growth_factor": 1.5,
                        }
                    }
                },
            )
            self.assertTrue(result.success, result.message)
            self.assertGreater(result.ninvalid, 0)
            self.assertEqual(result.njev, result.nit)
            accepted = [row for row in result.trials if row["accepted"]]
            self.assertGreater(accepted[0]["backtracks"], 0)
            next_trials = [row for row in result.trials if row["iteration"] == 1]
            self.assertLessEqual(
                next_trials[0]["step_limit"], accepted[0]["scaled_step_norm"]
            )
            np.testing.assert_allclose(result.x, [1, 1], atol=1e-8)
            full = solve_root(
                logarithmic,
                [10, 8],
                logarithmic_jacobian,
                method=method,
                options={"backtracking": None},
            )
            self.assertFalse(full.success)
            self.assertEqual((full.nit, full.ninvalid, full.ntrials), (0, 1, 1))
            np.testing.assert_array_equal(full.x, [10, 8])

    def test_rank_contract_and_overdetermined_root(self):
        """Strict Newton rejects rank loss while Gauss–Newton uses minimum-norm solutions."""
        singular = np.array([[1.0, 1.0], [2.0, 2.0]])
        result = solve_root(
            lambda x: singular @ x - [2, 4],
            [0, 0],
            lambda x: singular,
            method="newton_raphson",
            options={"backtracking": None},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.numerical_rank, 1)
        self.assertEqual(result.ntrials, 0)
        least_squares = solve_root(
            lambda x: singular @ x - [2, 4],
            [0, 0],
            lambda x: singular,
            method="gauss_newton",
            options={"backtracking": None},
        )
        self.assertTrue(least_squares.success, least_squares.message)
        np.testing.assert_allclose(least_squares.x, [1, 1])
        matrix = np.array([[1, 0], [0, 1], [1, 2]], dtype=float)
        target = matrix @ np.array([2, -3])
        over = solve_root(
            lambda x: matrix @ x - target,
            [0, 0],
            lambda x: matrix,
            method="gauss_newton",
            options={"backtracking": None},
        )
        self.assertTrue(over.success, over.message)
        np.testing.assert_allclose(
            over.x, np.linalg.lstsq(matrix, target, rcond=None)[0]
        )
        with self.assertRaisesRegex(ValueError, "square"):
            solve_root(lambda x: matrix @ x, [0, 0], method="newton_raphson")

    def test_nonzero_stationary_residual_is_not_a_root(self):
        """An inconsistent overdetermined system cannot pass merely by minimizing merit."""
        result = solve_root(
            lambda x: np.r_[x - [1, 2], 1.0],
            [0, 0],
            lambda x: np.array([[1, 0], [0, 1], [0, 0]]),
            method="gauss_newton",
        )
        self.assertFalse(result.success)
        self.assertAlmostEqual(result.residual_norm, 1.0)
        self.assertIn("Stationary", result.message)

    def test_numeric_derivatives_and_partial_budget(self):
        """Finite-difference probes share the call budget and use the exact accepted base."""
        for method in METHODS:
            result = solve_root(
                lambda x: x - [1, 2],
                [0, 0],
                method=method,
                options={"backtracking": None, "max_nfev": 2},
            )
            self.assertFalse(result.success)
            self.assertEqual((result.nfev, result.njev, result.ntrials), (2, 1, 0))
            self.assertEqual(
                [row["kind"] for row in result.evaluations],
                ["initial", "finite_difference"],
            )
            complete = solve_root(
                lambda x: x - [1, 2],
                [0, 0],
                method=method,
                options={"backtracking": None},
            )
            self.assertTrue(complete.success, complete.message)
            self.assertEqual(complete.nfev, len(complete.evaluations))

    def test_derivative_domain_errors_and_programming_errors(self):
        """Only explicit domain errors become controlled failures; probes are not retried."""

        def unavailable_probe(x):
            """Permit the initial state but reject any derivative perturbation."""
            if np.any(x):
                raise DomainError("Derivative state unavailable.")
            return np.array([-1.0, -1.0])

        result = solve_root(unavailable_probe, [0, 0], method="newton_raphson")
        self.assertFalse(result.success)
        self.assertEqual((result.nfev, result.njev, result.ntrials), (2, 1, 0))
        self.assertIn("derivative", result.message)

        def broken(x):
            """Raise a programming exception after a valid initial model call."""
            if np.any(x):
                raise RuntimeError("Unexpected model bug.")
            return np.array([-1.0, -1.0])

        with self.assertRaisesRegex(RuntimeError, "model bug"):
            solve_root(broken, [0, 0], lambda x: np.eye(2), method="gauss_newton")

    def test_exhausted_trial_budget_and_disabled_logs(self):
        """Failed calls count identically even when history storage is disabled."""

        def isolated(x):
            """Provide only one valid base point so every nonzero trial is rejected."""
            if np.any(x):
                raise DomainError("Outside the available model domain.")
            return np.array([-1.0, -1.0])

        for level in ("full", "summary", "none"):
            result = solve_root(
                isolated,
                [0, 0],
                lambda x: np.eye(2),
                method="gauss_newton",
                options={"history_level": level, "max_iterations": 3},
            )
            self.assertEqual(
                (result.ntrials, result.nfev, result.ninvalid, result.nrejected),
                (3, 4, 3, 3),
            )
            self.assertFalse(result.success)
            np.testing.assert_array_equal(result.x, [0, 0])
            if level == "none":
                self.assertEqual(result.trials, [])
                self.assertEqual(result.evaluations, [])

    def test_callback_copies_and_history_levels(self):
        """Stored logs do not affect numerical work, and callbacks cannot mutate state."""
        references = []
        for level in ("full", "summary", "none"):
            seen = []

            def callback(record):
                """Save the accepted point, damage the detached copy, and stop at move three."""
                seen.append(record["x"].copy())
                record["x"][:] = -999
                record["residual_norm"] = -1
                return len(seen) < 4

            result = solve_root(
                lambda x: x - [5, 3],
                [0, 0],
                lambda x: np.eye(2),
                method="newton_raphson",
                callback=callback,
                options={
                    "history_level": level,
                    "backtracking": {
                        "step_limits": {
                            "initial_limit": 0.1,
                            "max_limit": 1,
                            "growth_factor": 1.5,
                        }
                    },
                },
            )
            self.assertEqual(result.nit, 3)
            self.assertEqual(result.status, 5)
            np.testing.assert_array_equal(result.x, seen[-1])
            self.assertTrue(all(row["residual_norm"] > 0 for row in result.history))
            references.append(result)
        for result in references[1:]:
            np.testing.assert_array_equal(result.x, references[0].x)
            self.assertEqual(result.linear_algebra, references[0].linear_algebra)
            self.assertEqual(result.nfev, references[0].nfev)

    def test_aliases_and_explicit_none_have_consistent_defaults(self):
        """Aliases only select names; dictionary and dataclass policies remain equivalent."""
        for canonical, alias in (
            ("levenberg_marquardt", "lm"),
            ("powell_hybrid", "hybr"),
            ("newton_raphson", "newton"),
            ("gauss_newton", "gn"),
        ):
            results = [
                solve_root(
                    lambda x: x - [2, 3], [0, 0], lambda x: np.eye(2), method=name
                )
                for name in (canonical, alias)
            ]
            self.assertEqual(results[0].method, canonical)
            self.assertEqual(results[1].method, canonical)
            self.assertEqual(results[1].requested_method, alias)
            np.testing.assert_array_equal(results[0].x, results[1].x)
            self.assertEqual(results[0].ntrials, results[1].ntrials)
        for method in METHODS:
            default = solve_root(
                lambda x: x - [2, 3], [0, 0], lambda x: np.eye(2), method=method
            )
            self.assertEqual(default.step_policy, "backtracking")
            self.assertLessEqual(default.trials[0]["scaled_step_norm"], 1 + 2e-14)
            self.assertIsNone(NewtonOptions(backtracking=None).backtracking)
            self.assertIsNone(
                NewtonOptions(
                    backtracking={"step_limits": None}
                ).backtracking.step_limits
            )
        with self.assertRaises(TypeError):
            solve_root(lambda x: x, [1, 2], method="lm", options={"backtracking": None})

    def test_initial_root_and_large_gradient_range(self):
        """Newton does not require an unused overflowing gradient or scaled initial root."""
        for method in METHODS:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                initial = solve_root(
                    lambda x: np.zeros(2),
                    [1e300, -1e300],
                    method=method,
                    options={"x_scale": 1e-100},
                )
            self.assertTrue(initial.success)
            self.assertEqual(initial.njev, 0)
        target = np.array([1e-50, -2e-50])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = solve_root(
                lambda x: 1e200 * (x - target),
                [0, 0],
                lambda x: 1e200 * np.eye(2),
                method="newton_raphson",
                options={"backtracking": None, "xtol": 1e-100},
            )
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.x, target)
        small_gradient = solve_root(
            lambda x: 1e-20 * x - [1, 2],
            [0, 0],
            lambda x: 1e-20 * np.eye(2),
            method="newton_raphson",
            options={"backtracking": None},
        )
        self.assertTrue(small_gradient.success, small_gradient.message)

    def test_rounding_respects_a_small_cap_at_large_coordinates(self):
        """Rounded property-evaluation coordinates remain inside the requested envelope."""
        start = np.array([1e8, -1e8])
        result = solve_root(
            lambda x: x - (start + [1, 1]),
            start,
            lambda x: np.eye(2),
            method="newton_raphson",
            options={
                "max_iterations": 2,
                "xtol": 1e-16,
                "backtracking": {
                    "step_limits": {
                        "initial_limit": 0.01,
                        "max_limit": 0.01,
                    }
                },
            },
        )
        self.assertEqual(result.ntrials, 2)
        for trial in result.trials:
            length = np.linalg.norm(trial["x"] - trial["x_base"])
            self.assertLessEqual(length, 0.01 * (1 + 2e-14))

    def test_strict_rank_and_option_validation(self):
        """Rank-deficient linear models and ambiguous infinite policies fail explicitly."""
        model = LinearModel(np.ones((2, 2)), [1, 1], compute_gradient=False)
        with self.assertRaises(np.linalg.LinAlgError):
            _ = model.newton_raphson
        for invalid in (
            dict(backtracking=np.inf),
            dict(max_iterations=True),
            dict(x_scale=[1, 0]),
        ):
            with self.assertRaises((ValueError, TypeError)):
                NewtonOptions(**invalid)
        for value in (np.inf, None, -1, True):
            with self.assertRaises(ValueError):
                BacktrackingOptions(max_backtracks=value)
        for invalid in (
            dict(initial_limit=np.inf),
            dict(initial_limit=2, max_limit=1),
            dict(growth_factor=0.5),
        ):
            with self.assertRaises(ValueError):
                StepLimitOptions(**invalid)
        with self.assertRaises(ValueError):
            solve_root(lambda x: x, [1, 2], method="powell")


if __name__ == "__main__":
    unittest.main()
