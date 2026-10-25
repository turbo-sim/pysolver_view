"""Behavioral checks for the production SciPy SLSQP adapter.

Run from the repository root with:

    .venv/Scripts/python.exe -m unittest discover -s tests -p test_custom_slsqp.py -v

The tests compare ordinary problems with SciPy's public solver and exercise
the new controls using small problems whose trial points are easy to inspect.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Barrier
import unittest

import numpy as np
from scipy.optimize import _slsqp_py, minimize

from pysolver_view.solvers.slsqp import (
    DomainError,
    Evaluation,
    SLSQPOptions,
    minimize_slsqp,
)


class SLSQPWrapperTests(unittest.TestCase):
    def test_quadratic_matches_public_scipy(self):
        target = np.array([0.75, -1.25, 2.0])
        weights = np.array([1.0, 3.0, 0.5])

        def objective(x):
            return np.dot(weights, (x - target) ** 2)

        def gradient(x):
            return 2.0 * weights * (x - target)

        x0 = np.array([3.0, 2.0, -1.0])
        public = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 100},
        )
        result = minimize_slsqp(
            lambda x: Evaluation(objective(x)),
            x0,
            jac=lambda x: gradient(x)[None, :],
            options=SLSQPOptions(first_step_fraction=1.0, ftol=1e-11, maxiter=100),
        )
        self.assertTrue(public.success, public.message)
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.solver_success)
        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.x, public.x, atol=1e-8, rtol=0)
        self.assertAlmostEqual(result.fun, public.fun, places=10)

    def test_equalities_inequalities_and_bounds_match_public_scipy(self):
        def objective(x):
            return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2

        def gradient(x):
            return 2.0 * (x - np.array([1.0, 2.0]))

        def evaluate(x):
            return Evaluation(
                objective(x),
                equalities=np.array([x.sum() - 1.0]),
                inequalities=np.array([x[0] - 0.2]),
            )

        def jacobian(x):
            return np.vstack([gradient(x), [1.0, 1.0], [1.0, 0.0]])

        bounds = [(0.0, 3.0), (0.0, 3.0)]
        public = minimize(
            objective,
            [0.5, 0.5],
            jac=gradient,
            bounds=bounds,
            method="SLSQP",
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x.sum() - 1.0,
                    "jac": lambda x: np.ones(2),
                },
                {
                    "type": "ineq",
                    "fun": lambda x: x[0] - 0.2,
                    "jac": lambda x: np.array([1.0, 0.0]),
                },
            ],
            options={"ftol": 1e-11},
        )
        result = minimize_slsqp(
            evaluate,
            [0.5, 0.5],
            bounds=bounds,
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=1.0, ftol=1e-11),
        )
        self.assertTrue(public.success, public.message)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, public.x, atol=1e-9, rtol=0)
        np.testing.assert_allclose(result.x, [0.2, 0.8], atol=1e-9, rtol=0)

    def test_nonlinear_constraint_matches_public_scipy(self):
        def objective(x):
            return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

        def gradient(x):
            return np.array(
                [
                    -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                    200.0 * (x[1] - x[0] ** 2),
                ]
            )

        def constraint(x):
            return 1.0 - np.dot(x, x)

        bounds = [(0.0, 2.0), (-2.0, 2.0)]
        public = minimize(
            objective,
            [0.5, 0.5],
            jac=gradient,
            bounds=bounds,
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": constraint, "jac": lambda x: -2.0 * x}
            ],
            options={"ftol": 1e-10, "maxiter": 150},
        )
        result = minimize_slsqp(
            lambda x: Evaluation(objective(x), inequalities=[constraint(x)]),
            [0.5, 0.5],
            bounds=bounds,
            jac=lambda x: np.vstack([gradient(x), -2.0 * x]),
            options=SLSQPOptions(first_step_fraction=1.0, ftol=1e-10),
        )
        self.assertTrue(public.success, public.message)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, public.x, atol=1e-7, rtol=0)
        self.assertGreaterEqual(constraint(result.x), -1e-6)

    @staticmethod
    def _sqrt_problem(attempts):
        def evaluate(x):
            attempts.append(float(x[0]))
            if x[0] <= 0.0:
                raise DomainError("The square root requires x > 0.")
            return Evaluation((np.sqrt(x[0]) - 0.2) ** 2)

        def jacobian(x):
            return np.array([[1.0 - 0.2 / np.sqrt(x[0])]])

        return evaluate, jacobian

    def test_invalid_full_trial_is_halved_and_converges(self):
        attempts = []
        evaluate, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(
            evaluate,
            [0.36],
            bounds=[(-2.0, 2.0)],
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=1.0, ftol=1e-11),
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.x[0], 0.04, places=6)
        self.assertTrue(any(x < 0.0 for x in attempts))
        first_rejected = next(i for i, x in enumerate(attempts) if x < 0.0)
        self.assertAlmostEqual(
            attempts[first_rejected + 1],
            0.36 + 0.5 * (attempts[first_rejected] - 0.36),
            places=12,
        )
        self.assertGreater(len(result.events), 0)

    def test_disabled_domain_guard_returns_controlled_failure(self):
        attempts = []
        evaluate, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(
            evaluate,
            [0.36],
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=1.0, domain_backtracking=False),
        )
        self.assertFalse(result.success)
        self.assertFalse(result.solver_success)
        self.assertTrue(any(x < 0.0 for x in attempts))
        self.assertTrue(result.message)

    def test_invalid_initial_point_returns_controlled_failure(self):
        attempts = []
        evaluate, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(evaluate, [-0.1], jac=jacobian)
        self.assertFalse(result.success)
        self.assertFalse(result.solver_success)
        self.assertEqual(attempts, [-0.1])
        self.assertTrue(result.message)

    def test_fd_probes_respect_bounds_at_upper_endpoint(self):
        attempts = []

        def evaluate(x):
            attempts.append(float(x[0]))
            if not -1.0 <= x[0] <= 0.0:
                raise AssertionError("The evaluator was called outside the bounds.")
            return Evaluation((x[0] + 0.4) ** 2)

        result = minimize_slsqp(evaluate, [0.0], bounds=[(-1.0, 0.0)])
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.x[0], -0.4, places=5)
        self.assertLess(attempts[1], 0.0)

    def test_invalid_derivative_probe_stops_without_trying_opposite_side(self):
        attempts = []

        def evaluate(x):
            attempts.append(float(x[0]))
            if x[0] > 0.0:
                raise DomainError("Only x <= 0 has a model evaluation.")
            return Evaluation((x[0] + 0.4) ** 2)

        result = minimize_slsqp(evaluate, [0.0], bounds=[(-1.0, 1.0)])
        self.assertFalse(result.success)
        self.assertEqual(result.status, 12)
        self.assertEqual(attempts, [0.0, SLSQPOptions().finite_diff_step])
        self.assertEqual(result.derivative_rejections, 1)
        self.assertIn("not retried", result.message)
        self.assertFalse(result.diagnostics_available)

    def test_invalid_derivative_probe_is_not_shrunk(self):
        attempts = []

        def evaluate(x):
            attempts.append(float(x[0]))
            if abs(x[0]) >= 0.01:
                raise DomainError("The model domain is |x| < 0.01.")
            # A flat quartic keeps derivative truncation below the final
            # stationarity tolerance even with this intentionally large probe.
            return Evaluation(x[0] ** 4)

        result = minimize_slsqp(
            evaluate,
            [0.0],
            bounds=[(-1.0, 1.0)],
            options=SLSQPOptions(
                first_step_fraction=1.0, finite_diff_step=0.1, ftol=1e-5
            ),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, 12)
        self.assertEqual(attempts, [0.0, 0.1])
        self.assertEqual(result.fun, 0.0)
        self.assertIsNone(result.kkt)

    @staticmethod
    def _quadratic_with_trials(options):
        attempts = []

        def evaluate(x):
            attempts.append(float(x[0]))
            return Evaluation(x[0] ** 2)

        result = minimize_slsqp(
            evaluate,
            [1.0],
            jac=lambda x: np.array([[2.0 * x[0]]]),
            options=options,
        )
        return result, attempts

    def test_first_step_cap_does_not_damp_later_iterations(self):
        result, attempts = self._quadratic_with_trials(
            SLSQPOptions(first_step_fraction=0.25, step_growth_factor=None, ftol=1e-12)
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(attempts[1], 0.5, places=12)
        self.assertAlmostEqual(attempts[2], 0.0, places=12)

    def test_every_iteration_cap_and_first_step_cap_both_apply(self):
        result, attempts = self._quadratic_with_trials(
            SLSQPOptions(
                first_step_fraction=0.25,
                max_step_fraction=0.5,
                step_growth_factor=None,
                ftol=1e-12,
            )
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(attempts[1], 0.5, places=12)
        self.assertAlmostEqual(attempts[2], 0.25, places=12)
        self.assertAlmostEqual(attempts[3], 0.125, places=12)

    def test_disabled_step_growth_preserves_existing_trial_sequence(self):
        disabled, disabled_trials = self._quadratic_with_trials(
            SLSQPOptions(first_step_fraction=0.125, step_growth_factor=None, ftol=1e-12)
        )
        self.assertTrue(disabled.success, disabled.message)
        np.testing.assert_allclose(disabled_trials, [1.0, 0.75, 0.0], atol=1e-14)
        evaluate, jacobian = self._sqrt_problem([])
        recovered = minimize_slsqp(
            evaluate,
            [0.36],
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=1.0, step_growth_factor=None),
        )
        self.assertTrue(recovered.success, recovered.message)
        self.assertGreater(recovered.domain_rejections, 0)
        self.assertFalse(
            any(
                event["kind"] in ("step_cap", "step_restoration")
                for event in recovered.events
            )
        )

    def test_growth_caps_real_trials_then_restores_full_step(self):
        result, attempts = self._quadratic_with_trials(
            SLSQPOptions(first_step_fraction=0.125, step_growth_factor=2.0, ftol=1e-12)
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(
            attempts, [1.0, 0.75, 0.5625, 0.28125, 0.0], atol=1e-14
        )
        caps = [event for event in result.events if event["kind"] == "step_cap"]
        self.assertEqual([event["iteration"] for event in caps], [1, 2, 3])
        np.testing.assert_allclose(
            [event["fraction"] for event in caps], [0.125, 0.25, 0.5]
        )

    def test_growth_respects_permanent_cap_and_large_factors_do_not_overflow(self):
        for growth in (2.0, 1e308):
            with self.subTest(growth=growth):
                result, attempts = self._quadratic_with_trials(
                    SLSQPOptions(
                        first_step_fraction=0.125,
                        max_step_fraction=0.5,
                        step_growth_factor=growth,
                        ftol=1e-12,
                    )
                )
                self.assertTrue(result.success, result.message)
                caps = [event for event in result.events if event["kind"] == "step_cap"]
                self.assertGreater(len(caps), 3)
                expected = [0.125, 0.25] if growth == 2 else [0.125, 0.5]
                expected += [0.5] * (len(caps) - 2)
                np.testing.assert_allclose(
                    [event["fraction"] for event in caps], expected
                )
                self.assertTrue(np.all(np.isfinite(attempts)))

    def test_unit_growth_keeps_initial_cap(self):
        result, attempts = self._quadratic_with_trials(
            SLSQPOptions(
                first_step_fraction=0.25,
                max_step_fraction=0.5,
                step_growth_factor=1.0,
                ftol=1e-12,
            )
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(attempts[:4], [1.0, 0.5, 0.375, 0.28125])
        caps = [event for event in result.events if event["kind"] == "step_cap"]
        self.assertGreater(len(caps), 3)
        self.assertTrue(all(event["fraction"] == 0.25 for event in caps))

    def test_growth_and_domain_backtracking_work_together(self):
        attempts = []
        evaluate, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(
            evaluate,
            [0.36],
            jac=jacobian,
            options=SLSQPOptions(
                first_step_fraction=0.25, step_growth_factor=1.5, ftol=1e-11
            ),
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.x[0], 0.04, places=6)
        rejections = [
            event for event in result.events if event["kind"] == "trial_rejection"
        ]
        self.assertTrue(rejections)
        self.assertTrue(all(event["iteration"] > 1 for event in rejections))
        caps = [event for event in result.events if event["kind"] == "step_cap"]
        np.testing.assert_allclose(
            [event["fraction"] for event in caps[:2]], [0.25, 0.375]
        )
        restorations = [
            event for event in result.events if event["kind"] == "step_restoration"
        ]
        self.assertTrue(restorations)
        by_iteration = {event["iteration"]: event["fraction"] for event in caps}
        for event in restorations:
            self.assertAlmostEqual(
                event["next_fraction"], min(1.0, event["accepted_fraction"] * 1.5)
            )
            self.assertLess(
                event["accepted_fraction"], by_iteration.get(event["iteration"], 1.0)
            )
            next_iteration = event["iteration"] + 1
            if next_iteration in by_iteration:
                self.assertAlmostEqual(
                    by_iteration[next_iteration], event["next_fraction"]
                )
        self.assertGreater(result.domain_rejections, 0)
        self.assertLessEqual(result.kkt_ratio, 1.0)

    def test_step_growth_validation_and_default_iteration_budget(self):
        defaults = vars(SLSQPOptions())
        self.assertEqual(defaults["maxiter"], 200)
        self.assertEqual(defaults["step_growth_factor"], 1.5)
        for valid in (None, 1, 1.5, 1e308):
            with self.subTest(valid=valid):
                self.assertEqual(
                    vars(SLSQPOptions(step_growth_factor=valid))["step_growth_factor"],
                    valid,
                )
        for invalid in (True, np.bool_(False), "1.5", 0.9, -1, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "step_growth_factor"):
                    SLSQPOptions(step_growth_factor=invalid)

    def test_unexpected_evaluator_exception_propagates(self):
        def evaluate(x):
            if x[0] < 0.5:
                raise RuntimeError(
                    "Deliberate application bug, not a domain rejection."
                )
            return Evaluation(x[0] ** 2)

        with self.assertRaisesRegex(RuntimeError, "application bug"):
            minimize_slsqp(evaluate, [1.0], jac=lambda x: np.array([[2.0 * x[0]]]))

    def test_callback_receives_progress_and_exceptions_propagate(self):
        records = []
        result = minimize_slsqp(
            lambda x: Evaluation((x[0] - 0.4) ** 2),
            [1.0],
            callback=records.append,
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(records)
        self.assertTrue(result.history)

        def callback(record):
            raise RuntimeError("Deliberate callback failure.")

        with self.assertRaisesRegex(RuntimeError, "callback failure"):
            minimize_slsqp(
                lambda x: Evaluation((x[0] - 0.4) ** 2), [1.0], callback=callback
            )

    def test_iteration_limit_is_reported_as_failure(self):
        def evaluate(x):
            return Evaluation(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

        result = minimize_slsqp(
            evaluate,
            [-1.2, 1.0],
            options=SLSQPOptions(first_step_fraction=1.0, maxiter=1),
        )
        self.assertFalse(result.success)
        self.assertFalse(result.solver_success)
        self.assertEqual(result.status, 9)
        self.assertLessEqual(result.nit, 1)

    def test_does_not_replace_scipy_backend_on_success_or_failure(self):
        original = _slsqp_py.slsqp
        self._quadratic_with_trials(SLSQPOptions(first_step_fraction=0.5))
        self.assertIs(_slsqp_py.slsqp, original)
        evaluate, jacobian = self._sqrt_problem([])
        minimize_slsqp(
            evaluate,
            [0.36],
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=1.0, domain_backtracking=False),
        )
        self.assertIs(_slsqp_py.slsqp, original)

    def test_concurrent_solves_keep_their_controls_independent(self):
        ready = Barrier(4)
        original = _slsqp_py.slsqp

        def solve(index):
            target = float(index) - 1.5
            first_call = True
            attempts = []

            def evaluate(x):
                nonlocal first_call
                attempts.append(float(x[0]))
                if first_call:
                    first_call = False
                    ready.wait(timeout=15.0)
                return Evaluation((x[0] - target) ** 2)

            fraction = 0.1 * (index + 1)
            result = minimize_slsqp(
                evaluate,
                [target + 2.0],
                jac=lambda x: np.array([[2.0 * (x[0] - target)]]),
                options=SLSQPOptions(first_step_fraction=fraction, ftol=1e-12),
            )
            return target, fraction, result, attempts

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(solve, range(4)))
        for target, fraction, result, attempts in results:
            self.assertTrue(result.success, result.message)
            self.assertAlmostEqual(result.x[0], target, places=8)
            self.assertAlmostEqual(
                attempts[1], target + 2.0 - 4.0 * fraction, places=12
            )
        self.assertIs(_slsqp_py.slsqp, original)

    def test_unconstrained_rosenbrock_matches_public_scipy(self):
        def objective(x):
            return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

        def gradient(x):
            return np.array(
                [
                    -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                    200.0 * (x[1] - x[0] ** 2),
                ]
            )

        public = minimize(
            objective,
            [-1.2, 1.0],
            jac=gradient,
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 150},
        )
        result = minimize_slsqp(
            lambda x: Evaluation(objective(x)),
            [-1.2, 1.0],
            jac=lambda x: gradient(x)[None, :],
            options=SLSQPOptions(first_step_fraction=1.0, ftol=1e-11),
        )
        self.assertTrue(public.success, public.message)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, public.x, atol=1e-8, rtol=0)
        self.assertAlmostEqual(result.fun, public.fun, places=12)

    def test_domain_recovery_also_handles_later_major_iterations(self):
        attempts = []
        evaluate, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(
            evaluate,
            [0.36],
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=0.25, ftol=1e-11),
        )
        self.assertTrue(result.success, result.message)
        self.assertGreater(attempts[1], 0.0)
        rejections = [
            event for event in result.events if event["kind"] == "trial_rejection"
        ]
        self.assertTrue(rejections)
        self.assertTrue(all(event["iteration"] > 1 for event in rejections))
        self.assertAlmostEqual(result.x[0], 0.04, places=6)

    def test_exhausted_domain_backtracking_returns_last_accepted_point(self):
        attempts = []
        evaluate, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(
            evaluate,
            [0.36],
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=1.0, max_backtracks=0),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, 11)
        self.assertAlmostEqual(result.x[0], 0.36)
        self.assertAlmostEqual(result.fun, 0.16)

    def test_exhausted_fd_probes_fail_without_inventing_a_derivative(self):
        attempts = []

        def evaluate(x):
            attempts.append(float(x[0]))
            if x[0] != 0.0:
                raise DomainError("Only the initial point has a valid evaluation.")
            return Evaluation(1.0)

        result = minimize_slsqp(evaluate, [0.0], options=SLSQPOptions())
        self.assertFalse(result.success)
        self.assertFalse(result.solver_success)
        self.assertEqual(result.status, 12)
        self.assertEqual(result.x[0], 0.0)
        self.assertEqual(result.fun, 1.0)
        self.assertTrue(any(x > 0.0 for x in attempts))
        self.assertEqual(attempts, [0.0, SLSQPOptions().finite_diff_step])

    def test_native_merit_backtracking_remains_active_after_a_step_cap(self):
        attempts = []
        x0 = np.array([-1.2, 1.0])

        def evaluate(x):
            attempts.append(x.copy())
            return Evaluation(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)

        def jacobian(x):
            return np.array(
                [
                    [
                        -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                        200.0 * (x[1] - x[0] ** 2),
                    ]
                ]
            )

        result = minimize_slsqp(
            evaluate,
            x0,
            jac=jacobian,
            options=SLSQPOptions(first_step_fraction=0.5, ftol=1e-10),
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=2e-5, rtol=0)
        first_step, second_step = attempts[1] - x0, attempts[2] - x0
        np.testing.assert_allclose(first_step, -0.5 * jacobian(x0)[0], atol=1e-12)
        self.assertGreater(np.linalg.norm(second_step), 0.0)
        self.assertLess(np.linalg.norm(second_step), np.linalg.norm(first_step))
        np.testing.assert_allclose(
            second_step / np.linalg.norm(second_step),
            first_step / np.linalg.norm(first_step),
            atol=1e-12,
        )
        self.assertFalse(
            any(event["kind"] == "trial_rejection" for event in result.events)
        )

    def test_tiny_step_cap_does_not_misreport_stationarity(self):
        # Tiny caps can trigger native objective/step stopping criteria far
        # from stationarity. The wrapper must report the independent check.
        result, attempts = self._quadratic_with_trials(
            SLSQPOptions(first_step_fraction=1e-12)
        )
        self.assertTrue(result.solver_success)
        self.assertFalse(result.success)
        self.assertGreater(result.fun, 0.99)
        self.assertGreater(result.optimality, 1.9)
        self.assertLess(abs(attempts[1] - attempts[0]), 1e-10)

    def test_more_equalities_than_variables_is_a_controlled_failure(self):
        result = minimize_slsqp(
            lambda x: Evaluation(x[0] ** 2, equalities=[x[0], 2.0 * x[0]]),
            [0.5],
        )
        self.assertFalse(result.success)
        self.assertEqual(result.status, 2)
        self.assertEqual(result.njev, 0)
        self.assertEqual(result.x[0], 0.5)

    def test_scaled_inactive_constraint_cannot_fake_final_optimality(self):
        result = minimize_slsqp(
            lambda x: Evaluation(x[0] ** 2, inequalities=[1e-8 * x[0]]),
            [1.0],
            jac=lambda x: np.array([[2 * x[0]], [1e-8]]),
            options=SLSQPOptions(first_step_fraction=1e-12),
        )
        self.assertTrue(result.solver_success)
        self.assertFalse(result.success)
        self.assertGreater(result.optimality, 0.9)
        self.assertGreater(result.complementarity, 0.9)

    def test_stop_iteration_from_model_is_not_callback_cancellation(self):
        def evaluate(x):
            if x[0] < 0.5:
                raise StopIteration("unexpected model exception")
            return Evaluation(x[0] ** 2)

        with self.assertRaisesRegex(StopIteration, "unexpected model exception"):
            minimize_slsqp(evaluate, [1.0], jac=lambda x: np.array([[2 * x[0]]]))

    def test_changing_constraint_count_is_an_application_error(self):
        def evaluate(x):
            constraints = [1.0] if x[0] >= 0.5 else [1.0, 1.0]
            return Evaluation(x[0] ** 2, inequalities=constraints)

        with self.assertRaisesRegex(ValueError, "Constraint sizes changed"):
            minimize_slsqp(
                evaluate, [1.0], jac=lambda x: np.array([[2.0 * x[0]], [0.0]])
            )

    def test_nonfinite_results_require_explicit_domain_error_handling(self):
        for nonfinite in (np.nan, np.inf, -np.inf):
            with self.subTest(nonfinite=nonfinite):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    minimize_slsqp(lambda x: Evaluation(nonfinite), [1.0])

        def evaluate(x):
            return Evaluation(x[0] ** 2, inequalities=[np.nan if x[0] < 0.5 else 1.0])

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            minimize_slsqp(
                evaluate, [1.0], jac=lambda x: np.array([[2.0 * x[0]], [0.0]])
            )

    def test_nonfinite_jacobian_is_an_application_error(self):
        with self.assertRaisesRegex(ValueError, "Jacobian must be finite"):
            minimize_slsqp(
                lambda x: Evaluation(x[0] ** 2),
                [1.0],
                jac=lambda x: np.array([[np.nan]]),
            )

    def test_low_level_driver_does_not_call_high_level_solvers(self):
        # The numerical driver must stay independent of high-level solver facades.
        folder = str(Path(__file__).resolve().parents[1])
        script = """
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from pysolver_view.solvers.slsqp import Evaluation, minimize_slsqp
import pysolver_view
with patch.object(pysolver_view, 'OptimizationSolver', side_effect=AssertionError('Unexpected PSV solver')), patch.object(pysolver_view, 'NonlinearSystemSolver', side_effect=AssertionError('Unexpected PSV solver')):
    result = minimize_slsqp(lambda x: Evaluation((x[0] - 0.5)**2), [1.0])
assert result.success, result.message
assert abs(result.x[0] - 0.5) < 1e-5
print('Standalone solve passed without PSV solver calls.')
"""
        process = subprocess.run(
            [sys.executable, "-I", "-c", script, folder],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("Standalone solve passed", process.stdout)

    def test_recovery_preserves_failure_metadata(self):
        def evaluate(x):
            if x[0] < 0:
                raise DomainError(
                    "No real triangle",
                    metadata=dict(code="rotor_radicand", stage_index=1, value=-0.2),
                )
            return Evaluation((np.sqrt(x[0]) - 0.2) ** 2)

        result = minimize_slsqp(
            evaluate, [0.36], jac=lambda x: np.array([[1 - 0.2 / np.sqrt(x[0])]])
        )
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.domain_rejections, 0)
        # The default initial cap can emit a step_cap before the rejection.
        rejection = next(
            event for event in result.events if event["kind"] == "trial_rejection"
        )
        failure = rejection["failure"]
        self.assertEqual(failure["code"], "rotor_radicand")
        self.assertEqual(failure["stage_index"], 1)
        self.assertEqual(failure["value"], -0.2)
        self.assertTrue(failure["recoverable"])

    def test_history_kkt_and_step_use_the_reported_point(self):
        snapshots = []
        target = np.array([0.3, -0.4])
        result = minimize_slsqp(
            lambda x: Evaluation(np.dot(x - target, x - target)),
            [1.0, 0.8],
            jac=lambda x: (2 * (x - target))[None, :],
            options=dict(first_step_fraction=0.1),
            callback=snapshots.append,
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(snapshots), len(result.history))
        for index, entry in enumerate(result.history):
            expected = np.max(np.abs(2 * (entry["x"] - target)))
            self.assertAlmostEqual(entry["optimality"], expected)
            self.assertTrue(entry["diagnostics_available"])
            step = (
                0
                if index == 0
                else np.linalg.norm(entry["x"] - result.history[index - 1]["x"])
            )
            self.assertAlmostEqual(entry["step_norm"], step)
            self.assertEqual(snapshots[index]["kkt_ratio"], entry["kkt_ratio"])
        self.assertEqual(result.optimality, result.history[-1]["optimality"])
        self.assertEqual(result.kkt_ratio, result.history[-1]["kkt_ratio"])

    def test_failed_trial_keeps_kkt_of_last_accepted_design(self):
        attempts = []
        model, jacobian = self._sqrt_problem(attempts)
        result = minimize_slsqp(
            model,
            [0.36],
            jac=jacobian,
            options=dict(first_step_fraction=1.0, domain_backtracking=False),
        )
        self.assertEqual(result.status, 11)
        self.assertTrue(result.diagnostics_available)
        self.assertAlmostEqual(result.optimality, 2 / 3)
        self.assertEqual(result.optimality, result.history[-1]["optimality"])

    def test_unavailable_derivative_never_reports_zero_optimality(self):
        def model(x):
            if x[0] != 0:
                raise DomainError("No derivative probe is evaluable")
            return Evaluation(1.0)

        result = minimize_slsqp(model, [0.0], options=SLSQPOptions())
        self.assertEqual(result.status, 12)
        self.assertFalse(result.diagnostics_available)
        self.assertIsNone(result.history[0]["optimality"])
        self.assertIsNone(result.kkt)
        self.assertFalse(result.success)

    def test_log_reports_tolerances_and_kkt_with_no_duplicate_final_row(self):
        import io
        import logging

        stream = io.StringIO()
        logger = logging.Logger("kkt-test", logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))
        result = minimize_slsqp(
            lambda x: Evaluation(x[0] ** 2),
            [1.0],
            jac=lambda x: np.array([[2 * x[0]]]),
            logger=logger,
        )
        report = stream.getvalue()
        for heading in (
            "Step-norm",
            "Optimality",
            "Complement.",
            "KKT/tol",
            "Final KKT:",
            "native ftol",
        ):
            self.assertIn(heading, report)
        rows = [
            line
            for line in report.splitlines()
            if line.split() and line.split()[0].isdigit()
        ]
        self.assertEqual(len(rows), len(result.history))


if __name__ == "__main__":
    unittest.main()
