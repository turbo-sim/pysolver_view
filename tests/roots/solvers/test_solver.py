"""Behavioral checks for the experimental root solvers.

Run from the repository root with::

    python -m pytest tests/roots/solvers -q

The checks use actual model calls and returned histories, including failed
trials, rather than relying only on the final solver status.
"""

import unittest
import warnings

import numpy as np

from pysolver_view.nonlinear_system import NonlinearSystemProblem

from pysolver_view.roots.problems import BrownAlmostLinearProblem, get_problems
from pysolver_view.roots.solvers.solver import (
    DomainError,
    StepControlOptions,
    solve_root,
)

METHODS = ("lm", "hybr")


def rosenbrock(x):
    """Return the two-equation Rosenbrock residual used by the behavioral fixtures."""
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def rosenbrock_jacobian(x):
    """Return the analytic 2-by-2 Jacobian of the behavioral Rosenbrock fixture."""
    return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


class RootStepControlTests(unittest.TestCase):
    """Check accepted-state consistency, trial bounds, budgets, recovery and solver
    termination contracts.
    """

    def test_initial_root_does_not_require_representable_scaled_coordinates(self):
        """An initial root must succeed without constructing scaled coordinates that
        would overflow.
        """
        # No step geometry is needed when the supplied point is already a root.
        # Forming x / scale here would overflow despite both inputs being finite.
        for method in METHODS:
            with self.subTest(method=method), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                result = solve_root(
                    lambda x: np.zeros(2),
                    [1e300, -1e300],
                    method=method,
                    options={"x_scale": 1e-100},
                )
                self.assertTrue(result.success)
                self.assertEqual(result.nfev, 1)
                self.assertEqual(result.njev, 0)
                self.assertEqual(result.ntrials, 0)

    def assert_consistent_history(self, result, fun, scale=1.0):
        """Every trial starts at the last accepted state; reports match calls."""
        accepted = result.history[0]["x"].copy()
        history_index = 1
        for trial in result.trials:
            np.testing.assert_array_equal(trial["x_base"], accepted)
            displacement = trial["x"] - accepted
            self.assertAlmostEqual(
                trial["step_norm"], np.linalg.norm(displacement), places=12
            )
            self.assertAlmostEqual(
                trial["scaled_step_norm"],
                np.linalg.norm(displacement / scale),
                places=12,
            )
            self.assertLessEqual(
                trial["scaled_step_norm"], trial["radius"] * (1 + 2e-14)
            )
            if trial["valid"]:
                self.assertAlmostEqual(
                    trial["residual_norm"],
                    np.linalg.norm(fun(trial["x"])),
                    places=12,
                )
            if trial["accepted"]:
                accepted = trial["x"].copy()
                np.testing.assert_array_equal(
                    result.history[history_index]["x"], accepted
                )
                history_index += 1
        self.assertEqual(history_index, len(result.history))
        self.assertEqual(result.nit, len(result.history) - 1)
        self.assertEqual(result.ntrials, len(result.trials))
        self.assertEqual(result.nfev, len(result.evaluations))
        np.testing.assert_array_equal(result.x, accepted)
        np.testing.assert_allclose(result.fun, fun(result.x), rtol=0, atol=0)
        norms = np.array([row["residual_norm"] for row in result.history])
        self.assertTrue(np.all(np.diff(norms) < 0))

    def test_initial_radius_growth_and_persistent_cap_bound_actual_calls(self):
        """Audit actual model-call locations against the first radius, growth rate and
        persistent cap.
        """
        for method in METHODS:
            with self.subTest(method=method):
                calls = []

                def fun(x):
                    """Record each requested coordinate and return the linear residual
                    toward the target (3, 4).
                    """
                    calls.append(x.copy())
                    return x - np.array([3.0, 4.0])

                options = dict(
                    initial_radius=0.1,
                    max_radius=0.3,
                    growth_factor=1.5,
                    residual_tol=1e-10,
                )
                result = solve_root(
                    fun,
                    [0.0, 0.0],
                    jac=lambda x: np.eye(2),
                    method=method,
                    options=options,
                )
                self.assertTrue(result.success, result.message)
                self.assertLessEqual(np.linalg.norm(calls[1] - calls[0]), 0.1)
                self.assertEqual(len(calls), 1 + result.ntrials)
                for call, trial in zip(calls[1:], result.trials):
                    np.testing.assert_array_equal(call, trial["x"])
                    self.assertLessEqual(
                        np.linalg.norm(call - trial["x_base"]), 0.3 * (1 + 2e-14)
                    )
                radii = np.array([row["radius"] for row in result.trials])
                self.assertTrue(np.all(radii[1:] <= 1.5 * radii[:-1] * (1 + 2e-14)))
                self.assertTrue(np.any(np.isclose(radii, 0.3)))
                self.assert_consistent_history(result, lambda x: x - [3.0, 4.0])

    def test_unit_growth_keeps_initial_cap_after_success(self):
        """A growth factor of one must retain the initial radius throughout successful
        progress.
        """
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    lambda x: x - 1.0,
                    [0.0],
                    jac=lambda x: np.ones((1, 1)),
                    method=method,
                    options=dict(initial_radius=0.1, max_radius=2.0, growth_factor=1.0),
                )
                self.assertTrue(result.success, result.message)
                self.assertGreater(result.nit, 5)
                self.assertTrue(all(row["radius"] == 0.1 for row in result.trials))

    def test_rejected_nonlinear_trials_preserve_base_and_monotonic_history(self):
        """Rejected Rosenbrock trials must retain the accepted base and leave accepted
        norms strictly decreasing.
        """
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    rosenbrock,
                    [-1.2, 1.0],
                    rosenbrock_jacobian,
                    method=method,
                    options=dict(
                        initial_radius=100.0, max_radius=100.0, residual_tol=1e-9
                    ),
                )
                self.assertTrue(result.success, result.message)
                self.assertTrue(any(not row["accepted"] for row in result.trials))
                self.assert_consistent_history(result, rosenbrock)
                np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)

    def test_domain_recovery_and_hard_cap_apply_to_invalid_trials(self):
        """Undefined square-root trials must shrink/retry while every requested trial
        obeys the cap.
        """

        def fun(x):
            """Return the square-root residual, rejecting nonpositive inputs as
            undefined.
            """
            if x[0] <= 0:
                raise DomainError("Square root requires positive x.")
            return np.sqrt(x) - 0.1

        for method in METHODS:
            with self.subTest(method=method):
                calls = []

                def tracked(x):
                    """Capture even undefined trial coordinates before delegating to the
                    restricted residual.
                    """
                    calls.append(x.copy())
                    return fun(x)

                result = solve_root(
                    tracked,
                    [1.0],
                    jac=lambda x: np.array([[0.5 / np.sqrt(x[0])]]),
                    method=method,
                    options=dict(
                        initial_radius=2.0, max_radius=2.0, residual_tol=1e-10
                    ),
                )
                self.assertTrue(result.success, result.message)
                self.assertTrue(any(not row["valid"] for row in result.trials))
                for call, trial in zip(calls[1:], result.trials):
                    np.testing.assert_array_equal(call, trial["x"])
                    self.assertLessEqual(abs(call[0] - trial["x_base"][0]), 2.0)
                self.assert_consistent_history(result, fun)
                np.testing.assert_allclose(result.x, [0.01], atol=1e-9)

    def test_exhausted_domain_retries_return_last_accepted_point(self):
        """Exhausting undefined retries must return the valid initial state with
        accurate counters.
        """

        def fun(x):
            """Allow evaluation only at the initial point so every attempted move
            exercises domain recovery.
            """
            if x[0] != 1.0:
                raise DomainError("Only the initial point is evaluable.")
            return np.array([1.0])

        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    fun,
                    [1.0],
                    jac=lambda x: np.ones((1, 1)),
                    method=method,
                    options=dict(initial_radius=1.0, max_radius=1.0, max_iterations=5),
                )
                self.assertFalse(result.success)
                self.assertEqual(result.ntrials, 5)
                self.assertEqual(result.nit, 0)
                self.assertEqual(result.njev, 1)
                np.testing.assert_array_equal(result.x, [1.0])
                np.testing.assert_array_equal(result.fun, [1.0])
                for row in result.trials:
                    self.assertFalse(row["valid"])
                    np.testing.assert_array_equal(row["x_base"], [1.0])
                self.assert_consistent_history(result, fun)

    def test_unit_growth_does_not_restore_radius_after_domain_recovery(self):
        """With growth disabled, recovery may shrink the radius but subsequent progress
        must not restore it.
        """

        def fun(x):
            """Evaluate a square-root residual only above the artificial lower domain
            boundary.
            """
            if x[0] <= 0.2:
                raise DomainError("Model requires x > 0.2.")
            return np.sqrt(x) - 0.5

        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    fun,
                    [1.0],
                    jac=lambda x: np.array([[0.5 / np.sqrt(x[0])]]),
                    method=method,
                    options=dict(initial_radius=2.0, max_radius=2.0, growth_factor=1.0),
                )
                self.assertTrue(result.success, result.message)
                self.assertFalse(result.trials[0]["valid"])
                radii = np.array([row["radius"] for row in result.trials])
                self.assertTrue(np.all(np.diff(radii) <= 0))

    def test_stationary_nonzero_residual_is_not_a_root(self):
        """A zero least-squares gradient with a nonzero residual must return stationary
        failure.
        """
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    lambda x: np.array([x[0], 1.0]),
                    [0.0],
                    jac=lambda x: np.array([[1.0], [0.0]]),
                    method=method,
                )
                self.assertFalse(result.success)
                self.assertIn("Stationary", result.message)
                self.assertEqual(result.residual_norm, 1.0)
                self.assertEqual(result.ntrials, 0)

    def test_small_steps_and_large_xtol_do_not_report_false_success(self):
        """Tiny allowed motion or a loose step tolerance must not report root success
        above residual tolerance.
        """
        cases = (
            dict(initial_radius=1e-14, max_radius=1e-14),
            dict(initial_radius=0.1, max_radius=1.0, xtol=1.0),
        )
        for method in METHODS:
            for options in cases:
                with self.subTest(method=method, options=options):
                    result = solve_root(
                        lambda x: x - 10,
                        [1.0],
                        jac=lambda x: np.ones((1, 1)),
                        method=method,
                        options=options,
                    )
                    self.assertFalse(result.success)
                    self.assertGreater(result.residual_norm, 1.0)
                    self.assertIn("stagnation", result.message)

    def test_analytic_and_finite_difference_jacobians_reach_same_root(self):
        """Both derivative paths must reach the Rosenbrock root and correctly classify
        derivative probes.
        """
        for method in METHODS:
            for jac in (None, rosenbrock_jacobian):
                with self.subTest(method=method, analytic=jac is not None):
                    result = solve_root(
                        rosenbrock,
                        [-1.2, 1.0],
                        jac,
                        method=method,
                        options=dict(
                            initial_radius=0.25, max_radius=0.5, residual_tol=1e-9
                        ),
                    )
                    self.assertTrue(result.success, result.message)
                    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)
                    self.assert_consistent_history(result, rosenbrock)
                    kinds = [row["kind"] for row in result.evaluations]
                    self.assertEqual("finite_difference" in kinds, jac is None)

    def test_finite_difference_budget_counts_incomplete_jacobian(self):
        """A budget exhausted partway through a numerical Jacobian must count calls and
        preserve the base.
        """
        for method in METHODS:
            with self.subTest(method=method):
                calls = []

                def fun(x):
                    """Record calls to a linear residual so partial-Jacobian budget
                    consumption can be verified.
                    """
                    calls.append(x.copy())
                    return x - 1.0

                result = solve_root(
                    fun, [0.0, 0.0], method=method, options=dict(max_nfev=2)
                )
                self.assertFalse(result.success)
                self.assertEqual(result.nfev, 2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result.ntrials, 0)
                self.assertEqual(result.nit, 0)
                self.assertIn("budget", result.message)
                np.testing.assert_array_equal(result.x, [0.0, 0.0])

    def test_finite_difference_probes_have_separate_configured_distance(self):
        """Derivative probes must use finite_diff_step and x_scale independently of the
        trial radius.
        """
        scale = np.array([2.0, 3.0])
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    lambda x: x - [0.1, 0.2],
                    [0.0, 0.0],
                    method=method,
                    options=dict(
                        initial_radius=0.05,
                        max_radius=0.05,
                        finite_diff_step=0.2,
                        x_scale=scale,
                    ),
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_allclose(result.evaluations[1]["x"], [0.4, 0])
                np.testing.assert_allclose(result.evaluations[2]["x"], [0, 0.6])
                self.assertTrue(
                    all(
                        row["scaled_step_norm"] <= 0.05 * (1 + 2e-14)
                        for row in result.trials
                    )
                )
                self.assert_consistent_history(result, lambda x: x - [0.1, 0.2], scale)

    def test_invalid_derivative_probe_stops_without_fabricating_derivative(self):
        """An undefined derivative probe must terminate with its actual failure record
        and no trial move.
        """

        def fun(x):
            """Reject positive probe coordinates while retaining an evaluable initial
            residual.
            """
            if x[0] > 0.0:
                raise DomainError("Derivative probe outside model domain.")
            return x - 1.0

        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(fun, [0.0], method=method)
                self.assertFalse(result.success)
                self.assertEqual(result.nfev, 2)
                self.assertEqual(result.ntrials, 0)
                self.assertIn("derivative", result.message)
                self.assertFalse(result.evaluations[-1]["valid"])
                self.assertEqual(result.evaluations[-1]["kind"], "finite_difference")
                np.testing.assert_array_equal(result.x, [0.0])

    def test_analytic_residual_budget_preserves_last_accepted_state(self):
        """Analytic-derivative solves must respect the residual-call budget and return
        the accepted state.
        """
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    lambda x: x - 10.0,
                    [0.0],
                    jac=lambda x: np.ones((1, 1)),
                    method=method,
                    options=dict(initial_radius=0.1, max_radius=0.1, max_nfev=3),
                )
                self.assertFalse(result.success)
                self.assertEqual(result.nfev, 3)
                self.assertEqual(result.ntrials, 2)
                self.assertEqual(result.nit, 2)
                self.assert_consistent_history(result, lambda x: x - 10.0)

    def test_variable_scaling_preserves_normalized_trajectory(self):
        """Equivalent problems in different units must follow the same trajectory in
        scaled coordinates.
        """
        scale = np.array([1e-3, 1e3])
        for method in METHODS:
            with self.subTest(method=method):
                options = dict(
                    initial_radius=0.25,
                    max_radius=0.5,
                    growth_factor=1.5,
                    residual_tol=1e-9,
                )
                reference = solve_root(
                    rosenbrock,
                    [-1.2, 1.0],
                    rosenbrock_jacobian,
                    method=method,
                    options=options,
                )
                result = solve_root(
                    lambda x: rosenbrock(x / scale),
                    np.array([-1.2, 1.0]) * scale,
                    jac=lambda x: rosenbrock_jacobian(x / scale) / scale,
                    method=method,
                    options=dict(options, x_scale=scale),
                )
                self.assertTrue(reference.success, reference.message)
                self.assertTrue(result.success, result.message)
                self.assertEqual(result.nit, reference.nit)
                self.assertEqual(result.ntrials, reference.ntrials)
                np.testing.assert_allclose(
                    np.array([row["x"] / scale for row in result.history]),
                    np.array([row["x"] for row in reference.history]),
                    atol=1e-7,
                    rtol=1e-7,
                )
                self.assert_consistent_history(
                    result, lambda x: rosenbrock(x / scale), scale
                )

    def test_hybrid_reuses_jacobians_across_nonlinear_accepted_steps(self):
        """Hybrid must use secant models across accepted moves and require fewer full
        Jacobians than moves.
        """
        jacobian_calls = []

        def jacobian(x):
            """Record full derivative requests and return the exact diagonal derivative
            of x squared minus two.
            """
            jacobian_calls.append(x.copy())
            return np.diag(2.0 * x)

        result = solve_root(
            lambda x: x * x - 2.0,
            [1.0, 1.5],
            jacobian,
            method="hybr",
            options=dict(initial_radius=0.1, max_radius=0.5),
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.njev, len(jacobian_calls))
        self.assertLess(len(jacobian_calls), result.nit)
        self.assertGreater(result.nupdates, 0)
        self.assertTrue(
            any(
                row["accepted"] and row["jacobian_source"] == "broyden"
                for row in result.trials
            )
        )
        np.testing.assert_allclose(result.x, np.sqrt([2.0, 2.0]), atol=1e-8)
        self.assert_consistent_history(result, lambda x: x * x - 2.0)

    def test_hybrid_refreshes_after_rejected_steps_with_poor_model_agreement(self):
        """Two poor secant-model rejections must request fresh derivatives at the
        unchanged accepted base.
        """
        jacobian_calls = []

        def jacobian(x):
            """Record derivative refresh locations and return the exact arctangent
            Jacobian.
            """
            jacobian_calls.append(x.copy())
            return np.diag(1.0 / (1.0 + x * x))

        result = solve_root(
            np.arctan,
            [10.0, 5.0],
            jacobian,
            method="hybr",
            # Identity geometry deliberately triggers the poor-model branch;
            # automatic scaling can avoid those rejected steps altogether.
            options=dict(
                initial_radius=100.0, max_radius=100.0, jacobian_scaling=False
            ),
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.njev, len(jacobian_calls))
        self.assertGreater(result.njev, 1)
        # Observe a refresh after two failed trials at an already moved base.
        # The failed trial information may update the model without moving x.
        refreshed_after_rejections = False
        for index in range(2, len(result.trials)):
            previous, rejected, current = result.trials[index - 2 : index + 1]
            if (
                not previous["accepted"]
                and not rejected["accepted"]
                and previous["jacobian_source"] == "broyden"
                and rejected["jacobian_source"] == "broyden"
                and current["njev"] > rejected["njev"]
            ):
                np.testing.assert_array_equal(previous["x_base"], rejected["x_base"])
                np.testing.assert_array_equal(rejected["x_base"], current["x_base"])
                self.assertEqual(current["jacobian_source"], "evaluated")
                self.assertTrue(
                    any(
                        np.array_equal(point, current["x_base"])
                        for point in jacobian_calls[1:]
                    )
                )
                refreshed_after_rejections = True
        self.assertTrue(refreshed_after_rejections)
        self.assert_consistent_history(result, np.arctan)

    def test_hybrid_refreshes_misleading_stationary_secant_model(self):
        """A flat rejected-trial secant must trigger fresh derivatives before a false
        stationary exit.
        """
        jacobian_calls = []

        def jacobian(x):
            """Record refreshes and return the nonzero derivative at the retained
            quadratic base.
            """
            jacobian_calls.append(x.copy())
            return np.array([[2.0 * x[0]]])

        # The full trial from +1 to -1 has the same residual, producing a flat
        # secant model even though the derivative at the retained +1 is nonzero.
        # A fresh derivative should permit progress before the callback stops.
        result = solve_root(
            lambda x: x * x + 3.0,
            [1.0],
            jacobian,
            method="hybr",
            options=dict(initial_radius=2.0, max_radius=2.0),
            callback=lambda row: row["iteration"] < 1,
        )
        self.assertFalse(result.success)
        self.assertIn("callback", result.message)
        self.assertFalse(result.trials[0]["accepted"])
        self.assertEqual(result.nit, 1)
        self.assertEqual(len(jacobian_calls), 2)
        np.testing.assert_array_equal(jacobian_calls[0], [1.0])
        np.testing.assert_array_equal(jacobian_calls[1], [1.0])
        self.assertLess(result.residual_norm, result.history[0]["residual_norm"])
        self.assert_consistent_history(result, lambda x: x * x + 3.0)

    def test_hybrid_refreshes_when_finite_secant_matrix_overflows_gradient(self):
        """A finite secant matrix with an overflowing gradient must be discarded without
        corrupting the base.
        """

        def residual(x):
            """Flip a huge residual across the initial coordinate to force an unusable
            finite secant update.
            """
            return np.array([1e150 if x[0] == 0 else -1e150, x[1]])

        # A tiny rejected move changes the residual sign without changing its
        # norm. The secant entries remain finite, but their product with f
        # overflows; this unusable model must trigger a fresh Jacobian.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = solve_root(
                residual,
                [0.0, 0.0],
                jac=lambda x: np.diag([1e150, 1.0]),
                method="hybr",
                # Retain the unbalanced geometry that provokes this overflow.
                options=dict(
                    initial_radius=1e-20,
                    max_radius=1e-20,
                    max_iterations=3,
                    jacobian_scaling=False,
                ),
            )
        self.assertFalse(result.success)
        self.assertIn("budget", result.message)
        self.assertEqual(result.ntrials, 3)
        self.assertLessEqual(result.nfev, 4)
        self.assertGreater(result.njev, 1)
        self.assertEqual(result.nit, 0)
        self.assertTrue(np.all(np.isfinite(result.x)))
        self.assertTrue(np.all(np.isfinite(result.fun)))
        np.testing.assert_array_equal(result.x, [0.0, 0.0])
        np.testing.assert_array_equal(result.fun, [1e150, 0.0])

    def test_rank_deficient_consistent_problem(self):
        """Both methods must solve a consistent singular linear system using a
        minimum-norm step.
        """
        matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
        target = np.array([2.0, 4.0])
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    lambda x: matrix @ x - target,
                    [0.0, 0.0],
                    jac=lambda x: matrix,
                    method=method,
                    options=dict(initial_radius=0.1, max_radius=0.25),
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)
                self.assert_consistent_history(result, lambda x: matrix @ x - target)

    def test_overdetermined_consistent_problem(self):
        """Both custom methods must reach the exact root of a consistent
        three-equation/two-variable system.
        """
        matrix = np.array([[1.0, 2.0], [2.0, -1.0], [3.0, 1.0]])
        root = np.array([1.0, -2.0])
        for method in METHODS:
            with self.subTest(method=method):
                result = solve_root(
                    lambda x: matrix @ (x - root),
                    [0.0, 0.0],
                    jac=lambda x: matrix,
                    method=method,
                    options=dict(initial_radius=0.1, max_radius=0.5),
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_allclose(result.x, root, atol=1e-8)

    def test_driver_uses_qr_fast_path_and_rank_deficient_svd_fallback(self):
        """Driver operation counts must distinguish healthy QR solves from
        singular-model SVD fallback.
        """
        matrices = (
            (np.array([[2.0, 1.0], [1.0, 3.0]]), False),
            (np.array([[1.0, 1.0], [2.0, 2.0]]), True),
        )
        root = np.ones(2)
        for method in METHODS:
            for matrix, rank_deficient in matrices:
                with self.subTest(method=method, rank_deficient=rank_deficient):
                    result = solve_root(
                        lambda x: matrix @ (x - root),
                        np.zeros(2),
                        jac=lambda x: matrix,
                        method=method,
                        options=dict(initial_radius=10.0, max_radius=10.0),
                    )
                    self.assertTrue(result.success, result.message)
                    np.testing.assert_allclose(result.x, root, atol=1e-12)
                    counts = result.linear_algebra
                    self.assertEqual(counts["qr_factorizations"], 1)
                    self.assertEqual(counts["svd_factorizations"], int(rank_deficient))
                    self.assertEqual(
                        counts["triangular_solves"], int(not rank_deficient)
                    )

    def test_callback_receives_detached_records_and_can_stop_after_acceptance(self):
        """Mutating callback records must leave solver history intact, and False must
        stop at an accepted state.
        """
        for method in METHODS:
            with self.subTest(method=method):
                seen = []

                def callback(record):
                    """Save callback coordinates, mutate its copy, and request a stop
                    after three accepted records.
                    """
                    seen.append(record["x"].copy())
                    record["x"][:] = 999.0
                    record["residual_norm"] = -1.0
                    return len(seen) < 3

                result = solve_root(
                    lambda x: x - 10.0,
                    [0.0],
                    jac=lambda x: np.ones((1, 1)),
                    method=method,
                    callback=callback,
                    options=dict(initial_radius=0.1, max_radius=0.1),
                )
                self.assertFalse(result.success)
                self.assertIn("callback", result.message)
                self.assertEqual(result.nit, 2)
                self.assertEqual(len(result.history), 3)
                np.testing.assert_array_equal(result.x, seen[-1])
                self.assert_consistent_history(result, lambda x: x - 10.0)

    def test_initial_root_needs_no_jacobian_or_trial(self):
        """A supplied root must return after its initial residual without asking for
        derivatives or a trial.
        """
        for method in METHODS:
            with self.subTest(method=method):

                def forbidden_jacobian(x):
                    """Fail immediately if the solver unnecessarily requests a Jacobian
                    at the supplied root.
                    """
                    raise AssertionError("Jacobian should not be called at a root.")

                result = solve_root(
                    lambda x: x, [0.0], forbidden_jacobian, method=method
                )
                self.assertTrue(result.success)
                self.assertEqual(result.nfev, 1)
                self.assertEqual(result.njev, 0)
                self.assertEqual(result.ntrials, 0)

    def test_history_levels_preserve_numerical_results_and_audit_counters(self):
        """Full, summary and disabled logs must preserve the solve trajectory and all
        aggregate accounting.
        """
        for method in METHODS:
            reference = solve_root(
                rosenbrock,
                [-1.2, 1.0],
                rosenbrock_jacobian,
                method=method,
                options=dict(initial_radius=100.0, max_radius=100.0),
            )
            self.assertTrue(reference.success, reference.message)
            self.assertGreater(reference.nrejected, 0)
            self.assertEqual(
                reference.nrejected,
                sum(not row["accepted"] for row in reference.trials),
            )
            self.assertEqual(
                reference.max_trial_step_scaled,
                max(row["scaled_step_norm"] for row in reference.trials),
            )
            self.assertEqual(
                reference.max_radius_used,
                max(row["radius"] for row in reference.trials),
            )
            for level in ("summary", "none"):
                with self.subTest(method=method, history_level=level):
                    result = solve_root(
                        rosenbrock,
                        [-1.2, 1.0],
                        rosenbrock_jacobian,
                        method=method,
                        options=dict(
                            initial_radius=100.0,
                            max_radius=100.0,
                            history_level=level,
                        ),
                    )
                    np.testing.assert_array_equal(result.x, reference.x)
                    np.testing.assert_array_equal(result.fun, reference.fun)
                    for field in (
                        "success",
                        "status",
                        "message",
                        "nfev",
                        "njev",
                        "nit",
                        "ntrials",
                        "nupdates",
                        "nrejected",
                        "ninvalid",
                        "radius",
                        "max_trial_step_scaled",
                        "max_radius_used",
                        "max_radius_ratio",
                        "linear_algebra",
                    ):
                        self.assertEqual(result[field], reference[field], field)
                    for field in ("history", "trials", "evaluations"):
                        expected = (
                            []
                            if level == "none"
                            else [
                                {
                                    key: value
                                    for key, value in row.items()
                                    if key not in ("x", "x_base")
                                }
                                for row in reference[field]
                            ]
                        )
                        self.assertEqual(result[field], expected)

    def test_callbacks_remain_full_and_detached_at_every_history_level(self):
        """Callbacks must retain detached coordinates and diagnostics even when stored
        histories omit them.
        """
        for method in METHODS:
            for level in ("full", "summary", "none"):
                with self.subTest(method=method, history_level=level):
                    seen = []

                    def callback(record):
                        """Check complete callback data, mutate the detached record and
                        stop after four records.
                        """
                        seen.append(record["x"].copy())
                        self.assertGreaterEqual(record["residual_norm"], 0.0)
                        record["x"][:] = 999.0
                        record["residual_norm"] = -1.0
                        return len(seen) < 4

                    result = solve_root(
                        lambda x: x - [2.0, 3.0],
                        [0.0, 0.0],
                        jac=lambda x: np.eye(2),
                        method=method,
                        callback=callback,
                        options=dict(
                            initial_radius=0.1,
                            max_radius=0.1,
                            history_level=level,
                        ),
                    )
                    self.assertFalse(result.success)
                    self.assertIn("callback", result.message)
                    self.assertEqual(result.nit, 3)
                    self.assertEqual(result.ntrials, 3)
                    self.assertEqual(result.nfev, 4)
                    np.testing.assert_array_equal(result.x, seen[-1])
                    np.testing.assert_array_equal(result.fun, result.x - [2, 3])
                    self.assertTrue(
                        all(row["residual_norm"] > 0 for row in result.history)
                    )
                    if level == "none":
                        self.assertEqual(result.history, [])

    def test_disabled_logs_do_not_disable_failure_or_budget_accounting(self):
        """Disabling stored logs must retain undefined-trial counts, budgets and
        last-accepted-state behavior.
        """
        for method in METHODS:
            for level in ("full", "summary", "none"):
                with self.subTest(method=method, history_level=level):
                    calls = []

                    def restricted(x):
                        """Record all requests while allowing only the initial vector,
                        producing undefined trial failures.
                        """
                        calls.append(x.copy())
                        if np.any(x != 0.0):
                            raise DomainError("The attempted state is unavailable.")
                        return np.array([-1.0, -1.0])

                    result = solve_root(
                        restricted,
                        [0.0, 0.0],
                        jac=lambda x: np.eye(2),
                        method=method,
                        options=dict(max_iterations=3, history_level=level),
                    )
                    self.assertFalse(result.success)
                    self.assertEqual(result.ntrials, 3)
                    self.assertEqual(result.nrejected, 3)
                    self.assertEqual(result.ninvalid, 3)
                    self.assertEqual(result.nfev, len(calls))
                    self.assertEqual(result.nfev, 4)
                    self.assertEqual(result.nit, 0)
                    self.assertGreater(result.max_trial_step_scaled, 0)
                    self.assertLessEqual(result.max_radius_ratio, 1 + 2e-14)
                    np.testing.assert_array_equal(result.x, [0.0, 0.0])

                    derivative_result = solve_root(
                        lambda x: x - 1.0,
                        [0.0, 0.0],
                        method=method,
                        options=dict(max_nfev=2, history_level=level),
                    )
                    self.assertFalse(derivative_result.success)
                    self.assertEqual(derivative_result.nfev, 2)
                    self.assertEqual(derivative_result.ntrials, 0)
                    self.assertEqual(derivative_result.nrejected, 0)
                    self.assertEqual(derivative_result.max_trial_step_scaled, 0)
                    self.assertIn("budget", derivative_result.message)

    def test_nonfinite_initial_residual_is_controlled_failure(self):
        """NaN or infinite initial residuals must return a diagnostic without entering
        the solver loop.
        """
        for value in (np.nan, np.inf):
            for method in METHODS:
                with self.subTest(method=method, value=value):
                    result = solve_root(
                        lambda x: np.array([value]), [0.0], method=method
                    )
                    self.assertFalse(result.success)
                    self.assertIsNone(result.fun)
                    self.assertEqual(result.nfev, 1)
                    self.assertEqual(result.history, [])
                    self.assertIn("initial", result.message)

    def test_unexpected_model_errors_are_not_retried(self):
        """Unexpected application exceptions must propagate instead of being hidden as
        recoverable domain failures.
        """
        for error in (RuntimeError, ValueError):
            with self.subTest(error=error):

                def fun(x):
                    """Raise the selected programming-error type on a trial while
                    keeping the initial call valid.
                    """
                    if x[0] != 0:
                        raise error("Programming error.")
                    return x - 1.0

                with self.assertRaisesRegex(error, "Programming error"):
                    solve_root(fun, [0.0], jac=lambda x: np.ones((1, 1)))

    def test_input_validation(self):
        """Invalid controls, coordinates, method names and residual/Jacobian shapes must
        be rejected.
        """
        invalid_options = (
            dict(initial_radius=0),
            dict(initial_radius=np.inf),
            dict(initial_radius=2, max_radius=1),
            dict(max_radius=-1),
            dict(growth_factor=0.5),
            dict(shrink_factor=1),
            dict(acceptance_ratio=0.5),
            dict(residual_tol=False),
            dict(xtol=np.nan),
            dict(gtol=0),
            dict(max_iterations=True),
            dict(max_nfev=2.5),
            dict(finite_diff_step=-1),
            dict(x_scale=[1, 0]),
            dict(x_scale=[[1]]),
            dict(x_scale=[]),
            dict(history_level="accepted"),
            dict(history_level=None),
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    StepControlOptions(**options)
        for x0 in ([], [np.inf], [[1.0]], 1.0):
            with self.subTest(x0=x0):
                with self.assertRaises(ValueError):
                    solve_root(lambda x: x, x0)
        with self.assertRaises(ValueError):
            solve_root(lambda x: x, [1.0], method="dogleg")
        with self.assertRaises(TypeError):
            solve_root(lambda x: x, [1.0], options=object())
        with self.assertRaises(ValueError):
            solve_root(lambda x: x, [1.0], options=dict(x_scale=[1, 2]))
        with self.assertRaises(ValueError):
            solve_root(lambda x: np.array([1.0]), [1.0, 2.0])
        with self.assertRaises(ValueError):
            solve_root(lambda x: x, [1.0], jac=lambda x: np.ones(1))

    def test_residual_dimension_changes_are_application_errors(self):
        """Changing the residual-vector length during a solve must raise an application
        error.
        """

        def fun(x):
            """Change output length after leaving the initial point to exercise the
            fixed-shape contract.
            """
            return np.ones(1 if x[0] == 0 else 2)

        with self.assertRaisesRegex(ValueError, "Residual size changed"):
            solve_root(fun, [0.0], jac=lambda x: np.ones((1, 1)))


if __name__ == "__main__":
    unittest.main()
