"""Check timing accounting and audits without asserting speed rankings."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from .. import run_timing_benchmark as timing
from ..src.solvers import linear_model
from ..src.problems import BroydenTridiagonalProblem


class TimingAccountingTests(unittest.TestCase):
    """Check measurement identities, instrumentation cleanup, solver settings and
    standalone runners.
    """

    def test_callback_times_and_first_target_are_counted_once(self):
        """A deterministic clock must produce exact callback totals and only one
        first-target timestamp.
        """
        measurements = timing.Measurements(start_ns=1000)
        residual = measurements.residual(lambda x: np.zeros(2))
        jacobian = measurements.jacobian(lambda x: np.eye(2))
        with patch.object(
            timing,
            "perf_counter_ns",
            side_effect=[1100, 1150, 1200, 1300, 1370, 1400, 1490],
        ):
            residual(np.ones(2))
            residual(np.zeros(2))
            jacobian(np.zeros(2))
        self.assertEqual(measurements.residual_calls, 2)
        self.assertEqual(measurements.residual_ns, 120)
        self.assertEqual(measurements.jacobian_calls, 1)
        self.assertEqual(measurements.jacobian_ns, 90)
        self.assertEqual(measurements.first_target_call, 1)
        self.assertEqual(measurements.first_target_ns, 200)

    def test_failed_callback_work_is_counted_and_error_propagates(self):
        """Timing must include failed callback work while preserving the original
        exception.
        """

        def undefined(x):
            """Raise a synthetic model error to exercise failed residual-call
            accounting.
            """
            raise RuntimeError("model failed")

        measurements = timing.Measurements()
        with patch.object(timing, "perf_counter_ns", side_effect=[10, 90]):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                measurements.residual(undefined)(np.zeros(2))
        self.assertEqual(measurements.residual_calls, 1)
        self.assertEqual(measurements.residual_ns, 80)
        self.assertIsNone(measurements.first_target_call)

    def test_failed_solve_is_retained_with_missing_target_time(self):
        """Failed solves must remain in batch data with missing target time instead of
        being silently dropped.
        """
        problem = BroydenTridiagonalProblem(n=4)
        result = OptimizeResult(x=problem.x0, success=False, message="budget exhausted")
        with patch.object(timing, "run_solver", return_value=result):
            row, _ = timing.measure_batch(
                problem, "scipy_lm", timing.make_options(problem, "scipy_lm"), 2, True
            )
        self.assertFalse(row["all_root_success"])
        self.assertEqual(row["root_successes"], 0)
        self.assertIsNone(row["first_target_ns"])
        self.assertIsNone(row["first_target_call"])
        self.assertIn("budget exhausted", row["messages"])

    def test_controlled_and_summary_variants_keep_the_same_step_policy(self):
        """History storage variants must use identical initial, maximum and growth
        controls.
        """
        problem = BroydenTridiagonalProblem(n=4)
        for solver in timing.CONTROLLED_SOLVERS + timing.SUMMARY_SOLVERS:
            options = timing.make_options(problem, solver)
            if timing.solver_configuration(solver).method in ("lm", "hybr"):
                self.assertEqual(options.initial_radius, 0.1)
                self.assertEqual(options.max_radius, 1)
                self.assertEqual(options.growth_factor, 1.5)
            else:
                limits = options.backtracking.step_limits
                self.assertEqual(limits.initial_limit, 0.1)
                self.assertEqual(limits.max_limit, 1)
                self.assertEqual(limits.growth_factor, 1.5)
            self.assertEqual(
                options.history_level,
                "summary" if solver.endswith("_summary") else "full",
            )

    def test_svd_timer_restores_original_after_svd_exception(self):
        """A failing numerical kernel must still be timed and its original binding
        restored.
        """

        def failing_svd(*args, **kwargs):
            """Raise a synthetic factorization failure without evaluating any matrix
            operation.
            """
            raise RuntimeError("factorization failed")

        measurements = timing.Measurements()
        with patch.object(linear_model, "svd", failing_svd):
            with patch.object(timing, "perf_counter_ns", side_effect=[100, 280]):
                with self.assertRaisesRegex(RuntimeError, "factorization failed"):
                    with timing.measure_custom_kernels(
                        measurements, "pysolverview_lm", enabled=True
                    ):
                        self.assertIsNot(linear_model.svd, failing_svd)
                        linear_model.svd(np.eye(2))
            self.assertIs(linear_model.svd, failing_svd)
        self.assertEqual(measurements.svd_calls, 1)
        self.assertEqual(measurements.svd_ns, 180)

    def test_svd_timer_restores_original_after_outer_exception(self):
        """Leaving the timer context through an unrelated exception must restore the
        imported kernel.
        """
        original = linear_model.svd
        measurements = timing.Measurements()
        with self.assertRaisesRegex(RuntimeError, "outer failure"):
            with timing.measure_custom_kernels(
                measurements, "pysolverview_lm", enabled=True
            ):
                linear_model.svd(np.eye(2))
                raise RuntimeError("outer failure")
        self.assertIs(linear_model.svd, original)
        self.assertEqual(measurements.svd_calls, 1)

    def test_disabled_svd_timer_does_not_wrap_or_count(self):
        """Disabled instrumentation must preserve the kernel binding and leave its
        counters untouched.
        """
        original = linear_model.svd
        measurements = timing.Measurements()
        with timing.measure_custom_kernels(
            measurements, "pysolverview_lm", enabled=False
        ):
            self.assertIs(linear_model.svd, original)
            linear_model.svd(np.eye(2))
        self.assertEqual(measurements.svd_calls, 0)
        self.assertEqual(measurements.svd_ns, 0)

    def test_real_small_solve_has_additive_accounting_and_valid_result(self):
        """Real native/custom solves must have additive timing, valid roots and correct
        kernel availability markers.
        """
        for solver in timing.SOLVER_CONFIGURATIONS:
            with self.subTest(solver=solver):
                problem = BroydenTridiagonalProblem(n=4)
                options = timing.make_options(problem, solver)
                row, result = timing.measure_batch(problem, solver, options, 2, True)
                self.assertAlmostEqual(
                    row["residual_ns"] + row["jacobian_ns"] + row["solver_overhead_ns"],
                    row["total_ns"],
                    places=6,
                )
                self.assertGreater(row["residual_calls"], 0)
                self.assertGreater(row["jacobian_calls"], 0)
                self.assertLessEqual(row["first_target_call"], row["residual_calls"])
                self.assertLessEqual(row["first_target_ns"], row["total_ns"])
                self.assertLessEqual(
                    np.linalg.norm(problem.residual(result.x)), timing.TARGET
                )
                if solver.startswith("pysolverview_"):
                    self.assertGreater(
                        sum(row[k + "_calls"] for k in timing.KERNELS), 0
                    )
                    self.assertLessEqual(row["svd_ns"], row["solver_overhead_ns"])
                    if timing.solver_configuration(solver).method in ("lm", "hybr"):
                        self.assertLessEqual(
                            row["max_active_radius"], options.max_radius
                        )
                    else:
                        self.assertIsNone(row["max_active_radius"])
                        self.assertGreater(row["accepted_iterations"], 0)
                else:
                    self.assertIsNone(row["svd_ns"])
                    self.assertIsNone(row["svd_calls"])

    def test_clean_runs_mark_unmeasured_components_missing(self):
        """Clean runs must record only total time and mark unmeasured components as
        missing.
        """
        problem = BroydenTridiagonalProblem(n=4)
        for solver in ("scipy_lm", "pysolverview_lm"):
            with self.subTest(solver=solver):
                row, _ = timing.measure_batch(
                    problem, solver, timing.make_options(problem, solver), 1, False
                )
                self.assertGreater(row["total_ns"], 0)
                for field in (
                    "residual_ns",
                    "jacobian_ns",
                    "svd_ns",
                    "solver_overhead_ns",
                    "residual_calls",
                    "jacobian_calls",
                    "svd_calls",
                    "first_target_ns",
                    "first_target_call",
                ):
                    self.assertIsNone(row[field], field)

    @staticmethod
    def sample_row(solver, instrumented, total, residual=0, jacobian=0):
        """Build a synthetic batch row with controlled totals and components for
        summary-accounting tests.
        """
        return dict(
            problem="accounting_fixture",
            dimension=2,
            mode="generous",
            history_level="full",
            solver=solver,
            instrumented=instrumented,
            batch_size=5,
            total_ns=total * 1e6,
            residual_ns=residual * 1e6,
            jacobian_ns=jacobian * 1e6,
            solver_overhead_ns=(total - residual - jacobian) * 1e6,
            svd_ns=None if solver.startswith("scipy_") else 0.2e6,
            svd_calls=None if solver.startswith("scipy_") else 2,
            qr_ns=0,
            qr_calls=0,
            qr_update_ns=0,
            qr_update_calls=0,
            triangular_ns=0,
            triangular_calls=0,
            residual_calls=4,
            jacobian_calls=2,
            first_target_ns=total * 0.8e6,
            first_target_call=3,
            final_residual_norm=1e-10,
            all_native_success=True,
            all_root_success=True,
            root_successes=5,
            max_radius_ratio=1,
            messages="fixture",
            max_active_radius=10,
            max_trial_step_scaled=2,
        )

    def test_summary_uses_one_median_total_batch_and_distinguishes_missing_svd(self):
        """Summary components must come from one median-total batch and distinguish
        unavailable from measured kernels.
        """
        rows = []
        for solver, multiplier in (("scipy_lm", 1), ("pysolverview_lm", 2)):
            # Component medians intentionally differ from the components of
            # the median-total batch. A mixed row would not be additive.
            for total, residual, jacobian in (
                (4, 0.2, 0.2),
                (2, 0.1, 0.1),
                (3, 1, 0.5),
            ):
                rows.append(
                    self.sample_row(
                        solver,
                        True,
                        multiplier * total,
                        multiplier * residual,
                        multiplier * jacobian,
                    )
                )
            for total in (3, 1, 2):
                rows.append(self.sample_row(solver, False, multiplier * total))
        summary = {row["solver"]: row for row in timing.summarize_batches(rows)}
        native = summary["scipy_lm"]
        custom = summary["pysolverview_lm"]
        self.assertEqual(native["total_ms"], 3)
        self.assertEqual(native["residual_ms"], 1)
        self.assertEqual(native["jacobian_ms"], 0.5)
        self.assertEqual(native["solver_overhead_ms"], 1.5)
        self.assertEqual(native["clean_total_median_ms"], 2)
        self.assertEqual(native["clean_total_q25_ms"], 1.5)
        self.assertEqual(native["clean_total_q75_ms"], 2.5)
        self.assertEqual(custom["clean_time_relative_to_native_method"], 2)
        self.assertIsNone(native["custom_svd_ms"])
        self.assertIsNone(native["custom_svd_calls"])
        self.assertEqual(custom["custom_svd_ms"], 0.2)
        self.assertEqual(custom["custom_svd_calls"], 2)
        for row in summary.values():
            self.assertEqual(
                row["residual_ms"] + row["jacobian_ms"] + row["solver_overhead_ms"],
                row["total_ms"],
            )

    def test_solver_configurations_and_native_radius_formula(self):
        """Only supported configurations must be accepted, with matching native/custom
        initial-radius formulas.
        """
        self.assertEqual(
            set(timing.SOLVERS),
            {"scipy_lm", "scipy_hybr", "pysolverview_lm", "pysolverview_hybr"},
        )
        self.assertEqual(
            set(timing.CONTROLLED_SOLVERS),
            {
                "pysolverview_lm_controlled",
                "pysolverview_hybr_controlled",
                "pysolverview_newton_raphson_controlled",
                "pysolverview_gauss_newton_controlled",
            },
        )
        self.assertEqual(
            set(timing.SUMMARY_SOLVERS),
            {
                "pysolverview_lm_controlled_summary",
                "pysolverview_hybr_controlled_summary",
            },
        )
        problem = BroydenTridiagonalProblem(n=4)
        with self.assertRaisesRegex(ValueError, "Unknown solver"):
            timing.make_options(problem, "unknown_lm")
        with self.assertRaisesRegex(ValueError, "Unknown solver"):
            timing.run_solver(
                problem, "unknown_lm", {}, problem.residual, problem.gradient
            )
        custom = timing.make_options(problem, "pysolverview_lm")
        native = timing.make_options(problem, "scipy_lm")
        self.assertEqual(
            custom.initial_radius, 100 * np.linalg.norm(problem.x0 / problem.x_scale)
        )
        self.assertEqual(custom.growth_factor, 2)
        self.assertGreater(custom.max_radius, custom.initial_radius * 1e6)
        self.assertEqual(native["factor"], 100)
        np.testing.assert_array_equal(native["diag"], 1 / problem.x_scale)
        problem.x0 = np.zeros(problem.n)
        self.assertEqual(
            timing.make_options(problem, "pysolverview_hybr").initial_radius, 100
        )

    def test_multiword_methods_dispatch_without_identifier_parsing(self):
        """Full and protected multiword configurations must dispatch the exact method."""
        problem = BroydenTridiagonalProblem(n=4)
        for method in ("newton_raphson", "gauss_newton"):
            for suffix in ("", "_controlled"):
                solver = f"pysolverview_{method}{suffix}"
                options = timing.make_options(problem, solver)
                if suffix:
                    self.assertIsNotNone(options.backtracking)
                else:
                    self.assertIsNone(options.backtracking)
                with patch.object(timing.solver_module, "solve_root") as solve:
                    timing.run_solver(
                        problem, solver, options, problem.residual, problem.gradient
                    )
                self.assertEqual(solve.call_args.kwargs["method"], method)

    def test_plot_labels_use_full_names_and_separate_backend_identity(self):
        """Method abbreviations must not replace author-based names in plot labels."""
        for solver in timing.SOLVER_CONFIGURATIONS:
            configuration = timing.solver_configuration(solver)
            label = timing.solver_plot_label(solver)
            self.assertIn(timing.METHOD_NAMES[configuration.method], label)
            self.assertIn(
                "SciPy/MINPACK" if configuration.backend == "scipy" else "PySolverView",
                label,
            )
        self.assertIn(
            "Newton–Raphson", timing.solver_plot_label("pysolverview_newton_raphson")
        )
        self.assertIn(
            "Gauss–Newton", timing.solver_plot_label("pysolverview_gauss_newton")
        )

    def test_newton_audit_checks_actual_step_limit_and_allows_uncapped_steps(self):
        """Newton audits must enforce actual displacement without inventing a radius."""
        problem = SimpleNamespace(
            name="newton_audit",
            x0=np.zeros(2),
            x_scale=np.ones(2),
            residual=lambda x: np.zeros(2),
        )
        result = OptimizeResult(
            x=np.zeros(2),
            success=True,
            message="fixture",
            trials=[dict(x=np.array([0.2, 0]), x_base=np.zeros(2), step_limit=0.1)],
        )
        solver = "pysolverview_newton_raphson_controlled"
        with self.assertRaisesRegex(AssertionError, "active step limit"):
            timing.check_result(
                problem, solver, timing.make_options(problem, solver), result
            )
        result.trials[0]["step_limit"] = None
        solver = "pysolverview_newton_raphson"
        checked = timing.check_result(
            problem, solver, timing.make_options(problem, solver), result
        )
        self.assertIsNone(checked["max_active_radius"])
        self.assertEqual(checked["max_trial_step_scaled"], 0.2)

    def test_comparison_groups_keep_native_and_custom_methods_separate(self):
        """Requested figures must contain their intended backend/method comparisons."""
        rows = [
            dict(problem="fixture", dimension=100, solver=solver)
            for solver in timing.SOLVER_CONFIGURATIONS
        ]
        groups = {stem: group for stem, _, _, group in timing.comparison_groups(rows)}
        self.assertEqual(
            {r["solver"] for r in groups["timing_breakdown"]}, set(timing.SOLVERS)
        )
        self.assertEqual(
            {r["solver"] for r in groups["custom_timing_breakdown"]},
            set(timing.CUSTOM_SOLVERS),
        )
        self.assertEqual(
            {r["solver"] for r in groups["controlled_timing_breakdown"]},
            set(timing.CONTROLLED_SOLVERS),
        )

    def test_audit_rejects_active_user_cap_and_invalid_trial_distance(self):
        """Step audits must reject nonbinding-cap violations and trials outside the
        active trust radius.
        """
        problem = SimpleNamespace(
            name="audit_fixture",
            x_scale=np.ones(2),
            residual=lambda x: np.zeros(2),
        )
        options = timing.solver_module.StepControlOptions(
            initial_radius=1, max_radius=1
        )
        result = OptimizeResult(
            x=np.zeros(2),
            success=True,
            message="fixture",
            trials=[dict(x=np.array([0.1, 0]), x_base=np.zeros(2), radius=1)],
        )
        with self.assertRaisesRegex(
            AssertionError, "maximum radius bound became active"
        ):
            timing.check_result(problem, "pysolverview_lm", options, result)
        options = timing.solver_module.StepControlOptions(
            initial_radius=1, max_radius=1e12
        )
        result.trials[0]["radius"] = 0.01
        with self.assertRaisesRegex(
            AssertionError, "exceeded its intrinsic trust radius"
        ):
            timing.check_result(problem, "pysolverview_hybr", options, result)

    def test_final_residual_verification_does_not_trust_native_success(self):
        """Independent verification must reject a large residual even when the native
        result flag says success.
        """
        problem = BroydenTridiagonalProblem(n=4)
        result = OptimizeResult(
            x=problem.x0, success=True, message="false fixture success"
        )
        with self.assertRaisesRegex(AssertionError, "residual"):
            timing.check_result(
                problem, "scipy_lm", timing.make_options(problem, "scipy_lm"), result
            )

    def test_direct_runners_import_from_unrelated_working_directory(self):
        """Every run script must expose its CLI successfully when launched from outside
        the repository.
        """
        experiment = Path(timing.__file__).resolve().parent
        # The workspace above the repository contains no top-level development
        # package; this checks direct execution without creating temporary files.
        for name, argument in (
            ("run_convergence_benchmark.py", "--suite"),
            ("run_timing_benchmark.py", "--repetitions"),
            ("run_step_geometry.py", "--output"),
        ):
            with self.subTest(runner=name):
                script = experiment / name
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=script.parents[3],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(argument, result.stdout)


if __name__ == "__main__":
    unittest.main()
