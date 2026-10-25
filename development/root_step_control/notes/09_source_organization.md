# Source organization and refactor validation

The packaged implementation separates public configuration and dispatch,
numerical algorithms, shared evaluation/history mechanics, and benchmark
problems. The public `NonlinearSystemSolver` adapter adds observation streams without
changing step policies. Problems and study runners remain under development.
See [public integration](10_public_interface.md).

## Public interface

The package-level exports and `src.solver` entry point remain available:

```python
from pysolver_view import (
    BacktrackingOptions,
    DomainError,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
    solve_root,
)
```

`pysolver_view.solvers.root.solver` normalizes aliases and dispatches directly to one of four named
solver functions. `root.options`
defines the configuration dataclasses; trust-radius settings and optional
Armijo backtracking retain distinct semantics. `root.exceptions` defines the
shared domain-error protocol and internal budget/probe termination signals.
The problem classes and suite functions previously exported by `src.problems`
remain importable from that path, which is now a package.

Internal implementation paths have moved to the modules below. Runner scripts,
tests and timing instrumentation use the new paths directly.

## Four methods, two shared engines

The public dispatcher maps each canonical author name to a dedicated function.
The functions are also exported directly by `pysolver_view`:

| Entry point | Numerical choices | Shared engine |
| --- | --- | --- |
| `solve_levenberg_marquardt` | Damped least-squares step; fresh Jacobian after acceptance, model reuse after rejection. | `_solve_trust_region` |
| `solve_powell_hybrid` | Dogleg step; Broyden updates from valid secant pairs, refresh after poor agreement or unusable factors. | `_solve_trust_region` |
| `solve_newton_raphson` | Strict square, full-rank Newton direction; no least-squares stationarity test. | `_solve_line_search` |
| `solve_gauss_newton` | Least-squares direction with rank-deficient fallback; projected-change stationarity test. | `_solve_line_search` |

The engines receive the direction/step functions and the corresponding update,
shape and stationarity choices once. Their loops do not inspect method-name
strings to select algorithms. Powell's secant update and Gauss–Newton's
stationarity rule are named helpers. The linear model exposes distinct LM and
dogleg step functions; its `step(method, radius)` selector remains available for
existing model-level experiments and tests.

Each named entry point accepts the same options and callback conventions as its
`solve_root` selection. Direct results contain the canonical `method` and the
same numerical histories. The main dispatcher additionally records the requested
alias and human-readable method name.

## Algorithm and helper boundaries

| Module under `pysolver_view/solvers/root/` | Responsibility |
| --- | --- |
| `trust_region.py` | Levenberg–Marquardt and Powell hybrid entry points, Powell update helper, and shared trust-region acceptance/recovery. |
| `line_search.py` | Newton–Raphson and Gauss–Newton entry points, direction/stationarity helpers, and shared full-step or Armijo recovery. |
| `linear_model.py` | Local residual model, cached QR/SVD, Broyden factor updates, strict Newton/least-squares directions, LM and dogleg subproblems. |
| `evaluation.py` | Initial-vector preparation, residual shape/finiteness checks, analytic or numerical Jacobians, call budgets and probe accounting. |
| `history.py` | Accepted/evaluated/trial records, detached callback records and final results from the last accepted point. |
| `scaling.py` | Numerical Jacobian column equilibration from fresh derivatives, separately from fixed physical variable scales. |
| `trial_step.py` | Trial coordinates and bounds on actual physical and numerical scaled displacement, including floating-point rounding. |

The two drivers share bookkeeping while retaining their numerical decisions.
The history recorder does not decide whether a point is accepted or whether
the residual is a root. The evaluator propagates application errors and reports
recognized model-domain failures; the driver determines whether recovery is
possible. Method-specific diagnostics remain explicit when records/results are
built.

The trust-region trial helpers preserve physical clipping and check both
numerical and physical distances after coordinate rounding. The line-search
helper preserves its fraction and physical-displacement accounting for Armijo.
They share a module because they address the same geometry, but retain their
arithmetic order and failure messages. Scaling factors are chosen separately
by `JacobianScaling`; derivative probes continue to use fixed physical scales.

`linear_model.py` keeps its name because it owns persistent model/factor state
and Broyden updates in addition to solving linear subproblems.

## Problem organization

`src/problems/base.py` supplies benchmark metadata on top of the existing
`NonlinearSystemProblem` interface. Both mathematical problems and the duct use
this base. `mathematical.py` contains the standard systems and the constructed
domain/rectangular controls. `thermodynamics.py` contains the CoolProp model,
its analytic derivatives and physical diagnostics.

`suites.py` chooses the original, large and extension collections. It imports
the thermodynamic module only when a collection needs it. Mathematical problems
and solver use do not require CoolProp. Problem definitions import `DomainError`
from the packaged root API; roots and SLSQP share one `DomainError` class. The numerical modules never import benchmark
problems. Study-specific plotting, timing and report generation remain in the
`run_*.py` scripts.

## Module reorganization verification on 25 October 2026

The verification records below describe the module/dispatch/public-interface
refactors before Jacobian equilibration became the default. Their exact-history
comparisons and timings are historical evidence for those refactors; they do
not assert identical histories with the current scaling enabled. See the
[current scaling policy](07_usage_and_controls.md#jacobian-scaling-and-physical-limits)
for the default and its effect on the four methods.

- All 91 existing development tests pass after updating implementation imports.
- Fresh native/custom and custom-only convergence studies contain 100 and 112
  configurations respectively. Every summary and saved trace matches the
  pre-refactor run exactly, including rejected/undefined trials and failures.
- All 56 timing configurations were measured before and after the refactor:
  11 randomized rounds, five solves per batch, three warmups, one BLAS thread,
  with clean and instrumented batches measured separately.
- All 6,160 timed solves in each measurement reached the residual target.
  Residual/Jacobian calls, kernel counts, accepted/trial counts, backtracks,
  invalid evaluations and final residuals match across the timing comparison.
- The median across the 46 custom configurations of the after/before clean
  median-time ratio is 0.978, with range 0.905–1.047. The corresponding median
  for the 10 unchanged native SciPy configurations is 0.993.

These are sequential measurements on one machine. Their variation does not
establish a speedup; they show no broad timing regression from this extraction.

For the protected 100-variable duct, clean milliseconds per complete solve:

| Method | Before refactor | After refactor |
| --- | ---: | ---: |
| Levenberg–Marquardt | 60.188 | 57.499 |
| Powell hybrid | 14.337 | 13.855 |
| Newton–Raphson | 30.804 | 30.340 |
| Gauss–Newton | 30.679 | 30.956 |

The local validation artifacts are under
`results_refactor_validation/before/` and `results_refactor_validation/after/`,
each with `convergence/`, `custom/` and `timing/` outputs. Generated result
directories follow the repository ignore rules; this note records the findings
in version control. The existing 21-round timing report is a separate study.

To repeat the checks from the repository root:

```powershell
python -m unittest discover -s development/root_step_control/tests -t . -q
python development/root_step_control/run_convergence_benchmark.py --comparison both --no-plots --output development/root_step_control/results_refactor_validation/after/convergence --custom-output development/root_step_control/results_refactor_validation/after/custom
python development/root_step_control/run_timing_benchmark.py --repetitions 11 --batch-size 5 --warmups 3 --output development/root_step_control/results_refactor_validation/after/timing
```

## Explicit method dispatch verification

The subsequent extraction of four named entry points passes 94 development
tests. Two added tests compare every direct function with its `solve_root`
selection, including complete results and histories, and exercise the distinct
Newton/Gauss–Newton shape and rank contracts.

All 100 native/custom and 112 custom-only convergence summaries and saved traces
match the preceding implementation exactly. A separate timing check repeats
all 56 configurations before and after this extraction with seven randomized
rounds, three solves per batch, two warmups and one BLAS thread. All 2,352 timed
solves in each run reach the target; residual/Jacobian/kernel counts, iterations,
invalid evaluations, backtracks and final residuals remain identical.

The median after/before clean-time ratio across the 46 custom configurations is
0.919 (range 0.813–0.993), while the 10 unchanged SciPy configurations have median
ratio 0.926. Both groups becoming faster indicates environmental variability;
these short sequential measurements do not establish a speedup from dispatch.
They show no observed timing regression on this workload set.

Local artifacts are under `results_method_dispatch_validation/before/timing/`
and `results_method_dispatch_validation/after/`. The latter also contains the
`convergence/` and `custom/` runs. As with other generated study outputs, these
remain ignored by Git. The numerical policies and main solver default are
unchanged.

## Public integration validation

All four numerical methods now live in `pysolver_view/solvers/root/`. The
development API re-exports them; benchmark problems remain under `src/problems/`.
The public frontend delegates numerical derivatives to the core and translates
its evaluation and accepted-point events without extra model calls.

On 25 October 2026, the combined package, custom SLSQP study, and root development
suite passed **418 tests**, with ten Hessian tests excluded by the agreed scope.
This includes 94 development root checks and 29 public-root regressions.
Forward/central/complex derivatives, probe budgets, cancellation, shared domain
errors, authoritative results, history modes and solver reuse are covered.

The public-interface convergence rerun verifies all 100 native/custom cases.
The custom comparison retains 98/101 eligible roots, six unsupported rectangular
Newton configurations, and five correctly rejected inconsistent configurations.
All 51 eligible controlled cases succeed. All 11,760 timed solves in the
56-configuration main benchmark reach the target, as do 3,168 timed solves in
the separate direct/public comparison. Work counts agree across interfaces.
[Timing findings](06_time_performance.md) describe the measured interface cost
and the observed variation between timing sessions.

The four-solver demo succeeds on both of its problems. A separate isolated copy
of the package runs all four methods without importing development files.
Wheel construction is still unverified because poetry-core is unavailable;
this smoke check does not replace the release installation matrix.
