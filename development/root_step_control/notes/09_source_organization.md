# Source organization and refactor validation

The development implementation separates public configuration and dispatch,
numerical algorithms, shared evaluation/history mechanics, and benchmark
problems. This refactor changes organization without changing step policies.

## Public interface

The package-level exports and `src.solver` entry point remain available:

```python
from development.root_step_control import (
    BacktrackingOptions,
    DomainError,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
    solve_root,
)
```

`src.solver` normalizes aliases and selects a numerical driver. `src.options`
defines the configuration dataclasses; trust-radius settings and optional
Armijo backtracking retain distinct semantics. `src.exceptions` defines the
shared domain-error protocol and internal budget/probe termination signals.
The problem classes and suite functions previously exported by `src.problems`
remain importable from that path, which is now a package.

Internal implementation paths have moved to the modules below. Runner scripts,
tests and timing instrumentation use the new paths directly.

## Algorithm and helper boundaries

| Module under `src/solvers/` | Responsibility |
| --- | --- |
| `trust_region.py` | Levenberg–Marquardt and Powell hybrid iteration, model agreement, trust-radius changes, and Jacobian refresh decisions. |
| `line_search.py` | Newton–Raphson and Gauss–Newton iteration, optional Armijo backtracking, full-step commits, and envelope recovery. |
| `linear_model.py` | Local residual model, cached QR/SVD, Broyden factor updates, strict Newton/least-squares directions, LM and dogleg subproblems. |
| `evaluation.py` | Initial-vector preparation, residual shape/finiteness checks, analytic or numerical Jacobians, call budgets and probe accounting. |
| `history.py` | Accepted/evaluated/trial records, detached callback records and final results from the last accepted point. |
| `trial_step.py` | Trial coordinates and bounds on their actual scaled displacement, including floating-point rounding. |

The two drivers share bookkeeping while retaining their numerical decisions.
The history recorder does not decide whether a point is accepted or whether
the residual is a root. The evaluator propagates application errors and reports
recognized model-domain failures; the driver determines whether recovery is
possible. Method-specific diagnostics remain explicit when records/results are
built.

There are two trial-coordinate helpers. The trust-region helper preserves
subproblem clipping followed by coordinate-rounding correction. The line-search
helper preserves its fraction and actual-displacement accounting for Armijo.
They share a module because they address the same geometry, but retain their
arithmetic order and failure messages. Neither helper chooses variable scales.

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
directly from `exceptions.py`. The numerical modules never import benchmark
problems. Study-specific plotting, timing and report generation remain in the
`run_*.py` scripts.

## Verification on 25 October 2026

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
