# Development notes and studies

These are maintainer working notes for personal use. This directory is outside
`docs/source/` and is not included in the published documentation or navigation.

`development/` contains reproducible study scripts. Reusable numerical
algorithms, problem models and audit helpers are installed under
[`src/pysolver_view/`](../../src/pysolver_view/), and all tests live in
[`tests/`](../../tests/). Design notes and mathematical explanations live here
under [`docs/development/`](.).

## What is retained

| Script | Purpose beyond the demos | Generated output |
| --- | --- | --- |
| [Native Pounce NLP](../../development/pounce_solvers/run_native_solver_comparison.py) | Direct IPM/SQP versus custom SLSQP with first-step fractions 1.00/0.20; original-model feasibility and KKT audits | [Native report](../../development/pounce_solvers/results_native_comparison/REPORT.md) |
| [Pounce wrapper](../../development/pounce_solvers/run_wrapper_solver_comparison.py) | Same constrained model through the public backend, with native/wrapper parity checks | [Wrapper report](../../development/pounce_solvers/results_wrapper_comparison/REPORT.md) |
| [Root convergence](../../development/root_step_control/run_convergence_benchmark.py) | Broad native/custom comparisons, four-method controls, domain recovery and rectangular/inconsistent cases; audits actual trial distances | [Native/custom report](../../development/root_step_control/results_convergence/REPORT.md), [custom report](../../development/root_step_control/results_custom_convergence/REPORT.md) |
| [Root timing](../../development/root_step_control/run_timing_benchmark.py) | Randomized clean/instrumented timings, physical-call accounting and QR/SVD/linear-solve costs across mathematical and CoolProp workloads | [Timing report](../../development/root_step_control/results_time_perf/RESULTS.md) |
| [Interface timing](../../development/root_step_control/run_adapter_timing.py) | Direct numerical driver versus public interface overhead on three 100-variable problems | [Interface report](../../development/root_step_control/results_adapter_perf/RESULTS.md) |
| [Step geometry](../../development/root_step_control/run_step_geometry.py) | Independently verifies and draws the local LM and dogleg boundary steps; the trust-radius sensitivity demo addresses a different comparison | [Geometry figure](../../development/root_step_control/results_step_geometry/lm_hybrid_step_geometry.svg) |
| [SLSQP tolerance matrix](../../development/slsqp_controls/run_validation.py) | Twenty combinations of stopping/acceptance tolerances, model-domain recovery and inconsistent constraints with independent audits | [Validation report](../../development/slsqp_controls/results_validation/REPORT.md) |

Focused sensitivity and KKT examples live in `demos/`. Generated study results
are ignored and can be absent in a fresh checkout; use the commands below to
reproduce them.

## Run the studies

Install this checkout in the environment used to run the scripts:

```bash
poetry install -E thermodynamics
```

Run from the repository root:

```bash
python development/root_step_control/run_convergence_benchmark.py
python development/root_step_control/run_timing_benchmark.py
python development/root_step_control/run_adapter_timing.py
python development/root_step_control/run_step_geometry.py
python development/slsqp_controls/run_validation.py
poetry install --extras pounce
poetry run python development/pounce_solvers/run_native_solver_comparison.py --no-show
poetry run python -m development.pounce_solvers.run_wrapper_solver_comparison --no-show
```

Each script accepts `--output` and `--help`. Convergence has a separate
`--custom-output` for the four-method comparison. Defaults resolve output paths
beside each script. The root timing study defaults to 21 randomized rounds,
five solves per batch and three warmups; interface timing uses 11 rounds and
three solves per batch. Timing scripts set one BLAS thread before importing
NumPy. Run timed studies separately from tests and other computational work.
CoolProp is optional for the library but needed for the complete root studies.

Reusable SLSQP models are in
[`optimization/problems/slsqp_controls.py`](../../src/pysolver_view/optimization/problems/slsqp_controls.py).
Their [independent audit helpers](../../src/pysolver_view/diagnostics/slsqp_audit.py)
use fresh model values and derivatives with supplied multipliers. They audit
these unscaled inequality-only benchmark models and do not replace the shared
KKT assessor or estimate new multipliers. Explicit audit work is outside solver
call counters.

## Tests

All tests are selected by [`tests/run_tests.py`](../../tests/run_tests.py):

```bash
python tests/run_tests.py
python tests/run_tests.py --suite studies
```

The study tests cover accounting identities, restoration after instrumentation
errors, truthful failure/reporting states, actual step bounds, derivative
accuracy, KKT signs and domain recovery. The master runner's default discovers
the entire `tests/` tree once; the `studies` selection is a convenience for
these targeted checks.

## Topic guide

| Note | Content |
| --- | --- |
| [Architecture](architecture.md) | Canonical modules, ownership, compatibility and backend dependencies |
| [Results and accounting](results_and_accounting.md) | Authoritative points, events, counters, callbacks and detached histories |
| [Root interface](root_solvers.md) | Four methods, options, defaults, derivatives and direct/public usage |
| [LM and Powell hybrid mathematics](root_trust_regions.md) | Scaling, QR/SVD, damping, dogleg, Broyden updates and acceptance |
| [Newton and Gauss-Newton mathematics](root_line_search.md) | Rank requirements, Armijo, limits and full steps |
| [Thermodynamic models](thermodynamic_models.md) | Physical scaling, domain failures, state ownership and derivative limits |
| [KKT and SLSQP](optimization_kkt.md) | Signs, tolerances, multiplier fitting, accounting and independent audits |
| [Timing protocol](timing.md) | Workloads, instrumentation, fairness and interpretation |
| [Validation](validation.md) | Master test runner, packaging checks and study reproduction |
| [Remaining limitations](limitations.md) | Confirmed implementation gaps and unvalidated scope |

Completed migration plans and superseded defect lists have been retired.
[PROGRESS.md](../../PROGRESS.md) keeps brief completed and remaining checklists;
the notes here describe current behavior and how to reproduce validation.
