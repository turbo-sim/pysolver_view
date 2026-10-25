# Validation and reproduction

Run commands from the repository root using the installed checkout. Prefer
`poetry install -E thermodynamics` and `poetry run python ...`; the CoolProp extra
is optional for the package but needed for complete physical studies. The custom
SLSQP bridge and complete test suite require Python 3.11 or newer with SciPy
1.17.1. Do not add checkout paths to scripts to work around an old installation.

## Master test runner

[`tests/run_tests.py`](../../tests/run_tests.py) owns selection and defaults:

```bash
poetry run python tests/run_tests.py
poetry run python tests/run_tests.py --suite studies
```

Edit `TEST_SUITES`, `ENABLED_SUITES` and `PYTEST_OPTIONS` in that script to control
testing. The default package suite discovers all of `tests/` once; the study
suite selects three focused regression modules. The script forwards additional
pytest arguments, runs from the repository root, defaults to headless plotting,
and stores pytest cache under `build/`. Keep this control in Python rather than
adding test-selection logic to GitHub workflows.

Current defaults exclude Hessian cases. PyGMO, IPOPT and SNOPT execution is
outside routine validation. Optional CoolProp checks depend on its installation.
Do not interpret passing adapter tests as native solver validation.

| Area under `tests/` | Evidence |
| --- | --- |
| `roots/`, including `solvers/` | Independent residual checks, actual rounded trial limits, rank/shape failures, domain recovery, callbacks, budgets and Jacobian equilibration |
| `optimization/` | Native/custom SLSQP, independent KKT reconstruction, signs, bounds, dependent constraints, strict/optional KKT gates and domain/probe failure behavior |
| `differentiation/` | Supported first derivatives, shape and probe behavior; Hessians excluded by default |
| `reporting/` and domain result/history checks | Detached state, same-point observations, counters, mutation safety and passive presentation |
| `packaging/` | Installed-wheel checks and compatibility/import contracts |

For the last numerical consolidation, 502 tests passed and 10 Hessian cases
were deselected; the study subset had 27 passing checks. These are historical
results for that checkout/environment, not requirements on future test counts.
See [PROGRESS.md](../../PROGRESS.md) for completed milestones and remaining work.

## Study runners

```bash
poetry run python development/root_step_control/run_convergence_benchmark.py
poetry run python development/root_step_control/run_timing_benchmark.py
poetry run python development/root_step_control/run_adapter_timing.py
poetry run python development/root_step_control/run_step_geometry.py
poetry run python development/slsqp_controls/run_validation.py
```

Use each script's `--help` for selection and output options. All accept
`--output`; root convergence also accepts `--custom-output`. Defaults resolve
beside the script. Results are generated, ignored artifacts; the links in the
[study index](README.md) become available after running the commands.

| Study | Outputs and interpretation |
| --- | --- |
| Native/custom roots | `results_convergence/`: 100 configurations in the last full run; all met the independently checked target |
| Four custom methods | `results_custom_convergence/`: 112 cases; 98/101 eligible roots and 51/51 controlled roots passed; unsupported shapes and inconsistent systems remain explicit outcomes |
| Root timing | `results_time_perf/`: 56 configurations, 21 rounds, five solves/batch, three warmups; clean/instrumented modes produce 11,760 timed solves |
| Interface timing | `results_adapter_perf/`: 24 paired configurations, 11 rounds, three solves/batch; 3,168 timed solves |
| Step geometry | `results_step_geometry/`: PNG/SVG of independently checked LM and dogleg steps for one fixed local model |
| SLSQP controls | `results_validation/`: twenty-case tolerance/domain/inconsistency matrix; all twenty audits passed in the last run |

These counts describe the maintained configurations and last complete run. Read
the report, CSV/JSON and environment metadata together when changing settings.
The root convergence comparisons preserve failed unrestricted Newton-family
solves; successful timing workloads do not establish universal robustness.

Timing runners set one BLAS/OpenMP thread before numerical imports. Execute them
directly and sequentially, separately from tests and other computational work.
Separate clean total time, instrumented model/derivative work and kernel subsets.
Use paired interface measurements for adapter costs. The detailed
[timing protocol](timing.md) explains additivity, target checks and limitations.

The duct model uses actual CoolProp property/state/derivative calls, with
conservation, derivative and state-reset checks. It is a single-phase idealized
workload. The SLSQP audit uses fresh values and supplied multipliers for the
bundled unscaled inequality-only models; audit calls are outside solver counts.

## Installed artifacts and documentation

```bash
poetry run python -m build
poetry run python docs/build_docs.py --no-autobuild
```

Build an sdist and its wheel, install the wheel in a fresh environment with
SciPy 1.17.1, and execute
[`tests/packaging/wheel_smoke.py`](../../tests/packaging/wheel_smoke.py) from
outside the checkout using that environment's Python with `-I`. First check
without JAX or CoolProp, then install CoolProp and repeat with `--thermodynamics`.
This checks installed paths, compatibility identity, native/custom solves,
promoted models and inclusion of the SLSQP license. Source import success alone
does not establish artifact completeness.

The last consolidation built both artifacts and passed isolated wheel checks
with and without CoolProp, including the promoted SLSQP model/audit with
development imports blocked. Downstream application migration remains separate.

The documentation builder regenerates API pages and builds `docs/source/`.
These development notes live outside that Sphinx source tree: also check their
relative links, mathematical delimiters and executable examples when editing.
Documentation-only edits do not require repeating full numerical timing runs.
