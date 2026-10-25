# Root step control: folder guide

This development suite exercises the packaged Levenberg–Marquardt, Powell hybrid,
Newton–Raphson, and Gauss–Newton root solvers with explicit step controls.
It compares the first two with native SciPy and all four with each other.
Problems use the project's `NonlinearSystemProblem` interface.

Throughout the code, notes and results, **SciPy/MINPACK** means the native
solvers exposed by `scipy.optimize.root`. **PySolverView** means the custom
root implementations in `pysolver_view/solvers/root/`. Saved solver
identifiers use `scipy_*` and `pysolverview_*`, respectively.

## Folder structure

```text
root_step_control/
|-- README.md
|-- src/
|   |-- solver.py                # Compatibility re-export of packaged drivers
|   `-- problems/
|       |-- base.py              # Shared benchmark metadata/interface
|       |-- mathematical.py      # Standard systems and constructed controls
|       |-- thermodynamics.py    # Optional CoolProp duct problem
|       `-- suites.py            # Reproducible problem collections
|-- tests/                       # Numerical, problem, and study checks
|-- notes/                       # Mathematics, usage, design, measured findings
|-- run_convergence_benchmark.py
|-- run_timing_benchmark.py
|-- run_adapter_timing.py
|-- run_step_geometry.py
|-- results_convergence/
|-- results_custom_convergence/
|-- results_time_perf/
|-- results_adapter_perf/
`-- results_step_geometry/
```

The installable numerical implementation is in
[`pysolver_view/solvers/root/`](../../pysolver_view/solvers/root/).
Its `solver.py`, `options.py`, `trust_region.py`, `line_search.py`,
`linear_model.py`, `scaling.py`, `evaluation.py`, `history.py`, and `trial_step.py` retain the
same responsibilities described below. The public adapter is
[`custom_root.py`](../../pysolver_view/custom_root.py). See
[public usage and integration](notes/10_public_interface.md).


The `run_*.py` files contain their study-specific setup, execution, plotting,
reporting, and command-line handling. Shared benchmark problem definitions live in `src/problems/`; numerical functions
are imported from the installable package. Each study writes to a named
`results_<case>/` directory; use `--output` to keep an additional case separately.
Result directories contain outputs rather than implementation or tests.

| Source | Purpose |
| --- | --- |
| [solver.py](../../pysolver_view/solvers/root/solver.py) | Public `solve_root`, method aliases and driver dispatch; configuration re-exports |
| [options.py](../../pysolver_view/solvers/root/options.py) | All validated options, keeping trust-region and backtracking policies distinct |
| [exceptions.py](../../pysolver_view/solvers/root/exceptions.py) | Recognized model-domain failures and internal budget/probe signals |
| [trust_region.py](../../pysolver_view/solvers/root/trust_region.py) | Named LM and Powell hybrid functions, explicit step/update choices, shared trust-region engine |
| [line_search.py](../../pysolver_view/solvers/root/line_search.py) | Named Newton–Raphson and Gauss–Newton functions, explicit direction/rank choices, shared line-search engine |
| [linear_model.py](../../pysolver_view/solvers/root/linear_model.py) | QR/SVD factors, LM damping, dogleg steps and Broyden updates |
| [scaling.py](../../pysolver_view/solvers/root/scaling.py) | Default-enabled Jacobian column equilibration, independent of physical step scales |
| [evaluation.py](../../pysolver_view/solvers/root/evaluation.py) | Residual validation, analytic/numerical Jacobians and model-call budgets |
| [history.py](../../pysolver_view/solvers/root/history.py) | Histories, copied callbacks and accepted-state result assembly |
| [trial_step.py](../../pysolver_view/solvers/root/trial_step.py) | Trial coordinates and scaled displacement audits, including rounding |
| [base.py](src/problems/base.py) | Benchmark metadata using the existing `NonlinearSystemProblem` interface |
| [mathematical.py](src/problems/mathematical.py) | Standard multidimensional equations and constructed domain/rectangular controls |
| [thermodynamics.py](src/problems/thermodynamics.py) | Coupled CoolProp nitrogen duct, derivatives and physical diagnostics |
| [suites.py](src/problems/suites.py) | Original, large and extension problem collections; thermodynamic imports on demand |

Both numerical drivers share evaluation and history mechanics. Their acceptance,
stopping and Jacobian-refresh decisions remain explicit in their respective
modules. `linear_model.py` retains the cached factors and secant updates as well
as computing directions. `trial_step.py` bounds actual displacements using the
fixed caller input `x_scale`. Independent numerical factors adapt to fresh
Jacobian column norms by default; set `jacobian_scaling=False` to disable them.
LM and hybrid use the numerical geometry while retaining the physical cap.
Full-rank Newton/Gauss-Newton directions are unchanged in exact arithmetic.

`solve_root()` dispatches directly to four named functions: `solve_levenberg_marquardt`,
`solve_powell_hybrid`, `solve_newton_raphson` and `solve_gauss_newton`. Each binds its
numerical choices before entering one of two shared iteration engines. The
engines do not branch on method names.

The package-level imports, `src.solver` API and `src.problems` exports remain
available. The development imports are compatibility re-exports. Runners and internal
implementation imports use the packaged paths above. The
[organization and validation note](notes/09_source_organization.md) documents the
boundaries and the refactor checks.

## Run the studies

The stored convergence and timing reports predate default Jacobian equilibration.
They remain historical measurements; rerunning the scripts uses the current
defaults and may change their trajectories, counts, and timings. The
[radiative heat demo](../../demos/demo_eqn_solving/README.md) documents the
500-node validation of the new scaling behavior.

Run these commands from the `pysolver_view` repository root. The scripts also
work from another directory when given their absolute paths. Default output
directories are resolved relative to this experiment, independently of the
working directory. A relative `--output` path is resolved from the working
directory.

```powershell
python development/root_step_control/run_convergence_benchmark.py
python development/root_step_control/run_timing_benchmark.py
python development/root_step_control/run_step_geometry.py
python development/root_step_control/run_adapter_timing.py
```

| Runner | Default output | Saved results |
| --- | --- | --- |
| [run_convergence_benchmark.py](run_convergence_benchmark.py) | [results_convergence/](results_convergence) and [results_custom_convergence/](results_custom_convergence) | 100 native/custom configurations and 112 custom configurations, CSV summaries, JSON traces, reports, and PNG/SVG figures |
| [run_timing_benchmark.py](run_timing_benchmark.py) | [results_time_perf/](results_time_perf) | 56 configurations, raw batches, summary, metadata, timing report, and separate native/custom/controlled figures |
| [run_adapter_timing.py](run_adapter_timing.py) | [results_adapter_perf/](results_adapter_perf) | Paired direct/public timing of all four custom methods on three 100-variable problems |
| [run_step_geometry.py](run_step_geometry.py) | [results_step_geometry/](results_step_geometry) | LM/dogleg contour illustration in PNG and SVG |

The convergence runner defaults to both comparison families. The native/custom
family sweeps three custom step-control settings and two native initial-radius
settings on ten problems. The custom family compares all four methods with
common conservative controls and adds uncapped-backtracking/full-step Newton
policies. It includes a 20-variable duct, coupled rectangular examples, and a
10-variable logarithmic domain-recovery test, for fourteen problems overall.

Select `--comparison backends` or `--comparison custom` to run just one family.
Select `--suite original` for the seven mathematical systems, `--suite large`
for the three 100-variable systems, or `--suite extension` for the additional
custom examples.
For example, save a selected problem as a separate case:

```powershell
python development/root_step_control/run_convergence_benchmark.py --comparison backends --suite original --problems brown_almost_linear_n10 --output development/root_step_control/results_brown
```

Use `--custom-output` to choose the custom comparison's output directory.

The timing runner requests one BLAS thread before loading NumPy. Run its file
directly. Its defaults are 21 randomized rounds, five solves per batch, and three
warmups. A shorter run can be saved separately:

```powershell
python development/root_step_control/run_timing_benchmark.py --repetitions 3 --batch-size 1 --warmups 1 --output development/root_step_control/results_time_perf_quick_check
```

Each runner has `--help`. NumPy, SciPy, and Matplotlib are required. The complete
convergence suite and timing study also need optional CoolProp; it is not a new
dependency of the main package.

## Tests and direct use

Numerical and study checks are in [tests/](tests); public-interface regressions
are in [test_custom_root.py](../../tests/test_custom_root.py). Run tests separately
from timing measurements:

```powershell
python -m unittest discover -s development/root_step_control/tests -t . -v
```

Direct calls use the packaged numerical API:

```python
from pysolver_view import StepControlOptions, solve_root
from development.root_step_control.src.problems import ExtendedRosenbrockProblem

problem = ExtendedRosenbrockProblem(n=10)
result = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="powell_hybrid",  # or "levenberg_marquardt"
    options=StepControlOptions(
        initial_radius=0.1,
        max_radius=1.0,
        growth_factor=1.5,
        x_scale=problem.x_scale,
    ),
)
```

For Newton–Raphson or Gauss–Newton, import `NewtonOptions`,
`BacktrackingOptions`, and `StepLimitOptions` from the same package.
Both default to protected backtracking; `NewtonOptions(backtracking=None)`
selects full steps. The [Newton usage example](notes/08_newton_and_gauss_newton.md)
shows a conservative configuration. Canonical names are `levenberg_marquardt`,
`powell_hybrid`, `newton_raphson`, and `gauss_newton`. The short aliases `lm`,
`hybr`, `newton`, and `gn` remain accepted. Results report the canonical name in
`method` and the original spelling in `requested_method`.

The four named functions can also be imported directly from
`pysolver_view`. The development import re-exports them for compatibility.
Each accepts `fun, x0, jac=None, options=None,
callback=None`, with the last two arguments keyword-only. Direct calls return
the same numerical results and histories, including the canonical `method`.
`solve_root()` additionally normalizes aliases and supplies `requested_method`
and the display label `method_name`.

## Read the explanations and results

- [Convergence report](results_convergence/REPORT.md) and [timing results](results_time_perf/RESULTS.md).
- [LM and hybrid mathematics](notes/05_lm_and_hybrid_mathematics.md), including the [step-geometry figure](results_step_geometry/lm_hybrid_step_geometry.svg).
- [Newton–Raphson and Gauss–Newton](notes/08_newton_and_gauss_newton.md): directions, backtracking, full steps, and author-name aliases.
- [Four-method custom convergence comparison](results_custom_convergence/REPORT.md), including undamped and domain-recovery cases.
- [Controls, usage, and histories](notes/07_usage_and_controls.md) and [timing protocol](notes/06_time_performance.md).
- [Notes index](notes/README.md): thermodynamic policy, implementation boundaries, library integration, and other solvers.

Every named Python function, including internal callbacks and test fixtures,
has a docstring explaining its role. The numerical helpers also document their
coordinate conventions, array shapes, units, and failure behavior where relevant.

The custom methods are independent solver implementations; the reports explain
their differences from native MINPACK and the limits of the comparisons.
