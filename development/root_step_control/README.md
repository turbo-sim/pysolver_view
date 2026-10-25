# Root step control: folder guide

This experiment implements Levenberg–Marquardt, Powell hybrid,
Newton–Raphson, and Gauss–Newton root solvers with explicit step controls.
It compares the first two with native SciPy and all four with each other.
Problems use the project's `NonlinearSystemProblem` interface.

Throughout the code, notes and results, **SciPy/MINPACK** means the native
solvers exposed by `scipy.optimize.root`. **PySolverView** means the custom
root implementations in this experiment. Saved solver
identifiers use `scipy_*` and `pysolverview_*`, respectively.

## Folder structure

```text
root_step_control/
├── README.md
├── src/                         # Shared solvers and problem definitions
├── tests/                       # Numerical, solver, problem, and timing checks
├── notes/                       # Mathematics, usage, design, and recommendations
├── run_convergence_benchmark.py
├── run_timing_benchmark.py
├── run_step_geometry.py
├── results_convergence/         # Step-control cases, traces, plots, and report
├── results_custom_convergence/  # Four-method and Newton-policy comparisons
├── results_time_perf/           # Native/custom timings, settings, and plots
└── results_step_geometry/       # Illustrated LM and dogleg local-model steps
```

The `run_*.py` files contain their study-specific setup, execution, plotting,
reporting, and command-line handling. Shared solver functions and all problem
definitions live together in `src/`. Each study writes to a named
`results_<case>/` directory; use `--output` to keep an additional case separately.
Result directories contain outputs rather than implementation or tests.

| Source | Purpose |
| --- | --- |
| [solver.py](src/solver.py) | Shared iteration, step controls, histories, and failure handling |
| [linear_model.py](src/linear_model.py) | QR/SVD factors, LM damping, dogleg steps, and secant updates |
| [newton_solver.py](src/newton_solver.py) | Newton–Raphson/Gauss–Newton iteration, Armijo acceptance, and full-step modes |
| [newton_options.py](src/newton_options.py) | Validated backtracking and scaled step-limit options |
| [line_search.py](src/line_search.py) | Rounded trial geometry and pre-evaluation distance audit |
| [solver_common.py](src/solver_common.py) | Shared explicit domain-failure and evaluation-budget exceptions |
| [problems.py](src/problems.py) | Standard multidimensional mathematical systems |
| [problem_thermodynamics.py](src/problem_thermodynamics.py) | Coupled CoolProp nitrogen duct problem |

## Run the studies

Run these commands from the `pysolver_view` repository root. The scripts also
work from another directory when given their absolute paths. Default output
directories are resolved relative to this experiment, independently of the
working directory. A relative `--output` path is resolved from the working
directory.

```powershell
python development/root_step_control/run_convergence_benchmark.py
python development/root_step_control/run_timing_benchmark.py
python development/root_step_control/run_step_geometry.py
```

| Runner | Default output | Saved results |
| --- | --- | --- |
| [run_convergence_benchmark.py](run_convergence_benchmark.py) | [results_convergence/](results_convergence) and [results_custom_convergence/](results_custom_convergence) | 100 native/custom configurations and 112 custom configurations, CSV summaries, JSON traces, reports, and PNG/SVG figures |
| [run_timing_benchmark.py](run_timing_benchmark.py) | [results_time_perf/](results_time_perf) | 56 configurations, raw batches, summary, metadata, timing report, and separate native/custom/controlled figures |
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

All unit and integration checks are in [tests/](tests). Run them separately
from timing measurements:

```powershell
python -m unittest discover -s development/root_step_control/tests -t . -v
```

The existing public experiment import remains available:

```python
from development.root_step_control import StepControlOptions, solve_root
from development.root_step_control.src.problems import ExtendedRosenbrockProblem

problem = ExtendedRosenbrockProblem(n=10)
result = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="hybr",  # or "lm"
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
shows a conservative configuration. Method aliases are
`levenberg_marquardt`, `powell_hybrid`, `newton`, and `gn`.

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
