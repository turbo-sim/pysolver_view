# Timing native and custom root-finding methods

This experiment first compares **Levenberg-Marquardt** and **Powell hybrid**
between SciPy/MINPACK and the custom [driver](../../../pysolver_view/solvers/root/solver.py). A separate
comparison includes all four PySolverView methods: Levenberg-Marquardt,
Powell hybrid, Newton-Raphson, and Gauss-Newton. All plot labels use these
full names and identify the backend.

The custom-method comparison has unrestricted-user-control and protected-step
variants. The unrestricted variant keeps LM/Powell hybrid's intrinsic
trust-region adaptation with generous user radii and uses `backtracking=None`
for full-step Newton-Raphson/Gauss-Newton. Protected variants use a small initial
distance, a persistent maximum, and gradual growth. Every primary custom case
retains full histories; a secondary experiment compares full and summary
LM/Powell hybrid histories on the 100-variable workloads.

Read the generated [timing results](../results_time_perf/RESULTS.md) for
the measured numbers and [summary.csv](../results_time_perf/summary.csv) for the complete table.
The studies and problems stay in `root_step_control/`; they use the packaged
custom implementation through `NonlinearSystemSolver(library="pysolver")`.
The standard run contains **56 configurations**: ten configurations on each of
five workloads (two native, four custom unrestricted, four custom protected),
plus two summary-history configurations on each of three 100-variable workloads.
With 21 rounds, five solves per batch, and separate clean/instrumented modes,
this produces **11,760 timed solves**. Warmups and independent verification
are additional untimed work.

## Run it

From the `pysolver_view` repository root:

```powershell
python development/root_step_control/run_timing_benchmark.py
```

Run the file directly, rather than with `python -m`: it requests one BLAS/OpenMP
thread before NumPy/SciPy load. Imports and warmup are outside the timers.
The experiment requires the repository's usual NumPy, SciPy and Matplotlib
dependencies plus **CoolProp** for the thermodynamic workload. CoolProp is not
added as a dependency of the main package.

The defaults are 21 randomized/interleaved timing rounds, five solves per batch,
and three warmup solves per problem/solver. Each configuration has both clean
total-time measurements and separate instrumented measurements. For a quick
functional check in a separate output directory:

```powershell
python development/root_step_control/run_timing_benchmark.py --repetitions 3 --batch-size 1 --warmups 1 --output development/root_step_control/results_time_perf_quick_check
```

Changing `--output` changes the destination for generated results. The source
and these instructions remain here.
The optional `--problems NAME ...` argument selects named workloads for focused
checks. Solver identifiers are resolved through an explicit configuration
registry; multiword method names are never recovered by splitting underscores.

## A real thermodynamic workload

`HeatedGasDuctProblem` derives from the existing `NonlinearSystemProblem` class
and exposes `residual(x)` and `gradient(x)`. Ten- and fifty-cell cases have
interleaved unknowns $x=(T_1,p_1,\ldots,T_N,p_N)$: **20 and 100 coupled
variables and equations**, respectively.
Each cell receives prescribed heat and has a lumped pressure loss:

$$
h(T_i,p_i)-h(T_{i-1},p_{i-1})-q_i=0,
$$

$$
p_{i-1}-p_i-\frac{K G^2}{2\rho(T_i,p_i)}=0.
$$

The inlet is nitrogen at 320 K and 2 MPa, with $G=300$ kg/(m$^2$ s).
Both benchmark sizes use total heat input 100,000 J/kg and total resistance
coefficient 80, distributed uniformly over the cells. Thus the ten-cell case
has $q_i=10\,000$ J/kg and $K=8$, while the fifty-cell case has
$q_i=2\,000$ J/kg and $K=1.6$. Increasing resolution does not multiply the
physical heat and resistance by five. Residuals are normalized by 10,000 J/kg and
20,000 Pa. Fixed variable scales are 20 K and 100,000 Pa. The initial guess is
the inlet state repeated at every cell; no root is precomputed for the start.

This is an idealized low-speed heated duct model. It neglects kinetic-energy
changes and does not model choking or two-phase flow. It is a reproducible
property-evaluation workload, not validation of a full turbomachinery model.

Each residual evaluation performs $N$ real CoolProp HEOS pressure/temperature
state updates and reads enthalpy and density. Each Jacobian evaluation performs
$N$ new state updates and obtains four analytic property derivatives per cell.
There are no sleeps, dummy calculations or cross-call residual caches. The
low-level fluid object is constructed outside timing and reused sequentially;
every update supplies the complete requested pressure and temperature.
[CoolProp low-level API](https://coolprop.org/coolprop/LowLevelAPI.html#partial-derivatives).

Analytic derivatives are checked against central differences. Post-solve checks
verify the equation residuals, increasing temperature and decreasing pressure.
The ten-cell outlet is approximately 413.94 K and 1.78836 MPa; detailed physical
errors and state diagnostics are in [metadata.json](../results_time_perf/metadata.json).

Two additional cases use the established Broyden tridiagonal system with **20
and 100 variables**. A fifth case is MINPACK's **100-variable discrete integral
equation**, whose Jacobian is genuinely dense. Its standard discretization
couples every variable to every equation; this complements the locally coupled
Broyden system. Their inexpensive residuals make solver overhead and dense
factorization costs visible. Equations, standard starts, analytic Jacobians and
source references are in [mathematical.py](../src/problems/mathematical.py). All five cases use the
project's nonlinear problem interface.

## What each timing means

| Quantity | Measurement |
| --- | --- |
| Clean total | Wall time around the full solve with no residual, Jacobian or kernel timers |
| Residual | Accumulated time inside `problem.residual(x)` during an instrumented solve |
| Jacobian | Accumulated time inside analytic `problem.gradient(x)` during an instrumented solve |
| Solver overhead | Instrumented total minus residual and Jacobian time |
| Instrumented total | Full solve with callback timers, target detection, and custom kernel timing enabled |
| Custom kernel subsets | SVD, QR factorization, QR update and triangular-solve times; already included in solver overhead |

**Solver overhead is not pure linear-algebra time.** It includes factorization,
matrix operations, step selection, convergence tests, Python/native bookkeeping,
callback overhead, public-interface reporting preparation, detached results,
and custom history storage. Native MINPACK
does not expose separate QR timing through SciPy's public interface. Isolating
that time would require instrumenting the native source/build. This experiment
does not estimate it or report it as zero.

The kernel timers temporarily replace only imported callables in the local
linear-model module, run sequentially, and restore them afterward,
including on exceptions. SciPy's installed code and native routines are untouched.
These measured kernels are disjoint subsets of solver overhead. Matrix products,
norms, condition estimation, damping/dogleg searches and Python logic remain
outside them, so the sum is not the complete linear-algebra cost. Native kernel
timings remain unavailable rather than zero.

All model calls use analytic Jacobians. Thus residual and Jacobian timings do
not nest or double-count finite-difference probes. Physical property calls made
while building a Jacobian belong to Jacobian time. Per-call averages and call
counts are reported alongside cumulative times.

## Fairness and interpretation

- All solvers use identical starts, normalized equations, analytic derivatives
  and fixed variable scales.
- Native MINPACK uses `factor=100` and `diag=1/x_scale`. The custom initial
  radius is $100\|x_0/s\|_2$ (100 at zero), with growth factor 2 and a maximum
  of at least $10^{12}$. Every generous-radius custom trace is audited to confirm
  that the user maximum never binds. Normal trust-region adaptation and rejection remain
  enabled; removing them would change the algorithms.
- The custom solvers retain full histories in the main comparison. Native
  SciPy does not expose equivalent complete histories. These are timings of
  implementations as used, including their diagnostic costs, not isolated
  mathematical algorithms.
- Active-control runs on all five problems use
  `initial_radius=0.1`, `max_radius=1`, `growth_factor=1.5` in the custom solvers.
  For Newton-Raphson/Gauss-Newton, the equivalent distance settings are
  `initial_limit=0.1`, `max_limit=1`, `growth_factor=1.5` inside
  `BacktrackingOptions`. Matching these bounds does not make Armijo acceptance
  identical to trust-region acceptance. A shortened Newton-family step carries
  its reduced distance limit forward according to the documented recovery policy.
  PySolverView summary-history runs use the same controls and residual target.
  They remove coordinate storage only and are labeled separately from the full-history
  speed comparison. The initial radius restricts these solves; the persistent
  cap also binds on the duct workload. Native generous-radius timings provide
  context here; native MINPACK does not expose the same persistent-cap policy.
- Full-history step bounds are audited from actual trial and accepted-base
  coordinates against either a radius or a line-search limit. Full-step
  Newton-family cases have no imposed bound and may increase the residual.
  Scalar summary records are checked against their recorded radii, and
  history-level tests separately verify identical numerical results.
- Success is independently checked using $\|F\|_2\leq10^{-8}$. Native stopping
  rules can continue below that target. Instrumented runs record the time and
  residual-call index of the first **evaluated point** meeting it, without
  interrupting the solve. That point is not necessarily a native accepted
  iterate. Native status is retained separately.
- Each reported breakdown comes from the batch with the median total, so its
  components add to that total. Independently taking medians of all components
  would generally lose additivity. Clean total-time quartiles show variation
  among batch averages; they are not confidence intervals.
- Clock calls, first-target norm checks and kernel wrappers perturb small timings.
  Use clean medians for speed ratios. The CSV includes the instrumented/clean
  ratio to make this perturbation visible.
- Imports, problem construction, solver-option construction, warmup, post-solve
  verification, and study report/plot generation are excluded. Custom totals
  include public solver construction, option copying, convergence-report
  preparation, and detached result assembly, with printing/live plotting off.
  Native timings call scipy.optimize.root directly. Each model call includes
  its necessary state updates and array assembly.

The results apply to this machine and these workload sizes. Actual expensive
flash calculations, mixtures or different property input pairs can change the
balance between model cost and solver overhead.

## Measured outcome on 25 October 2026

The maintained run completed **56 configurations and 11,760 timed solves**.
Every solve met the independently checked residual target. The maximum audited
protected-step ratio was `1.0000000000000018`, within the floating-point audit
tolerance. The raw table contains 2,352 batch averages. These timing workloads
all converged; robustness conclusions require the separate difficult-start
convergence studies, which retain failures and domain rejections.

The following are **clean median milliseconds per solve**, with full histories:

| 100-variable workload | PySolverView Levenberg-Marquardt, generous | PySolverView Powell hybrid, generous | Newton-Raphson, full step | Gauss-Newton, full step |
| --- | ---: | ---: | ---: | ---: |
| Broyden tridiagonal | 2.7126 | 3.8002 | 2.4857 | 2.6873 |
| Dense discrete integral equation | 2.3256 | 2.0024 | 2.1277 | 2.2002 |
| Heated nitrogen duct | 7.0999 | 4.4108 | 6.4176 | 6.7389 |

| 100-variable workload, protected steps | Levenberg-Marquardt | Powell hybrid | Newton-Raphson | Gauss-Newton |
| --- | ---: | ---: | ---: | ---: |
| Broyden tridiagonal | 16.5418 | 5.5971 | 6.3782 | 7.2520 |
| Dense discrete integral equation | 7.2159 | 2.4139 | 3.4873 | 4.5972 |
| Heated nitrogen duct | 92.2530 | 27.0018 | 51.3572 | 50.5545 |

Newton-Raphson and Gauss-Newton have nearly the same direction on these square
full-rank problems; small timing differences between them should not be
interpreted as a general algorithmic advantage. Protected Newton-family runs
avoid LM's damped SVD search and are faster than protected LM on these three
workloads. Powell hybrid remains faster in these protected cases: it reuses
and updates its Jacobian instead of evaluating a fresh one after each move.
On the 100-variable duct, protected LM uses 28 Jacobian calls, the Newton-family
methods use 27, and Powell hybrid uses one.

The native comparison remains separate. On the 100-variable duct, native
SciPy/MINPACK LM and Powell hybrid take 8.4895 and 5.3236 ms,
respectively, versus 7.0999 and 4.4108 ms through the custom
public interface with generous radii. Native Powell hybrid is faster on the
inexpensive 100-variable mathematical problems. These totals compare interfaces
as used, with different diagnostic and stopping costs; they do not isolate
algorithmic performance. See the report for quartiles and all measured components.

The separate [interface timing runner](../run_adapter_timing.py) compares the
same custom core directly and through NonlinearSystemSolver, with identical
full-history policies. Its [report](../results_adapter_perf/RESULTS.md) and
[CSV](../results_adapter_perf/summary.csv) quantify interface cost without
attributing that cost to residuals or numerical kernels.

## Files and checks

- [run_timing_benchmark.py](../run_timing_benchmark.py): complete timing study,
  including configuration, measurements, checks, reporting, plotting, and CLI.
- [thermodynamics.py](../src/problems/thermodynamics.py): thermodynamic equations and derivatives.
- [RESULTS.md](../results_time_perf/RESULTS.md): measured tables and plot.
- [summary.csv](../results_time_perf/summary.csv): aggregate timings, counts, checks and ratios.
- [raw_batches.csv](../results_time_perf/raw_batches.csv): every measured batch average.
- [metadata.json](../results_time_perf/metadata.json): versions, numerical-library configuration,
  requested thread environment, settings, starts and physical diagnostics.
- [timing_breakdown.svg](../results_time_perf/timing_breakdown.svg): exportable timing figure.
- [custom_timing_breakdown.svg](../results_time_perf/custom_timing_breakdown.svg):
  all four PySolverView methods with generous/full-step policies.
- [controlled_timing_breakdown.svg](../results_time_perf/controlled_timing_breakdown.svg):
  all four PySolverView methods with protected steps.
- [summary_timing_breakdown.svg](../results_time_perf/summary_timing_breakdown.svg):
  secondary history-storage comparison.

The stacked bars use additive instrumented components; black diamonds and
error bars show clean total medians and quartiles. A failed residual target is
marked explicitly and excluded from successful-solve timing ratios.

Run the functional tests separately from a timing session:

```powershell
python -m unittest discover -s development/root_step_control/tests -t . -v
```

They test conservation, derivatives, state resets, timing accounting and
instrumentation cleanup. They make no assertions about which solver is faster.

## Source organization verification

During the earlier module organization, a separate
11-round timing comparison repeated all 56 configurations before and after the
refactor. The historical 21-round study above remains a separate measurement.
See [source organization and validation](09_source_organization.md) for the fresh
numbers and their limits; counters and convergence traces were preserved.

## Public interface overhead

The direct/public experiment uses 24 configurations: all four methods, generous
or full steps and controlled steps, on each of the three 100-variable problems.
Eleven interleaved rounds, three solves per batch, and clean/instrumented modes
produce **3,168 timed solves**. Every solve reaches the target; residual and
Jacobian calls and accepted-iteration counts agree between interfaces.

With full histories, public/direct clean-time ratios span 1.10-1.57 on the two
cheap mathematical workloads. Most differences are about 0.44-1.20 ms per solve.
For the thermodynamic duct, ratios span 0.98-1.18. A negative difference for
controlled LM is measurement variation, not evidence that adding an adapter
accelerates numerical work. Controlled Newton and Gauss-Newton timings also vary
despite matching directions and work counts. Raw batches and quartiles are saved.

The public interface pays for option resolution, accepted-point recording,
common reports and detached results. These are real costs, most visible when
residuals are inexpensive. The separate main timing session had appreciably
higher absolute times, including on native SciPy; do not subtract numbers from
different sessions to estimate adapter cost. Use the paired experiment for that.

The integration is usable for these workloads, but has not removed Python
bookkeeping costs. A focused future optimization could reduce repeated history
copying and format reports lazily, preserving detached data and callback
contracts. Direct `psv.solve_root` is available when the common frontend is not
needed. These observations do not justify changing residual targets or disabling
diagnostics silently. See [the interface report](../results_adapter_perf/RESULTS.md).
