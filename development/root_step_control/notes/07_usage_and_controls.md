# Solver usage, step controls, and histories

Explicit first-step and persistent maximum-step controls are feasible for both
Levenberg-Marquardt (LM) and Powell's hybrid root-finding method (`hybr`). This
experiment implements independent dense trust-region variants and compares them
with `scipy.optimize.root(method="lm")` and `root(method="hybr")`.
The solvers use QR-based Gauss-Newton solves and hybrid rank-one QR updates,
with explicit trial-distance controls and SVD for damped or deficient-rank steps.
The [folder guide](../README.md) locates the shared source, runners, tests,
results, and notes. Timing comparisons use the custom solvers and native
SciPy LM/hybrid solvers.
See [the measured results](../results_convergence/REPORT.md) and [the complete summary](../results_convergence/summary.csv).
For thermodynamic models, backend details, and next steps, read the
[design and recommendation notes](README.md).
For the equations, derivations, and step geometry, see the
[LM and hybrid mathematics note](05_lm_and_hybrid_mathematics.md).
For residual, Jacobian, solver-overhead and total-time comparisons, see the
[timing experiment](06_time_performance.md).

The [timing report](../results_time_perf/RESULTS.md) compares custom and native execution
times on five workloads, including three 100-variable systems. Active
conservative controls and summary-history settings are separate experiments.
See [the report](../results_convergence/REPORT.md) for settings, native comparisons and limits.
The native/custom LM and Powell hybrid study is separate from the
[four-method custom comparison](../results_custom_convergence/REPORT.md).
The latter retains full-step failures, unsupported square-only Newton cases,
and inconsistent systems rather than treating every configuration as a root.

## Newton-family methods and author-name aliases

The same `solve_root` entry point also accepts `newton_raphson` and
`gauss_newton` with `NewtonOptions`. Both use Armijo backtracking by default.
`backtracking=None` gives full steps, including valid residual increases;
`BacktrackingOptions(step_limits=None)` only removes the displacement envelope.
The [Newton-family mathematics and examples](08_newton_and_gauss_newton.md)
describe their separate options and rank requirements.

The aliases `lm`, `hybr`, `newton`, and `gn` resolve to `levenberg_marquardt`,
`powell_hybrid`, `newton_raphson`, and `gauss_newton` respectively.
The result's `method` contains the canonical author name; `requested_method`
retains the supplied spelling. The low-level `solve_root` default is `levenberg_marquardt`; the public
`NonlinearSystemSolver` retains native SciPy `hybr` as its default.
Legends use Levenberg-Marquardt, Powell hybrid, Newton-Raphson, and
Gauss-Newton. Aliases do not create extra algorithms or change the backend.

The trust-region descriptions and radius fields below refer to LM/Powell
hybrid. Newton-family histories instead record the line-search fraction and
optional step limit. Their full-step mode deliberately does not promise
monotone residual reduction.

## What transfers from turbodash

The neighboring `turbodash/turbodash/optimization/slsqp.py` driver controls the
fraction of SLSQP's proposed direction with `first_step_fraction`,
`max_step_fraction`, and `step_growth_factor`. Its private backend consistently
changes the stored step, trial coordinates, and line-search bookkeeping. It
retains SciPy's native merit line search and accumulated solver state. Recognized
`DomainError` failures cause shorter real model evaluations; recovery also
reduces the next iteration's cap. The backend is pinned to SciPy 1.17.1.

A direction fraction is not an absolute distance limit: a small fraction of a
huge Newton step can still be huge. Here the control is the distance itself in
fixed, caller-selected variable coordinates:

$$
z_i = p_i/s_i, \qquad \|z\|_2 \leq \Delta \leq \Delta_{\max}.
$$

`x_scale` supplies $s$. Consequently every trial also satisfies
$|p_i| \leq s_i\Delta_{\max}$. Residual scaling is separate and belongs in the
model. The same radius controls apply before evaluating accepted, rejected, and
undefined trials. Numerical-derivative probes have a separate distance setting.
The default Jacobian equilibration described below changes the numerical model
coordinates, while retaining these physical distance limits and probe settings.

## What SciPy already provides

The source review targets SciPy **1.17.1**, the version used for this experiment.
Documentation links below describe the public APIs; tagged source links fix the
implementation being discussed.

| Interface | Exposed controls | Relevant limitation |
| --- | --- | --- |
| [`root(method="lm")`](https://docs.scipy.org/doc/scipy/reference/optimize.root-lm.html) | `factor`, `diag` | Initial radius and scaling; no persistent maximum or growth control |
| [`root(method="hybr")`](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html) | `factor`, `diag` | Same limitation; Powell hybrid uses dogleg steps and Jacobian updates |

MINPACK starts with $\Delta_0=\mathrm{factor}\|Dx_0\|_2$, or
$\Delta_0=\mathrm{factor}$ when that norm is zero. Thus a fixed `factor` is not
an absolute first-step distance, and the radius can subsequently grow. `diag`
changes the metric; multiplying all its entries by a common constant generally
also multiplies the initial radius, cancelling the intended reduction in
physical step size. MINPACK's LM parameter routine also allows approximately
10% tolerance in satisfying its trust-radius equation. See the initialization,
radius updates, and `lmpar` in [SciPy's MINPACK source](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__minpack.c).

SciPy exposes neither an accepted-iteration callback for LM/hybrid nor a
suspended, mutable MINPACK trial state comparable to the SLSQP backend. See
[`_root.py`](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/_root.py)
and [`_minpack_py.py`](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/_minpack_py.py).
Clipping `x` inside the residual wrapper would evaluate a different point from
the one MINPACK is judging. Repeated short solves would discard native state.
Preserving the exact native algorithms while exposing these controls would
require maintaining a modified native implementation.

## Use the experiment

Run from the `pysolver_view` repository root. The solvers use NumPy and public
SciPy APIs, without a compiler or new solver dependency. Plotting uses
Matplotlib. The default complete benchmark and thermodynamic timing workloads
also require optional **CoolProp**; `--suite original` runs the seven mathematical
systems without it.

```python
from pysolver_view import StepControlOptions, solve_root
from development.root_step_control.src.problems import ExtendedRosenbrockProblem

problem = ExtendedRosenbrockProblem(n=10)

options = StepControlOptions(
    initial_radius=0.1,
    max_radius=1.0,
    growth_factor=1.5,
    x_scale=problem.x_scale,
    jacobian_scaling=True,  # Default: balance Jacobian columns numerically.
    residual_tol=1e-8,
)
result = solve_root(problem.residual, problem.x0, jac=problem.gradient,
                    method="lm", options=options)  # or method="hybr"
print(result.success, result.x, result.residual_norm)
```

Existing pysolver_view problem objects can be used directly:

```python
result = solve_root(problem.residual, x0,
                    jac=getattr(problem, "gradient", None),
                    method="hybr", options=options)
```

The experiment returns an `OptimizeResult`. Its `x` and `fun` always refer to the
last accepted iterate, including on failed trials or budget exhaustion. Supply
`callback(record)` to monitor accepted iterates, including the initial point;
return `False` to stop. The callback receives a copy.

| Control | Meaning |
| --- | --- |
| `initial_radius` | Absolute scaled radius for the first trial, independent of $x_0$ |
| `max_radius` | Persistent bound on every solver trial distance |
| `growth_factor` | Maximum multiplicative radius growth after a well-predicted boundary step; 1 disables regrowth |
| `shrink_factor` | Multiplicative radius reduction after poor or undefined trials |
| `acceptance_ratio` | Minimum ratio of actual to predicted merit reduction for acceptance |
| `x_scale` | Positive fixed characteristic distance per variable; scalar broadcasts |
| `jacobian_scaling` | Boolean, default `True`: balance the columns of `J * x_scale` in numerical coordinates; physical distance limits and derivative probes still use fixed `x_scale` |
| `residual_tol` | Required residual two-norm for root success |
| `xtol`, `gtol` | Stagnation/stationarity termination thresholds; never imply root success |
| `max_iterations` | Budget of trial attempts, including rejected attempts |
| `max_nfev` | Budget of all residual calls, including derivative probes and undefined trials |
| `finite_diff_step` | Forward probe distance in scaled coordinates when `jac` is omitted |
| `history_level` | `"full"` (default), `"summary"` for scalar histories, or `"none"` to disable stored histories |

Choose scales before selecting radii. For example, `x_scale=[100, 0.01]` and
`max_radius=0.2` imply trial component limits of 20 and 0.002 in the original
units, with an additional joint Euclidean constraint. A cap limits motion; it
does not impose variable bounds or guarantee membership of a physical domain.

## Jacobian scaling and physical limits

All four custom methods enable `jacobian_scaling=True` by default. To disable
it through the common interface, use
`NonlinearSystemSolver(..., library="pysolver", options={"jacobian_scaling": False})`.
For direct calls, use `StepControlOptions(jacobian_scaling=False)` for LM/hybrid
or `NewtonOptions(jacobian_scaling=False)` for Newton/Gauss-Newton. This is a
custom-backend option; native SciPy options and behavior are unchanged.

Write $A_0=J\,\mathrm{diag}(s)$ and $C=\mathrm{diag}(c)$, where $c$ contains
positive numerical factors. The solver factors $A=A_0C$ and converts a numerical
step $v$ to the caller-scaled displacement $z=Cv$. The initial nonzero column
norms define a fixed reference: their median in logarithmic coordinates.
Each factor is that reference norm divided by the largest norm seen for its
column on fresh Jacobian evaluations, clipped to
$[\sqrt{\epsilon_{\mathrm{mach}}},1/\sqrt{\epsilon_{\mathrm{mach}}}]$.
Zero columns keep factor one until a nonzero derivative appears. No extra model
or derivative evaluations are needed. Hybrid freezes $C$ during Broyden updates
and changes it only when evaluating a fresh Jacobian.

LM and hybrid generate their candidates inside $\|v\|_2\leq\Delta$. If
$\|Cv\|_2>\Delta$, the candidate is shortened along the same direction to
honor the physical limit as well. Both norms are checked after coordinate
rounding. This preserves a physical first-step cap of `initial_radius` and a
persistent cap of `max_radius`; it is not an exact optimization over the
intersection of the two trust regions.

For full-rank Newton and Gauss-Newton, this invertible change of coordinates
leaves the exact physical linear direction unchanged. Finite-precision solves
and numerical-rank decisions can still differ. Rank-deficient Gauss-Newton's
minimum-norm choice can change because its norm is now measured in numerical
coordinates. Backtracking and its physical step limits retain their meanings.
Scaling can improve poorly balanced Jacobians, but does not replace sensible
residual normalization, physical limits, or domain-error handling.

## Algorithm and failure handling

With the numerical Jacobian $A=J\,\mathrm{diag}(s)C$, both trust-region methods
model $\frac12\|F+Av\|_2^2$ inside $\|v\|_2\leq\Delta$, followed by the
physical-distance safeguard described above. With scaling disabled, $C=I$.

- **LM** obtains the unconstrained Gauss-Newton step from pivoted QR and a
  triangular solve when the factor passes its conditioning checks. Only when
  damping or a rank/conditioning fallback is needed does it compute an SVD.
  Damping brackets $(A^TA+\lambda I)v=-A^TF$ using spectral coefficients; their
  norm is evaluated without repeated multiplication by the right singular
  vectors. The numerical candidate lies on the feasible side of the radius equation.
  The implementation never explicitly forms the normal equations.
- **Hybrid** follows the dogleg Cauchy-to-Gauss-Newton path to the radius boundary.
  A stable quadratic intersection computes the boundary step directly.
  Its fast Gauss-Newton path uses QR; a minimum-norm SVD step handles deficient
  rank. Scaled Broyden rank-one updates incorporate valid trials,
  including rejected ones. Two consecutive agreement ratios below 0.1 trigger
  a fresh Jacobian. Approximate models are also refreshed before declaring
  stationarity, step stagnation, or a non-descent model. Undefined trials cannot
  supply secant information and retain the current model. QR factors are updated
  with the Broyden matrix, with periodic refactorization to limit accumulated
  update error. Refactorizing a matrix does not itself evaluate a new Jacobian.
- Predicted reduction uses the actual rounded numerical displacement after
  enforcing both distance limits.
  Accepted points strictly reduce the residual norm. Poor trials shrink the
  radius; good steps near the boundary may expand it up to the maximum.
  LM reuses its Jacobian and available factors on retries and recomputes them after acceptance;
  hybrid updates or refreshes the Jacobian as described above.
- Recognized `DomainError` and nonfinite residuals cause rejection and radius
  reduction. No surrogate residual or penalty is supplied. Other exceptions
  propagate, so programming errors are not silently treated as domain failures.
  An invalid initial point, invalid derivative probe, or invalid Jacobian stops
  with a diagnostic. All retries are bounded by the trial/evaluation budgets.
- Only a residual norm at or below `residual_tol` means success. A stationary
  least-squares point with a nonzero residual, tiny steps, or exhausted budgets
  means failure. This matters especially for LM and overdetermined systems.

For hybrid, the update in the current numerical coordinates is

$$
A_{\mathrm{new}} = A + \frac{(y-Av)v^T}{v^Tv},\qquad
y=F(x_{\mathrm{trial}})-F(x_{\mathrm{base}}).
$$

Here $v=C^{-1}\mathrm{diag}(s)^{-1}(x_{\mathrm{trial}}-x_{\mathrm{base}})$
uses the actual rounded trial. Holding $C$ fixed through secant updates keeps
the Jacobian model and its factors in the same coordinates.

This retains the defining dogleg/Broyden structure of Powell's hybrid method.
It differs from native MINPACK HYBRJ in its factorization safeguards, shared configurable
radius/acceptance policy, residual-based success, and omission of native
slow-progress counters that could stop deliberately restricted steps. The poor
agreement counter resets after a fresh Jacobian. It is an experimental
Powell-style hybrid variant, not a patch or exact reproduction of SciPy's native
solver. See the [HYBRJ source](https://netlib.org/minpack/hybrj.f).

Both custom methods support dense square and overdetermined systems; SciPy's
native `hybr` requires a square system. The experiments have no sparse linear
algebra, automatic residual scaling, bounds, robust loss, or global convergence guarantee.
Residuals and variables still need sensible scaling. Damped LM still performs
dense SVDs, and Python bookkeeping can be material for inexpensive residuals.
The [timing experiment](06_time_performance.md) compares the custom solvers with
native MINPACK on the same workloads; no universal speed advantage is implied. `njev` counts full Jacobian evaluations; `nupdates` counts hybrid
rank-one updates. `linear_algebra` separates QR, QR-update, triangular-solve,
condition-check and SVD work by operation counts.

The public numerical primitives are SciPy's [pivoted QR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html),
[QR update](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_update.html),
and [triangular solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html).
Using these compiled kernels does not call a native root solver.

## Histories, tests, and reproducibility

- `history`: initial point and accepted iterates, with residual and step norms.
  Its `radius` is the radius available for the next trial.
- `trials`: every attempted step, its accepted base, actual distance, radius
  used, agreement ratio, domain validity, and acceptance decision.
  Trust-region `scaled_step_norm` is the physical norm $\|z\|_2$;
  `numerical_step_norm` separately records $\|C^{-1}z\|_2$.
- `evaluations`: every residual call, tagged as initial, trial, or finite
  difference. Derivative probes are intentionally outside the trial-radius
  guarantee and use their own distance setting.

These lists contain coordinate copies with `history_level="full"`.
`"summary"` retains the same scalar diagnostics, without `x` or `x_base` arrays.
`"none"` returns empty lists. All three levels preserve the solve, budgets,
`nit`, `ntrials`, `nfev`, `njev`, `nupdates`, `nrejected`, `ninvalid`, and the
maximum scaled trial distance, used radius and step/radius ratio. `ninvalid`
counts undefined solver trials, not a failed initial evaluation or derivative
probe. A callback always receives a complete detached accepted-point record,
including coordinates, regardless of the storage level. Use full histories
when auditing actual evaluation locations; summary logs cannot reconstruct them.

Results also expose `jacobian_scaling`, the final detached factor vector
`jacobian_scale`, and `jacobian_scale_updates`. The factors multiply `x_scale`
for numerical linear algebra; they do not replace the physical scales used
for reported step sizes. Updates count fresh nonzero Jacobians used to update
the factors, not Broyden updates or extra model calls. If no Jacobian is needed,
the factors remain one and the update count is zero.

The cap is subject only to floating-point rounding (eight machine epsilons in
the scaled norm). Coordinate-rounding checks reserve room for addition to large
base coordinates. A step too small to represent is reported as stagnation.

SciPy LM/hybrid residual-call histories include rejected trials and repeated
calls. Their consecutive evaluation displacement is **not** an accepted step
or a measurement of the native trust radius. The benchmark separates that
quantity from custom trial distances measured from a known accepted base.
Baseline final results use SciPy's returned `result.x` and independently check
the residual; native success flags alone are insufficient.

```powershell
python -m unittest discover -s development/root_step_control/tests -t . -v
python development/root_step_control/run_convergence_benchmark.py --output development/root_step_control/results_convergence
```

The benchmark problems subclass the project's `NonlinearSystemProblem` and
expose `residual(x)` and `gradient(x)`. They are established multidimensional
systems, with equations, starts, dimensions, and references in [mathematical.py](../src/problems/mathematical.py).
The benchmark compares them against SciPy LM/hybrid at `factor=100`
and `factor=0.1`, plus three configurations of each custom solver. It writes
`summary.csv`, `traces.json` (including versions, settings and full histories),
and PNG/SVG figures. Use `--no-plots` for numerical checks only, or
`--problems powell_badly_scaled_n2 brown_almost_linear_n10` to select cases.
The extended suite adds 100-variable Broyden tridiagonal, dense discrete integral
equation, and 50-cell thermodynamic duct systems. See the benchmark help and
[timing instructions](06_time_performance.md) for selecting these workloads.

The public `NonlinearSystemSolver` now supports `library="pysolver"` for all four
methods. Final points come from the backend, and accepted-iteration observations
are available through `update_on="iteration"`. Native SciPy remains the default.
See [public usage, option precedence, and callback contracts](10_public_interface.md).
