# KKT assessment and controlled SLSQP

## Entry points and organization

- `src/pysolver_view/diagnostics/kkt.py`: numerical assessment, multiplier estimation,
  and independent reconstruction from supplied multipliers. No model evaluations,
  histories, callbacks, or application dependencies.
- `src/pysolver_view/optimization/solvers/slsqp/driver.py`: the controlled reverse-communication driver and its
  domain-failure policy. `_scipy_backend.py` isolates the compiled SLSQP interface;
  `SCIPY_LICENSE.txt` is explicitly included in source/wheel configuration.
- `src/pysolver_view/optimization/backends/custom.py`: inequality sign conversion and solver
  option mapping. The facade owns model and derivative accounting.
- `OptimizationSolver`: existing public problem, scaling, result, plot, and
  report interface, now also supporting `library="pysolver", method="slsqp"`.

The custom bridge requires **SciPy 1.17.1**. Its runtime guard is local to the
custom solver; dependency metadata and the lockfile select this version on Python 3.11 or
newer. Python 3.10 retains a SciPy 1.15 range for native methods and custom
roots. Run the complete suite on Python 3.11 or newer.

```python
import pysolver_view as psv
from pysolver_view.optimization.problems import StudyProblem

problem = StudyProblem("constrained_rosenbrock")
x0 = problem.initial
solver = psv.OptimizationSolver(
    problem,
    library="pysolver",
    method="slsqp",
    max_iterations=200,
    tolerance=1e-9,
    update_on="iteration",
    print_convergence=False,
    extra_options={
        "first_step_fraction": 0.25,
        "max_step_fraction": 1.0,
        "step_growth_factor": 1.5,
        "domain_backtracking": True,
        "feasibility_tol": 1e-7,
        "optimality_tol": 1e-4,
    },
)
x = solver.solve(x0)
assert solver.success
solver.print_optimization_report(
    tol=1e-4,
    include_kkt_conditions=True, include_multipliers=True,
)
```

`tolerance` maps to native `ftol` and the two independent acceptance tolerances
unless overridden. `max_iterations` maps to `maxiter`. `extra_options` wins.
The effective options are retained in `result.effective_options`.
Step fractions limit the proposed SLSQP direction; they are not absolute physical
step lengths. Native merit backtracking remains active. The controls do not
guarantee admissible thermodynamic states: models should raise `psv.DomainError`
only for recognized domain failures. Programming errors propagate.

Analytic gradients are preferred. Otherwise the facade supplies bounded numerical
derivatives using its `derivative_method` and `derivative_abs_step`; if no explicit
step is supplied, custom SLSQP uses `finite_diff_step` (default $10^{-7}$).
Finite-difference/complex-step probes are model calls, never iterations or
callbacks. Failed probes stop differentiation without retry and carry rejection
metadata. Complex-step requires a model that preserves imaginary perturbations.

Fixed variables must be eliminated before using custom SLSQP. Existing scaling
is still based on finite variable bounds; independent variable scaling is a
separate limitation. Finite physical bounds and meaningful scaling remain the
caller's responsibility when enabling `problem_scale`.

## Mathematical conventions

Public inputs use $h(x)=0$, $g(x)\leq0$, and $l\leq x\leq u$:

$$
L=f+\lambda^T h+\mu^Tg-z_L^T(x-l)+z_U^T(x-u),
\qquad \mu,z_L,z_U\geq0.
$$

The shared checker reports infinity-norm stationarity, maximum primal violation
(including every finite bound), maximum negative inequality/bound multiplier,
and maximum absolute complementarity product. Equality multipliers are free.
It estimates sign-feasible multipliers with a minimax linear program, then
reconstructs residuals from the original coefficients. Supplied complete
multiplier sets can be checked separately with `psv.evaluate_kkt_residuals`.
Missing bound multipliers are not silently assumed to be a complete native set.

`psv.assess_kkt` returns availability/reason, coordinates, component residuals,
tolerances, multiplier source, ordered multiplier arrays, and pass/fail status.
Absolute tolerances are expressed in the supplied variable, objective, and
constraint scales. A successful assessment is evidence of first-order KKT
conditions, not a general proof of a minimum. Constraint qualifications and
derivative quality still matter.

The low-level driver retains its nonnegative-inequality convention.
The adapter negates inequality values/Jacobian rows. Public result and iteration
history KKT dictionaries negate the driver's equality multipliers and inequality
complementarity products back to public signs. `backend_result.kkt` and its raw
history retain driver signs; `backend_result.fitness` uses public signs to support
authoritative final values. Native QP multipliers remain separate from independent
KKT estimates.

## Reports, histories, and acceptance

`evaluate_kkt_conditions(x_norm, tol)` remains available as an on-demand facade
adapter. It uses same-point values and derivatives, counts diagnostic work, and
does not publish progress or invoke callbacks. Legacy report fields remain,
with the following conventions:

- First-order optimality now uses infinity-norm stationarity.
- Bound-only feasibility and sign-feasible nonunique multiplier sets are checked.
- All finite inequality/bound multiplier slots are available, without a guessed
  active-set threshold. Missing bounds are omitted from the multiplier report.
- The legacy printed dual row remains the signed minimum multiplier ($\geq-\tau$);
  the shared `dual_feasibility` field is the nonnegative residual ($\leq\tau$).
- Unavailable multiplier fits produce unavailable diagnostics and an explicit
  report reason; they never pass by substituting zeros.
- Named constraint satisfaction is recomputed at the requested tolerance from
  numerical residuals, rather than trusting stale display metadata.
- Explicit `print_optimization_report(to_console=True)` works independently of
  whether live convergence printing was enabled.

Custom accepted-point histories reuse driver KKT diagnostics without another LP
or derivative request. Trial observation rows can have unavailable optimality.
The authoritative final fitness avoids an extra model evaluation at termination.
On-demand reports may obtain missing values/derivatives and estimate multipliers;
repeated same-point, same-tolerance report components reuse the facade assessment.

For native SciPy, `success` remains native success and KKT is an independent audit.
For custom SLSQP, `result.solver_success`, `result.feasible`, `result.kkt_passed`,
and strict `result.success` remain distinct. Custom success requires native
success plus independently passing feasibility and first-order optimality by default.
`OptimizationSolver(check_kkt=False)` disables automatic assessment and that
optimality gate; native success and feasibility remain required, while
`result.kkt_passed` becomes `None`. Explicit KKT/multiplier reports still assess
on demand. The low-level equivalent is `SLSQPOptions(check_kkt=False)`.
The [timing demo](../../demos/demo_kkt_conditions/run_kkt_timing.py) compares a constrained
Rosenbrock problem with checks enabled/disabled and separates assessor time from
total solve time. Solve-time `kkt_time` and `kkt_evaluations` are recorded.

Custom callbacks receive initial/accepted coordinates and a callback count,
regardless of the selected function/gradient/iteration observation stream.
Returning `False` or raising `StopIteration` cancels with a structured result
at the accepted point. Arrays and records are detached copies. Native SciPy
SLSQP callbacks still indicate major iterations without guaranteed acceptance.


## Residuals and acceptance ratio

| Measure | Definition |
| --- | --- |
| Stationarity | $\|\nabla f+J_h^T\lambda+J_g^T\mu-z_L+z_U\|_\infty$ |
| Primal feasibility | Maximum of absolute equality residuals, positive inequality residuals and finite-bound violations |
| Dual feasibility | Maximum negative part of inequality and finite-bound multipliers |
| Complementarity | Maximum absolute multiplier-times-slack over all inequalities and finite bounds |

The shared assessment aggregates these as

$$
r_{\mathrm{opt}}=\max(r_{\mathrm{stat}},r_{\mathrm{dual}},r_{\mathrm{comp}}),
\qquad
\mathrm{kkt\_ratio}=\max\left(
\frac{r_{\mathrm{primal}}}{\tau_{\mathrm{feas}}},
\frac{r_{\mathrm{opt}}}{\tau_{\mathrm{opt}}}\right).
$$

An available assessment passes when the ratio is at most one. Preserve all
components: an aggregate alone does not explain a failure.

## Multiplier fitting, units and independent checks

The minimax LP fits stationarity and complementarity together, using free
equality multipliers and nonnegative inequality/bound multipliers. It includes
all inequalities and every finite bound; it does not guess an active set from
an arbitrary proximity cutoff. Columns are equilibrated and the objective
gradient rescales the fit; residuals are then reconstructed in the original
supplied units. Zero-gradient cases and absent bounds have explicit treatment.

The fit can find sign-feasible multipliers in dependent systems without assuming
uniqueness. It still costs work, uses dense arrays and depends on conditioning
and derivative quality. Dual feasibility of fitted multipliers does not establish
that the native solver's multipliers have the same signs. Native QP multipliers
and independently estimated multipliers remain separate sources.

Absolute KKT tolerances depend on variable, objective and constraint scales.
For $\hat x=D^{-1}(x-a)$, $\hat f=f/q_f$ and $\hat h_i=h_i/q_i$,
physical equality multipliers satisfy
$\lambda_i=(q_f/q_i)\hat\lambda_i$. Inequalities transform analogously;
bound multipliers additionally depend on the variable scale. Always record
coordinate and ordering conventions before interpreting sensitivities.

The [independent study audit](../../src/pysolver_view/diagnostics/slsqp_audit.py)
reconstructs residuals using fresh values/derivatives and supplied multipliers.
It covers the bundled unscaled inequality-only `StudyProblem` models. It does
not call the production checker or fit new multipliers. This explicit audit
work is outside solver counters; it is not a general application KKT adapter.

The [named KKT demo](../../demos/demo_kkt_conditions/README.md) explains a known
solution, feasible nonoptimal and infeasible points, and ordered named metadata.
The [validation guide](validation.md) locates current tests and the twenty-case
tolerance/domain-recovery matrix. Real IPOPT/SNOPT/PyGMO execution remains outside
routine validation; a native SciPy comparison does not validate those backends.
