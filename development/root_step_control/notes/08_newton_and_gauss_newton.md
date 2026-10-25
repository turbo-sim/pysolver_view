# Newton–Raphson and Gauss–Newton with optional backtracking

These development solvers separate the choice of direction from the decision
to accept a trial. Both use backtracking by default. Setting
`NewtonOptions(backtracking=None)` selects full, undamped steps.
They use public numerical kernels, not MINPACK or an optimization solver.

## Names and scope

| Canonical method | Alias | Display name |
| --- | --- | --- |
| `newton_raphson` | `newton` | Newton–Raphson |
| `gauss_newton` | `gn` | Gauss–Newton |
| `levenberg_marquardt` | `lm` | Levenberg–Marquardt |
| `powell_hybrid` | `hybr` | Powell hybrid |

Aliases select the same implementation and defaults. They do not select a
different backend. Native SciPy/MINPACK is called separately by the comparison
scripts. There is no separate damped-Newton method name: damping is a policy
of the Newton–Raphson method.

Newton–Raphson requires a square, numerically nonsingular Jacobian.
Gauss–Newton also handles overdetermined and numerically rank-deficient models.
Both still solve a root-finding problem: a stationary point with residual norm
above the target is a failure, even if it is a least-squares minimum.

## Linear directions

For residual $r=F(x)$ and Jacobian $J$, choose fixed positive variable
distances $s$ and write

$$
S=\operatorname{diag}(s),\qquad p=Sz,\qquad A=JS.
$$

The application normalizes its residual equations separately. Scaling a
temperature variable does not normalize an energy residual.

Newton–Raphson solves

$$
Az=-r.
$$

Gauss–Newton solves the local least-squares problem

$$
\min_z\frac12\|r+Az\|_2^2.
$$

When $A$ is square and nonsingular these directions agree in exact arithmetic.
Consequently the two methods can have matching convergence curves on square
benchmarks. Their distinction becomes important for rectangular systems and
rank deficiency; matching square-system curves are not evidence of a missing
method.

The implementation uses guarded pivoted QR and a triangular solve when
appropriate. An SVD provides numerical-rank checks and a minimum-norm
Gauss–Newton direction otherwise. Strict Newton does not silently substitute a
truncated pseudoinverse for a singular linear solve. Neither path explicitly
forms or inverts $A^TA$.

A fresh Jacobian is obtained after each committed move. Rejected trials at one
base reuse the direction and factorization. These two methods do not use the
Broyden updates employed by Powell hybrid.

### Default Jacobian equilibration

`NewtonOptions(jacobian_scaling=True)` is the default. The fixed caller scales
$s$ still define physical step limits and derivative-probe distances. A
separate positive diagonal matrix $C$ balances columns of $A=JS$ for numerical
linear algebra: solve with $AC$ for a direction $v$, then recover $z=Cv$.
The factors use a fixed initial median reference and running maximum column
norms from fresh Jacobians. See the
[shared scaling policy](07_usage_and_controls.md#jacobian-scaling-and-physical-limits)
for zero-column handling and finite-precision limits.

Because $C$ is invertible, a unique square Newton solution satisfies
$ACv=-r$ if and only if $Az=-r$. Full-column-rank Gauss-Newton likewise has
the same unique physical least-squares direction in exact arithmetic. The
benefit is numerical column balance; roundoff and rank decisions may change.
For rank-deficient Gauss-Newton, minimizing $\|v\|_2$ instead of $\|z\|_2$
can select a different minimum-norm solution. This distinction is deliberate
and does not imply equivalence of deficient-rank trajectories.

Line-search slopes, step limits, and stagnation checks use the converted
physical direction $z$ and the original $A$. This transformation does not
add damping to full Newton or change the meaning of `backtracking=None`.
Use `NewtonOptions(jacobian_scaling=False)` or the public
`options={"jacobian_scaling": False}` to disable it. The switch also applies
to custom LM/hybrid through `StepControlOptions`, but does not affect SciPy.
No extra residual or Jacobian calls are made to compute the factors.

Gauss–Newton's `gtol` guard checks whether the modeled residual change is
negligible: $\|Az\|_2\leq\mathrm{gtol}\min(1,\|r\|_2)$ above the root target.
This avoids declaring stationarity solely because Jacobian units make $A^Tr$
small. Newton–Raphson has no gradient-based early exit before its linear solve.

## Backtracking acceptance

Define

$$
\phi(x)=\frac12\|F(x)\|_2^2,\qquad d=r^TAz.
$$

An exact Newton direction satisfies $d=-\|r\|_2^2<0$ away from a root.
An exact Gauss–Newton direction is descending when it has a nonzero modeled
residual reduction. Numerical descent must still be checked.

If a scaled step limit $\Delta$ is enabled, the first fraction is

$$
\alpha_0=\min(1,\Delta/\|z\|_2).
$$

Without that envelope the search starts at $\alpha_0=1$. Each trial has
$x_{\mathrm t}=x+\alpha Sz$. A valid accepted trial must satisfy sufficient
decrease:

$$
\phi(x_{\mathrm t})\leq\phi(x)+c_1\alpha d.
$$

The default Armijo parameter is $c_1=10^{-4}$. Rejected or recognized undefined
trials reduce the fraction by the default factor $\beta=0.5$. The base point
and Jacobian remain fixed during this search. Unexpected programming errors
propagate instead of being converted to recovery events.

In finite precision, use the actual rounded displacement for the model
prediction and cap audit. Require positive actual reduction as well as the
Armijo inequality. An unevaluable initial point or derivative probe terminates;
trial backtracking does not imply derivative-probe recovery.

## Initial limit and gradual recovery

The envelope is an additional control beyond classical backtracking:

1. Start with `initial_limit` and never exceed `max_limit`.
2. If a search backtracks, carry forward at most the accepted scaled distance.
3. If the initially permitted trial succeeds and uses at least 80% of the
   available limit, grow it by `growth_factor` up to the persistent maximum.
4. Otherwise retain the current limit.

The default limits are 1 initially, 100 maximum, and growth factor 2.
The conservative comparison uses 0.1 initially, 1 maximum, and growth factor
1.5. Those distances are meaningful only together with `x_scale`.

`BacktrackingOptions(step_limits=None)` retains classical backtracking but
removes the envelope. It can still shorten a step. A zero retry budget permits
one trial which must pass acceptance; it is not an undamped solver.

## Undamped execution

`backtracking=None` disables both the search and its envelope. The fraction
is one and every valid finite trial is committed, including one that increases
the residual norm. For example, Newton–Raphson on the Rosenbrock equations can
first move from $(-1.2,1)$ to approximately $(1,-3.84)$, increasing the residual,
before reaching the root.

An undefined full trial stops at the last valid committed state. There is no
hidden shorter retry. Evaluation budgets, finite-coordinate checks, and
residual-based termination remain active. Small steps alone are not success.

Use `None` rather than infinity to request this policy. Infinite displacement
or retry limits would not mathematically remove the acceptance test.

## Using the development API

```python
from pysolver_view import (
    BacktrackingOptions,
    NewtonOptions,
    StepLimitOptions,
    solve_root,
)
from development.root_step_control.src.problems import ExtendedRosenbrockProblem

problem = ExtendedRosenbrockProblem(n=10)
options = NewtonOptions(
    x_scale=problem.x_scale,
    jacobian_scaling=True,
    backtracking=BacktrackingOptions(
        step_limits=StepLimitOptions(
            initial_limit=0.1, max_limit=1.0, growth_factor=1.5
        )
    ),
)
result = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="newton_raphson",
    options=options,
)
```

Use `method="gauss_newton"` with the same options for the least-squares
direction. For a full-step comparison, pass
`NewtonOptions(x_scale=problem.x_scale, backtracking=None)`.
The result exposes `jacobian_scaling`, the final relative factors in
`jacobian_scale`, and `jacobian_scale_updates`. Reported `scaled_step_norm`
continues to mean the step size measured with the fixed physical `x_scale`.

## Interpreting the comparisons

The [custom convergence report](../results_custom_convergence/REPORT.md)
compares all four methods, including separate full-step Newton-family cases.
The [native comparison](../results_convergence/REPORT.md) compares the custom
Levenberg–Marquardt/Powell hybrid implementations with SciPy/MINPACK.
The [timing report](../results_time_perf/RESULTS.md) separates residual,
Jacobian, selected numerical kernels, other solver overhead, and total time.

Custom accepted histories and native residual-evaluation traces are different
observations. A native consecutive evaluation displacement is not an accepted
step or a measurement of its internal trust radius. Failed solves and
unsupported dimensions remain visible rather than being treated as successful
timings.

See [LM/hybrid mathematics](05_lm_and_hybrid_mathematics.md) for the
trust-region methods and [usage and histories](07_usage_and_controls.md) for
shared evaluation, callback, and reporting conventions.
