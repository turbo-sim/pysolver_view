# Packaged root solvers and common convergence histories

The four custom methods are available through `NonlinearSystemSolver` with
`library="pysolver"`. Numerical code lives in
[`src/pysolver_view/roots/solvers/`](../../src/pysolver_view/roots/solvers); the adapter
is [`backends/custom.py`](../../src/pysolver_view/roots/backends/custom.py).
Benchmark problems live in the installed `roots.problems` package. Convergence
sweeps, timing runners and generated results remain under `development/`.

## Public usage

```python
import numpy as np
import pysolver_view as psv

class Equations(psv.NonlinearSystemProblem):
    """A small square system with a positive root at (1, 2)."""

    def residual(self, x):
        """Return the two nonlinear equations."""
        return np.array([x[0] ** 2 - 1, x[1] ** 2 - 4])

    def gradient(self, x):
        """Return the analytic residual Jacobian."""
        return np.diag(2 * x)

solver = psv.NonlinearSystemSolver(
    Equations(),
    library="pysolver",
    method="powell_hybrid",
    tolerance=1e-8,
    max_iterations=500,
    options={
        "x_scale": [1.0, 2.0],
        "jacobian_scaling": True,
        "initial_radius": 0.1,
        "max_radius": 1.0,
        "growth_factor": 1.5,
    },
    update_on="iteration",
    print_convergence=False,
)
x = solver.solve([3.0, 4.0])
assert solver.success
print(solver.result.method, solver.result.nfev, solver.result.nit)
```

`levenberg_marquardt` uses the same radius fields. For `newton_raphson` or
`gauss_newton`, replace those fields with:

```python
options = {
    "x_scale": [1.0, 2.0],
    "backtracking": {
        "step_limits": {
            "initial_limit": 0.1,
            "max_limit": 1.0,
            "growth_factor": 1.5,
        },
    },
}
```

`backtracking=None` selects full valid Newton-family steps, even if the residual
increases. `backtracking={"step_limits": None}` retains Armijo backtracking but
removes the distance envelope. Neither configuration is a universal robustness
guarantee. Bounds and distances apply to scaled coordinates $z_i=p_i/s_i$;
choose scales from physical characteristic changes. Normalize residual equations
in the problem itself. Derivative probes have separate distance settings.

`jacobian_scaling=True` is the default for all four custom methods. It balances
the columns of `J * x_scale` before numerical linear algebra, independently of
the fixed physical `x_scale`. To turn it off, add `"jacobian_scaling": False`
to `options`; direct calls also accept `StepControlOptions(jacobian_scaling=False)`
or `NewtonOptions(jacobian_scaling=False)`. The option must be a boolean and
does not change native SciPy/MINPACK behavior.

Numerical factors use a fixed initial reference norm and running maximum column
norms from fresh Jacobians. Hybrid keeps them fixed during Broyden updates.
LM and hybrid can therefore choose different directions and convergence histories;
their actual physical steps still obey the configured initial and maximum radii.
Full-rank Newton/Gauss-Newton directions are unchanged in exact arithmetic,
although conditioning, roundoff, and numerical-rank decisions can differ.
Rank-deficient Gauss-Newton's minimum-norm choice can change. The
[trust-region mathematics](root_trust_regions.md#how-the-numerical-factors-are-chosen)
describes the coordinate transformation and physical safeguard.

Canonical names and aliases:

| Full method | Short alias | Shape/rank support |
| --- | --- | --- |
| `levenberg_marquardt` | `lm` | At least as many equations as variables; damped/rank-deficient least squares |
| `powell_hybrid` | `hybr` | Custom dogleg/Broyden variant also accepts overdetermined systems |
| `newton_raphson` | `newton` | Square, numerically full-rank Jacobian required |
| `gauss_newton` | `gn` | At least as many equations as variables; minimum-norm fallback |

Native SciPy remains the default library, with default method `hybr`. Its full
LM/Powell aliases select native MINPACK. Existing positional constructor arguments
and `solve()` returning a vector are preserved. The lower-level `psv.solve_root`
continues to default to `levenberg_marquardt`. Custom roots use public SciPy
linear algebra; they do not use `minimize`, a private MINPACK bridge, or the
SLSQP driver's pinned private backend.

## Options, budgets, and derivatives

Explicit custom dictionary options override constructor defaults. Otherwise
`tolerance` fills `residual_tol`, and `max_iterations` fills the trial-attempt
budget. A supplied `StepControlOptions` or `NewtonOptions` dataclass supplies all
settings. `result.effective_options` records the resolved values. Native option
mapping retains its existing behavior.

The custom numerical core owns numerical differentiation when `gradient` is
absent. `2-point`, `3-point`, and `cs` are supported; complex step requires a
model that preserves imaginary perturbations. The constructor's
`derivative_abs_step` selects physical probe distances. An explicit custom
`finite_diff_step` instead uses $h_i=\epsilon s_i$ unless an explicit absolute
step is also supplied. Dataclasses default to scaled forward differences.
Automatic Jacobian scaling does not change these derivative-probe distances
and does not request additional model or derivative evaluations.

- `nfev` counts every physical residual call, including failed calls and probes.
- `derivative_nfev` is the probe subset, not an extra cost to add to `nfev`.
- `njev` counts complete Jacobian requests; `failed_njev` marks failed requests.
- `nit` counts accepted moves. The initial observation is iteration zero.
- `ntrials` and `max_iterations` concern trial attempts, including rejections.
  `max_nfev` also charges derivative probes. Probes never advance iterations.

No reporting callback re-evaluates the model. Undefined derivative probes stop
with a diagnostic; recovery does not fabricate derivatives. Recognized
`psv.DomainError` is shared by custom roots and SLSQP. Trial recovery follows
the selected method's controls; an undefined initial residual returns failure
at the initial point with `fun=None`. Unrelated model exceptions propagate.

## Results, observations, and callbacks

`result` is authoritative and detached from `backend_result`. Custom success
requires the configured residual threshold. Stationarity with a nonzero residual
is not root success. Native success and the existing extra residual check remain
separate. Solver reuse clears run state.

Custom results report `jacobian_scaling`, the final numerical factors in
`jacobian_scale`, and the number of fresh nonzero Jacobians used to update
them in `jacobian_scale_updates`. The factors are relative to `x_scale`;
they are not new physical step limits. Trust-region trial records distinguish
`scaled_step_norm`, measured using fixed physical `x_scale`, from
`numerical_step_norm`, measured in the equilibrated coordinates. Only the
final factors are stored, so this vector does not reconstruct every earlier
numerical metric.

`update_on="iteration"` records initial and accepted custom points.
`"function"` includes valid rejected trials; `"gradient"` observes requested
Jacobians at their residual base. Numerical probes never become rows.
`iteration_history` separately retains accepted custom states regardless of
the selected observation stream. Native LM/hybrid do not expose iterations.

Custom `callback_func(x, callback_count)` runs at the initial state and accepted
moves, independently of `update_on`. It receives a detached vector; returning
exactly `False` or raising `StopIteration` requests cancellation at that state.
The native interface retains observation callbacks without a new cancellation
contract. Low-level `solve_root` callbacks instead receive a detached diagnostic
dictionary; returning exactly `False` stops them.

`history_level="full"` stores coordinates, `"summary"` stores scalar records,
and `"none"` disables stored histories. Plotting and reporting never change the
numerical solution. Printing/live plotting can still incur costs even with
history disabled; disable them explicitly when timing the numerical core.

## Defaults and direct calls

The frontend defaults to native SciPy `hybr`, `tolerance=1e-6` and
`max_iterations=100`. Direct `psv.solve_root` instead defaults to custom LM.
Direct `StepControlOptions`/`NewtonOptions` default to `residual_tol=1e-8`,
500 trial attempts, 5,000 residual calls, scaled finite-difference distance
`1e-7`, and full histories. An explicit dataclass carries all those settings;
it does not inherit missing values from the frontend constructor.

Trust-region defaults are `initial_radius=1`, `max_radius=100`, growth 2,
shrink 0.25 and acceptance ratio 0.1. Newton-family defaults enable Armijo
backtracking (`armijo_c1=1e-4`, contraction 0.5, at most 30 backtracks) and
physical step limits 1 initially, 100 maximum, growth 2. The canonical
[option definitions](../../src/pysolver_view/roots/solvers/options.py) are the
source of truth for all fields and validation.

```python
from pysolver_view import StepControlOptions, solve_root
from pysolver_view.roots.problems import ExtendedRosenbrockProblem

problem = ExtendedRosenbrockProblem(n=10)
result = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="lm",
    options=StepControlOptions(x_scale=problem.x_scale, residual_tol=1e-8),
)
assert result.success
```

The [four-solver demo](../../demos/demo_eqn_solving/demo_custom_root_solvers.py)
uses the public frontend. The [trust-region](root_trust_regions.md) and
[Newton-family](root_line_search.md) notes explain the algorithms. Shared
[result/accounting contracts](results_and_accounting.md) and the
[validation guide](validation.md) explain observations and reproducible checks.
