# Packaged root solvers and common convergence histories

The four custom methods are available through `NonlinearSystemSolver` with
`library="pysolver"`. Numerical code lives in
[`pysolver_view/solvers/root/`](../../../pysolver_view/solvers/root/); the adapter
is [`custom_root.py`](../../../pysolver_view/custom_root.py). Benchmark problems,
convergence sweeps, timing runners, and their results remain under development.
The old development imports re-export the packaged functions for compatibility;
there is only one numerical implementation.

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
[scaling and controls note](07_usage_and_controls.md#jacobian-scaling-and-physical-limits)
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

## Example and validation

The single [four-solver demo](../../../demos/demo_eqn_solving/demo_custom_root_solvers.py)
defines its own two- and eight-variable problems and uses only the installed
package. Run:

```powershell
python demos/demo_eqn_solving/demo_custom_root_solvers.py
python demos/demo_eqn_solving/demo_custom_root_solvers.py --no-show
```

Solid Newton-Raphson and dashed Gauss-Newton traces expose overlap. For a square
nonsingular $J$, both solve $Jp=-F$, since the unique least-squares minimizer has
zero linearized residual. Equal scales, limits, and acceptance choices therefore
usually produce matching histories up to roundoff. Rank-deficient models,
stopping rules, and rectangular systems can differ. All eight demo solves reach
$\|F\|_2\leq10^{-8}$.

The full convergence and timing runners now exercise the public custom interface.
Their SciPy comparisons still call `scipy.optimize.root` directly, so custom
timing includes additional common-interface functionality. The separate
[`run_adapter_timing.py`](../run_adapter_timing.py) isolates that distinction
using direct and public calls to the same custom core on three 100-variable
workloads. [Timing protocol and findings](06_time_performance.md).

Integration regressions are in [`tests/test_custom_root.py`](../../../tests/test_custom_root.py).
The scaling-specific checks are in
[`tests/test_root_jacobian_scaling.py`](../../../tests/test_root_jacobian_scaling.py).
The default-scaling follow-up passes 455 package/root tests, excluding 10
noncritical Hessian checks. The 212-case convergence rerun retains 100/100
eligible backend roots, 98/101 eligible custom roots, and all 51/51 eligible
controlled roots. The remaining three eligible failures use undamped methods;
unsupported rectangular Newton configurations and inconsistent systems remain
explicit outcomes. This rerun did not repeat the historical timing studies.

The [radiative heat demo](../../../demos/demo_eqn_solving/README.md) tests 500
collocation nodes and 1000 variables. All six Newton/LM/hybrid configurations
converge with the new default, including controlled LM in 12 accepted iterations
and 13 residual calls. Numerical scaling repairs that LM case but does not
guarantee fewer calls for every method or problem.

The development numerical tests remain in [`tests/`](../tests/). Wheel building,
dependent-package migration, and difficult phase-boundary property models are
separate release/application checks.
