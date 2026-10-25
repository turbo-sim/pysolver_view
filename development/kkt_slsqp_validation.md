# Shared KKT assessment and controlled SLSQP

Implemented on 25 October 2026. Turbodash's files were copied, not changed or
deleted. Updating its imports and dependency is deferred until the user releases
and adopts the package. No version bump or automatic commit is part of this work.

## Entry points and organization

- `pysolver_view/diagnostics/kkt.py`: numerical assessment, multiplier estimation,
  and independent reconstruction from supplied multipliers. No model evaluations,
  histories, callbacks, or application dependencies.
- `pysolver_view/solvers/slsqp.py`: the copied generic controlled driver and its
  domain-failure policy. `_slsqp_backend.py` isolates the compiled SLSQP interface;
  `SCIPY_LICENSE.txt` is explicitly included in source/wheel configuration.
- `pysolver_view/custom_optimization.py`: inequality sign conversion and solver
  option mapping. The facade owns model and derivative accounting.
- `OptimizationSolver`: existing public problem, scaling, result, plot, and
  report interface, now also supporting `library="pysolver", method="slsqp"`.

The custom bridge requires **SciPy 1.17.1**. Its runtime guard is local to the
custom solver; the native backend and package imports keep their existing
dependency range. Use a Python version supported by that SciPy release and
install `scipy==1.17.1` in the environment used to run the custom driver.

```python
solver = psv.OptimizationSolver(
    problem,
    library="pysolver",
    method="slsqp",
    max_iterations=200,
    tolerance=1e-9,
    update_on="iteration",
    extra_options={
        "first_step_fraction": 0.25,
        "max_step_fraction": 1.0,
        "step_growth_factor": 1.5,
        "domain_backtracking": True,
        "feasibility_tol": 1e-7,
        "optimality_tol": 1e-6,
    },
)
x = solver.solve(x0)
solver.print_optimization_report(
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
separate roadmap item. Finite physical bounds and meaningful scaling remain the
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

The low-level copied driver retains its nonnegative-inequality convention.
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
with the following deliberate corrections:

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
The [timing demo](../demos/demo_kkt_conditions/run_kkt_timing.py) compares a constrained
Rosenbrock problem requiring 21 iterations with checks enabled/disabled and separates assessor time from
total solve time. Solve-time `kkt_time` and `kkt_evaluations` are recorded.

Custom callbacks receive initial/accepted coordinates and a callback count,
regardless of the selected function/gradient/iteration observation stream.
Returning `False` or raising `StopIteration` cancels with a structured result
at the accepted point. Arrays and records are detached copies. Native SciPy
SLSQP callbacks still indicate major iterations without guaranteed acceptance.

## Named example and validation

Run from the repository root:

```powershell
python demos/demo_kkt_conditions/demo_named_kkt.py
```

The [demo guide](../demos/demo_kkt_conditions/README.md) explains the known convex
solution, equality/inequality/bound multipliers, feasible-but-nonoptimal points,
and infeasible points. Problem classes already support `variable_names` and
`constraint_data`; explicit `pack`/`unpack` helpers map dictionaries to numerical
arrays in a stable order. Named metadata must preserve the numerical order within
each constraint type. Solvers themselves continue operating on arrays.

**Fresh result:** 279 tests passed, with 10 Hessian tests deselected. A headless
rerun of 15 in-scope demo scripts completed 43 successful solves; the mixed
comparison excluded only its IPOPT entry in memory. Generated figures/logs were
kept outside the repository. Interactive windows were not tested.

```powershell
python -m pytest tests/test_kkt.py tests/test_kkt_reporting.py tests/test_custom_slsqp.py tests/test_custom_slsqp_backend.py tests/test_custom_slsqp_kkt_tolerances.py tests/test_custom_slsqp_integration.py tests/test_named_kkt_demo.py tests/test_result_history_optimization.py tests/test_result_history_root.py tests/test_optimization.py tests/test_nonlinear_system.py tests/test_differentiation.py -k "not hessian" -q -p no:cacheprovider
```

Validation covers analytical multidimensional KKT systems, fixed/opposing bounds,
redundant constraints, supplied multipliers, scaling, failed fits, named reports,
probe accounting, recovery, native-merit consistency, callback cancellation,
strict versus native success, and the local SciPy version guard. The original
history/root/first-derivative regression suite is also retained. PyGMO, IPOPT,
and SNOPT execution remains excluded as agreed.

An installation wheel was not built: this environment lacks `poetry-core`, and
the configured package source could not supply it. Wheel installation and
license-presence verification in the built artifact remain release checks.

## Combined validation before topic commits

On 25 October 2026, the package tests, SLSQP study tests and root development
tests passed together: **387 passed, 10 Hessian cases deselected**. This includes
the optional-KKT regressions and all 94 root development checks. PyGMO, IPOPT
and SNOPT execution remains outside the agreed validation scope. Installed-wheel
validation and Turbodash consumer migration remain pending.

```powershell
python -m pytest tests development/slsqp_controls/tests development/root_step_control/tests -k "not hessian" -q -p no:cacheprovider
```
