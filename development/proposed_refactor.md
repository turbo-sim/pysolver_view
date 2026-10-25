# Proposal: shared solver implementations in PySolverView

**Implementation update (25 October 2026):** Result/history repairs, the shared
KKT checker, and the custom SLSQP backend are implemented. See
[KKT/SLSQP usage and validation](kkt_slsqp_validation.md). The review below records
the original findings and design; Turbodash imports/dependencies and deletion of
its copies remain deferred until the user releases and adopts pysolver.

Status: implementation plan only. Reviewed against the local PySolverView and
Turbodash checkouts on 25 October 2026. No solver implementation, package move,
dependency change, or application migration is included in this proposal.

## Iteration and evaluation accounting

Finite-difference and complex-step model evaluations are **never solver
iterations**. Keep them in total model/function/residual calls and, optionally,
a separate derivative-probe count. A complete gradient/Jacobian request counts
as one derivative request, not one iteration and not one gradient request per
probe. Record incomplete derivative requests explicitly if they fail.

Only actual solver iteration events may advance an iteration counter or appear
as iteration-history rows. Derivative probes must not trigger iteration
callbacks, change accepted-iterate state, or consume an iteration budget; they
may consume the function-evaluation budget. If a backend does not expose its
iterations, leave that count unavailable and plot evaluation counts under their
own labels. Do not infer iterations from function or gradient calls.

Probe counts are a subset of model-call totals, not extra calls to add to those
totals. Trial/rejection counters remain separate; numerical derivative probes
are not line-search or trust-region trial attempts.

## Migration validation scope

SciPy SLSQP is the agreed proxy check for the existing solver infrastructure.
There is **no requirement to install, run, benchmark, or test the PyGMO backend,
IPOPT, or SNOPT** for this migration. Their wrappers remain supported, and we
will assume they continue to work if SciPy SLSQP works correctly through the
updated infrastructure. This is an accepted project assumption, not a claim
that those backends have been independently validated.

References below to backend, compatibility, consumer, or cross-backend checks
exclude PyGMO, IPOPT, and SNOPT. Their validation is not a completion or release
gate. Tests for custom SLSQP, root solvers, shared KKT mathematics, and the
Turbodash application remain in scope.

Hessian helpers are optional, noncritical functionality for pysolver and are
applicable to IPOPT as well as other second-order methods through suitable
backend interfaces. There is no requirement to check that the existing Hessian
helpers work correctly, add Hessian regressions, or extend their integration
for this migration. Their validation is not a completion or release gate.
Required first-derivative and first-order KKT checks remain in scope.

## Numerical compatibility policy

Changing the convergence behavior of current solvers is allowed. Defaults,
option mappings, step/recovery policies, iterates, and evaluation counts need
not reproduce historical PySolverView runs. Keep changes reviewable and
document effective settings, but do not retain an inferior numerical policy
solely to match an old trace.

Acceptance is based on independently verified convergence to the intended
root residual or optimization feasibility/optimality targets, with correct
diagnostics and respected controls/budgets. Do not weaken those targets to make
a changed run pass. Historical results are references, not exact replay
requirements. Investigate lost convergence and material performance regressions;
a different successful history is acceptable. Public API, callback, and
serialized-field compatibility are separate concerns.

## Recommendation

Yes. Move the reusable SLSQP driver, its SciPy bridge, and the generic KKT
assessment into PySolverView. Expose them through a new
`library="pysolver"` selection while preserving the existing SciPy and
PyGMO selections. Promote the custom LM and hybrid root implementations through
the same library name in a subsequent stage.

This gives PySolverView responsibility for numerical methods, evaluation
accounting, step controls, and solver diagnostics. Turbodash retains responsibility
for turbine physics, optimization configuration, physical interpretation,
application callbacks, and exports. The dependency should run in one direction:
Turbodash imports PySolverView; PySolverView never imports Turbodash.

The user-facing problem classes, plotting functions, reports, and
`solve(x0)` returning a solution vector can remain familiar. Achieving this
requires adapting their internals: current result handling and KKT reporting
have problems that a new dispatch entry alone would preserve.

Do not move both Turbodash directories wholesale. Most of its current
`development/optimization/` is a turbine validation study and should remain
beside the turbine code.

## What the review found

| Current component | Finding | Consequence |
| --- | --- | --- |
| [Optimization dispatch](../pysolver_view/optimization.py), `OPTIMIZATION_LIBRARIES` and `VALID_LIBRARIES_AND_METHODS` | Already separates library and method. | Add a PySolverView adapter; preserve existing selections. |
| [Optimization wrappers](../pysolver_view/optimization_wrappers.py) | Return only `(success, message)`; the SciPy wrapper discards the native result. | Return an internal structured result with the authoritative solution. |
| `OptimizationSolver.solve` | Selects the final point from `x_last_norm` or the evaluation cache. | A reporting or trial point must not determine the returned solution. |
| [NonlinearSystemSolver](../pysolver_view/nonlinear_system.py) | Calls SciPy directly and saves `self.x_last` rather than `solution.x`. | Add root dispatch and use the backend's final point. |
| [Turbodash SLSQP driver](../../turbodash/turbodash/optimization/slsqp.py) | Most numerical code already has no turbine or UI dependency; accepts a combined evaluation and optional Jacobian. | A useful reusable core already exists. |
| `normalize_solver_options` in that same file | Handles Turbodash YAML aliases, retired settings, and the turbine adapter's `[0,1]` coordinates. | Separate application configuration migration from generic solver option validation. |
| [Private SLSQP bridge](../../turbodash/turbodash/optimization/_slsqp_backend.py) | Requires exactly SciPy 1.17.1 and checks pending-trial workspace assumptions. | Preserve this guard and isolate it from other backends. |
| [Custom root implementation](root_step_control/src/solver.py) and [linear model](root_step_control/src/linear_model.py) | Development code, independent of MINPACK, with explicit accepted/trial histories and step controls. | Move numerical runtime code into the installable package before exposing it publicly. |
| [Turbodash KKT assessment](../../turbodash/turbodash/optimization/kkt.py) | Generic array-based assessment, including finite bounds and sign-constrained multiplier estimation. | Use it as the basis of one shared KKT service. |

The new library name identifies the owner of the driver and its controls.
Custom SLSQP still uses SciPy's compiled SLSQP algorithm through a private
reverse-communication interface. Custom LM and hybrid are separate Python
implementations using public numerical kernels. They are not MINPACK wrappers,
and none of the root methods should be routed through `scipy.optimize.minimize`.

## Move and keep boundaries

| Existing content | Proposed destination or treatment |
| --- | --- |
| `turbodash/optimization/slsqp.py` numerical driver, `SLSQPOptions`, `Evaluation` | PySolverView's packaged solver modules. Validate convergence and numerical contracts during extraction; exact historical trajectories are not required. |
| `turbodash/optimization/_slsqp_backend.py` | Private module adjacent to the packaged SLSQP driver. |
| `turbodash/optimization/SCIPY_LICENSE.txt` | Move with the derived bridge; include it in source distributions and wheels. |
| `turbodash/optimization/kkt.py` | Shared PySolverView diagnostics module, usable by every optimization backend. |
| Generic SLSQP and KKT tests | PySolverView tests, including recovery, multiplier signs, bounds, stopping, and unavailable diagnostics. |
| `normalize_solver_options` application compatibility logic | Turbodash configuration adapter. Generic options should not know Turbodash YAML conventions. |
| `turbodash/optimization/turbine.py` | Keep in Turbodash: response extraction, variable normalization, domain-error translation, physical reevaluation, plotting, and YAML exports. |
| Turbodash app jobs, presets, callbacks, cancellation, browser checks | Keep in Turbodash; update imports and validate behavior during migration. |
| `development/optimization/verify_kkt.py`, `kkt_checks.py`, `kkt_report.py` | Keep the turbine matrix and its independent physical audit in Turbodash. Extract only demonstrably generic helpers if useful. |
| `case_poor_start.yaml`, turbine demos, `historical_benchmark.json` | Keep with Turbodash. Historical results retain their original provenance. |
| Root `src/solver.py` and `src/linear_model.py` | Promote to PySolverView runtime modules after the shared contracts are ready. |
| Root problem definitions, `run_*.py` studies, notes, saved results | Keep under development; update imports to exercise the installed numerical implementation. |

The turbine audit must continue reconstructing quantities independently.
Replacing its independent checker with a call to the production KKT service
would remove much of its value.

One possible package structure is:

```text
pysolver_view/
    optimization.py
    optimization_wrappers.py
    nonlinear_system.py
    nonlinear_system_wrappers.py
    solvers/
        __init__.py
        common.py                 # Explicit domain error and common result/event types
        slsqp.py                  # Generic evaluation/recovery driver
        _slsqp_backend.py          # Validated private SciPy bridge
        SCIPY_LICENSE.txt
        root.py                   # Custom LM/hybrid dispatch and iteration driver
        linear_model.py
    diagnostics/
        __init__.py
        kkt.py
tests/
    ...                           # Numerical, adapter, and packaging regressions
development/
    root_step_control/
    slsqp_step_control/            # Only portable studies, if added
```

Use a separate `solvers/` package because `optimization.py` is already a public
module. Do not replace it with a directory as an incidental part of this work.
Portable studies should follow the existing `README.md`, `src/`, `tests/`,
`run_*.py`, and `results_<case>/` arrangement. Keep study-specific execution
code in its runner rather than making thin wrapper scripts.

## Proposed public selection and compatibility

The intended high-level interface is:

```python
# Proposed API: not yet implemented.
optimizer = OptimizationSolver(
    problem,
    library="pysolver",
    method="slsqp",
    extra_options={
        "first_step_fraction": 0.25,
        "max_step_fraction": 1.0,
        "step_growth_factor": 1.5,
        "feasibility_tol": 1e-6,
        "optimality_tol": 1e-4,
    },
)
x_opt = optimizer.solve(x0)

root_solver = NonlinearSystemSolver(
    root_problem,
    library="pysolver",
    method="hybr",
    options={"initial_radius": 0.1, "max_radius": 1.0},
)
x_root = root_solver.solve(x0)
```

Keep `library="scipy"` as the default. Add the root `library` parameter
without shifting existing positional arguments, preferably as a new
keyword-only argument at the end. Register capabilities by both problem
family and library: initially SLSQP for optimization, LM/hybrid for roots.
The [Newton extension plan](root_finder_extension.md) remains a separate
algorithm-development task and is not a prerequisite for this refactor.

Expose the low-level generic SLSQP entry point as well. Turbodash can first
use that entry point with its existing combined evaluator; it need not replace
its entire physical adapter in the same change. A later adapter to
`OptimizationProblem` can make turbine cases use `OptimizationSolver` directly.
Both paths must call the same numerical driver.

Record the driver library, underlying numerical engine, engine version, and
controls separately in results and exports. Updating Turbodash's current
`solver.library="scipy"` metadata should preserve the distinction between a
PySolverView driver and the SciPy SLSQP engine.

Preserve legacy Turbodash configuration interpretation at the Turbodash
boundary. In particular, its existing YAML `library: scipy` currently selects
the controlled driver. Do not reinterpret saved cases as native public SciPy
runs during migration. New explicit configuration can use the PySolverView name.

## Contracts to establish before integration

### Results and accepted points

Introduce an internal result contract containing the final `x`, values at that
point, backend status/message, native termination flag where available, counters,
and method-specific diagnostics. Preserve raw backend results separately.
Optimization values and root residual vectors have different meanings and
must remain clearly identified.

Update every existing wrapper to return its real result or champion point.
PyGMO wrappers must explicitly return the evolved population's solution;
do not rely on whichever model evaluation happened last. Preserve its existing
deep-copy compatibility when adapting problem objects.

Keep `solve(x0)` returning the solution vector, with a richer `result`
attribute available for inspection. Failed recovery or cancellation must return
the last accepted valid point when one exists. If the initial point is invalid,
report that no valid accepted solution is available.

### Evaluation, derivatives, and reporting

Separate evaluation from progress recording. Build each history row from one
point-consistent snapshot containing coordinates, values, derivatives when
available, counters, and diagnostics. Reporting must not recursively trigger
another reporting event or choose the final solution.

Choose one owner of numerical derivatives for each execution path. The custom
drivers can retain their own differences initially; analytic derivatives pass
through adapters with the correct signs and coordinate transformation. Merely
passing `OptimizationSolver.gradient` or `NonlinearSystemSolver.gradient` as an
always-present Jacobian would hide numerical probes from the custom driver's
budgets and bypass its documented derivative policy.

Account separately for actual model calls, valid/invalid trials, derivative
probes, accepted iterations, and optional diagnostic reevaluations. Sharing a
combined objective/constraint evaluation should avoid repeated thermodynamic
calls. Cache keys must preserve distinct representable solver points; reporting
and diagnostics must not silently replace a requested point with a nearby one.

Retain each documented derivative scheme or reject an unsupported request
explicitly. Both custom drivers currently terminate on an invalid derivative
probe. Adding derivative recovery is a separate numerical change.

Unify the explicitly recognized domain-error type, including optional
structured metadata and compatible import aliases. The current root and SLSQP
exceptions have different base classes and payloads. Turbodash should continue
translating only recoverable model-domain failures at its boundary; arbitrary
`ValueError`, programming errors, and consistency failures must propagate.
Handle callback cancellation only around callback execution, so an exception
raised by the model is not mistaken for a user stop.

### Histories and callbacks

Keep accepted-iterate history distinct from evaluation and rejection histories.
Preserve existing `(x, count)` callbacks through an adapter and offer richer
record callbacks separately. Document coordinates and event types.

Custom SLSQP stops on callback `StopIteration`; custom roots currently stop on
a return value of exactly `False`. Translate these deliberately through the
high-level adapter, preserving low-level behavior. Pass detached snapshots so
callbacks cannot modify the running solver state. Preserve Turbodash's stop
handling and export/history alignment.

Native SciPy LM/hybrid do not provide iteration callbacks through `root`;
their evaluation traces cannot be presented as accepted-iterate histories.
Expose that capability difference instead of manufacturing events.
[SciPy root documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)

### Scaling, constraints, and controls

PySolverView's optimization convention is $g(x)\leq0$. The existing SLSQP
driver uses $c(x)\geq0$. Convert once at the adapter:

$$
c=-g,\qquad J_c=-J_g.
$$

Equality rows retain their order and sign. Keep explicit mappings for variable,
equality, inequality, and finite-bound indices.

For an affine coordinate transformation $x=b+Du$, use $J_u=J_xD$ exactly once.
Turbodash already normalizes its variables; passing its normalized evaluator
through another normalization would change the problem. KKT values,
derivatives, bounds, and reported point must all use the same coordinates and
objective/constraint scaling.

The existing bounds-based normalization also needs validation: infinite-width
bounds cannot define this affine scale, and a fixed physical variable must not
be represented by a fictitious free normalized interval. Initially use
unscaled coordinates or require explicit finite scales for unbounded variables,
and reject unsupported fixed-variable normalization. Eliminating fixed
variables requires reconstruction of the physical solution and consistent
derivatives and diagnostics.

Rescaling does not automatically preserve acceptance under unchanged absolute
tolerances. Positive constraint rescaling changes primal residual magnitudes,
even when transformed multipliers preserve stationarity and complementarity.
Variable and objective scaling also change numerical stationarity measures.
State the units and transform tolerances when claiming equivalent checks;
do not present normalized multipliers as physical sensitivities without the
corresponding objective and constraint scaling transformation.

SLSQP step fractions multiply the QP direction; they do not bound the absolute
physical displacement. Root radii bound the scaled norm
$\|p/x_{\mathrm{scale}}\|_2$. Keep these controls distinct. For fragile
thermodynamics, retain meaningful variable scaling, recognized domain-failure
recovery, and explicit reporting of actual physical movements. An absolute
SLSQP step cap would be a later feature requiring consistent bridge updates
and its own validation.

Map generic tolerances and budgets explicitly. For example, SLSQP
`tolerance` maps to `ftol`, not automatically to all KKT thresholds.
Custom root `max_iterations` currently counts attempted trials, while `nit`
counts accepted moves; native root limits and SLSQP major-iteration limits have
different meanings. Reject conflicting aliases and never silently ignore
unsupported controls.

## KKT assessment: repair and consolidate

KKT checking is largely independent of the solver that produced a point.
It can work with SciPy, PyGMO, and the custom driver provided the point, scaling,
constraint signs, and derivatives agree. However, the present PySolverView
checker should not be carried forward unchanged.

### Confirmed problems in the current PySolverView code

These findings concern existing behavior, not a hypothetical new backend:

| Finding | Evidence from inspection and small read-only reproductions |
| --- | --- |
| Bound-only feasibility can be incorrectly reported as satisfied. | `evaluate_kkt_conditions` adds bounds to its inequality vector but checks feasibility only when `self.N_ineq > 0`. With no user inequalities, bounds $[-1,1]^2$, $x=(2,0)$, and zero gradient, it reports zero inequality violation although the bound violation is one. |
| Dependent active constraints can cause a false dual-feasibility failure. | Multipliers come from unconstrained `np.linalg.lstsq`. At a fixed first coordinate with gradient $(1,0)$, the lower/upper multipliers become $(0.5,-0.5)$ although $(1,0)$ satisfies stationarity and nonnegativity. Opposing active inequalities show the same issue. |
| Diagnostics can reenter the progress recorder. | The recorder calls KKT, which requests `gradient`, which calls the recorder again when `update_on="gradient"`. An initial progress request produced two history rows. |
| A history row can combine values from different points. | Constraint violation is read before KKT refreshes the evaluation cache. For $g(x)=x_1\leq0$, a cache at $(2,0)$ followed by a gradient/report at $(-1,0)$ recorded violation two although the refreshed inequality was minus one. |

Other issues to address during integration: progress constraint violation omits
bounds; the checker assumes bounds can be unpacked; one tolerance controls both
active-set selection and assessment; progress uses `50*tolerance`; and the
reported stationarity measure alone does not establish all KKT conditions.
These are reasons to centralize diagnostics and make their inputs explicit.

### Shared mathematical convention

For the public PySolverView convention $h=0$, $g\leq0$, and $l\leq x\leq u$,
define

$$
L=f+\lambda^T h+\mu^T g-z_L^T(x-l)+z_U^T(x-u),
\qquad \mu,z_L,z_U\geq0.
$$

Check primal feasibility, stationarity
$\|\nabla f+J_h^T\lambda+J_g^T\mu-z_L+z_U\|_\infty$,
dual feasibility, and complementarity, including every finite bound.
Exclude absent infinite bounds before forming products.

Retain the Turbodash assessment's existing $c\geq0$ convention internally
during extraction. Its equality multiplier changes sign when reported in the
public convention above: $\lambda_{\mathrm{public}}=-\lambda_{\mathrm{core}}$.
With $c=-g$, the inequality multiplier retains its sign. Verify these mappings
with analytic examples rather than guessing from array positions.

Use the existing Turbodash linear minimax formulation as the shared estimator:
unrestricted equality multipliers, nonnegative inequality/bound multipliers,
and stationarity/complementarity residuals involving all constraints. This
avoids arbitrary active-set cutoffs and the unconstrained least-squares sign
problem. Recompute the residuals from the returned multipliers and retain
explicit failure/unavailable diagnostics if estimation fails.

This remains a numerical first-order assessment. Dependent constraints can
make multipliers nonunique, derivative errors limit accuracy, and a passing
assessment does not prove a minimum or global optimality. A failed check at a
degenerate constrained minimum also needs interpretation.

Keep native multipliers separate from independently estimated multipliers.
SciPy SLSQP reports multipliers from its QP approximation and omits bound
multipliers from that public result.
[SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-slsqp.html)

### Acceptance and cost policy

Expose native termination, feasibility, `kkt_passed`, diagnostic availability,
and final application acceptance separately. Preserve the controlled SLSQP
policy: success requires native success, feasibility, and the independent
first-order limits. A tiny imposed step must not turn into a false success.

For existing SciPy/PyGMO users, retain their success policy initially and expose
strict KKT acceptance as an explicit option. Changing all backends' success
semantics by default would be a separate compatibility change. Unavailable
derivatives or multipliers must never be reported as zero optimality.

Provide separate feasibility and optimality tolerances. Label norms and scaled
units, and retain the legacy KKT-report interface through a documented adapter.
The legacy stationarity 2-norm and the proposed infinity norm are different;
do not replace values under unchanged names without documenting the transition.
Likewise, preserve or explicitly migrate multiplier ordering and signed dual
metrics used by reports.

KKT estimation and fresh derivatives cost time. Support an explicit diagnostic
frequency, such as final-only or accepted-iterate reporting, and document any
changes to Turbodash's diagnostic schedule during extraction. Reuse derivatives at
the exact accepted point where appropriate. If independent fresh derivatives
are requested, account for their calls and time separately. Reporting must
not request gradients at every rejected thermodynamic trial.

Root methods require a separate residual-based acceptance policy, not an
optimization KKT gate. A stationary least-squares point with a nonzero residual
is not a solved root. Also reconcile the existing root wrapper's
`10*tolerance` acceptance check and its message with an explicitly documented
root target.

## Dependencies and packaging

PySolverView currently declares Python `>=3.10,<3.14` and SciPy `^1.15.2`.
Turbodash pins SciPy `1.17.1`, and the private bridge rejects every other version.
SciPy 1.17.1 requires Python 3.11 or newer.
[SciPy 1.17.1 package metadata](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/pyproject.toml)

Recommended initial policy:

1. Preserve broad PySolverView compatibility for existing backends.
2. Load/check the private SLSQP bridge only when that method is selected.
3. Document and test custom SLSQP on Python 3.11-3.13 with SciPy 1.17.1; provide
   a reproducible supported-environment constraint file.
4. Preserve Turbodash's exact SciPy pin during migration and make the required
   new PySolverView release an explicit runtime dependency of its optimization
   path. Decide whether that path remains optional from actual import behavior;
   do not rely on an indirect dependency through another package.
5. Raise a clear unsupported-environment error before model evaluations for
   custom SLSQP. Native backends and custom roots must still import and run
   where their own dependencies are supported.

Do not weaken the version check based on matching function signatures alone.
The bridge depends on workspace layout and state semantics. Supporting another
SciPy release requires reviewing the corresponding source and rerunning recovery
and numerical regressions.
[Validated SciPy wrapper source](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/_slsqp_py.py)

Check installed wheels in a clean environment, including the license file.
Runtime code must not import `development`, Turbodash, CoolProp, Dash, or an
uninstalled PyGMO backend. Thermodynamic benchmark dependencies remain optional.

## Implementation sequence

| Phase | Work | Completion gate |
| --- | --- | --- |
| 1. Capture contracts and regressions | Add focused cases for the confirmed KKT/reporting problems, final-result selection, sign/scaling conversions, and callbacks. Record baseline controlled SLSQP and turbine outputs. | Failures are reproducible; intended compatibility behavior is written down. |
| 2. Extract the generic numerical core | Move SLSQP, its private bridge/license, and the generic KKT service into PySolverView. Separate Turbodash configuration migration; validate convergence and documented controls rather than historical traces. | Generic SLSQP/KKT tests pass through packaged imports, including unsupported-version checks. |
| 3. Repair result and reporting infrastructure | Introduce explicit backend results, point-consistent evaluation snapshots, nonrecursive diagnostics, and accepted/evaluation history separation. Update native adapters. | Final solutions are independent of plotting, callbacks, cache state, and last trial. SciPy SLSQP and applicable custom-solver tests pass; PyGMO/IPOPT/SNOPT validation is excluded. |
| 4. Add the PySolverView optimization backend | Register `library="pysolver", method="slsqp"`; adapt combined evaluations, signs, bounds, derivatives, options, and callback records. Reuse the shared KKT service in legacy reports. | Direct generic driver and high-level adapter agree on numerical outcomes and acceptance. |
| 5. Migrate Turbodash | Point its physical adapter at the packaged driver; preserve configuration aliases, errors, result schema, callbacks, and exports. Add temporary compatibility re-exports where needed. | Turbine numerical tests, independent KKT matrix, and relevant app checks pass. |
| 6. Promote root methods | Move LM/hybrid runtime code into the package, add root dispatch and adapters, and update development imports. Keep Newton-family work separate. | Existing root controls, domain handling, counts, histories, and convergence benchmarks remain verified. |
| 7. Document and release | Document backend capabilities, tolerances, coordinates, diagnostics, dependencies, and migration examples; validate distributions and consumer installation. | Both packages work with the released dependency set before duplicate implementations are removed. |

Keep changes in reviewable stages and distinguish numerical policy changes from
code movement when evaluating results. Recovery and algorithm tuning may improve
during the refactor; matching the previous trajectory is not required.
Remove duplicate solver
implementations only after the consumer migration passes; temporary re-exports
must delegate to a single implementation.

## Validation plan

### Generic numerical and adapter checks

- Unconstrained, equality, inequality, mixed, and bound-constrained analytic
  optima; incorrect multiplier signs; redundant constraints; inactive
  constraints; and bounds without user inequalities.
- Unbounded and partially bounded variables. The current SLSQP driver rejects
  fixed bounds: initially reject them clearly through the adapter too, or add
  explicit fixed-variable elimination with reconstruction tests as separate
  work. The generic KKT service can still assess fixed variables.
- Positive constraint rescaling, objective scaling, and anisotropic variable
  scaling; compare physical solutions and correctly transformed multipliers.
- Native success with unacceptable feasibility or stationarity, unavailable
  derivatives, nonfinite data, iteration limits, and user cancellation.
- Small first steps, persistent fraction caps, recovery growth, invalid initial
  points, exhausted domain recovery, invalid derivative probes, and unexpected
  exceptions that must propagate.
- Native result differing from the final evaluated trial; reporting enabled or
  disabled; exact point/value alignment; no duplicate initial row; callback
  mutation protection; no diagnostic recursion.
- Analytic versus numerical Jacobians, exact model-call accounting, diagnostic
  call accounting, and stopping within evaluation budgets.
- Existing SciPy SLSQP selection and output interfaces. Do not run PyGMO,
  IPOPT, or SNOPT checks, even if installed; their continued operation is assumed
  under the agreed proxy-validation policy.

Move generic tests with the core and retain application tests with their
application. Adapt tests that monkeypatch module internals to the new ownership
boundary without weakening what they verify.

### Turbodash acceptance

Run its numerical optimizer/API tests, KKT reporting/tolerance/history tests,
domain/error tests, and the existing 52-case independent validation matrix.
Check the poor-start recovery cases as well as nominal designs. Validate
cancellation, progress display, result serialization, and YAML exports in the
optimization app.

Assess physical final designs, normalized residuals, KKT assessments,
accepted histories, rejection metadata, and meaningful call counts. Do not
require identical trajectories, counts, or floating-point solutions. Where
multiple roots or optima exist, use the case's physical admissibility and
solution-quality criteria rather than automatically accepting any stationary
point. Retain
unsuccessful outcomes: a correctly rejected optimization can still pass the
independent audit. Use fresh output directories after changing implementation
or settings; historical results are not a current acceptance oracle.

### Root and performance checks

Run the existing root unit tests and multidimensional convergence/step-control
studies against the packaged implementation. Include the 100-variable dense
and thermodynamic cases. Test that failures return the accepted point, that
probe counts remain within budget, and that nonzero stationary residuals fail
the root target.

Benchmark the direct packaged driver against the high-level adapter and native
SciPy with comparable numerical controls. Measure model/derivative time,
solver overhead, and KKT/reporting overhead separately. Record changes in
evaluation counts and numerical policy alongside timings, but do not require
the old evaluation sequence. For an adapter-only overhead comparison, use the
same current method/settings to isolate that cost. Do not describe total solver overhead as pure linear
algebra time.

## Evidence gathered for this proposal

This review inspected both codebases and their existing development notes.
Small read-only reproductions confirmed the four PySolverView KKT/history
issues listed above. The existing Turbodash `tests/test_slsqp.py` suite was run
with its project environment: **41 tests passed**.

The focused Turbodash `tests/test_kkt.py` and `tests/test_kkt_tolerances.py`
suites were also run: **69 tests passed**.

The full turbine acceptance matrix, application/browser checks, and root
benchmark suite were not rerun for this documentation task. They remain
implementation gates, not results already established for the proposed
integration. The PyGMO suite, including IPOPT and SNOPT, is excluded from
migration validation; its wrappers are assumed to work based on SciPy SLSQP.

The recommended first deliverable is therefore a packaged generic SLSQP/KKT
core plus the result/reporting fixes and a high-level PySolverView adapter.
Turbodash can then consume it without losing its physical diagnostics, and the
custom root methods can join the same infrastructure in a separate stage.
