# PySolverView feature review and recommended direction

**Implementation update (25 October 2026):** Result/history repairs, the shared
KKT checker, and the custom SLSQP backend are implemented. See
[KKT/SLSQP usage and validation](kkt_slsqp_validation.md). The review below records
the original findings and design; Turbodash imports/dependencies and deletion of
its copies remain deferred until the user releases and adopts pysolver.

Reviewed 25 October 2026. This note evaluates the current local implementation,
its original purpose, and changes worth making beyond new solvers and KKT.
It proposes work; it does not implement it.

Related notes: [refactor plan](proposed_refactor.md) and
[KKT comparison and design](kkt_comparison_and_design.md).

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

## Numerical behavior policy

The refactor may change convergence behavior, default tuning, option mappings,
step sequences, and evaluation counts. Exact reproduction of current
PySolverView runs is not a goal. Keep changes that produce verified convergence
to the intended problem targets and correct diagnostics, and investigate lost
convergence or material performance regressions.

Use historical traces for explanation and comparison, not as exact acceptance
snapshots. Public API compatibility is a separate concern: familiar calls can
remain supported without guaranteeing identical convergence histories.
Numerical observation must still be trustworthy: aliases for one method and
enabling a plot must not secretly select a different numerical policy.

## Overall judgment

The original idea is good and remains the package's strongest justification:
define a numerical problem once, run SLSQP, IPOPT, SNOPT, or another supported
method, and inspect their progress through consistent records and plots.
Expensive thermodynamic models especially benefit from shared evaluation,
derivative, and diagnostic handling.

The weakness is not that this functionality is meaningless. It is that some
current abstractions hide differences between algorithms or let presentation
change stored data. Reliable observations are more important than exposing a
large list of solver names.

I recommend keeping convergence monitoring as a first-class purpose, even
after adding custom methods. The library should provide a consistent view
of different solvers while identifying what each backend can actually report.
The custom controls are an additional capability, not a reason to discard
the original monitoring infrastructure.

## Feature-by-feature assessment

| Existing feature | Judgment | Recommendation |
| --- | --- | --- |
| Common optimization problem interface | Important; retain. | Keep physical model definition separate from backend adaptation and per-run state. |
| Common root problem interface | Important; retain. | Share recording/result concepts with optimization, but retain distinct residual and acceptance semantics. |
| SciPy and PyGMO dispatch | Central to the original purpose. | Preserve SLSQP/IPOPT/SNOPT comparison and make supported capabilities explicit. |
| Combined objective/constraint evaluation | Particularly valuable for expensive models. | Preserve one model evaluation producing all responses and their ordering. |
| Evaluation caching | Valuable, but current approximate point matching is risky. | Use exact solver-point identity by default; do not merge small steps or derivative probes. Make caching optional for stateful/stochastic models. |
| Analytic and numerical derivatives | Important. | Assign one owner, validate shapes, respect bounds/domain rules, and count probe evaluations accurately. |
| Central differences, complex step, JAX helpers | Useful optional tools. | Retain only with documented prerequisites; do not imply thermodynamic property calls necessarily support complex values or tracing. |
| Hessian helpers | Applicable to IPOPT and other second-order methods; optional and noncritical for pysolver. | Retain as optional utilities. No validation of old helpers, new Hessian tests, or integration work is required for this migration. |
| Variable normalization | Important for conditioning. | Separate coordinate scaling from bounds; support explicit finite scales and correct chain rules. |
| `problem_scale` as a way to control steps | Indirect and potentially misleading. | Keep scaling for conditioning; use explicit step controls where available. Do not promise that increasing it reliably reduces physical steps for every method. |
| Function/gradient counters | Important. | Define actual model calls, derivative calls/probes, diagnostic calls, and cache hits separately. |
| `update_on="function"/"gradient"` | Useful as a choice of observation stream, misleading as a universal iteration definition. | Retain compatibility but label the stream correctly and add accepted-iteration records only where known. |
| Step norm | Useful for stagnation and movement diagnostics. | Specify coordinates and event type. Difference between consecutive evaluations is not necessarily an accepted solver step. |
| Live plotting | Useful but optional. | Make it a throttled observer, independent of solver correctness and acceptance. |
| Post-run plots and figure export | Important to the core purpose. | Retain, add comparison helpers, and ensure plotting never mutates history. |
| `plot_improvement_only` | Useful derived display; current implementation is problematic. | Compute a separate best-so-far series. For constrained problems, distinguish best feasible objective from all evaluated objectives. |
| Objective and constraint reports | Important for engineering interpretation. | Build from point-consistent snapshots and include bound violations. |
| KKT and multiplier reports | Important, requiring repair. | Consolidate as described in the separate KKT note. |
| Callbacks | Important for application integration. | Specify event type, coordinates, cancellation, exception handling, and mutation protection. |
| Generic `tolerance` | Convenient shorthand, not a common accuracy guarantee. | Keep explicit backend mappings and expose feasibility, optimality, and root-residual targets separately. |
| Generic `max_iterations` | Convenient but currently ambiguous across methods. | Preserve backend-specific budgets and report their actual meaning. |
| Backend-specific `extra_options` | Essential escape hatch. | Retain, validate conflicts, and store effective options with the result. |
| Tuned IPOPT/SNOPT defaults | Potentially useful for particular applications, problematic as universal defaults. | Put application-tuned settings in named opt-in profiles; document numerical implications. |
| Many advertised SciPy methods | Useful only when adapter support is complete. | Add capability tests; do not treat membership in a method list as evidence of working support. |
| Population algorithms | Potentially useful, currently incomplete result semantics. | Either complete their result/history contract or explicitly label them experimental. |
| Console and file logging | Useful. | Separate recording from rendering and use per-run logger settings. |
| General plotting/style helpers | Convenient, peripheral. | Retain as optional utilities; avoid turning the numerical core into a general graphics framework. |
| Automatic import-time process configuration | Problematic. | Stop changing exception hooks, JAX platform/precision, and logging policy merely because the package is imported. |
| Built-in example problems and tests | Important. | Keep portable examples and strengthen adapter/history/scaling tests, not only final-solution tests. |

## Concrete findings beyond KKT

### Plotting currently modifies the observations

In [optimization.py](../pysolver_view/optimization.py),
`_plot_convergence_callback` assigns the stored objective list directly to a
local variable and overwrites entries when `plot_improvement_only=True`.

A read-only in-memory reproduction during this review gave:

```text
Stored objective history before plotting: [3.0, 5.0, 2.0]
Stored objective history after plotting:  [3.0, 3.0, 2.0]
```

This is a correctness bug in a convergence-viewing package. Preserve raw
history and compute the cumulative minimum into a separate array. Objective
increases can be legitimate during constrained optimization; hiding them by
default would remove useful information even if the storage bug were fixed.

For constrained comparisons, a very low objective at an infeasible trial must
not become the reported best feasible solution.

### Final solutions depend on reporting state

Optimization wrappers return only `(success, message)`, and
`OptimizationSolver.solve` retrieves the final coordinates from reporting/cache
state. [NonlinearSystemSolver](../pysolver_view/nonlinear_system.py) likewise
uses `self.x_last` instead of the native returned coordinates.

The backend result must be authoritative. A last evaluation can be a trial,
and a last report can be an earlier point. This should be fixed even if no
custom solver is ever added.

### A common axis does not imply common iterations

The optimization plot says “Number of iterations” but uses function or
gradient counts according to `update_on`. These are different events and can
differ substantially across SLSQP, IPOPT, and SNOPT.

Keep function-evaluation histories: they are often the most portable
observations available. Label them as evaluations, and do not infer accepted
iterations merely from a derivative request.

Native SciPy LM/hybrid do not provide iteration callbacks through `root`.
Their evaluation histories remain useful, but must not be relabeled accepted
iterations.
[SciPy root documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)

### Option defaults obscure what accuracy was requested

In [optimization_wrappers.py](../pysolver_view/optimization_wrappers.py),
the IPOPT wrapper sets `dual_inf_tol=1e6`, `compl_inf_tol=1e6`, and
`tol=1e-3`, while the generic tolerance maps to constraint violation.
The SNOPT wrapper sets `Major optimality tolerance=1.0` while mapping the
generic tolerance to feasibility.

These are concrete configured choices, not evidence that IPOPT or SNOPT
always returns inaccurate solutions. IPOPT has multiple interacting
termination tests. However, users should not interpret the common tolerance
as a uniform first-order accuracy target. IPOPT distinguishes overall,
constraint, dual, and complementarity thresholds.
[IPOPT options](https://coin-or.github.io/Ipopt/OPTIONS.html)

Replace broad comments such as “will not converge if tolerance is too tight”
with scoped observations about tested problems. Provide transparent defaults,
named application profiles, and a saved record of all effective options.

### Optional backend availability is unnecessarily coupled

The wrappers import `pygmo` and `pygmo_plugins_nonfree` in one `try` block.
Missing the nonfree plugin makes the combined availability flag false even
when ordinary PyGMO is installed.

Detect those independently. Only methods requiring the plugin should require
it. Similarly, the private SLSQP version restriction should affect that method,
not importing the entire package or using native SciPy.

### Declared methods exceed the adapter's uniform contract

The SciPy adapter supplies gradients, bounds, and dictionary constraints
generically, but does not supply Hessians/Hessian products. Its method list
includes methods needing additional information. Constraint/bound support
also differs by method.

Validate method capabilities before model evaluation. For in-scope methods,
implement required inputs or give a clear unsupported-combination error.
Completing or testing Hessian-based paths is not a migration requirement.
Hessian helpers can also support IPOPT when adapted to its derivative interface;
the current IPOPT wrapper selects a limited-memory Hessian approximation.
SciPy documents these differences per method.
[SciPy minimize documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

For PyGMO population methods, current wrappers return empty strings for success
and message. A completed search budget is not proof of convergence, but it
should still have a meaningful termination reason, a champion point, and
properly typed status fields. SNOPT success currently depends on matching
message text; retain raw information and prefer structured codes when the
installed interface exposes them.

### Imports configure the host application

[__init__.py](../pysolver_view/__init__.py) installs an IPython exception hook
when available, sets `JAX_PLATFORM_NAME`, enables JAX 64-bit mode, and changes
the JAX logger level. These should be explicit user actions or opt-in helpers.
The caller may already have an application-wide policy.

Keep optional differentiation support, but load/configure it deliberately.
Wildcard public exports also make the supported API harder to identify;
move toward explicit exports without abruptly breaking old imports.

## What a consolidated convergence view should mean

Provide a common record format with honestly different available fields.

| Field group | Recommended meaning |
| --- | --- |
| Identity | Run ID, problem ID, backend/method, engine version, settings and scales. |
| Event | Model evaluation, derivative probe, accepted iteration, rejection, diagnostic, or termination. |
| Coordinates | Exact solver point and, when transformed, physical point. |
| Progress | Objective or residual norm, feasibility including bounds, optional KKT components. |
| Work | Actual model calls, Jacobian calls, probe calls, diagnostic calls, and elapsed time. |
| Method-specific data | Radius, fraction, recovery reason, native iteration number, or population generation when exposed. |
| Availability | Missing values when unknown; never fabricated zeros or guessed acceptance. |

Use model-call count and elapsed time as broadly available comparison axes.
Native accepted iterations are useful where exposed but need not be identical
work units across methods. Separate solver work from optional diagnostic and
rendering overhead, and preserve raw backend counters alongside common ones.

Compare the same physical problem, initial point, scaling, derivative policy,
and external acceptance target. Display failure and budget exhaustion rather
than dropping unsuccessful curves. Do not imply that equal option names
produce equivalent algorithmic stopping rules.

A useful first comparison helper would overlay objective and feasibility, with
optional KKT components, against evaluations or time for several saved results.
It should not require rerunning the models to redraw a figure.

## Recommended organization

Keep the public solver classes as convenient facades, but separate their
responsibilities internally:

```text
Problem definition
    ↓
Evaluation service: scaling, cache, derivatives, counters
    ↔ Backend adapter: algorithm-specific options and execution
    ↓
Point-consistent records and authoritative final result
    ↓
Optional diagnostics → history recorder → reports / plots / exports
```

The evaluator should not draw figures, and a plot should not evaluate physics.
Diagnostics should consume explicit snapshots. The high-level facade can
coordinate these parts without forcing all solvers through the same numerical
loop.

Add a backend capability registry: constraints, bounds, derivative requirements,
accepted callbacks, cancellation, domain recovery, multipliers, and step
controls. Validate capabilities within the agreed migration scope; existing
Hessian helpers and PyGMO/IPOPT/SNOPT execution are explicitly excluded.

Keep low-level custom solvers usable without the plotting facade. Sharing
result and event concepts is useful; requiring identical step options or
identical callbacks from every native backend is not.

Store per-run state separately from reusable problem definitions. In
particular, the current facade writes `problem_scale` onto the supplied problem.
Avoid shared mutable scaling when the same model object is reused. Define
whether solver instances can be reused and reset all run state if supported.
Stateful thermodynamic models also need explicit sequential/concurrent usage
rules; exact-point caching alone cannot make them thread-safe.

## Priorities beyond new solvers and KKT

| Priority | Recommended work | Why |
| --- | --- | --- |
| First: correctness | Authoritative final results; immutable raw history; point-consistent records; nonrecursive reporting. | These determine whether users can trust the package's main output. |
| First: honest comparisons | Correct event/axis labels; explicit counters and budgets; separate native status from external acceptance. | Consolidation should make comparisons clearer, not hide differences. |
| Next: model safety | Exact caching, one derivative owner, finite scaling, bound-aware probes, explicit recognized-domain errors. | Prevents infrastructure from breaking fragile or expensive evaluations. |
| Next: backend contracts | Capability validation, independent optional dependencies, documented option mappings, completed population statuses. | Reduces surprises and unsupported advertised functionality. |
| Next: usability | Serializable result/history schema, saved effective options, comparison plots, throttled live updates. | Directly advances the original convergence-viewing goal. |
| Later: focused expansion | Optional derivative checks, sparse adapters where justified, continuation/warm-start interfaces where actually supported. | Useful additions once the core observations and results are reliable. |

Keep the scope bounded. A package rename, generic plugin framework, new
dashboard framework, automatic solver-selection system, and replacement
implementations of IPOPT/SNOPT are not prerequisites. Additional solver names
are lower priority than making the shared observations correct and reproducible.
Validate that infrastructure with SciPy SLSQP; continued IPOPT/SNOPT wrapper
operation is assumed rather than tested during this migration.

## Verification and evidence limits

This review inspected the current problem interfaces, wrappers, numerical
differentiation, reports, plotting code, imports, and existing tests.
The objective-history mutation was reproduced in memory without solving or
editing a problem. Earlier confirmed KKT/history defects and test outcomes
are documented in the linked KKT note.

The option/default, dependency-coupling, Hessian-wiring, and population-status
findings come from source inspection. They are not claims that every listed
method or every IPOPT/SNOPT installation was executed during this review.

Implementation regressions should prove that plotting/reporting does not
change solutions or raw histories; final rejected trials do not replace the
solution; cache-sensitive short steps stay distinct; counters match known
model calls; unsupported capabilities fail clearly; and saved records can be
replotted without model access. Test equivalent physical formulations through
SciPy SLSQP and the custom driver, plus import isolation without optional
dependencies. Do not install or exercise PyGMO, IPOPT, or SNOPT for these checks.

The recommendation is to strengthen the original purpose rather than replace
it: a reliable common view of solver behavior, with custom controlled methods
available through the same user-facing infrastructure.
