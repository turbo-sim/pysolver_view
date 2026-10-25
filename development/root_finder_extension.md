# Implementation plan: Newton and Gauss–Newton root solvers

**Status: implemented in development on 25 October 2026.** The two methods,
options, and aliases below are available in
[root_step_control/](root_step_control). Production-package integration remains
separate. This note retains the design and validation plan; current usage and
mathematics are in the [Newton-family note](root_step_control/notes/08_newton_and_gauss_newton.md).

Validation completed: 91 development tests; 100 native/custom LM/hybrid
convergence cases; 112 custom comparison configurations; and 11,760 timed
solves across 56 configurations. Failed and unsupported convergence cases are
retained in the [custom report](root_step_control/results_custom_convergence/REPORT.md);
timing outcomes and limitations are in the
[timing report](root_step_control/results_time_perf/RESULTS.md).

**Numerical compatibility policy:** the extension and shared-code refactor may
change convergence behavior of existing methods. Historical trajectories,
iteration counts, and residual-call counts need not match. Validate convergence
against the intended residual target, physical admissibility where relevant,
and documented step/budget contracts. Investigate lost convergence and material
performance regressions, not every difference in a successful history.
Alias equivalence, accurate counters, and independence from passive reporting
remain correctness requirements within the new implementation.

The intended methods are `newton_raphson` and `gauss_newton`. Both enable
backtracking by default; setting `backtracking=None` selects the corresponding
undamped, full-step method. Damped Newton is a step-policy configuration of
Newton–Raphson, not a third solver. This separates the linear direction from
step acceptance and avoids duplicating solver implementations.

They will share scaling, evaluation tracking, histories and root-success
semantics with the existing LM and Powell-style hybrid solvers. Descriptive
author-name aliases for those existing methods are proposed below.
The immediate application is multidimensional thermodynamic systems whose
property evaluations can fail after an excessive displacement.

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

## 1. Method definitions and intended behavior

Let $F:\mathbb{R}^n\to\mathbb{R}^m$ be the supplied residual vector and
$J=\partial F/\partial x$. Keep variable and residual scaling separate.
For fixed characteristic variable distances $s_i>0$, define

$$
S=\operatorname{diag}(s),\qquad p=Sz,\qquad A=JS,\qquad r=F(x).
$$

The application supplies any normalization of physical equations through
$F$ and its consistent Jacobian.

| Canonical method | Dimensions | Direction | Step acceptance |
| --- | --- | --- | --- |
| `newton_raphson` | Square systems, $m=n$, with a numerically nonsingular Jacobian | Solve $Az=-r$ | Backtracking with scaled limits by default; `backtracking=None` takes full valid steps, including residual increases. |
| `gauss_newton` | Square or overdetermined systems, $m\geq n$ | Minimize $\lVert r+Az\rVert_2$ | The same configurable backtracking policy; `backtracking=None` selects full-step Gauss–Newton. |

For a square nonsingular Jacobian, Newton and Gauss–Newton give the same
direction in exact arithmetic. Their implementations should therefore share
the corresponding factorization and solve logic. Gauss–Newton has additional
meaning for overdetermined or numerically rank-deficient models.

Full-step Newton–Raphson is a useful reference and local solver when a good
initial guess is available. Newton–Raphson with backtracking is the priority
for fragile thermodynamic models. Gauss–Newton adds the unregularized
least-squares direction, with or without line-search damping. Backtracking
scales a computed direction; it does not introduce LM regularization into
the linear model.

## 2. Existing code to reuse

| Current component | Reuse or required change |
| --- | --- |
| [src/solver.py](root_step_control/src/solver.py) | Preserve the public `solve_root` entry point and aliases. Share budgets and evaluation/history mechanics through the `solvers/` helpers; retain the `DomainError` re-export. |
| [src/solvers/linear_model.py](root_step_control/src/solvers/linear_model.py) | Reuse guarded QR and SVD factors and the existing `gauss_newton` direction. Add an explicit strict Newton direction/rank contract. |
| [src/problems/mathematical.py](root_step_control/src/problems/mathematical.py) | Reuse the standard mathematical systems and 100-variable cases. Add only problems needed to exercise missing behavior. |
| [src/problems/thermodynamics.py](root_step_control/src/problems/thermodynamics.py) | Reuse the 20- and 100-variable CoolProp duct models and their physical checks. |
| [tests/](root_step_control/tests) | Extend the existing behavioral, numerical and accounting checks. |
| [run_convergence_benchmark.py](root_step_control/run_convergence_benchmark.py) | Add eligible method/configuration cases and policy-aware audits. |
| [run_timing_benchmark.py](root_step_control/run_timing_benchmark.py) | Add explicit method registrations while preserving clean/instrumented measurement separation. |

The existing driver always constructs a trust-region step and requires
residual reduction for acceptance. Adding a name to its method selector would
therefore not produce full Newton. The new methods need an explicit full-step
or backtracking execution path.

## 3. API and options

Keep `solve_root(..., method=..., options=..., callback=...)` and retain
`method="lm"` and `method="hybr"` with their `StepControlOptions` interface.
Default tuning may change when documented and validated; existing trajectories
do not have to be reproduced.

### Method names and author-name aliases

Use the following explicit root-method registry:

| Canonical identifier | Additional accepted alias | Display name |
| --- | --- | --- |
| `newton_raphson` | `newton` | Newton–Raphson |
| `gauss_newton` | `gn` | Gauss–Newton |
| `lm` | `levenberg_marquardt` | Levenberg–Marquardt |
| `hybr` | `powell_hybrid` | Powell hybrid |

The correct spellings are **Levenberg–Marquardt** and **Powell hybrid**.
SciPy identifies `hybr` as a modification of the Powell hybrid method and
`lm` as Levenberg–Marquardt; MINPACK's HYBRD documentation uses the same
Powell attribution.
[SciPy root documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html),
[MINPACK HYBRD documentation](https://www.math.utah.edu/software/minpack/minpack/hybrd.html)

Keep `lm` and `hybr` as canonical identifiers for compatibility with existing
results and scripts. Their longer aliases call exactly the same implementation
with identical defaults, counters, and stopping behavior. An alias identifies
the method family; it does not make the custom implementation identical to
SciPy/MINPACK. Keep backend provenance separate.

Normalize supported string names once, before dispatch and option validation,
using an explicit mapping. Store the canonical identifier and, optionally,
the requested name in the result. Use readable display names in reports.
Do not introduce a separate `damped_newton` entry: damping is selected through
options. Avoid bare `powell` or `gauss` aliases because they are ambiguous
across numerical methods; `powell_hybrid` names the root-finding method clearly.

These are PySolverView API aliases, not new names accepted directly by
`scipy.optimize.root`. A future SciPy adapter may map the applicable aliases
to `lm`/`hybr`; it must reject unsupported Newton-family selections rather
than redirecting them to a different algorithm.

### Options and default resolution

Introduce one Newton-family options object for both additional methods,
rather than treating trust-region radius settings as line-search parameters.
A practical proposed split is:

- `NewtonOptions`: common scaling, residual tolerance, stagnation thresholds,
  budgets, finite-difference distance, history level and a `backtracking` field.
- `BacktrackingOptions`: Armijo constant, contraction factor, maximum number
  of backtracks and optional scaled step-limit settings.
- `StepLimitOptions`: initial limit, persistent maximum, growth factor and
  recovery carry-forward policy.

Use the same policy contract for both Newton-family methods:

| Configuration | Meaning | Acceptance |
| --- | --- | --- |
| `backtracking` omitted | Construct default `BacktrackingOptions`, including its default step limits. | Valid trial with sufficient merit reduction. |
| `backtracking=BacktrackingOptions(...)` | Use the explicitly configured search and optional limits. | Valid trial with sufficient merit reduction. |
| `backtracking=BacktrackingOptions(step_limits=None)` | Classical backtracking with no displacement envelope; start each search at fraction 1. | Valid trial with sufficient merit reduction; shortening remains possible. |
| `backtracking=None` | Full-step method: fraction 1, no line search and no displacement envelope. | Commit every valid finite trial, even if the residual increases. |

Prefer **`None`**, not infinity, to disable backtracking. An infinite
displacement limit only removes a cap; it does not disable the Armijo test.
An infinite retry count would mean unlimited retries, not an undamped method.
Do not overload a numerical option with either interpretation.

Use a dataclass default factory for `backtracking` so omission creates a fresh
`BacktrackingOptions`, whereas explicit `None` remains disabled. Likewise,
an omitted `step_limits` inside enabled backtracking constructs its default
`StepLimitOptions`; explicit `step_limits=None` disables only the envelope.
Parse nested dictionaries into the same validated option objects and preserve
these distinctions. `options=None` means default options and therefore
backtracking enabled; use `options={"backtracking": None}` to request full steps.

Require finite, nonnegative integer `max_backtracks`. Zero permits one initial
trial and no retries, but still requires Armijo acceptance; it is not full-step
mode. Enabled step limits require finite positive limits and a finite valid
growth factor. Reject infinity, NaN, and ambiguous disabling values explicitly.

A proposed protected call is:

```python
# Development API; production-package integration remains separate.
result = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="newton_raphson",
    options=NewtonOptions(
        x_scale=problem.x_scale,
        residual_tol=1e-8,
        backtracking=BacktrackingOptions(
            armijo_c1=1e-4,
            contraction_factor=0.5,
            max_backtracks=30,
            step_limits=StepLimitOptions(
                initial_limit=0.1,
                max_limit=1.0,
                growth_factor=1.5,
            ),
        ),
    ),
)
```

To select either undamped method:

```python
# Identical policy switch for both direction choices.
full_newton_options = NewtonOptions(
    x_scale=problem.x_scale,
    backtracking=None,
)
full_newton = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="newton_raphson",
    options=full_newton_options,
)
full_gauss_newton = solve_root(
    problem.residual,
    problem.x0,
    jac=problem.gradient,
    method="gauss_newton",
    options={"x_scale": problem.x_scale, "backtracking": None},
)
```

Use the current scaled-distance defaults of 1 initially, 100 maximum and
growth factor 2 as the proposed general protected defaults. The example uses
more conservative values; applications must choose limits from physical scales.
An explicit full-step request must not inherit these limits.

Validate option combinations before iteration. Step limits belong to enabled
backtracking; reject a conflicting request to impose them in full-step mode.
Do not silently ignore a requested cap, shorten an undamped step, or fall back
to LM. The Newton-family `backtracking` option does not alter LM/hybrid's
existing trust-region behavior; reject it for those methods.
A finite-cap fail-fast mode could be added later, with an explicit name and
documented semantics.

## 4. Linear algebra and direction construction

Use the existing fixed scaled coordinates throughout.

For Newton:

1. Require a square residual/Jacobian pair.
2. Compute a fresh Jacobian at each committed iterate.
3. Use the guarded QR solve when suitable.
4. When QR conditioning requires investigation, use a rank-revealing SVD and
   verify numerical full rank before computing the Newton direction.
5. Return a diagnostic on numerical singularity or an unusable linear solve.
   A truncated pseudoinverse direction belongs to Gauss–Newton, not strict
   Newton.

For Gauss–Newton:

1. Accept $m\geq n$.
2. Recompute derivatives after each committed iterate.
3. Use the existing QR path for healthy full-column-rank models.
4. Use the numerical-rank minimum-norm SVD solution otherwise.

Do not form or invert $A^TA$. Cache the factorization at an accepted base and
reuse it for every backtracking attempt. These methods do not use Broyden
updates; their `nupdates` count remains zero.

Keep gradient calculation separate from the direction solve where possible.
Full Newton does not need a least-squares stationarity check before solving.
A small $A^Tr$ can reflect scaling rather than a non-root stationary point.
Likewise, avoid making an unnecessary overflowing gradient product prevent
an otherwise usable Newton solve. For line-search directional derivatives,
$r^T(Az)$ avoids explicitly constructing $A^Tr$.

For Gauss–Newton, a fresh-model stationary point above the residual target
remains a failure diagnostic. Numerical rank and linear-solve residuals
should be available in result diagnostics.

## 5. Full-step execution path: `backtracking=None`

At each current iterate:

1. Check the residual target and remaining budgets.
2. Evaluate the fresh Jacobian and compute the strict Newton direction.
3. Form the physical trial `x_trial = x + S @ z` with step fraction 1.
4. Check finite coordinates and representability, then evaluate the residual.
5. Commit a valid finite trial even if its residual norm increases.
6. Record the actual rounded displacement and continue from the new point.

An undefined trial terminates at the last valid committed state, with the
failed evaluation recorded. Full-step Newton has no hidden shorter retry.
Unexpected application exceptions continue to propagate.

Retain the residual target, trial/evaluation budgets and clear stagnation
diagnostics. A sufficiently small step above the residual target is failure,
not success. Numerical singularity is also a diagnostic, not an instruction
to change algorithms without the caller's request.

Full-step Gauss–Newton follows the same commit policy, using its least-squares
direction and rank handling.

## 6. Backtracking for Newton–Raphson and Gauss–Newton

Use the merit function

$$
\phi(x)=\frac12\|F(x)\|_2^2.
$$

For the proposed raw scaled direction $z$, the model directional derivative is

$$
d=r^TAz.
$$

For an exact nonsingular Newton solve, $d=-\|r\|_2^2<0$ away from a root.
Verify descent numerically before starting the search. A non-descent direction
must produce a diagnostic, or an explicitly documented fallback if one is
introduced in a subsequent change.

With active scaled limit $\Delta_k$, start from

$$
\alpha_0=\min\left(1,\frac{\Delta_k}{\|z\|_2}\right).
$$

Without a configured limit, start from $\alpha_0=1$. Construct each trial as
$x_{\mathrm t}=x+\alpha Sz$ before evaluating properties. Backtrack through

$$
\alpha_{j+1}=\beta\alpha_j,\qquad 0<\beta<1.
$$

A valid trial must satisfy the Armijo condition

$$
\phi(x_{\mathrm t})\leq\phi(x)+c_1\alpha d,
\qquad 0<c_1<1.
$$

Use default $c_1=10^{-4}$ and $\beta=0.5$. Recognized undefined evaluations
are rejected and shortened without assigning a fictitious residual.
Permit at most 30 reductions after the initial trial by default; global
evaluation and trial budgets take precedence.

Implementation details:

- Check the actual rounded displacement before calling the model. Reuse the
  existing coordinate-rounding protections so every protected trial obeys
  its active limit.
- Use the actual displacement in the linear prediction. In finite precision,
  compare actual merit reduction against $-c_1 r^TAz_{\mathrm{actual}}$.
  Require strictly positive actual reduction as well.
- Compute merit differences with the existing stable norm-difference approach.
- Keep the base point, residual and Jacobian fixed during rejected retries.
- Stop on an unrepresentable displacement or exhausted search; return the
  last accepted state.

### Recovery and gradual restoration of the step limit

Use a persistent envelope with
$\Delta_0=\texttt{initial_limit}$ and
$\Delta_k\leq\Delta_{\max}=\texttt{max_limit}$.

The proposed carry-forward policy is:

1. If backtracking was needed, retain the lesson for the next outer iteration:
   set $\Delta_{k+1}=\min(\Delta_k,\|z_{\mathrm{accepted}}\|_2)$.
2. If the initial permitted trial was accepted without backtracking and used
   at least $0.8\Delta_k$, allow
   $\Delta_{k+1}=\min(\Delta_{\max},\gamma\Delta_k)$.
3. Otherwise retain the current limit.

Here $\gamma=\texttt{growth_factor}\geq1$. A factor of one prevents regrowth.
This carry-forward envelope is an explicit PySolverView control policy,
additional to classical backtracking. With step limits disabled, classical
backtracking starts again from $\alpha_0=1$ at each new base.

A step limit bounds displacement rather than physical validity. Numerical
derivative probes retain their separate distance setting and failure behavior.

## 7. Shared state, histories and termination

Extract a small set of common evaluation/accounting helpers from the current
driver so both driver families use the same contracts. The shared surface
should cover copied model inputs, shape/finite checks, recognized domain
failures, budgets, history storage, callbacks and final result construction.
Keep the LM/hybrid trust-region decisions in their existing path.

Preserve:

- `nfev`: every residual call, including failed evaluations and derivative probes.
- `njev`: full Jacobian requests, including incomplete numerical derivatives.
- `nit`: committed moves.
- `ntrials` and the existing `max_iterations` option: solver trial attempts,
  including rejected line-search attempts, matching the current budget convention.
  This is a trial budget, not the reported accepted-iteration count. Derivative
  probes never advance it; they consume only the residual-evaluation budget.
  Label this distinction explicitly and consider a clearer `max_trials` name
  during the API refactor.
- `history_level`: full, summary or none, with identical numerical behavior.
- Detached callback records, including the initial point and every committed
  iterate; `False` requests a stop.
- Root success only when $\|F(x)\|_2\leq\texttt{residual_tol}$.

Extend result records with the canonical method, requested alias, resolved step
policy, step fraction, number of backtracks and decision reason. A committed full-step iterate may have
negative actual reduction. Document that `accepted` means the driver committed
the point; it does not imply residual reduction for full-step modes.

Keep trust-region-specific fields meaningful. Use separate optional
line-search limit diagnostics, or a clearly named common active-step-bound
field. Mark an inactive/unavailable bound as `None`. Do not report an
artificial trust radius for full Newton.

Update history and benchmark audits to apply monotonicity and cap requirements
only to policies that promise them. Full-step failure or divergence should
remain visible in the results.

## 8. File changes and implementation sequence

| Phase | Work and files | Completion check |
| --- | --- | --- |
| 1. Contracts | Finalize the two new methods, alias registry, option validation, default policies and result meanings. | Tests specify aliases, omitted versus explicit `None`, full-step acceptance, protected acceptance and incompatible dimensions. |
| 2. Shared mechanics | Share evaluation/accounting through `src/solvers/evaluation.py` and records/results through `src/solvers/history.py`; keep exceptions in `src/exceptions.py`. Preserve re-exports of `DomainError`. | Existing correctness coverage remains passing, with obsolete trajectory expectations updated; LM/hybrid convergence and control contracts are verified. |
| 3. Directions | Extend `src/solvers/linear_model.py` with strict Newton/rank diagnostics and shared factorization reuse. Make unnecessary gradient work optional where required. | Independent linear-solve and least-squares references agree. |
| 4. Full-step modes | Add the Newton-family loop in `src/solvers/line_search.py` and dispatch from `src/solver.py`. | Pure Newton and full Gauss–Newton follow their stated commit policies. |
| 5. Protected modes | Keep bounded Armijo acceptance in `src/solvers/line_search.py` and trial construction in `src/solvers/trial_step.py` and select it for either method when `backtracking` is enabled. | Every evaluated trial respects the active limit; domain recovery and budgets are correct. |
| 6. Experiments | Extend the existing `run_*.py` scripts and policy-aware audits. | Scripts run independently and save complete cases to new `results_<case>/` folders. |
| 7. Documentation | Update exports, usage guide, mathematics note and comparisons. Every new function/class needs an explanatory docstring. | Imports, commands, links and method descriptions match actual behavior. |

The new shared modules should contain code reused across methods. Keep
study-specific orchestration, plots and reports in the existing runner scripts.
Keep new methods experimental until their thermodynamic cases have been reviewed.

Implementation detail: both driver families now use shared residual/Jacobian
accounting, history/callback/result handling and trial-coordinate helpers in
`src/solvers/`. Options and exceptions have dedicated modules. The trust-region
and line-search loops retain their own acceptance, stopping and Jacobian-refresh
policies. Problems are grouped by mathematical and thermodynamic definitions,
with shared benchmark metadata and separate suite selection.

The structural refactor reproduced all summaries and stored traces exactly in
fresh 100-case native/custom and 112-case custom studies. The
[source organization and validation note](root_step_control/notes/09_source_organization.md)
records the architecture, tests and timing comparison.

## 9. Validation plan

### Numerical and behavioral tests

Add tests under `root_step_control/tests/` for:

- Every alias resolving to the intended canonical method, with identical
  trajectories, counters, defaults and options. Retain separate backend labels.
- Omitted options enabling backtracking for both new methods; explicit
  `backtracking=None` selecting full steps for dictionary and dataclass inputs.
- `step_limits=None` preserving backtracking, and `max_backtracks=0` still
  rejecting a trial that fails Armijo. Reject infinite/invalid options and
  Newton-family options supplied to LM/hybrid.
- Square linear systems: one-step full Newton, analytic solution, correct
  counters and step fraction 1.
- Newton/Gauss–Newton equivalence on well-conditioned nonsingular square models,
  using independent solves as references.
- Rank deficiency: strict Newton reports numerical singularity; consistent
  Gauss–Newton problems reach an appropriate minimum-norm root.
- Consistent overdetermined systems and inconsistent overdetermined systems.
  For example, $F(x)=(x_1-1,x_2-2,1)^T$ has minimum residual norm 1 and must
  never be reported as a root.
- The coupled Rosenbrock system at $(-1.2,1)$: full Newton's first trial is
  approximately $(1,-3.84)$ and increases the residual. It must be committed
  in full-step mode, detecting accidental reuse of monotone acceptance.
- Armijo acceptance and rejection against independently evaluated merit
  changes; no Jacobian recomputation during retries at one base.
- Actual initial/persistent step limits, growth, recovery carry-forward,
  coordinate rounding and large raw Newton directions.
- Undefined trials: full mode stops; protected mode shortens and retries;
  unexpected programming errors propagate in both modes.
- Analytic and numerical Jacobians, invalid derivative probes, partial
  derivative budgets and line-search budget exhaustion.
- Callback isolation, early stop, full/summary/disabled histories and accurate
  counters even when histories are disabled.
- Poor scaling and numerical range: no inappropriate gradient-stationarity
  exit for a usable square Newton system.

Retain existing LM/hybrid correctness coverage. Update tests tied to obsolete
tuning or historical trajectories; do not weaken residual targets or step
contracts to make them pass. Add focused checks that exercise the
new behavior rather than tests that merely repeat implementation formulas.
Rerun the existing 100-configuration LM/hybrid study after extracting shared
mechanics. Verify residual targets and step audits, and investigate lost
convergence or material performance regressions before evaluating the new
methods. Different successful histories and evaluation counts are acceptable.

### Multidimensional problem studies

Use `NonlinearSystemProblem` subclasses for every additional benchmark.
Start with the seven existing standard systems, the 100-variable Broyden and
dense integral systems, and the 20-/100-variable thermodynamic ducts.
Add a coupled consistent overdetermined problem and an inconsistent example
to distinguish least-squares stationarity from root success.

Compare good initial guesses, standard starts and selected difficult starts.
Record unsupported method/problem combinations explicitly; square-only Newton
must not be counted as a convergence failure on an overdetermined problem.

For the duct problems, add parameterized operating points and deliberately
challenging but valid initial guesses. Check physical equation errors,
property failures and maximum temperature/pressure displacements. A duct
case that never produces a property failure only establishes accuracy and
step control, not a demonstrated recovery benefit.

## 10. Benchmark and timing updates

Replace the timing runner's current method-name parsing before registering
the new methods. Splitting solver identifiers by underscores cannot correctly
recover multiword names such as `newton_raphson`, `gauss_newton`,
`levenberg_marquardt` and `powell_hybrid`.
Use an explicit configuration registry carrying backend, canonical method,
step policy, history mode and display label. Aliases are alternative spellings,
not additional solver cases in numerical or timing comparisons.

Continue distinguishing **SciPy/MINPACK** from the custom **PySolverView**
implementation. Native LM/hybrid provide comparison backends; do not present
them as native pure-Newton equivalents.

Save new studies to separate folders, for example:

```text
results_newton_convergence/
results_newton_step_limits/
results_newton_time_perf/
```

Report residual-target success separately from native status and include:

- Residual/Jacobian calls, committed iterations, trial attempts and backtracks.
- Invalid property evaluations, maximum scaled/physical displacements and
  any active-limit excess.
- Residual histories that can show increases in full-step modes.
- Clean total time and separate instrumented residual, Jacobian, factorization
  subset and solver-overhead times.
- First evaluated residual-target crossing, retaining the existing distinction
  from accepted native iterates.

Use the existing interleaved timing protocol, common starts/scales/tolerances
and one requested BLAS thread. Compare equivalent policies and logging levels.
For a full Newton baseline, mark step controls inactive; for controlled
comparisons, state the active limits and globalization. Retain failures rather
than presenting timing ratios as successful root solves.

## 11. Completion criteria and priority

The extension is ready for wider application testing when all existing and new
tests pass, the canonical method/option contracts are enforced, protected
trial caps are independently audited from actual coordinates, and root success
is independently verified.

Full-step modes must remain identifiable in histories and results. The default
thermodynamic recommendation is `newton_raphson` or `gauss_newton` with
backtracking enabled, physical scaling, a small initial limit and a persistent
maximum. Disabling backtracking is an explicit choice for both methods.

Implement the full-step foundation first, then backtracking and its controls,
then the expanded convergence/timing studies. This order isolates the linear
direction calculation from acceptance and recovery behavior.

Related explanations and primary references:

- [Current LM/hybrid mathematics](root_step_control/notes/05_lm_and_hybrid_mathematics.md)
  and [thermodynamic step policy](root_step_control/notes/01_thermodynamic_step_policy.md).
- [SUNDIALS KINSOL mathematics](https://sundials.readthedocs.io/en/latest/kinsol/Mathematics_link.html)
  for Newton iteration, scaling and globalization.
- [PETSc nonlinear solvers](https://petsc.org/release/manual/snes/) for the
  separation between Newton steps, line searches and trust regions.
- [SciPy nonlinear least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
  for least-squares solver scope and termination considerations.
