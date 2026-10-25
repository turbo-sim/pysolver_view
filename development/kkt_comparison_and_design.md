# KKT checks: current functionality and recommended shared design

**Implementation update (25 October 2026):** Result/history repairs, the shared
KKT checker, and the custom SLSQP backend are implemented. See
[KKT/SLSQP usage and validation](kkt_slsqp_validation.md). The review below records
the original findings and design; Turbodash imports/dependencies and deletion of
its copies remain deferred until the user releases and adopts pysolver.

Reviewed 25 October 2026. This is an evaluation and design note, not an
implementation. It supplements the [refactor plan](proposed_refactor.md) and
the [package feature review](pysolverview_feature_review.md).

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

Exact reproduction of historical PySolverView convergence is not required.
The shared checker and revised solver settings may change stopping points,
histories, and evaluation counts. Judge them by independently verified
convergence and the intended feasibility/optimality thresholds. A previously
incorrect success may correctly become a rejection; never preserve a false
certificate or loosen a target just to reproduce an old outcome.

Document changed tolerance meanings, norms, and defaults. Compatibility of
public names and report fields remains a separate design decision. Pointwise
mathematical correctness and reconstruction tests remain required; tests should
not demand old iterate sequences or old diagnostic-call counts.

## Scope and source of the comparison

The older implementation is in
[PySolverView optimization.py](../pysolver_view/optimization.py), principally
`evaluate_kkt_conditions`, its report methods, and progress recording.
The newer numerical assessment is in
[Turbodash kkt.py](../../turbodash/turbodash/optimization/kkt.py).
Its integration with stopping, derivatives, history, and recovery is in
[Turbodash slsqp.py](../../turbodash/turbodash/optimization/slsqp.py).

The distinction matters: the KKT evaluator and the SLSQP driver have different
responsibilities. Some functionality described below belongs to the driver
around the checker, rather than the checker itself.

## Functionality available in each implementation

| Feature | PySolverView | Turbodash |
| --- | --- | --- |
| Problem inputs | A solver method obtains values and derivatives through its own fitness/gradient cache. | `assess_kkt` receives explicit point, gradient, constraint values/Jacobians, bounds, and tolerances. |
| Inequality convention | $g(x)\leq0$. | $c(x)\geq0$. |
| Equality multipliers | Unrestricted through an unconstrained linear least-squares fit. | Explicitly unrestricted in the multiplier-estimation linear program. |
| Inequality selection | Uses constraints with $|g_i|\leq\mathrm{tol}$ as an approximate active set. | Includes all inequalities, with complementarity penalizing use of inactive constraints. |
| Variable bounds | Appends lower and upper bounds as inequalities. | Includes every finite bound, with explicit finite-bound masks. |
| Bound-only feasibility | Implemented incorrectly: feasibility is skipped when there are no user inequalities. | Checked independently of the number of user inequalities. |
| Multiplier estimation | `np.linalg.lstsq` fits stationarity for selected constraints. Nonnegativity is checked afterward. | A linear minimax problem fits stationarity and complementarity together with nonnegative inequality/bound multipliers. |
| Dependent constraints | Minimum-norm unconstrained multipliers can have incorrect signs even when a valid KKT multiplier set exists. | Can find sign-feasible multipliers without requiring uniqueness; conditioning still matters. |
| Stationarity | Euclidean norm of the reconstructed Lagrangian gradient. | Infinity norm of the reconstructed Lagrangian gradient. |
| Primal feasibility | Separate equality and inequality measures, subject to the bounds bug above. | Maximum equality, inequality, and finite-bound violation. |
| Dual feasibility | Reports the minimum selected inequality multiplier, and tests it against a negative tolerance. | Reports the maximum negative part of inequality/bound multipliers, a nonnegative residual. |
| Complementarity | Maximum absolute multiplier-times-residual over selected active inequalities/bounds. | Maximum absolute multiplier-times-slack over all inequalities and finite bounds. |
| Tolerances | One `tol` is used for active-set selection and all checks; progress passes `50*tolerance`. | Separate `feasibility_tol` and `optimality_tol`; native SLSQP `ftol` remains separate. |
| Combined assessment | Returns individual residuals and Boolean checks; optimization success is not gated by these checks. | Returns `optimality`, `kkt_residual`, `kkt_ratio`, `available`, and `passed`. |
| Scaling safeguards in the fit | No dedicated column/residual equilibration in the multiplier fit. | Equilibrates columns and rescales by gradient magnitude before the LP, then reconstructs in the original supplied units. |
| Input validation | Relies largely on surrounding solver assumptions. | Checks shapes, finite values, bound validity, and positive finite tolerances. |
| Unavailable assessment | No structured unavailable-result contract comparable to the newer checker. | Failed multiplier estimation/nonfinite reconstruction produces `available=False` and missing diagnostics. Invalid input raises an error. |
| Multiplier output | Equality array and dictionary of selected inequality indices, including appended bounds. | Separate equality, inequality, lower-bound, and upper-bound arrays; absent-bound entries are zero. |
| Native solver multipliers | Wrappers discard native result objects rather than preserving their multiplier information. | Driver retains native QP multipliers separately from independently estimated KKT multipliers. |
| Text reports | Dedicated KKT, constraint, variable, and multiplier reports. | Structured diagnostics and driver logging; turbine/app adapters export and display them. |
| History integration | KKT calls can trigger evaluations, derivative calls, and recursive progress reporting. | Driver attaches same-point assessments to initial/accepted records and protects records with copies. |
| Final acceptance | Usually takes the backend success flag; KKT reports are separate. | Requires native success, feasibility, and available first-order diagnostics within tolerance. |
| Independent physical audit | No focused equivalent in the package's existing optimization tests. | A separate turbine validation study checks physical signs/scales, derivatives, histories, and exported results. |

These features do not imply identical numerical meanings. For example, the
older signed minimum multiplier and the newer nonnegative dual residual cannot
be substituted under the same field name without an explicit conversion.

## Confirmed errors in the older implementation

The previous review reproduced these issues with small examples:

1. With no user inequalities, bounds $[-1,1]^2$, zero objective gradient, and
   $x=(2,0)$, inequality feasibility was reported as satisfied with zero
   violation. The actual bound violation is one.
2. For a fixed first coordinate and objective gradient $(1,0)$, the
   unconstrained multiplier fit selected lower/upper multipliers
   $(0.5,-0.5)$ and failed dual feasibility. The valid combination $(1,0)$
   exists. Opposing active inequalities exhibited the same problem.
3. An initial progress request with `update_on="gradient"` generated two
   history rows because reporting requested KKT, KKT requested derivatives,
   and derivatives reentered reporting.
4. A progress row could combine a new point with an old constraint violation.
   For $g=x_1\leq0$, a cache at $(2,0)$ followed by a gradient/report at
   $(-1,0)$ recorded violation two even though the refreshed constraint was
   minus one.

The first two affect assessment correctness. The other two arise from coupling
diagnostics to mutable solver/reporting state. Replacing only the multiplier
formula would not fix them.

Additional limitations are the arbitrary active-set cutoff, assumptions about
available bounds, and lack of a clear unavailable-diagnostics state. A single
threshold also conflates active-set identification, feasibility, stationarity,
dual feasibility, and complementarity.

## What is worth retaining from each

Retain PySolverView's user-facing report functionality and its ability to
evaluate diagnostics for different optimization backends. These are central
to the original goal of a consistent convergence view.

Use Turbodash's explicit numerical evaluator, sign-constrained multiplier fit,
finite-bound treatment, structured diagnostics, and separate acceptance
thresholds as the starting point for the common implementation.

Do not assume the newer checker is free or infallible. Its LP adds work, uses
dense intermediate arrays, and depends on derivative quality and conditioning.
Nonnegative fitted multipliers make dual feasibility mostly a check on the fit
and reconstruction; it is not evidence that the native solver's own multipliers
have the right signs. Keep those two multiplier sources separate.

The turbine audit should remain independent. Calling the production checker
again from that audit would not independently validate its formulas.

## Recommended features in the refactored PySolverView

### One numerical assessment shared by optimization backends

Provide a generic `diagnostics/kkt.py` with two conceptual operations:

- **Estimate multipliers** when useful multipliers are not supplied.
- **Evaluate residuals** for a specified multiplier set.

The existing minimax estimator can serve the first operation. The second
should reconstruct residuals from original inputs for both independently
estimated and native multipliers. Native assessments with missing bound
multipliers must be marked incomplete or explicitly completed through a
documented estimation procedure, not silently treated as complete.

Support SciPy, PyGMO, and custom optimization drivers through adapters. Do not
require any particular optimizer or its internal state.

### Explicit mathematical convention

Keep the public PySolverView convention $h=0$, $g\leq0$, $l\leq x\leq u$:

$$
L=f+\lambda^T h+\mu^T g-z_L^T(x-l)+z_U^T(x-u),
\qquad \mu,z_L,z_U\geq0.
$$

Report the four nonnegative residual measures:

| Measure | Definition |
| --- | --- |
| Stationarity | $\|\nabla f+J_h^T\lambda+J_g^T\mu-z_L+z_U\|_\infty$ |
| Primal feasibility | Maximum of $|h_i|$, $\max(g_i,0)$, and finite-bound violations. |
| Dual feasibility | Maximum negative part of $\mu,z_L,z_U$. |
| Complementarity | Maximum of $|\mu_i g_i|$, $|(z_L)_j(x_j-l_j)|$, and $|(z_U)_j(u_j-x_j)|$ for finite bounds. |

During extraction, the newer checker's internal convention can remain
unchanged. Convert $c=-g$ and $J_c=-J_g$ at one boundary. If equality residuals
retain their sign, its equality multipliers must be negated for the public
Lagrangian above. Inequality and bound multiplier signs remain unchanged.

Include coordinates, objective/constraint scales, constraint ordering, and
derivative provenance in the diagnostic context. Absolute tolerance decisions
are not invariant to arbitrary rescaling. Physical sensitivity interpretation
requires transforming multipliers with objective/constraint scaling too.

### Stable output and explicit acceptance policy

Return point identity, availability/reason, the four residuals, tolerances,
multiplier source, full ordered multiplier arrays, and the Lagrangian gradient.
Use missing values for unavailable metrics rather than zero.

Initially preserve the newer aggregation:

$$
r_{\mathrm{opt}}=\max(r_{\mathrm{stat}},r_{\mathrm{dual}},r_{\mathrm{comp}}),
\qquad
\mathrm{kkt\_ratio}=
\max\left(\frac{r_{\mathrm{primal}}}{\tau_{\mathrm{feas}}},
\frac{r_{\mathrm{opt}}}{\tau_{\mathrm{opt}}}\right).
$$

An available assessment passes when this ratio is at most one. Keep individual
components visible; an aggregate alone is insufficient for diagnosing failures.
Separate component tolerances could be added later if actual use requires them.

Keep `solver_success`, `feasible`, `kkt_passed`, and application `success`
distinct. Preserve strict custom-SLSQP acceptance. Offer an explicit strict
policy for existing backends rather than silently changing all users' success
semantics during migration.

### Organization and history

```text
Backend adapter → accepted point / evaluation snapshot
                                ↓
                         KKT assessment
                                ↓
                  result + recorder + presentation
```

The checker must not evaluate the model, update counters, invoke callbacks, or
write history. The evaluation layer supplies same-point values and derivatives.
The recorder stores a detached snapshot; plot/report code only reads it.

Keep existing `evaluate_kkt_conditions` and report entry points as compatibility
adapters, with documented changes to norms and multiplier indexing. An explicit
on-demand assessment may obtain missing derivatives, but must do so outside
the numerical checker and account for the cost.

Support final-only and accepted-iterate diagnostic schedules. Do not force a
new Jacobian at every trial just to draw a plot. Timing should separate model
evaluation, derivative evaluation, diagnostic estimation, and rendering.

### Reverse communication and what it does not change

KKT assessment needs pointwise values and derivatives, not bidirectional access
to the numerical algorithm. It can be run after a conventional SciPy/PyGMO
solve. Native SLSQP multipliers describe its QP approximation and omit bound
multipliers from the public result.
[SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/optimize.minimize-slsqp.html)

The custom driver uses access to pending SLSQP trials to shorten steps and
recover from recognized domain errors consistently with internal state. That
access enables step control. It is not a prerequisite for a shared KKT checker.
It does make independent checks particularly useful, because small controlled
steps can satisfy native stopping tests without adequate stationarity.

Root solvers retain their residual-based acceptance rules. Least-squares
stationarity with nonzero residual is not root success.

## Validation and limits

Retain tests for bound-only feasibility, opposing normals, redundant
equalities, inactive constraints, positive rescaling, tiny gradients, unavailable
fits, nonfinite inputs, and reconstruction from returned multipliers. Add
sign/scaling adapter checks for SciPy SLSQP and the custom driver, plus the
four older implementation defects. No PyGMO/IPOPT/SNOPT execution is required.
Test that enabling plots never changes an assessment or its source history.

During the preceding refactor review, Turbodash's focused KKT/tolerance suites
passed **69 tests**, and its SLSQP suite passed **41 tests**. Those are prior
review results; they were not rerun for this note. The full turbine matrix and
SciPy SLSQP integration remain future validation gates. Other native backends
are excluded by the agreed migration validation scope.

A passing numerical KKT assessment is evidence of first-order conditions under
the supplied scaling and derivative accuracy. It does not establish second-order
sufficiency or global optimality; degeneracy and constraint qualifications also
affect interpretation. Neither checker currently supplies a general
second-order optimality certificate, and adding one is not necessary for the
proposed consolidation.
