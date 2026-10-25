# How the PySolverView solvers are implemented

## The direct answer

The new root solvers are **independent Python implementations of the solver
iteration**, using NumPy and public SciPy linear algebra. They do not wrap
MINPACK, call `scipy.optimize.minimize`, patch SciPy, launch solver executables,
or access hidden native MINPACK state. The benchmark separately calls SciPy's
native `root` methods for comparison.

Your turbodash SLSQP driver is different: it does interact directly with a
compiled SciPy SLSQP routine and its mutable state. This distinction affects
both what the current experiments prove and what must be maintained.

| Component | Who owns the iteration and step decision? | Native solver dependence |
| --- | --- | --- |
| Public SciPy `root("lm")` / `root("hybr")` | MINPACK | SciPy calls its compiled MINPACK routines |
| Turbodash custom SLSQP | Native SLSQP, with the Python driver controlling pending evaluations | Private `_slsqplib` interface and workspace, pinned to SciPy 1.17.1 |
| Experimental `solve_root(method="lm")` | Python driver in [trust_region.py](../../../pysolver_view/solvers/root/trust_region.py), model in [linear_model.py](../../../pysolver_view/solvers/root/linear_model.py) | Public `scipy.linalg` operations; no native LM solver call |
| Experimental `solve_root(method="hybr")` | The same driver and model modules | Public `scipy.linalg` operations; no native HYBRD/HYBRJ call |
| Experimental `solve_root(method="newton_raphson")` / `"gauss_newton"` | [line_search.py](../../../pysolver_view/solvers/root/line_search.py), sharing the QR/SVD model | Public linear algebra; optional Armijo backtracking or full valid steps |

SciPy's linear algebra can itself use compiled numerical libraries, but that
does not mean the native MINPACK solver is running. `OptimizeResult` is simply
the result container used to expose familiar fields and additional histories.

## Why SLSQP permits this particular wrapper

SLSQP uses reverse communication: the native routine advances, returns a request
for values or derivatives, and resumes when the driver supplies them. Your
backend holds the solve state and workspace between those calls. Before a
pending trial is evaluated, it can shorten the stored step and consistently
update coordinates and associated line-search quantities. The native merit
line search and accumulated optimization state remain involved. See the local
[SLSQP driver](../../../../turbodash/turbodash/optimization/slsqp.py) and
[backend](../../../../turbodash/turbodash/optimization/_slsqp_backend.py).

Those SLSQP controls limit **fractions of a proposed direction**. They do not by
themselves guarantee an absolute maximum distance. A fraction of a very large
direction can still be too large for a property evaluation. An absolute scaled
cap would be an additional capability to evaluate for that driver.

Native MINPACK is exposed through a synchronous call whose internal iterations
invoke the user's residual/Jacobian functions. SciPy does not expose the
equivalent suspended trial and mutable trust-region state. Its public root
interface also excludes callbacks for LM and hybrid. A residual callback is
therefore not permission to replace the requested point with a shorter one:
the native algorithm would judge a residual associated with the wrong point.
See [SciPy's root interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)
and the [1.17.1 MINPACK bindings](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/_minpack_py.py).

Native `factor` sets the initial scaled radius relative to the initial point;
`diag` sets variable scaling. Neither public root option provides the persistent
maximum used in this experiment. This remains the exposed API in the currently
served documentation, labeled SciPy 1.18.0 at review time. [LM options](https://docs.scipy.org/doc/scipy/reference/optimize.root-lm.html),
[hybrid options](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html).

## Experimental LM

The code forms the numerical linear model $r+Az$, where
$A=JSC$, $S=\operatorname{diag}(\texttt{x\_scale})$, and $C$ contains positive
Jacobian-based numerical factors. `jacobian_scaling=True` is the default;
setting it to `False` makes $C=I$. Physical step limits and derivative probes
continue to use the fixed caller scales $S$. This option belongs to the custom
solvers and does not change SciPy/MINPACK.

LM computes a candidate minimizing the squared model norm inside
$\|z\|_2\leq\Delta$. PySolverView first computes a pivoted economic QR
factorization and checks the triangular factor's conditioning. For an adequately
conditioned model it obtains the Gauss–Newton step by a triangular solve. An SVD
is computed only if damping is necessary or the QR path is unsuitable. Neither
path explicitly forms the normal equations. If the Gauss–Newton step fits, it
uses that step; otherwise it searches for a nonnegative damping parameter satisfying

$$
(A^TA+\lambda I)z=-A^Tr,\qquad\|z\|_2\leq\Delta.
$$

The candidate physical displacement is $p=SCz$. If $\|S^{-1}p\|_2>\Delta$,
the driver shortens it along the same ray to retain the physical cap. Both
norms are audited after adding the displacement to the current coordinates.
The final clipped trial is generally not the exact minimizer over the
intersection of the numerical and physical trust regions. The equations
above describe the numerical LM subproblem; the physical clipping is an
additional safeguard. Predicted reduction uses the actual rounded numerical
displacement $(SC)^{-1}(x_{\mathrm{trial}}-x_{\mathrm{base}})$.

The condition check uses LAPACK's triangular reciprocal-condition estimator
through SciPy's [DTRCON interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dtrcon.html),
with threshold $\sqrt{\epsilon}$. It avoids computing another SVD merely to
estimate conditioning. If the installed SciPy does not expose TRCON, the model
uses the conservative SVD path. This is a fallback for compatibility, not a
promise of the same fast-path timings on older SciPy versions.

For $A=U\Sigma V^T$, the damped spectral coefficients are
$c_i=-\sigma_i(U^Tr)_i/(\sigma_i^2+\lambda)$. Orthogonality gives
$\|Vc\|_2=\|c\|_2$, so the damping search works on these coefficients and
multiplies by $V$ only when constructing its final step. This avoids repeated
dense matrix-vector products inside the scalar search. The feasible side of
the bracket is retained.

Available QR/SVD factors are reused during retries at the same accepted point. The radius is
enforced before evaluating a trial. Actual versus predicted reduction governs
acceptance and the next radius. A new Jacobian is evaluated after an accepted
move. Numerical factors then update using running maximum column norms and a
fixed initial reference; rejected retries retain the same factors. No extra
model or derivative calls are needed for scaling. See the
[scaling policy](07_usage_and_controls.md#jacobian-scaling-and-physical-limits).
This is an LM trust-region implementation, not a translation of every
MINPACK decision or stopping rule.

## Experimental hybrid

The hybrid variant computes a dogleg step between the Cauchy and Gauss-Newton
steps in numerical coordinates, applies the same physical-distance safeguard,
then updates its numerical Jacobian using trial secant information:

$$
A_{\mathrm{new}}=A+\frac{(y-Az)z^T}{z^Tz},\qquad
y=r(x_{\mathrm{trial}})-r(x_{\mathrm{base}}).
$$

The secant displacement is the actual rounded numerical step
$z=(SC)^{-1}(x_{\mathrm{trial}}-x_{\mathrm{base}})$, including any clipping.
The numerical metric $C$ is frozen throughout Broyden updates. It changes only
when fresh derivatives are evaluated and the model is rebuilt, keeping the
secant pairs and factorization in consistent coordinates. Periodic QR
refactorization alone does not change the scaling.

A valid rejected trial can also supply this update. Undefined trials cannot.
Two consecutive poor agreement ratios trigger a fresh Jacobian; a misleading
approximate model is also refreshed before declaring stationarity or
stagnation. These dogleg/Broyden ingredients make it a Powell-style hybrid.

PySolverView maintains QR factors after each Broyden update. With
$A[:,P]=QR$ and update $uv^T$, the corresponding factor update uses $u$ and
$v[P]$. A triangular solve supplies the next Gauss–Newton step, subject to
conditioning safeguards. Periodic refactorization limits accumulated update
error; this rebuilds factors of the current approximate matrix without calling
the application Jacobian. A poorly conditioned updated QR requests an exact
Jacobian refresh; an exact rank-deficient or poorly conditioned model uses SVD
fallback. A stable solution of the dogleg segment's quadratic radius equation
computes the boundary intersection directly.

The official [QR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html)
and [QR-update](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_update.html)
contracts define these matrix operations. They do not establish equivalence of
the surrounding solver to MINPACK. The PySolverView hybrid still differs in its shared
configurable acceptance/radius policy, conditioning and deficient-rank handling,
residual-based success, and progress tests. It omits native slow-progress
counters that could reject intentionally restricted movement. Inspect
[HYBRJ](https://netlib.org/minpack/hybrj.f),
[DOGLEG](https://netlib.org/minpack/dogleg.f),
[LMDER](https://netlib.org/minpack/lmder.f), and
[LMPAR](https://netlib.org/minpack/lmpar.f) for the native implementation.

## Why a short implementation is not equivalent to MINPACK

LM and Powell-style hybrid have compact mathematical cores. A readable prototype
can implement those cores while delegating matrix factorizations to compiled
libraries. MINPACK additionally implements its own numerical kernels, derivative
variants, scaling logic, progress tests, workspaces and long-established edge-case
behavior. Source-line counts therefore compare different scopes and say little
about reliability or numerical equivalence. PySolverView separates the driver and
factorization model under `pysolver_view/solvers/root/`. Benchmark code stays
under development.

The following historical source-line counts were recorded on 25 October 2026
before the Jacobian-equilibration addition. They include comments and blank
lines and illustrate the different scopes; they are not current package totals:

| Source scope | Lines |
| --- | ---: |
| PySolverView public dispatcher, options and exceptions | 345 |
| PySolverView [trust_region.py](../../../pysolver_view/solvers/root/trust_region.py), LM/hybrid driver | 364 |
| PySolverView [line_search.py](../../../pysolver_view/solvers/root/line_search.py), Newton-family driver | 349 |
| Shared evaluation, histories and trial-coordinate handling | 312 |
| PySolverView [linear_model.py](../../../pysolver_view/solvers/root/linear_model.py), subproblems and factor management | 346 |
| Combined numerical modules for all four custom methods, excluding package initializers | 1,716 |
| MINPACK analytic-Jacobian [LMDER](https://netlib.org/minpack/lmder.f), main routine | 452 |
| LMDER plus its numerical helper closure | 1,358 |
| MINPACK analytic-Jacobian [HYBRJ](https://netlib.org/minpack/hybrj.f), main routine | 440 |
| HYBRJ plus its numerical helper closure | 1,460 |
| Union of those two MINPACK paths, shared helpers counted once | 2,369 |

The custom count excludes the public frontend/adapter, package initializers,
and the shared DomainError definition used by both roots and SLSQP.

The LMDER closure includes [LMPAR](https://netlib.org/minpack/lmpar.f),
[QRFAC](https://netlib.org/minpack/qrfac.f),
[QRSOLV](https://netlib.org/minpack/qrsolv.f),
[ENORM](https://netlib.org/minpack/enorm.f) and
[DPMPAR](https://netlib.org/minpack/dpmpar.f).
The HYBRJ closure includes QRFAC, ENORM and DPMPAR plus
[DOGLEG](https://netlib.org/minpack/dogleg.f),
[QFORM](https://netlib.org/minpack/qform.f),
[R1UPDT](https://netlib.org/minpack/r1updt.f) and
[R1MPYQ](https://netlib.org/minpack/r1mpyq.f).
These are Netlib Fortran sources, not the size of SciPy's complete translated
MINPACK file, wrappers, finite-difference variants or test suite. PySolverView delegates
factorizations and updates to compiled SciPy/LAPACK routines whose source is
excluded from these Python counts. The Newton-family modules add methods and
controls outside the scope of the two MINPACK paths. The shared modules serve
both driver families, so these totals do not imply equivalent implementations.

Tests compare subproblem solutions, factor updates, residual accuracy and actual
trial bounds. They provide evidence for this implementation's behavior; they do
not prove every MINPACK safeguard has been reproduced. The distinction remains
important even when PySolverView is competitive on the measured workloads.

## What the validation establishes

The [report](../results_convergence/REPORT.md) compares native and custom implementations explicitly.
Within each custom method, the three configurations vary the step controls;
those comparisons are the clearest evidence about those controls. Native versus
custom comparisons also include algorithm and stopping-rule differences.
The recorded convergence and timing findings below predate default Jacobian
equilibration. They support the configurations used in those studies and do
not establish identical histories or timings with the new numerical metric.

The test suite checks actual trial limits, accepted-state consistency,
root-success conditions, Jacobian update behavior, and problem derivatives.
It does not establish equivalence to MINPACK, robustness across thermodynamic regimes, or
superior performance for large sparse systems. The mathematical benchmarks
contained no domain failures in the original LM/hybrid study at its tested
standard starts. The separate
[four-method study](../results_custom_convergence/REPORT.md) includes a
coupled logarithmic-domain example that exposes failed full Newton steps and
successful backtracking. This is a mathematical guard case, not a claim about
all thermodynamic domains. The [timing suite](06_time_performance.md)
adds real CoolProp property evaluations in 20- and 100-variable duct models,
and measures QR/SVD work separately from residual and Jacobian time. It is a
single-phase workload, not a test of difficult flash calculations or phase crossings.

If preserving the exact MINPACK algorithm is the priority, the next route is a
small maintained native backend change that exposes the required radii and
recovery semantics, followed by native regression tests. If rapid experimentation
with physical step policies is the priority, the current independent solvers
are a useful, readable development platform. Both routes should keep native
backend identity visible to users.
