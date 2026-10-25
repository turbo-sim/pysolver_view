# Why unscaled LM fails in the 500-point radiative heat demo

## Implemented follow-up: default numerical Jacobian scaling

The custom root methods now accept `jacobian_scaling=True` by default. The
500-node demo has been rerun with this feature: all six Newton, LM, and hybrid
configurations converge. Controlled LM now uses 12 accepted iterations and 13
model evaluations, with residual norm about $4.5\times10^{-11}$ and no rejected
trials. Its first physical scaled step remains capped at 0.1. Controlled hybrid
also converges, in 26 iterations and 30 calls; scaling is not a speed improvement
for every method. Full-rank Newton directions are mathematically unchanged.

The numerical factors follow fresh Jacobian column norms, with running maxima
and a fixed initial normalization. Hybrid freezes them between fresh Jacobians.
They do not replace `x_scale`, change numerical-derivative probes, or reweight
the residual. Both numerical and physical trust bounds are enforced on the
actual trial displacement. See the
[usage note](../root_step_control/notes/07_usage_and_controls.md) and
[current demo results](../../demos/demo_eqn_solving/README.md).

The experiments and measurements below document the **unscaled implementation**
that motivated this change. Their runners explicitly set
`jacobian_scaling=False` to keep that evidence reproducible. The saved initial,
intermediate, and failed states remain available for the independent LM
subproblem audit. The demo itself defaults to the improved scaling; use
`--no-jacobian-scaling` to reproduce its original geometry.

## Original investigation

Investigation date: 2026-10-25. The user changed `NUM_POINTS` from 49 to 500,
giving 1000 unknowns in
[`demo_radiative_heat_collocation.py`](../../demos/demo_eqn_solving/demo_radiative_heat_collocation.py).
The original numerical experiments left the demo and production solver
unchanged; the subsequent implemented follow-up is described above.

**The failure is reproducible. The main causes are the mesh-dependent
least-squares weighting of the boundary equations and the choice of variable
scaling. We found no incorrect LM damping/subproblem calculation in the tested
states.** With numerical scaling disabled, the custom method uses fixed
caller-specified variable scales and recovers from domain violations by reducing
the trust radius. Those policies cannot guarantee escape from an unfavorable
descent path toward a domain boundary.

## What actually happens

The model uses constant conductivity and distributed radiation:

$$
T''+50-10(T^4-1)=0,\qquad T(0)=T(1)=1,\qquad T>0.
$$

All temperatures initially equal 0.3; all normalized heat fluxes initially equal
zero. At 500 nodes, the residual contains 500 Fourier-law equations, 498
interior energy equations, and only two boundary equations. LM minimizes the
unweighted sum of their squares while trying to find a zero.

In the failed controlled run, both endpoint temperatures move **away** from
their prescribed value of one:

| Accepted iteration | Left endpoint temperature | Residual 2-norm |
|---|---:|---:|
| 0 | 0.300 | 26.761 |
| 5 | 0.270 | 15.71 |
| 10 | 0.00341 | 7.385 |
| 33 | 8.77e-13 | 7.375 |

The final residual block norms are approximately 6.595 for Fourier's law, 2.985
for energy balance, and 1.414 for the two boundary conditions. Interior
temperatures remain around 0.33--0.40. This is a seriously incorrect state, not a
converged root with a slightly noisy final residual.

Further LM directions point toward negative endpoint temperatures. The model
correctly raises `DomainError`; the solver shrinks the radius, accepts tiny
positive-domain moves, and eventually reports step stagnation. The result is
correctly marked unsuccessful. The initial-0.1 run records 33 accepted moves,
68 model calls, and 34 invalid trials.

The same failure occurs with both small and large initial radii. It already
appears at 100 nodes in the tested mesh sweep; 250 and 251 nodes both fail, so
this is not a simple odd/even grid issue.

## Why increasing the number of nodes changes the path

### Boundary conditions lose relative influence

For residual blocks $F_c,F_e,F_b$, the merit function is

$$
\phi(z)=\tfrac12\left(\|F_c(z)\|_2^2+
\|F_e(z)\|_2^2+\|F_b(z)\|_2^2\right).
$$

Adding spatial nodes adds more interior residual terms, while the number of
boundary terms stays two. At the common uniform guess, the boundary share of
the squared residual norm falls from **1.43% at 49 nodes** to **0.137% at 500
nodes**. Thus the discretized least-squares problem being followed away from the
root changes with the mesh. A reduction in the total norm does not guarantee
improvement of every equation, physical admissibility of the next trial, or
progress toward the correct boundary temperatures.

The problem remains a consistent root problem. Positive row weights preserve
its exact roots, but change the LM merit function and therefore its path.

### The Jacobian becomes less evenly scaled

| Initial Jacobian diagnostic | 49 nodes | 500 nodes |
|---|---:|---:|
| 2-norm condition number | 8.97e3 | 1.33e6 |
| Smallest column norm | 1.11 | 11.52 |
| Largest column norm | 486 | 52,494 |
| Largest/smallest column norm | 436 | 4,555 |

The dense spectral derivative matrix couples every nodal variable. Its column
magnitudes vary strongly, particularly near the clustered endpoints. Setting
`x_scale=1` treats all variable displacements equally despite these differences
in their effect on the residual.

Conditioning contributes to sensitivity, but it is not evidence of an unusable
linear solve here. The tested matrices are full rank. Multiplying just the two
boundary rows by $\sqrt{500}$ improves the condition number only from 1.33e6 to
9.36e5, yet restores nonlinear convergence. Other methods and the reformulated
LM cases reach original residual norms near $10^{-11}$ or smaller. Roundoff
does not explain a stalled norm of 7.375.

### A fixed radius has a different meaning on a larger mesh

The limit is $\|\Delta z\|_2$, not a per-node temperature change. A uniform
temperature change of size $a$ across $n$ nodes has norm $|a|\sqrt n$. Therefore
the same radius 0.1 allows a smaller typical per-node change after refinement.

Also, the first full Newton direction grows from norm 32.5 at 49 nodes to
104.8 at 500 nodes. The demo's baseline radius 100 therefore **clips the first
LM/hybrid direction at 500 nodes**; the original README's statement that the
baseline admits the full first direction was specific to the 49-node example.

This alone does not explain the failure: increasing the baseline radius to
1000 still fails, and multiplying initial/maximum radii by
$\sqrt{500/49}$ still fails. A different step length cannot reliably fix a
descent direction that points out of the admissible domain.

## Independent check of the LM algorithm

[`run_lm_subproblem_audit.py`](run_lm_subproblem_audit.py) checks six trust-region
subproblems at the initial state, accepted iteration 10, and the failed final
state, using both large and small radii. These states are preserved in
[`lm_audit_states.json`](lm_audit_states.json), so rerunning the demo cannot
overwrite the evidence.

The production solver uses an SVD plus safeguarded Newton iterations for its
damping equation. The independent calculation uses a different LAPACK SVD
driver and Brent's method in logarithmic damping coordinates. Both solve

$$
\min_{\|p\|_2\leq\Delta}\tfrac12\|F+Jp\|_2^2.
$$

For the full-rank cases checked, the optimality conditions are

$$
(J^TJ+\lambda I)p=-J^TF,\qquad \lambda\geq0,
\qquad \lambda(\|p\|_2-\Delta)=0.
$$

The relative differences between the independently computed and production
steps are **2.4e-15 to 4.98e-12**. The relative radius discrepancies are below
5.2e-16. The damping equation requires 6--26 iterations, below its 100-iteration
limit. At the stalled state, the independently optimal step itself points
toward negative boundary temperatures.

The audit also records normal-equation certificates. Their relative residuals
can be around 2e-8 near stagnation because this reconstruction involves
cancellation in $J^TJp$; the independent step comparison is much tighter.
The production solver does not form normal equations to calculate its steps.
This audit supports correctness of the inspected subproblems, not a proof that
every possible LM problem is covered.

## Comparison with native SciPy/MINPACK

[`run_backend_comparison.py`](run_backend_comparison.py) uses the original
physical residual and analytic Jacobian. Native calls retain the positive
temperature check; exceptions are reported, never replaced with fake residuals.
Actual model calls are counted outside the backend.

| 500-node configuration | Outcome | Original residual norm | Model calls |
|---|---|---:|---:|
| SciPy LM, default variable scaling | Root found | 2.69e-12 | 30 |
| SciPy LM, `diag=ones` | Aborts on a negative-temperature trial | No returned solution | 5 |
| Custom LM, identity scaling, radius 0.1 | Stagnates at domain boundary | 7.375 | 68 |
| Custom LM, fixed initial-Jacobian column scaling | Root found | 7.78e-12 | 20 |

MINPACK's default Jacobian-based variable scaling differs materially from the
custom solver's fixed `x_scale=1`. The native algorithm and the custom one also
have different radius and acceptance policies; this is an ablation of scaling,
not an assertion that their full trajectories should match. SciPy documents
its variable scales and initial-bound convention in
[`root(method='lm')`](https://docs.scipy.org/doc/scipy/reference/optimize.root-lm.html);
its [`least_squares` documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html)
also describes the Jacobian-column scaling used by native LM and `leastsq`.

The successful custom scaling experiment uses fixed
$s_j=\operatorname{median}_k\|J_0[:,k]\|_2/\|J_0[:,j]\|_2$.
It is not dynamic MINPACK scaling. Its scale values range from approximately
0.011 to 50.0. Consequently a radius of 0.1 now measures a **different variable
metric**; this is evidence about scaling, not proof of the same physical step
protection as the original configuration.

## Formulation changes that restore convergence

[`run_spatial_scaling.py`](run_spatial_scaling.py) changes the demo formulation
inside isolated experimental wrappers, with numerical scaling explicitly disabled.
All rows below use an initial radius of 0.1. Each successful result is audited
against the **original unweighted residual equations** with the original
$10^{-8}$ root target.

| Variant at 500 nodes | Accepted moves / model calls | Original residual norm |
|---|---:|---:|
| Original residual weighting | 33 / 68, failed | 7.375 |
| Multiply the two boundary rows by 4 | 15 / 17 | 1.07e-11 |
| Multiply the two boundary rows by 10 | 14 / 15 | 4.68e-12 |
| Multiply the two boundary rows by $\sqrt{500}$ | 15 / 17 | 5.37e-12 |
| Divide PDE rows by $\sqrt{500}$ | 14 / 16 | 3.46e-10 |
| Weight PDE rows by square roots of spatial quadrature weights | 15 / 18 | 5.30e-12 |
| Eliminate prescribed boundary temperatures from unknowns | 16 / 19 | 6.60e-12 |

All successful variants in this table have zero invalid trials. A boundary
weight of 2 still fails, showing that a small arbitrary adjustment is not
enough. Quadrature weights are positive Clenshaw-Curtis weights appropriate to
the Chebyshev nodes; they approximate spatially integrated squared residuals
instead of letting the merit grow merely because more nodes were added.

The spatial experiment requests a weighted residual target of $10^{-10}$, then
checks the original equations separately. A smaller weighted norm is not by
itself evidence of a better physical solution. The successful unweighted audits
show these fixes did not just hide errors by loosening the convergence test.

Boundary elimination fixes the endpoint temperatures to one throughout the
solve and leaves 998 unknowns. It preserves the boundary-value problem but
changes the initial endpoints: only the interior temperatures remain uniformly
wrong. The row-weighting experiments preserve the entire original initial
guess, including its incorrect endpoints.

## Recommendations from the original investigation

1. **Make the demo formulation consistent across meshes.** For prescribed
   Dirichlet temperatures, enforce them exactly by eliminating them from the
   unknown vector. Use dimensionless, physically weighted residuals, with an
   RMS or spatial-quadrature convention that does not dilute boundary data as
   resolution increases. If keeping the all-nodes-wrong start is important, keep
   the endpoint unknowns and normalize the residual blocks explicitly instead.
2. **Keep separate physical acceptance checks.** Report equation residuals and
   boundary errors in their original units/scales in addition to the solver's
   weighted merit. Do not declare success by changing a plotting norm or merely
   increasing the iteration budget.
3. **Automatic Jacobian-based scaling for the custom solver (now implemented).**
   The existing fixed `x_scale` can already perform an initial column-scaling
   experiment. A polished dynamic option needs a clear policy for updates and
   for how trust radii transform. Automatic numerical scaling must not silently
   change the meaning of a user's physical step limit; separate physical
   displacement protection may be appropriate.
4. **Improve domain-stagnation diagnostics.** Repeated invalid trials while
   accepted temperatures approach a domain boundary deserve a more specific
   message than generic small-step stagnation. Radius contraction correctly
   prevents invalid evaluations from being accepted, but cannot impose a
   feasible tangent direction. Explicit bounds or a different constrained
   strategy are separate algorithmic features.
5. **Retain mesh-refinement regressions.** Include medium and larger meshes,
   blockwise physical residual checks, and raw-trial histories. Tests for a
   fixed benchmark should explicitly choose their resolution rather than
   inheriting the demo's editable `NUM_POINTS` global.

The initial recommendation prioritized formulation changes and proposed scaling
as an opt-in capability. The subsequent user-approved implementation instead
enables numerical scaling by default while preserving physical limits; the
original formulation now converges. Residual normalization and boundary treatment
remain useful formulation considerations. Rewriting LM's SVD/damping code is
not supported by this evidence.
An initially small step remains useful for fragile models, but it cannot
compensate for an unfavorable merit function or guarantee convergence on its
own. The 49-node test demonstrated useful behavior; it was insufficient to
establish robustness under spatial refinement.

## Reproduce the investigation

From the repository root, using an environment with the current checkout:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python development/radiative_heat_investigation/run_backend_comparison.py
python development/radiative_heat_investigation/run_spatial_scaling.py
python development/radiative_heat_investigation/run_lm_subproblem_audit.py
```

The matching `results_*.json` files contain numerical evidence. Saved timings
include diagnostic overhead and were not measured as an isolated performance
benchmark. Dense 1000-variable QR/SVD work is considerably more expensive than
the original 98-variable example; that cost is separate from the convergence
failure. Existing demo outputs are not regenerated by these scripts.
