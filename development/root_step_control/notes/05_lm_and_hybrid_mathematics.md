# The mathematics of LM and Powell's hybrid root solver

Both methods use a local linear approximation of the residual equations and
limit how far they trust that approximation. Levenberg-Marquardt (LM) chooses
a damped least-squares step. Powell's hybrid method chooses a step along a
dogleg path and reuses derivative information through secant updates.

This note derives those ideas and connects them to
[trust_region.py](../../../pysolver_view/solvers/root/trust_region.py) and [linear_model.py](../../../pysolver_view/solvers/root/linear_model.py).
The equations describe the algorithm families; details explicitly labeled
**PySolverView** describe our experimental implementation. It is an independent
implementation, not the exact native MINPACK iteration used by SciPy.
MINPACK's [LMDER](https://netlib.org/minpack/lmder.f) solves nonlinear least
squares, while [HYBRJ](https://netlib.org/minpack/hybrj.f) seeks a zero of a
square system using a modified Powell hybrid method.

## 1. From root finding to a local least-squares model

We want to solve

$$
F(x)=0,\qquad F:\mathbb{R}^n\longrightarrow\mathbb{R}^m.
$$

For an ordinary square root problem, $m=n$. PySolverView also accepts $m>n$, although an
overdetermined system need not have an exact root. At the current accepted point,
write $r=F(x)$ and $J=\partial F/\partial x$. For a physical displacement $p$,

$$
F(x+p)\approx r+Jp.
$$

The nonlinear merit function is

$$
\phi(x)=\frac12\|F(x)\|_2^2.
$$

A root is a global minimum with $\phi=0$. An arbitrary local minimum need not
be a root: $J^Tr=0$ can hold while $r\ne0$. This is why a small
gradient or a small step is insufficient evidence that a root was found.

Variables such as temperature and pressure have different units. Define
characteristic distances $s_i>0$ and

$$
S=\operatorname{diag}(s_1,\ldots,s_n),\qquad u=S^{-1}p,\qquad A_0=JS.
$$

Here $u$ is the dimensionless physical displacement. `x_scale` supplies $s$;
these caller-selected values remain fixed throughout a solve. The physical
distance control is

$$
\|u\|_2=\|S^{-1}p\|_2\leq\Delta.
$$

The custom solvers also enable `jacobian_scaling=True` by default. Positive
numerical factors $c_i$ form $C=\operatorname{diag}(c)$ and define

$$
u=Cz,\qquad p=SCz,\qquad A=A_0C=JSC.
$$

The local linear algebra uses the numerical coordinate $z$. The residual model
is $r+Az$, and its squared-norm model is

$$
q(z)=\frac12\|r+Az\|_2^2
    =\frac12\|r\|_2^2+g^Tz+\frac12z^TBz,
\qquad g=A^Tr,\quad B=A^TA.
$$

When $A=JSC$, $g$ is the gradient of the merit function in numerical coordinates.
Hybrid will also use an approximate $A$; its $g$ is then the gradient of the
current model, which can differ from the true merit gradient.

The numerical trust region is

$$
\|z\|_2\leq\Delta
\quad\Longleftrightarrow\quad
\|(SC)^{-1}p\|_2\leq\Delta.
$$

It is a sphere in numerical coordinates and generally an ellipsoid in physical
coordinates. A candidate computed inside this region is shortened along its
ray if needed to enforce the separate physical cap $\|Cz\|_2\leq\Delta$.
Both use the same configured `radius`, but different metrics. The controls
remain distances, not time steps or fractions of a Newton direction. With
`jacobian_scaling=False`, $C=I$ and the two metrics coincide.

### How the numerical factors are chosen

On the first fresh Jacobian with a nonzero column, let
$a_{0j}=\|(A_0)_{:j}\|_2$ and choose a fixed reference

$$
a_{\mathrm{ref}}=
\exp\!\left(\operatorname{median}_{j:a_{0j}>0}\log a_{0j}\right).
$$

For an even number of nonzero columns this is the geometric mean of the two
middle norms. It is fixed for the solve, so changing the overall Jacobian
magnitude does not repeatedly reset the coordinate convention. At each fresh
Jacobian evaluation $k$, retain the largest observed norm per column and set

$$
\bar a_{kj}=\max_{0\leq\ell\leq k}\|(A_{0,\ell})_{:j}\|_2,\qquad
c_{kj}=\operatorname{clip}\!\left(
\frac{a_{\mathrm{ref}}}{\bar a_{kj}},
\sqrt{\epsilon_{\mathrm{mach}}},
\frac{1}{\sqrt{\epsilon_{\mathrm{mach}}}}\right).
$$

A column that has always been zero uses $c_j=1$ until a nonzero derivative
is observed. Stable logarithmic norm calculations avoid intermediate overflow
in this policy. Factors do not grow once a column has been observed nonzero.
Scaling needs no additional model or derivative calls. LM updates it with
fresh derivatives after acceptance; hybrid keeps $C$ fixed during all Broyden
updates and recomputes it only with a fresh application Jacobian. A periodic
QR refactorization of the approximate model does not change $C$.

Balancing columns changes the LM damping metric and the hybrid dogleg path;
it does not make every Jacobian well conditioned or guarantee convergence.
The fixed physical $S$ still governs every solver trial's physical cap,
reported scaled step size, small-step termination, and derivative probes.

Residual scaling is a separate choice. If physical equations are $R(x)=0$,
the application might supply $F(x)=W R(x)$ with fixed positive diagonal $W$.
That preserves the exact roots but changes the residual norm, merit landscape,
and stopping tolerance. The supplied Jacobian must then also include $W$.
PySolverView does not choose these residual weights automatically.

## 2. Newton and Gauss-Newton steps

For a nonsingular square Jacobian, the Newton step solves

$$
Jp_{\mathrm N}=-r
\quad\Longleftrightarrow\quad
Az_{\mathrm N}=-r.
$$

It would cancel the residual if the linearization were exact. When the model
is nonlinear or the Jacobian is nearly singular, that prediction can require
an excessively large displacement.

Gauss-Newton generalizes the linear solve to least squares:

$$
z_{\mathrm{GN}}\in\arg\min_z\frac12\|r+Az\|_2^2.
$$

For full column rank, its first-order condition is

$$
A^TAz_{\mathrm{GN}}=-A^Tr.
$$

This is a mathematical identity, not an instruction to form and invert $A^TA$.
Forming the normal matrix squares the 2-norm condition number for full column
rank. PySolverView instead uses a column-pivoted QR factorization,

$$
AP=QR,\qquad Q^TQ=I,\qquad
Rw=-Q^Tr,\qquad z_{\mathrm{GN}}=Pw,
$$

where $P$ permutes columns. Solving the triangular system gives the step.
For $m>n$, the part of $r$ perpendicular to the columns of $Q$ contributes a
constant to the least-squares objective and does not affect this solve.

**PySolverView:** a triangular reciprocal-condition estimate guards the QR path.
If the factor is unsuitable, an SVD supplies a solution of minimum norm with
respect to the numerical rank. Singular values at or below
$\epsilon_{\mathrm{mach}}\max(m,n)\sigma_{\max}$ are treated as zero for that
Gauss-Newton solve. This numerical-rank decision is an approximation for tiny
but mathematically nonzero singular values.

Both methods use $z_{\mathrm{GN}}$ as the numerical candidate when it fits
inside the numerical radius. The separate physical cap may still shorten it.
Their main difference appears when the numerical candidate needs restriction.

## 3. Levenberg-Marquardt: solve the constrained quadratic model

The trust-region formulation of LM solves

$$
\min_z q(z)\qquad\text{subject to}\qquad\|z\|_2\leq\Delta.
$$

Introduce a multiplier $\lambda\geq0$ using the Lagrangian

$$
\mathcal{L}(z,\lambda)
=q(z)+\frac{\lambda}{2}\bigl(\|z\|_2^2-\Delta^2\bigr).
$$

Because $B=A^TA$ is positive semidefinite, the optimality conditions are

$$
(A^TA+\lambda I)z=-A^Tr,\qquad
\lambda\geq0,\qquad
\|z\|_2\leq\Delta,\qquad
\lambda\bigl(\|z\|_2^2-\Delta^2\bigr)=0.
$$

If a minimum-norm Gauss-Newton step fits, $\lambda=0$ is sufficient.
Otherwise choose $\lambda>0$ so that $\|z(\lambda)\|_2=\Delta$.
For positive $\lambda$, the shifted matrix is positive definite even when
$A$ has deficient rank.

These equations also minimize the penalized model

$$
\frac12\|r+Az\|_2^2+\frac{\lambda}{2}\|z\|_2^2.
$$

In trust-region LM, $\lambda$ is chosen to enforce the radius; it is not a
user-selected fixed regularization parameter. Its effect is directional:

$$
z(\lambda)\longrightarrow z_{\mathrm{GN}}
\quad\text{as }\lambda\downarrow0,
\qquad
z(\lambda)\approx-\frac{g}{\lambda}
\quad\text{for large }\lambda.
$$

The first limit uses the minimum-norm solution in exact arithmetic when rank
is deficient. Damping moves the step toward a scaled steepest-descent
direction; it does not merely multiply a Newton step by a scalar.

In physical coordinates, the corresponding equation is

$$
\bigl(J^TJ+\lambda (SC)^{-2}\bigr)p=-J^Tr.
$$

Thus the combined numerical scales $SC$ determine the relative damping penalty
on motion in each variable. In the large-damping limit,
$p\approx-\lambda^{-1}(SC)^2J^Tr$, which generally differs from an unscaled
steepest-descent step in physical coordinates. This equation describes the
numerical LM candidate before the independent physical-distance safeguard.

If $\|Cz\|_2>\Delta$, the custom driver replaces the candidate by a shorter
point along the same ray. In exact arithmetic this retains positive model
reduction for a descent candidate. It is generally **not** the exact minimizer
of $q$ over the intersection $\|z\|_2\leq\Delta$, $\|Cz\|_2\leq\Delta$.
The LM optimality equations above apply to the numerical subproblem, not
necessarily to the final physically clipped trial.

### How PySolverView computes a damped step

For the thin SVD $A=U\Sigma V^T$, put $b=U^Tr$. Then

$$
z(\lambda)=-Vd(\lambda),\qquad
d_i(\lambda)=\frac{\sigma_i b_i}{\sigma_i^2+\lambda}.
$$

Since the columns of $V$ are orthonormal,

$$
h(\lambda)=\|z(\lambda)\|_2=\|d(\lambda)\|_2
=\left(\sum_i
\frac{\sigma_i^2b_i^2}{(\sigma_i^2+\lambda)^2}\right)^{1/2}.
$$

When $h>0$, its derivative is

$$
h'(\lambda)
=-\frac{1}{h(\lambda)}
\sum_i\frac{d_i(\lambda)^2}{\sigma_i^2+\lambda}.
$$

The step length decreases as damping increases. PySolverView brackets a solution of
$h(\lambda)-\Delta=0$, uses safeguarded Newton iterations, and retains a
feasible bracket endpoint. If a Newton proposal leaves the bracket, it uses
its midpoint. A previously computed damping value can initialize a retry.

The code normalizes singular values by $\sigma_{\max}$ and searches in
$\mu=\lambda/\sigma_{\max}^2$ to improve numerical scaling. With
$a_i=\sigma_i/\sigma_{\max}$ and $e_i=a_ib_i/\sigma_{\max}$, it evaluates
$d_i=e_i/(a_i^2+\mu)$. The length calculation therefore needs only scalar
spectral coefficients during the search; multiplication by $V$ is needed
once for the final step.

PySolverView stops the search when it has a feasible length within a relative
$10^{-10}$ of the radius, or a numerical/iteration safeguard stops progress.
It returns the feasible candidate, which can be shorter if the search stops
early. A final rounding correction enforces the cap. These are finite-precision
approximations to the ideal optimality conditions above.

The SVD is lazy: a well-conditioned Gauss-Newton step that fits the radius
needs only QR. Damped retries at the same model reuse the SVD. This is not
MINPACK's numerical implementation: its [LMPAR](https://netlib.org/minpack/lmpar.f)
works from pivoted QR factors and permits approximately 10% tolerance in its
radius equation. See the [backend note](02_implementation_and_backend_boundaries.md)
for the distinction.

## 4. Powell's hybrid method: the dogleg step

The hybrid method combines a Newton-like step with a descent direction.
For a square nonsingular fresh Jacobian, its Gauss-Newton point is the Newton
point. When the Jacobian model has been updated approximately, it is a
quasi-Newton point.

Unlike LM, the dogleg step generally approximates the constrained quadratic
minimizer. It searches a particular broken line:

$$
0\longrightarrow z_{\mathrm C}\longrightarrow z_{\mathrm{GN}}.
$$

The first segment follows the negative model gradient. Along $z=-\alpha g$,

$$
q(-\alpha g)
=q(0)-\alpha g^Tg+\frac{\alpha^2}{2}\|Ag\|_2^2.
$$

Minimizing this one-dimensional quadratic gives the unrestricted Cauchy point

$$
\alpha_{\mathrm C}=\frac{g^Tg}{\|Ag\|_2^2},\qquad
z_{\mathrm C}=-\alpha_{\mathrm C}g.
$$

For $g\ne0$ with $g=A^Tr$, the denominator is positive in exact arithmetic.
The code also handles numerical degeneracies. The step has three cases:

| Condition | Dogleg step |
| --- | --- |
| $\lVert z_{\mathrm{GN}}\rVert_2\leq\Delta$ | $z=z_{\mathrm{GN}}$ |
| Gauss-Newton is outside and $\lVert z_{\mathrm C}\rVert_2\geq\Delta$ | $z=-\Delta g/\lVert g\rVert_2$ |
| Cauchy is inside and Gauss-Newton is outside | $z=z_{\mathrm C}+\tau(z_{\mathrm{GN}}-z_{\mathrm C})$, with $0<\tau<1$ chosen on the boundary |

For the last case, let $v=z_{\mathrm{GN}}-z_{\mathrm C}$. Expanding the
boundary condition gives

$$
(v^Tv)\tau^2+2(z_{\mathrm C}^Tv)\tau+
\|z_{\mathrm C}\|_2^2-\Delta^2=0.
$$

The positive root selects the intersection on the Cauchy-to-Gauss-Newton
segment. PySolverView computes an equivalent normalized expression. Define

$$
u=\frac{v}{\|v\|_2},\qquad
w=\frac{z_{\mathrm C}}{\Delta},\qquad
\beta=w^Tu,\qquad \gamma=1-w^Tw.
$$

Writing $z=z_{\mathrm C}+\Delta t u$, the boundary equation becomes
$t^2+2\beta t-\gamma=0$. Its nonnegative root is evaluated as

$$
t=
\begin{cases}
\displaystyle\frac{\gamma}{\sqrt{\beta^2+\gamma}+\beta},
&\beta\geq0,\\
\sqrt{\beta^2+\gamma}-\beta,
&\beta<0.
\end{cases}
$$

The first branch avoids subtracting nearly equal numbers. The implementation
also scales the vectors before forming their difference. The boundary step
is obtained directly from this geometric calculation.
MINPACK's [DOGLEG routine](https://netlib.org/minpack/dogleg.f) implements the
same broad gradient/Gauss-Newton construction, with its own arithmetic and
scaling details.

### A two-variable picture

![LM and dogleg steps for one local least-squares model](../results_step_geometry/lm_hybrid_step_geometry.png)

The figure uses $r=(3,3)^T$, $A=\operatorname{diag}(1,3)$, and $\Delta=1.5$.
Here $A$ is an already specified numerical model; the illustration has no
additional physical clipping. It isolates the two numerical subproblems.
The unconstrained solution is $z_{\mathrm{GN}}=(-3,-1)^T$, outside the circle.
The Cauchy point is $z_{\mathrm C}=(-15/41,-45/41)^T$, inside it.
The hybrid step follows the second dogleg segment to the circle. LM follows a
different damping curve to the constrained quadratic minimum. Both obey the
same radius, while their directions differ.

| Step | $z_1$ | $z_2$ | $\lVert z\rVert_2$ | $\lVert r+Az\rVert_2$ |
| --- | ---: | ---: | ---: | ---: |
| Gauss-Newton | -3.000000 | -1.000000 | 3.162278 | 0 |
| Cauchy | -0.365854 | -1.097561 | 1.156931 | 2.650357 |
| Dogleg | -1.048930 | -1.072262 | 1.500000 | 1.963077 |
| LM | -1.227737 | -0.861778 | 1.500000 | 1.820127 |

This is a local linear-model illustration, not an additional nonlinear
convergence benchmark. For this model LM gives the lowest attainable $q$
inside the circle, with $\lambda\approx1.443520$. That does not imply it will
give the larger reduction in the actual nonlinear residual, for which the
model is only approximate.
The [SVG](../results_step_geometry/lm_hybrid_step_geometry.svg) is suitable for export.
The [plotting runner](../run_step_geometry.py) reconstructs the figure
and independently checks the displayed steps against PySolverView.

## 5. Why hybrid also needs a Jacobian update

The dogleg path determines a trial step. Reusing derivative information is a
separate part of the hybrid algorithm. After a valid trial, define

$$
z=(SC)^{-1}(x_{\mathrm{trial}}-x_{\mathrm{base}}),\qquad
y=F(x_{\mathrm{trial}})-F(x_{\mathrm{base}}).
$$

We want the next Jacobian model to reproduce the observed residual change:
$A_{\mathrm{new}}z=y$. The rank-one Broyden update is

$$
A_{\mathrm{new}}
=A+\frac{(y-Az)z^T}{z^Tz}.
$$

Multiplying by $z$ verifies the secant condition immediately. Among all
corrections $E$ satisfying $Ez=y-Az$, this correction has minimum Frobenius
norm. It changes the model just enough, in that matrix norm, to match this
one observed displacement. This choice depends on the current scaling $SC$.
Both $S$ and $C$ stay fixed while this approximate model is updated. A new
numerical metric is introduced only alongside a freshly evaluated Jacobian,
with new factors built in that metric. The secant step uses the actual
rounded physical coordinates after any radial clipping.

A rejected but valid trial still contains derivative information. PySolverView may use
its secant pair without accepting its location. The residual anchoring the
next model remains $F(x_{\mathrm{base}})$ on rejection and changes to
$F(x_{\mathrm{trial}})$ on acceptance. An undefined trial provides no usable
residual difference and therefore no secant update.

To avoid refactorizing the entire matrix after every update, write

$$
a=\frac{y-Az}{\|z\|_2},\qquad
b=\frac{z}{\|z\|_2},\qquad A_{\mathrm{new}}=A+ab^T.
$$

If $AP=QR$, then

$$
A_{\mathrm{new}}P=QR+a(P^Tb)^T.
$$

Thus the QR-update routine must receive the permuted right-hand vector.
This is `direction[self.perm]` in the code. SciPy's public
[qr_update](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr_update.html)
updates factors for precisely this rank-one matrix operation.

For square dense systems, factor updates take quadratic work rather than a
fresh cubic factorization. **PySolverView** refactorizes the current approximate matrix
every 25 successful model updates to limit factorization drift. This does
not obtain new derivatives. A fresh application Jacobian is requested
instead after two consecutive valid trials with agreement ratio below 0.1,
an unhealthy update, or a suspect approximate model before a stagnation or
stationarity exit. The fixed 0.1 refresh threshold is separate from the
configurable acceptance threshold.

LM uses a fresh Jacobian after an accepted move and reuses the current one
on rejected retries. Hybrid's secant reuse can therefore save property
derivatives, at the cost of sometimes needing to refresh an inaccurate model.

## 6. Accepting a trial and changing the radius

Neither a good local-model step nor a radius bound guarantees that the true
residual decreases. Once a finite trial residual $r_{\mathrm t}$ is available,
compute the predicted and actual reductions:

$$
\operatorname{pred}
=q(0)-q(z)
=-r^TAz-\frac12\|Az\|_2^2,
$$

$$
\operatorname{ared}
=\frac12\|r\|_2^2-\frac12\|r_{\mathrm t}\|_2^2,\qquad
\rho=\frac{\operatorname{ared}}{\operatorname{pred}}.
$$

The denominator must be positive. A $\rho$ near one means the local prediction
was accurate; a negative $\rho$ means the trial increased the residual norm.
The merit function uses the actual residual even when $A$ is an approximation.

**PySolverView** accepts a valid trial only if

$$
\operatorname{ared}>0
\quad\text{and}\quad
\rho\geq\eta,
\qquad \eta=\texttt{acceptance\_ratio}.
$$

Therefore every accepted move strictly decreases $\|F\|_2$; the sequence of
all trial evaluations need not decrease. The default is $\eta=0.1$.
The next radius follows

$$
\Delta_{\mathrm{next}}=
\begin{cases}
\beta_{\mathrm s}\Delta,
&\text{undefined trial or }\rho<0.25,\\
\min(\Delta_{\max},\beta_{\mathrm g}\Delta),
&\rho\geq0.75\text{ and }\max(\|z\|_2,\|Cz\|_2)\geq0.8\Delta,\\
\Delta,
&\text{otherwise}.
\end{cases}
$$

Here $\beta_{\mathrm s}=\texttt{shrink\_factor}$ and
$\beta_{\mathrm g}=\texttt{growth\_factor}$. The defaults are 0.25 and 2.
Acceptance and radius adjustment are distinct decisions: with the default
settings, a trial with $0.1\leq\rho<0.25$ is accepted but the next radius
shrinks. A very accurate interior step does not automatically expand the
radius, because neither boundary restricted it. Either an active numerical
boundary or an active physical boundary can justify radius growth.

Before evaluation, PySolverView checks the actual displacement after conversion back to
physical coordinates and floating-point addition. The predicted reduction
and secant update use the actual numerical displacement
$z=(SC)^{-1}(x_{\mathrm{trial}}-x_{\mathrm{base}})$. The driver audits both
physical and numerical norms, permitting only
rounding-scale radius excess, bounded by eight machine epsilons in relative
scaled norm, and reports stagnation if no suitable movement is representable.

These thresholds are the PySolverView policy. Sharing the same algorithm family with
MINPACK does not imply identical acceptance rules, radius updates, or stopping
behavior.

## 7. What the controls mean for thermodynamic equations

`initial_radius` sets $\Delta_0$ independently of the magnitude of $x_0$.
`max_radius` imposes $\Delta\leq\Delta_{\max}$ on every solver trial, including
rejected or undefined ones. Together they imply

$$
|p_i|\leq s_i\Delta\leq s_i\Delta_{\max}.
$$

For example, scales of 20 K and 100,000 Pa with $\Delta_0=0.1$ limit the
first trial to at most 2 K in temperature and 10,000 Pa in pressure. The
joint Euclidean constraint is stronger than those component bounds: both
maxima cannot be used simultaneously. With many cells, simultaneous motion
in many coordinates consumes the same shared radius.

A small radius makes the local model more plausible, but it is not a physical
domain constraint. Near a property boundary, even a tiny trial can fail.
For a differentiable residual at an interior valid point with nonzero true
gradient, a sufficiently short true descent step reduces the merit function
by Taylor's theorem. A secant direction, noisy derivative, stationary
non-root, or domain boundary can invalidate that argument.

Consequently the useful operational requirement is: bound the **first trial**
before calling the property model, and require the **first accepted move**
to reduce the residual. A failed evaluation supplies no merit ratio.
PySolverView rejects recognized domain failures, shrinks the radius, and retries from the accepted
base without inventing a residual value. The budgets still apply.

Good, sufficiently long steps can gradually restore the radius through
$\beta_{\mathrm g}$, up to the persistent cap. Setting
$\beta_{\mathrm g}=1$ prevents regrowth even after a failure-induced shrink.
These controls do not cover finite-difference derivative probes: without an
analytic Jacobian, PySolverView uses the separate `finite_diff_step` distance, and
undefined derivative probes terminate the solve. See the
[thermodynamic policy note](01_thermodynamic_step_policy.md) for practical
scaling and domain-adapter choices.

## 8. Stopping and the connection to the implementation

PySolverView declares root success only when

$$
\|F(x)\|_2\leq\texttt{residual\_tol}.
$$

If this is not satisfied, a fresh-model numerical gradient
$\|A^Tr\|_\infty$ below `gtol`, an unrepresentable or sufficiently small step,
or an exhausted budget is a failure diagnostic. Jacobian scaling changes this
gradient metric as well as the numerical trust geometry.
The scaled small-step test is

$$
\|u\|_2=\|S^{-1}p\|_2\leq
\texttt{xtol}\bigl(\texttt{xtol}+\|S^{-1}x\|_2\bigr).
$$

For an approximate hybrid model, the driver first tries fresh derivatives
before concluding model stationarity or step stagnation. Neither this
refresh nor a monotone residual history guarantees eventual convergence.

| Mathematical operation | PySolverView implementation |
| --- | --- |
| Physical Jacobian coordinates $A_0=JS$ | `_solve_trust_region`: `jacobian * scale` |
| Numerical factors $C$ and Jacobian $A=A_0C$ | `JacobianScaling.update` and `_solve_trust_region` |
| QR or numerical-rank Gauss-Newton solve | `LinearModel.gauss_newton` |
| Damped LM radius equation | `LinearModel._lm` |
| Cauchy point and sphere intersection | `LinearModel._dogleg` |
| Broyden correction and residual anchoring | `_update_powell_jacobian` after a valid trial |
| Maintained QR and conditioning checks | `LinearModel.update` |
| Physical/numerical distance audit after rounding | `prepare_equilibrated_trial` |
| Actual/predicted reduction, acceptance, radius | `_solve_trust_region` |
| Root-only success and failure diagnostics | `_solve_trust_region` |

The [linear-model tests](../tests/test_linear_model.py) check LM steps against an
independent scalar root solve and optimality residuals, dogleg geometry,
rank-deficient models, and updated QR factors. The
[driver tests](../tests/test_solver.py) check actual trial distances, accepted-state
consistency, failures, and stopping. The [experiment report](../results_convergence/REPORT.md)
shows what these methods did on the multidimensional problems; mathematical
structure alone does not establish equivalence to MINPACK or reliability on
all thermodynamic models.
