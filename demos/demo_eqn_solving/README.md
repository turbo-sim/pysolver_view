# Equation-solving demos

Start with [demo_radiative_heat_collocation.py](demo_radiative_heat_collocation.py)
for a nonlinear boundary-value problem with many coupled unknowns. The shorter
[demo_custom_root_solvers.py](demo_custom_root_solvers.py) introduces the public
custom root interface on algebraic systems. The original
[demo_nonlinear_system.py](demo_nonlinear_system.py) compares equation solving
with an equality-constrained optimization formulation.
The [radiative SLSQP comparison](demo_radiative_heat_slsqp.py) applies the general
root-to-optimization adapter to the existing collocation problem.

## Radiative heat: Newton and constrained SLSQP

Run the eight-solve comparison with:

```powershell
python demos/demo_eqn_solving/demo_radiative_heat_slsqp.py
python demos/demo_eqn_solving/demo_radiative_heat_slsqp.py --no-show

# Smaller spatial problem for a quick check.
python demos/demo_eqn_solving/demo_radiative_heat_slsqp.py --degree 48 --no-show
```

The new script imports `RadiativeHeatProblem`, `NUM_POINTS` and the independent
reference solve from `demo_radiative_heat_collocation.py`. It reuses the physical
equations, spatial discretization, initial guess and analytic Jacobian. The
default remains 500 nodes and 1000 unknowns, starting from uniform temperature
0.3 and zero flux.

The comparison runs both Newton-Raphson and custom SLSQP with initial direction
fractions 1.0 (full), 0.3, 0.2 and 0.1. Both methods grow their direction caps
by `step_growth_factor=1.5` toward 1.0 after accepted moves. These are initial
fractions, not fixed damping factors, absolute step radii or per-node bounds.
Newton uses `backtracking=None` for all four runs and can accept valid steps
that increase the residual. SLSQP retains its native merit line search and
custom domain recovery: its full-step setting starts with a cap of 1.0 but
can still shorten a proposed direction.

Matching fractions use the same color in every plot. Newton uses solid lines
and circles; SLSQP uses dashed lines and squares, with white marker faces.

The SLSQP problem is created with the installed general adapter:

```python
optimization_problem = psv.as_optimization_problem(
    problem, n_variables=initial.size
)
```

The adapter supplies objective zero and the original residual vector as
equality constraints. It performs no model evaluation during conversion and
does not modify the root problem. The analytic optimization Jacobian consists
of a zero objective row followed by the original residual Jacobian. No extra
physical bounds or residual scaling are introduced in this comparison.

Each final state is checked against the original residual two-norm, with default
target $10^{-8}$, and the independent adaptive boundary-value solution.
Optimizer success and root acceptance are reported separately: the objective
is always zero, so its value or stationarity cannot establish a root. The
optimizer's maximum constraint violation also differs from the residual
two-norm used for the common root check.

Convergence plots show accepted residual two-norms against accepted-move index
and cumulative model calls, together with accepted step sizes. Model-call totals
include work between accepted points; the curves do not reconstruct every
rejected trial. SLSQP's native major-iteration count is retained separately from
the accepted-move count. JSON retains accepted states, root trial records and
SLSQP events. Reference solves and explicit post-solve checks are outside the
reported solver call counts. The scalar objective is not used as a convergence
curve because it contains no progress information here.

Results are saved beside the new script in `results_radiative_heat_slsqp/`:
`summary.csv`, `results.json`, `solution.png/.svg` and `convergence.png/.svg`.
Use `--output PATH` to choose another directory. The CLI also exposes
`--degree`, `--initial-temperature`, `--heating`, `--radiation`, `--tolerance`
and `--max-iterations`. Changing these settings can expose failed solves, which
remain visible in the results. Dense SLSQP on the default 1000-variable system
can be substantially more expensive than the direct root solve.

## Nonlinear conduction with radiation

Run from the repository root:

```powershell
python demos/demo_eqn_solving/demo_radiative_heat_collocation.py

# Save the same figures and results without opening plot windows.
python demos/demo_eqn_solving/demo_radiative_heat_collocation.py --no-show
```

The demo compares **Newton-Raphson, Levenberg-Marquardt, and Powell hybrid**,
each with a full/large initial step and with an initial scaled step limit of
**0.1**. All six use `NonlinearSystemSolver(library="pysolver")`, the same
uniform cold initial guess, the same residual equations, and the same analytic
Jacobian. The problem derives from `NonlinearSystemProblem`. These are the custom
root implementations in the package; they do not call `scipy.optimize.minimize`
or wrap MINPACK.

The editable `NUM_POINTS` global currently selects **500 spatial nodes and
1000 nonlinear unknowns**. The `--degree` option overrides the polynomial
degree, so `--degree 48` gives a smaller 49-node example.
The coordinate is one-dimensional; the algebraic root problem is not scalar.

### Physical model

Consider a thin rod with constant thermal conductivity, uniform internal heat
generation, and radiative exchange with a surrounding enclosure. Both rod ends
are held at the enclosure temperature. With constant area $A$ and perimeter $P$,
the dimensional steady balance is

$$
kA\frac{d^2 T_{\mathrm{dim}}}{dx_{\mathrm{dim}}^2}
+\dot q'''A-\epsilon\sigma P
\left(T_{\mathrm{dim}}^4-T_\infty^4\right)=0.
$$

Write $x=x_{\mathrm{dim}}/L$ and $T=T_{\mathrm{dim}}/T_\infty$. The implemented
dimensionless problem is

$$
T''+Q-R(T^4-1)=0,\qquad 0<x<1,\qquad T(0)=T(1)=1,
$$

with

$$
Q=\frac{\dot q'''L^2}{kT_\infty},\qquad
R=\frac{\epsilon\sigma P L^2T_\infty^3}{kA}.
$$

The defaults are $Q=50$ and $R=10$. Conductivity is constant: **radiation supplies
the nonlinearity**. The net radiation term removes heat for $T>1$ and supplies
heat for $T<1$. This is a distributed exchange model, not a radiating boundary
condition or a full radiative-transfer calculation. Nonpositive absolute
temperatures raise `DomainError`.

The initial guess is $T=0.3$ and zero conductive flux everywhere, including the
endpoints. It deliberately violates the boundary temperatures and energy
balance. All solvers receive this same guess; none receives a reference solution
or a preliminary warm-up solve.

### Collocation construction

The layout follows the local Nozzlex
[heat-equation example](../../../nozzle_modeling/nozzlex/examples/single_phase/heat_equation_collocation.py)
and its
[Chebyshev helpers](../../../nozzle_modeling/nozzlex/nozzlex/functions/nozzle_model_solver.py):
build the differentiation matrix once, pack nodal fields into one vector, and
include boundary equations in the square residual system. This demo implements
that pattern with NumPy; importing Nozzlex, JAX, or Optimistix is unnecessary.

A degree-$N$ interpolating polynomial uses $N+1$ Chebyshev-Lobatto nodes.
The matrix $D$ maps nodal values to nodal first derivatives. Define normalized
conductive flux $v=-T'/Q$ and pack $z=[T_0,\ldots,T_N,v_0,\ldots,v_N]$.
The equations are

$$
\begin{aligned}
(DT)_i/Q+v_i &=0 && i=0,\ldots,N,\\
(Dv)_i+(R/Q)(T_i^4-1)-1 &=0 && i=1,\ldots,N-1,\\
T_0-1&=0,\qquad T_N-1=0.
\end{aligned}
$$

There are $(N+1)+(N-1)+2=2(N+1)$ equations and unknowns. `gradient()` supplies
the exact Jacobian; in particular, the radiation block differentiates to
$\operatorname{diag}(4RT^3/Q)$ before selecting interior rows. The demo exposes
`residual()`, `gradient()`, `initial_guess()`, `unpack()`, and `interpolate()` so
the problem can also be used in other experiments.

### What the two step profiles mean

| Method | Full / large steps | Initial limit 0.1 |
|---|---|---|
| Newton | `backtracking=None`: full Newton steps, even when the residual increases. | Default Armijo backtracking, with `step_limits.initial_limit=0.1`. |
| LM and hybrid | `initial_radius=100`, allowing large initial steps. | `initial_radius=0.1`. |

LM and hybrid intrinsically use a trust region. **Their baseline retains
acceptance tests, rejection, and radius recovery**; it is not an algorithm with
all step management disabled. Radius 100 can still restrict the initial direction.
At 500 nodes, the full first Newton direction has physical scaled norm about
104.8; LM and hybrid also apply their numerical trust-region geometry.

All other controls use their defaults: maximum limit/radius 100 and growth
factor 2, plus each method's default recovery and acceptance settings. The
residual target is $\|F\|_2\leq10^{-8}$ and the budget is 500 trial attempts.

With `x_scale=1`, the bound applies to $\|\Delta z\|_2$ for the entire
dimensionless temperature/flux vector. It is **not a 10% Newton-step fraction**
and not a separate allowance of 0.1 at every node. Its interpretation changes
with mesh size or variable scaling. The first attempted step may be smaller
than 0.1 if the algorithm needs it. Later limits can grow, and rejected trials
can still increase the residual even when accepted iterates decrease it.

### Jacobian-based numerical scaling

All four custom root methods now default to `jacobian_scaling=True`. The solver
uses Jacobian column norms to balance the variables numerically, updating the
factors when it computes a fresh Jacobian. This changes LM and hybrid directions
and their trust-region geometry. Hybrid holds the factors fixed during Broyden
updates. Full-rank Newton and Gauss-Newton directions are mathematically
unchanged; finite-precision results may differ.

Numerical scaling is separate from the caller's `x_scale`: the physical
displacement cap, finite-difference probes, and residual convergence test retain
their definitions. LM/hybrid shorten the numerical trial along its direction
when necessary to respect the physical cap as well. This extra clipping does
not claim to solve the intersection of both trust-region constraints exactly.
No residual rows or boundary equations are reweighted.

To disable it through the public interface, pass
`options={"jacobian_scaling": False}`. To compare this demo directly:

```powershell
python demos/demo_eqn_solving/demo_radiative_heat_collocation.py --no-show
python demos/demo_eqn_solving/demo_radiative_heat_collocation.py --no-show --no-jacobian-scaling --output results_heat_unscaled
```

The unscaled 500-node LM runs are expected to fail near the positive-temperature
domain boundary. See the [root solver mathematics](../../docs/development/root_trust_regions.md)
for the distinction between numerical scaling and physical step limits. Scaling restores convergence
on this example; it is not a general domain-safety guarantee.

### Reading the figures and output

`solution.png` shows temperature and normalized conductive flux, together with
the wrong initial guess and an independent reference. The six converged fields
overlap because they solve the same equations. The reference peak temperature
is approximately 1.56229 times ambient.

`convergence.png` shows four complementary histories:

- Residual norm at initial/accepted states versus accepted iterations.
- Residual norm at every model evaluation, including rejected trials.
- Actual scaled displacement of every trial, measured from its accepted base.
- Temperature error relative to the reference at accepted states.

The dotted lines denote the residual target and the initial step limit. Every
computed point has a marker. Methods use `plasma` colors sampled from 0.2 to 0.8;
Dashed lines with square markers denote the baseline; solid lines with circles
denote the controlled profile.
All solver markers have white faces and are drawn above all the connecting lines.
`set_plot_options()` supplies the common style. A symmetric log scale preserves
exact zeros, with a linear region below $10^{-14}$. Undefined evaluations, if
encountered in other parameter choices, are gaps rather than invented residuals;
their counts and reasons remain in `results.json`.

The console prints an ASCII table of success, accepted iterations, model calls,
Jacobian calls, rejected trials, first displacement, peak/final residuals, and
off-grid temperature error. Model calls are not relabeled as iterations.
Analytic derivatives avoid finite-difference probes in this comparison.

The recorded default run gives:

| Method | Full / large: iterations / calls | Initial 0.1: iterations / calls |
|---|---:|---:|
| Newton | 10 / 11 | 12 / 13 |
| LM | 13 / 19 | 12 / 13 |
| Powell hybrid | 41 / 54 | 26 / 30 |

All six converge with default numerical scaling. The initial residual norm is
approximately 26.8. Baseline peak norms are about 7057 (Newton), 5890 (LM), and
2534 (hybrid); all controlled runs stay below the initial norm at every
evaluation. Controlled LM reaches $4.5\times10^{-11}$ in 12 accepted iterations
and 13 calls, with no rejected trials. All controlled accepted histories are
monotonic in this run. **Small initial steps reduce the large excursions; they
do not universally reduce work.** Hybrid without numerical scaling already
converged in 20 iterations and 22 calls on this mesh; the new default is not a
speed improvement for every method. Exact counts can change with numerical
libraries.

### Independent validation and outputs

The reference is computed by
[SciPy `solve_bvp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html),
which uses adaptive piecewise fourth-order collocation. It solves in
temperature/slope coordinates, without the spectral matrix or the custom
drivers. Its requested relative residual tolerance is $10^{-10}$; that is not
the same metric as the custom algebraic residual tolerance. Reference work and
post-solve audits are excluded from each custom solver's counters.

Temperature and flux errors are checked on 1001 points, including positions
between spectral nodes. The default maximum temperature errors are below
$3\times10^{-10}$. Tests additionally check polynomial differentiation, the
analytic Jacobian against complex step, the exact nonradiating parabolic
solution, spatial refinement, positive-temperature bounds, and integrated
energy conservation. Run them with:

```powershell
python -m pytest tests/roots/test_radiative_heat_demo.py -q
```

Results are written to `results_radiative_heat/` beside the script:
`solution.png/.svg`, `convergence.png/.svg`, `summary.csv`, and `results.json`.
JSON includes resolved solver options, final numerical scaling factors, versions,
final fields, accepted
histories, all trials, and all model evaluations. Existing files with these
names are replaced on rerun. Generated results follow the repository's ignore
rules; the source and this explanation are sufficient to reproduce them.

For a quick physical or spatial sensitivity, change the exposed parameters:

```powershell
# Recover linear conduction by switching off radiation.
python demos/demo_eqn_solving/demo_radiative_heat_collocation.py --radiation 0

# Change spatial resolution or the common initial guess.
python demos/demo_eqn_solving/demo_radiative_heat_collocation.py --degree 32 --initial-temperature 0.5
```

Changing the problem may change convergence and spatial accuracy. The demo
reports failures honestly and plots the last accepted state with a failed label;
it does not switch algorithms or warm-start a failing run.
