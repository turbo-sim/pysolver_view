# Understanding the KKT reports

## Optional checks and timing

Use `OptimizationSolver(..., check_kkt=False)` to disable automatic KKT work;
the default is `True`. This option belongs directly on the solver, not inside
`extra_options`. For direct `minimize_slsqp` calls, use
`SLSQPOptions(check_kkt=False)` instead.

With checks disabled, no automatic multiplier estimation runs, history optimality
is unavailable, and `result.kkt_passed` is `None`. Custom SLSQP still requires
native success and feasibility, but does not enforce the independent optimality
gate. A tiny controlled step may therefore produce native success without
stationarity. Native SciPy's success semantics remain native in both modes.

Explicit `evaluate_kkt_conditions` or requested KKT/multiplier reports still
perform an on-demand audit. This permits a cheap solve followed by one final
assessment. Such an assessment does not retrospectively change `result.success`;
inspect its returned `passed` flag explicitly.

Time a longer constrained Rosenbrock solve with and without automatic checks:

```powershell
python demos/demo_kkt_conditions/run_kkt_timing.py
```

The demo uses 50 warmed, alternating pairs per backend, verifies the analytical
solution, prints a timing table, and plots total and assessor time. Use
`--repeats 100`, `--no-show`, or `--output PATH` as needed.
Saved [results](../../development/slsqp_controls/results_kkt_timing/REPORT.md)
include raw samples, initial runs, environment versions, and paired overhead.
Printing, plotting, construction, and post-solve audits are excluded. History and
result bookkeeping remain included. These are solve-time measurements, not
isolated native linear-algebra timings.

`result.kkt_time` is the time in numerical KKT assessment during the solve, in
seconds; `result.kkt_evaluations` counts those assessments. Disabled checks report
zero for both. Explicit audits after solving may increase the solver's diagnostic
counters; the result's timing fields remain snapshots of solve-time work.

The timing problem minimizes $100(y-x^2)^2+(1-x)^2$ subject to $x+y\leq1$,
starting at $(-1.2,1)$. Its known boundary optimum is approximately
$(0.618796,0.381204)$ with objective $0.145607$ and an active inequality.
The script derives the reference solution from the cubic stationarity equation.
It requires at least 10 native iterations; the recorded solve takes **21**.

The comparison uses full first/max fractions, no growth ramp, no domain
recovery, and native `ftol=1e-12`. It asserts matching model-evaluation
trajectories, native iterations, and model-call counts across all four
configurations. Callback histories have different semantics across backends,
so actual model points define trajectory matching. The custom KKT run may need
one additional final Jacobian, which is counted and included in timing.
Gradient observations ensure SciPy's enabled run performs actual assessments;
the script asserts that enabled runs have nonzero KKT work.

On the updated recorded run, median total times were **2.25/30.66 ms** for
SciPy checks off/on and **3.57/34.48 ms** for custom SLSQP. All configurations
used 21 iterations and 30 model evaluations. SciPy performed 21 Jacobian
calls and 21 KKT assessments; custom performed 21 Jacobian calls without KKT
and 22 with KKT, with 22 assessments. The console uses aligned ASCII tables
and includes iteration counts. This cheap model emphasizes diagnostic overhead;
percentages need not transfer to costly physical models.


## Named problem and reports

Start with the named example:

```powershell
python demos/demo_kkt_conditions/demo_named_kkt.py
```

It runs SciPy SLSQP and the controlled `library="pysolver"` SLSQP backend on the
same three-variable problem. It prints design variables, constraints, KKT
residuals, and multipliers, and checks them against an analytic solution. Each
clearly labeled run prints its convergence history once, followed by its final
report. Similar final tables reflect the same solution from two different
backends, not duplicated printing. The controlled backend requires the SciPy
version supported by its private SLSQP bridge; other backends remain independent.

At the end, a comparison figure shows objective and maximum constraint violation
against total model calls, including derivative and diagnostic evaluations.
SciPy history records major-iteration points; custom history records accepted
points. The feasibility axis uses a symmetric logarithmic scale to retain exact
zeros. The two deliberately bad designs are separate diagnostic examples and
are not added to these histories. No files are saved automatically.

## Read the example as a sequence of checks

1. Read the problem definition and expected solution printed at startup.
2. Check the named design and constraint tables: both solvers should return
   `stream_a=1.2`, `stream_b=0.8`, and `reserve=0`, with objective `0.84`.
3. Check each KKT row rather than only the optimizer's success flag.
4. Read the multiplier table: `total_delivery` has multiplier `0.2`,
   `stream_a_capacity` has multiplier `0.6`, and the reserve lower bound has
   multiplier `1`. The inactive restrictions have zero multipliers.
5. Inspect the two deliberately bad designs. A feasible design can fail
   stationarity; a negative reserve fails primal feasibility even though the
   nonlinear constraints are satisfied and its objective is smaller.

The script raises an assertion if these independent expectations are not met.
Its diagnostic evaluations leave the optimized design and history intact.

## What is being solved?

The variables are $a$ (`stream_a`), $b$ (`stream_b`), and $r$ (`reserve`):

$$
\min_{a,b,r}\ \frac12\left((a-2)^2+(b-1)^2+(r+1)^2\right)
$$

subject to

$$
a+b-2=0,\qquad a-1.2\leq0,\qquad b-2.5\leq0,
\qquad 0\leq a,b,r\leq3.
$$

At $(1.2,0.8,0)$, the objective gradient is $(-0.8,-0.2,1)$.
Using the public Lagrangian convention

$$
L=f+\lambda h+\mu^T g-z_L^T(x-l)+z_U^T(x-u),
\qquad \mu,z_L,z_U\geq0,
$$

the stationarity equation is

$$
(-0.8,-0.2,1)+0.2(1,1,0)+0.6(1,0,0)-(0,0,1)=0.
$$

Feasibility, multiplier signs, and complementarity also hold. This particular
problem is strictly convex with affine constraints, so its solution is the
unique global minimum. For general nonlinear problems, passing first-order KKT
checks alone does **not** establish a local or global minimum. Derivative
accuracy and constraint qualifications also matter.

## What each diagnostic means

| Diagnostic | Interpretation |
| --- | --- |
| Stationarity | Infinity norm of the reconstructed Lagrangian gradient. A feasible point can still fail this check. |
| Primal feasibility | Largest equality, positive inequality, or finite-bound violation. |
| Dual feasibility | Largest negative part of inequality and bound multipliers; equality multipliers may have either sign. |
| Complementarity | Largest absolute multiplier-times-slack product. An inactive restriction must have zero multiplier at an exact KKT point. |
| Availability | Whether valid diagnostics could be computed. Unavailable is not a pass. |

The legacy text report retains its signed dual-feasibility presentation: it
shows the minimum inequality/bound multiplier and compares it with a negative
tolerance. The shared numerical assessment instead exposes the nonnegative
`dual_feasibility` residual described above. These checks are equivalent, but
their displayed numbers have different meanings. The report's inequality
feasibility row also includes finite-bound violations.

The generic checker estimates multipliers independently at the supplied point.
These are distinct from the optimizer's native QP multipliers. Tolerances apply
in the supplied objective, constraint, and variable scaling. The demo avoids
scaling to make its analytic values directly comparable to the report.

SciPy's native termination status and a KKT assessment answer different
questions. The controlled SLSQP backend requires its independent feasibility
and optimality criteria as well as native success. For a SciPy solve, assess
KKT explicitly when your application requires those conditions.

## Named variables and constraints

The existing problem/solver interfaces support named reports through
`problem.variable_names` and `problem.constraint_data`. Numerical solvers still
consume and return arrays. `NamedAllocationProblem.pack()` and `unpack()` show
the small application adapter needed to use dictionaries safely:

```python
initial_design = {"reserve": 0.6, "stream_b": 1.5, "stream_a": 0.5}
x0 = problem.pack(initial_design)
solver.solve(x0)
named_solution = problem.unpack(solver.x_final)
```

Always pack by the declared name order, not by an incoming dictionary's order.
The example rejects missing or unexpected keys. This is adequate for named
engineering models without requiring a dictionary-aware numerical backend.

Constraint metadata is refreshed inside `fitness()` at the evaluated point.
The numerical array contains residuals in the order
`[objective, equalities..., inequalities...]`, with inequalities nonpositive.
The report metadata carries descriptive names plus physical values and targets;
`normalized_mismatch` connects those entries to the numerical residuals.
Equality and inequality names must follow the corresponding numerical row order.

## Other files

`demo_print_kkt.py` demonstrates report formatting on HS71, and
`demo_print_report.py` demonstrates reports with variable scaling. The named
example above is the recommended starting point for understanding the checks.
Some older exploratory scripts use optional optimization backends.
