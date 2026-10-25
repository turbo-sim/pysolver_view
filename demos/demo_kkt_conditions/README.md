# KKT demonstrations

This folder contains two runnable examples. Both use `set_plot_options()` and
ASCII console output. Custom SLSQP requires **SciPy 1.17.1**.

| Script | Purpose |
|---|---|
| [demo_named_kkt.py](demo_named_kkt.py) | Named variables/constraints, known solution and multipliers, reports, histories, and diagnosing bad designs. |
| [run_kkt_timing.py](run_kkt_timing.py) | Measure optional KKT overhead on a longer constrained Rosenbrock solve. |

Run commands from the repository root.

## Start with the named tutorial

```powershell
python demos/demo_kkt_conditions/demo_named_kkt.py
```

Two clearly labeled runs compare SciPy and custom SLSQP. Each prints progress
and a full variable, constraint, KKT, and multiplier report. The final figure
compares objective and feasibility against total model calls. Similar final
reports reflect the same solution from two backends.

The problem minimizes

$$
f=\tfrac12[(a-2)^2+(b-1)^2+(r+1)^2]
$$

subject to $a+b=2$, $a\leq1.2$, $b\leq2.5$, and $0\leq a,b,r\leq3$.
The unique minimum is $(1.2,0.8,0)$ with objective $0.84$. Expected multipliers
are $0.2$ for the equality, $0.6$ for the active capacity, and $1$ for the
reserve lower bound. Other inequality/bound multipliers are zero. The script
asserts these independently derived expectations.

The last section audits a feasible-but-nonoptimal design and a bound-infeasible
design. It confirms that diagnostics do not replace the optimized design or add
history rows. Custom history records accepted points; native SciPy callbacks
record major-iteration observations without guaranteed acceptance.

## Optional tutorial features

```powershell
# Solve without automatic checks, then explicitly audit the final solution.
python demos/demo_kkt_conditions/demo_named_kkt.py --final-only

# Solve in normalized coordinates and explain the changed bound multiplier.
python demos/demo_kkt_conditions/demo_named_kkt.py --scaled

# Save both complete reports in an explicit directory.
python demos/demo_kkt_conditions/demo_named_kkt.py --save-report output/kkt_reports

# Combine features, including headless execution.
python demos/demo_kkt_conditions/demo_named_kkt.py --scaled --final-only --no-show --save-report output/kkt_reports
```

No files are written unless `--save-report` is supplied. It creates
`scipy_kkt_report.txt` and `pysolver_kkt_report.txt` in the chosen directory,
replacing these files on a subsequent run. No logger is created automatically.

`--final-only` passes `check_kkt=False`. The solve performs no automatic
assessments, and `result.kkt_passed` remains `None`. The tutorial then explicitly
audits the final solution and checks the returned `passed` flag, without
retrospectively changing solver success. The two bad-design assessments remain
separate educational checks after solving. Direct low-level calls can use
`SLSQPOptions(check_kkt=False)`.

`--scaled` maps physical bounds $[0,3]$ to solver bounds $[0,1]$. The physical
solution is unchanged. The reserve bound multiplier becomes $3$ because its
bound residual uses solver coordinates. Equality and capacity residual units
are unchanged, so their multipliers remain $0.2$ and $0.6$. KKT residuals and
absolute tolerances must always be interpreted in their supplied scales.

## Read the assessment

Public inequalities use $g\leq0$ and

$$
L=f+\lambda^Th+\mu^Tg-z_L^T(x-l)+z_U^T(x-u),
\qquad \mu,z_L,z_U\geq0.
$$

| Check | Meaning |
|---|---|
| Stationarity | Infinity norm of the Lagrangian gradient. |
| Primal feasibility | Largest equality, inequality, or finite-bound violation. |
| Dual feasibility | Inequality/bound multipliers must be nonnegative; equality multipliers are free. |
| Complementarity | Multiplier-times-slack products must vanish. |
| Availability | Unavailable diagnostics are not a pass. |

The text report displays dual feasibility as the signed minimum multiplier;
the shared `dual_feasibility` field is its nonnegative violation. Estimated KKT
multipliers are distinct from native QP multipliers. First-order KKT is not a
general proof of a minimum. This tutorial's strictly convex objective and affine
constraints do give a unique global minimum.

Names come from `variable_names` and `constraint_data`. Numerical solvers use
arrays; `pack()` and `unpack()` map dictionaries using declared name order.
Constraint names follow their numerical row order within each type. Metadata
shows physical values/targets; the fitness vector contains numerical residuals.

## Measure overhead separately

```powershell
python demos/demo_kkt_conditions/run_kkt_timing.py
python demos/demo_kkt_conditions/run_kkt_timing.py --repeats 50 --no-show
```

This uses constrained Rosenbrock requiring 21 iterations in the recorded run.
Full fractions, no ramp, and disabled recovery give matching model trajectories
across both backends and KKT on/off settings. Tables show actual assessment
counts, any extra final Jacobian, total solve time, and assessor time. Native
SciPy and custom SLSQP have different automatic diagnostic schedules.

The benchmark alternates warmed pairs, excludes construction/printing/plotting
and explicit post-solve audits, and verifies the known solution. Histories and
result bookkeeping remain timed. `result.kkt_time` records solve-time assessor
seconds; `result.kkt_evaluations` counts assessments. Explicit later audits do
not change these result snapshots. Disabled checks report zero for both.

Use `--output PATH` to choose a results folder. The
[generated report](results_kkt_timing/REPORT.md)
contains measurements; samples, versions, and the timing plot accompany it.
Cheap mathematical models emphasize percentage overhead compared with expensive
physical property evaluations.

See [implementation/validation notes](../../development/kkt_slsqp_validation.md)
for details. Step-control sensitivity belongs in
[demo_slsqp_controls](../demo_slsqp_controls/README.md).
