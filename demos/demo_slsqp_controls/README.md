# Controlled SLSQP sensitivity demo

Run from the repository root using SciPy **1.17.1**:

```powershell
python demos/demo_slsqp_controls/run_sensitivity.py
```

Use `--no-show` to save figures without opening windows, or `--output PATH` to
choose another results directory. The script prints one summary per run, opens
two figures (one per Rosenbrock problem), and saves the complete data and sensitivity
figures in [results_sensitivity](results_sensitivity/REPORT.md).
It does not change the solvers or any Turbodash file.

## Problems

All models have **two variables**, exact analytical gradients, and explicit
`OptimizationProblem` definitions in the shared
[slsqp_controls.py](../../src/pysolver_view/optimization/problems/slsqp_controls.py).

1. Unconstrained Rosenbrock, $f=100(y-x^2)^2+(1-x)^2$, starting at $(-1.2,1)$.
   There are no artificial box bounds to hide its large initial QP direction.
   The known minimum is $(1,1)$ with $f=0$.
2. The same objective with $x+y\leq1$. Its solution is approximately
   $(0.618796,0.381204)$ with $f=0.145607$. The constraint is active with a positive
   multiplier; this makes the final KKT test more informative than an inactive
   constraint placed around the unconstrained minimum.
## Controls and comparisons

The script varies **only the first-step fraction**:

```python
first_step_fraction = [1.0, 0.1, 0.016, 0.005, 0.001]
step_growth_factor = 1.5
max_step_fraction = 1.0
domain_backtracking = False
```

The native SciPy reference has no custom step control. Six configurations run on
each of the two Rosenbrock problems, giving **12 runs**. No maximum-fraction
analysis or square-root solve is performed by this sensitivity script. The shared
problem definitions retain the square-root model for the independent maintenance
validation and existing recovery tests.

The selected fractions came from an exploratory 121-point logarithmic search
per problem over $10^{-6}$ to $0.05$, a coarse comparison up to 1, and local
refinement around the smooth constrained case. The logarithmic search record
is saved as `results_sensitivity/fraction_search.json`. This is an educational
selection of measured behaviors, not a uniformly spaced sweep or a claim that
smaller fractions always perform better.

Colors are sampled from **plasma between 0.2 and 0.8**, in the listed fraction
order; SciPy is black. Each figure shows all raw model attempts, the first 12
attempts, recorded iteration objectives, and first-order optimality convergence histories. `set_plot_options`
applies the package style. No history is replaced by a best-so-far envelope.

Native stopping is fixed at `ftol=1e-12`, the iteration budget is 400,
feasibility tolerance is `1e-6`, and optimality tolerance is `1e-4`.

## What to read

- Raw objectives include rejected trials. Large spikes here are real model
  evaluations even when the accepted iterates look smooth.
- Recorded iteration objectives mean accepted points for the custom driver.
  Native SciPy's major-iteration callback can report a trial that its line search
  subsequently rejects, so its observation stream has different semantics.
- Console, CSV, and Markdown report both raw objective increases and recorded
  increases, together with peak objective divided by the initial objective.
  Monotonicity ignores only changes at or below
  $10^{-10}\max(1,|f_{\mathrm{previous}}|)$; raw data is preserved.
- The bottom-right panel shows first-order optimality versus model calls, with
  a target of `1e-4`. The measure is the maximum of stationarity, dual violation,
  and complementarity. Every recorded point is assessed after the solve using
  the same mathematical definition for both backends; this work adds no solver
  model calls or iterations. Unavailable diagnostics remain gaps. Final primal
  feasibility and the combined KKT ratio remain in JSON/CSV/Markdown.
- Every objective and optimality curve has a marker at every point. Fractions
  use no more than two significant digits in the plot labels. JSON retains
  component residuals and estimated final multipliers.
- The independent audit reconstructs KKT from fresh model values, checks
  analytical derivatives with two independently sampled finite-difference sizes,
  and checks history consistency. Audit work is excluded from solver call counts.

## Observed results

All 12 selected runs converge and pass KKT plus the independent audit.

| Problem | First fraction | Calls | Peak / starting objective | Raw objective increases |
|---|---:|---:|---:|---:|
| Unconstrained | SciPy / 1 | 49 | $8.70\times10^9$ | 11 |
| Unconstrained | 0.1 | 51 | $6.77\times10^5$ | 11 |
| Unconstrained | 0.005 | 50 | 8.44 | 4 |
| Unconstrained | 0.001 | 58 | 1 | 5 |
| Constrained | SciPy / 1 | 30 | $6.80\times10^7$ | 4 |
| Constrained | 0.016 | 22 | 1 | **0** |
| Constrained | 0.005 | 40 | 1.31 | 7 |
| Constrained | 0.001 | 51 | 9.14 | 7 |

The constrained fraction **0.016** decreases at every model evaluation, including
all trials. The unconstrained fraction **0.001** has monotonic accepted objectives
and no trial objective above its initial value, but still has five local raw
increases between successive attempts. The plots and metrics distinguish these
claims explicitly.

Smaller is not universally smoother: constrained 0.001 overshoots more than
0.016, despite a smaller first movement. The 1.5 growth policy and subsequent
BFGS directions influence the rest of the solve. A selected fraction is not a
monotonicity guarantee for other starts, problems, or future solver changes.
First-order KKT alone is not a general proof of global optimality.
