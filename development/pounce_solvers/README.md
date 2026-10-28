# Pounce NLP solver comparisons

Run from the repository root in the installed Poetry environment:

```powershell
poetry install --extras pounce
poetry run python development/pounce_solvers/run_native_solver_comparison.py --no-show
poetry run python -m development.pounce_solvers.run_wrapper_solver_comparison --no-show
```

`run_native_solver_comparison.py` calls `pounce.Problem` directly for the
interior-point and active-set SQP algorithms. It compares them with public
PySolverView custom SLSQP using first-step fractions 1.00 and 0.20.
`run_wrapper_solver_comparison.py` repeats that comparison using
`OptimizationSolver(library="pounce", method="ipm")` and `method="sqp"`, and
checks both against fresh native reference solves. Module execution lets the
wrapper script import shared study helpers normally, without path workarounds.

The default installed `CoupledRosenbrockProblem` has 24 variables, four linear
equalities, twelve nonlinear disk inequalities, finite bounds, and exact first
derivatives. Twelve Rosenbrock terms are coupled by neighboring first-variable
differences. Every three first variables have mean 0.8, while each pair obeys
$a_i^2+b_i^2\leq1+0.08\sin(i)$. The common start violates the nonlinear rows and
equalities; $a_i=0.8$, $b_i=0.5$ is strictly inside the disks and meets the
equalities.

The SLSQP first-step fraction limits its proposed QP direction, rather than an
absolute coordinate displacement. `step_growth_factor=None` restores the full
cap after the first major iteration, isolating the initial-step limit. Pounce
does not expose an equivalent initial fraction through its Python interface.

Pounce uses limited-memory Hessians for IPM and damped BFGS for SQP. Bounds are
not relaxed and internal objective/constraint scaling is disabled. Pounce uses
stopping tolerance $10^{-8}$; SLSQP uses `ftol=1e-12`. These native stopping
criteria have different meanings: SLSQP's objective/step tolerance is not a KKT
residual threshold. With `ftol=1e-8`, its stationarity residuals exceed the
common audit threshold. `--tolerance` and `--slsqp-ftol` expose both choices.
Fresh original-model audits use maximum feasibility $10^{-7}$ and first-order
KKT tolerance $10^{-5}$. Custom SLSQP uses those same explicit acceptance
thresholds. First-order checks do not establish a global optimum or prove a
second-order minimum.

Pounce's intermediate callback supplies metrics without an iterate vector. The
studies retain the last objective-callback point, following Pounce's own
`minimize` facade, and check that its original objective matches the reported
value. Such records are native observations with reconstructed coordinates,
rather than guaranteed accepted points. Restoration records or objective
mismatches remain in JSON/CSV and are excluded from coordinate-based plots.
Custom SLSQP records its initial and accepted iterates. Each authoritative
returned solution is independently audited.

The wrapper comparison checks native status and iteration count, final
coordinates/objective, and every regular callback coordinate. The absolute
point/objective parity threshold is $10^{-10}$. This checks interface fidelity
on one moderate NLP; it is not a broad robustness or speed benchmark.

Generated results are kept separately in ignored directories:

| Script | Result directory |
| --- | --- |
| `run_native_solver_comparison.py` | `results_native_comparison/` |
| `run_wrapper_solver_comparison.py` | `results_wrapper_comparison/` |

Each directory contains `convergence.png`, `convergence.pdf`, detailed
`comparison.json`, per-solver CSV histories, `summary.csv`, and `REPORT.md`.
Recorded objectives are actual values, never best-so-far envelopes. Plots
compare outer iterations rather than equivalent computational work. Fresh
post-solve audits are separate from model-call counts; timings are not ranked.
Results record the Git commit, dirty-tree flag, and dependency versions.

Both scripts accept `--blocks`, `--tolerance`, `--slsqp-ftol`, `--max-iterations`,
`--output`, and `--no-show`. Use `--blocks 24` for 48 variables, choose a custom
result directory with `--output PATH`, or omit `--no-show` to display the plot.
