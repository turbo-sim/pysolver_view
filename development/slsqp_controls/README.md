# SLSQP control studies and independent KKT audit

This folder adapts reusable maintenance ideas from
[Turbodash's optimization development folder](../../../turbodash/development/optimization/README.md)
to small mathematical problems. It imports the packaged solver rather than
copying another optimizer implementation. Turbodash files remain unchanged.

## What was worth carrying over?

| Turbodash material | Decision here |
|---|---|
| `kkt_checks.py`: independent residual/sign reconstruction | Adapted in `src/checks.py`; does not call production KKT assessment |
| Fresh final derivatives and half-step sensitivity | Adapted with analytical-reference checks and two independently sampled Jacobians |
| History objective/constraint/step consistency | Adapted; all custom history certificates are independently reconstructed |
| Separate native, feasibility, and optimality gates | Retained in the validation matrix; failed optimizations can pass the audit |
| `verify_kkt.py`: explicit tolerance profiles, impossible constraints | Adapted in `run_validation.py`, without physical/application dependencies |
| `kkt_report.py`: summaries, convergence, sensitivity plots | Adapted to the simpler demo's JSON/CSV/Markdown and plotted comparisons |
| Meanline inputs, YAML, physical output selectors, application exports | Keep in Turbodash; not meaningful generic library demonstrations |
| Parallel workers and resumable 52-case turbine runs | Not ported: these small studies run quickly, and fresh reruns avoid stale evidence |
| Historical turbine benchmark | Keep in Turbodash; it is not a correctness oracle for generic solvers |

The independent audit intentionally does not reuse the production residual
reconstruction. It uses supplied multiplier arrays, original analytical model
coefficients, and fresh finite-difference samples. Near the square-root domain
boundary, the independent probe is limited relative to distance from the
boundary; otherwise a fixed absolute difference measures a different local
regime. Both absolute and scaled derivative discrepancies are exported.

## Folder guide

- `src/problems.py`: multidimensional analytical problems and model-attempt logs.
- `src/checks.py`: fresh derivative and independent KKT/history audit functions.
- `tests/test_study.py`: nine checks for derivatives, analytical multipliers,
  sign errors, actual trial shortening, and recovery versus native failure.
- `run_validation.py`: study-specific tolerance/acceptance matrix.
- `results_validation/`: generated 20-case audit table and JSON.
- `results_sensitivity/`: generated 12-case first-fraction comparison, full data,
  compact tables, two plasma-colored Rosenbrock figures and the coarse fraction-search record.
- [Demo runner](../../demos/demo_slsqp_controls/run_sensitivity.py): owns its
  execution, plotting, export, and sensitivity configuration code.

Run from the repository root with SciPy 1.17.1:

```powershell
python demos/demo_slsqp_controls/run_sensitivity.py --no-show
python development/slsqp_controls/run_validation.py
python -m pytest development/slsqp_controls/tests -q
```

## Validation results

The current sensitivity study solves constrained and unconstrained Rosenbrock
using first fractions 1, 0.1, 0.016, 0.005, and 0.001; growth is fixed at 1.5,
maximum fraction at 1, and recovery is disabled. Each problem also runs native
SciPy. All 12 selected runs converge and pass independent audit. Local refinement
found constrained fraction 0.016 with monotonically decreasing raw evaluations.
Unconstrained fraction 0.001 has no above-start trial overshoot and monotonic
accepted objectives, but raw rejected attempts can increase locally. The main
script no longer includes maximum-fraction sweeps or square-root runs. The
separate validation matrix and recovery unit tests retain those generic problems.
The tolerance matrix covers four problems and five profiles. All **20 audits
pass**, including correctly rejected inconsistent constraints and native-success
cases rejected by the configured KKT thresholds.

The updated study suite has **nine passing tests**, including raw versus
accepted-point monotonicity for the selected examples. Generated results are
evidence from this run, not assertions that all future trajectories must be
identical. No meanline, PyGMO, IPOPT, or SNOPT model was run. No solver
implementation was changed.

See the [demo guide](../../demos/demo_slsqp_controls/README.md) for measured
improvements and counterexamples. First-order KKT checks are not a general proof
of a minimum. An audit pass means reporting and mathematical reconstruction are
consistent; it does not mean an optimizer converged.
