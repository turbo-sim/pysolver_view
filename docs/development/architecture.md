# Implementation boundaries

These notes describe the installed implementation. For public installation and
compatibility details, see the [source-layout guide](../source/source_layout.md).
The repository, distribution and import name are `pysolver_view`;
`library="pysolver"` selects a backend.

## Repository ownership

| Location | Responsibility |
| --- | --- |
| `src/pysolver_view/roots/` | Root frontend, run evaluation, method aliases, problem contracts/models, backend adapters and numerical solvers |
| `src/pysolver_view/optimization/` | Optimization frontend, run evaluation, coordinate/constraint adaptation, problems, backends and custom SLSQP |
| `src/pysolver_view/diagnostics/` | KKT mathematics and independent benchmark audits |
| `src/pysolver_view/differentiation/` | Finite differences, complex step, Hessian helpers and optional JAX support |
| `src/pysolver_view/reporting/` | Tables, text, plots, style and logging helpers |
| `src/pysolver_view/{results,history,exceptions,validation}.py` | Shared records, contracts and validation utilities |
| `tests/` | Domain, study, import and packaging checks; selection in `run_tests.py` |
| `demos/` | Focused public examples and their outputs |
| `development/` | Five reproducible study runners and generated results |
| `docs/source/` | Sphinx user and generated API documentation |
| `docs/development/` | Current implementation explanations, mathematics, protocols and limitations |

Reusable numerical code belongs in the installed package. Study runners own
their configuration, instrumentation, plotting and report generation. They do
not supply an alternative import tree. Benchmark models live one per module in
`roots/problems/`; `suites.py` selects collections. SLSQP study models live in
`optimization/problems/slsqp_controls.py`.

## Frontends, evaluation and reporting

`roots/solver.py` and `optimization/solver.py` coordinate public configuration,
run state and backend dispatch. Evaluation contexts own same-point caches and
observable model/derivative counters. Backend adapters translate options,
coordinates, signs, events and returned results. Numerical drivers decide
acceptance, recovery and termination.

The independent KKT core takes numerical arrays, bounds and tolerances. It does
not know about models, callbacks or histories. `optimization/kkt_adapter.py`
acquires and converts the required data; reporting formats the supplied records.
An explicit diagnostic request can acquire missing model data and count that
work. Passive formatting and plotting do not evaluate models.

## Custom roots

All modules below are in `src/pysolver_view/roots/solvers/`.

| Module | Responsibility |
| --- | --- |
| `solver.py`, `options.py` | Alias dispatch and validated method-specific options |
| `trust_region.py` | Named LM and Powell hybrid entry points, shared trial loop and Powell model-update policy |
| `line_search.py` | Named Newton-Raphson and Gauss-Newton entry points, shared full-step/Armijo loop and separate direction/stationarity policies |
| `linear_model.py` | Persistent local model, QR/SVD factors, Broyden updates and linear subproblems |
| `evaluation.py` | Vector/residual/Jacobian validation, numerical probes and budgets |
| `scaling.py` | Numerical Jacobian equilibration, independent of physical variable scales |
| `trial_step.py` | Actual rounded trial coordinates and displacement checks |
| `history.py` | Detached accepted/trial/evaluation records and final results |

LM recomputes the Jacobian after accepted moves and reuses it during retries.
Powell hybrid also learns from valid secant pairs, including rejected trials.
Newton-Raphson requires a square full-rank model; Gauss-Newton permits
overdetermined and rank-deficient least-squares directions. Both refresh the
Jacobian after accepted moves. The shared engines receive their numerical
policies explicitly rather than choosing them by method-name strings in the loop.

These are independent dense implementations using public SciPy linear algebra.
They do not patch MINPACK or call a private native root bridge. Native
`library="scipy"` still delegates to SciPy root and remains the default.

## Optimization backends

The registry in `optimization/backends/registry.py` dispatches `scipy`,
`pysolver`, `pygmo` and `pygmo_nlopt`. Custom optimization currently means SLSQP.
Its driver, options, private compiled bridge and retained SciPy license are in
`optimization/solvers/slsqp/`. The bridge has an exact SciPy 1.17.1 guard;
unlike custom roots, it depends on private solver state.

`pygmo.py` contains IPOPT and population-method adaptation. NLOpt has its own
adapter in `pygmo_nlopt.py`. SNOPT's nonfree plugin loads through
`pygmo_nonfree.py` only when requested. Separation reflects dependency and
translation boundaries; it does not imply equivalent numerical validation for
all listed methods. See [remaining limitations](limitations.md).

## Compatibility and optional dependencies

Legacy package modules forward to canonical objects. Keep class, option,
exception and function identity across supported import paths. New internal
imports use the canonical domain modules. Tests patch defining modules, such as
`roots.backends.scipy.root`, rather than compatibility re-exports.

JAX loads lazily; host precision/platform configuration requires explicit
`configure_jax(...)`. CoolProp is optional and loads when constructing the
heated-duct problem. PyGMO availability is independent of its nonfree plugin.
Importing the package does not install a global exception hook or configure the
host's logging or JAX policy. Calling the legacy logger helper is a separate
operation with a remaining ownership limitation.
