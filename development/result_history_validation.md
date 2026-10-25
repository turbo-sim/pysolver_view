# Phase 1: result and history validation

Implemented and checked on 25 October 2026. This covers production result and
history handling; it does not integrate the custom solvers or replace the
legacy KKT multiplier estimator.

## Tests before implementation

The first regression cases were written and executed against the existing code
before the fixes:

| Initial set | Baseline | Examples of exposed behavior |
| --- | --- | --- |
| Optimization: 12 cases | 7 failed, 5 passed | Last trial replacing the final solution; recursive initial rows; stale constraints; plotting changing stored objectives; zero complex-step gradients; callback mutation. |
| Native roots: 17 cases | 17 failed | Last observation replacing the final solution; stale finite-difference base; probes changing residual state; callback mutation; missing explicit result/accounting fields. |

The root baseline includes assertions for the new interface as well as
reproductions of incorrect existing behavior. Additional cases were added
during implementation and review; they are not claimed to have been run
against the old code.

## What changed

- Backend results own the final point. `solve()` still returns the physical
  solution vector; `result` and `backend_result` expose facade and native data.
- Evaluation and Jacobian acquisition no longer invoke the progress recorder
  recursively. Observations use one consistent point, and missing derivatives
  are not computed solely to print a progress row.
- Cache keys are exact. Finite-difference and complex-step evaluations neither
  become iteration events nor replace base-point snapshots.
- Stored coordinates, residuals, constraint metadata, and callback arguments
  are detached from mutable model/callback arrays.
- The improvement-only plot computes its own curve without modifying raw
  objective observations.
- SciPy SLSQP major-iteration events have a separate history. They carry
  `accepted=None` because SciPy's callback does not establish final line-search
  acceptance.
- Native LM/hybrid iteration counts remain `None`. Function and gradient
  observation plots use evaluation labels.
- Reusing a solver resets its per-run result, counters, and histories.
- Optimization finalization adds a history row only when the returned point
  differs exactly from the last recorded point. That row is labeled `Final
  solution` in the console. Existing observations, including genuine repeated
  solver events, remain unchanged. Final optimality from cached derivatives,
  feasibility, objective, and call totals appear in the footer and `result`;
  no extra derivative evaluation is required for reporting. There is no option
  to restore the artificial duplicate final row.

The old iteration-limit test was corrected to check native iterations rather
than using gradient counts with an arbitrary margin.

## Final checks

From the repository root:

```powershell
python -m pytest tests/test_result_history_optimization.py tests/test_result_history_root.py tests/test_optimization.py tests/test_nonlinear_system.py tests/test_differentiation.py -k "not hessian" -q -p no:cacheprovider
```

Result: **92 passed, 10 Hessian cases deselected**.
The command was run with the neighboring Turbodash environment's Python,
which supplies pytest. It includes 46 focused new result/history cases,
existing SciPy optimization/root cases, and first-derivative checks.
No PyGMO, IPOPT, SNOPT, or Hessian execution was required.

### Demo follow-up

Two additional regressions failed before the fixes: trust-constr's gradient-first
requests hid its function history, and Nelder–Mead ignored `max_iterations`.
An unperturbed model evaluation needed by a solver Jacobian now counts as an
ordinary function evaluation and can publish a function observation. Perturbed
finite-difference/complex-step samples remain derivative calls and never become
progress observations. Diagnostic-only requests remain nonpublishing.

Nelder–Mead now receives `maxiter`; its demonstration uses a budget of 500 and
converges in 247 iterations. Derivative-free methods do not receive an unused
Jacobian. Fully infinite bounds are omitted from SciPy arguments, and the BFGS
Wolfe demo explicitly declares unbounded variables. Finite bounds are not
silently removed. Scaling, comparison, and tolerance axes now label the actual
evaluation counts rather than calling them iterations.

The headless demo rerun completed 14 scripts and 41 successful solves without
runtime warnings. Trust-region sensitivity histories contain 34–40 observations
instead of two. IPOPT/SNOPT demos were skipped; the mixed comparison ran only
its SciPy entries, filtered in memory. The standalone numerical Hessian demo
also executed successfully as part of this requested demo check; this does not
expand the migration's Hessian validation requirements. Interactive window
behavior was not tested, and generated files were isolated from existing figures.

Both [short examples](../demos/result_history/README.md) also ran using the
workspace's default Python with Matplotlib's headless backend:

| Example | Observed solution | Checks |
| --- | --- | --- |
| Constrained SLSQP quadratic | Approximately $(1,0)$; objective 2. | Feasibility, facade result, raw-history preservation; 4 native major iterations and 12 facade model calls, including 8 derivative calls. |
| Coupled native hybrid equations | Approximately $(0.707107,0.707107)$; residual norm $2.22\times10^{-16}$. | Same-point residual histories, backend/facade agreement, unavailable iteration count; 17 model calls including 8 derivative calls. |

These counts describe the checked environment, not fixed expectations for
future SciPy releases or changed tuning.

## Remaining boundaries

- KKT reporting is now free of recursive progress events, but the old
  multiplier estimator and its known mathematical limitations remain for Phase 2.
- Optimization wrapper return types changed from status tuples to structured
  results. High-level `solve()` retains its return convention; direct callers
  of low-level wrappers should use result attributes.
- Non-SciPy wrappers were adapted statically to return their evolved solutions.
  They were not executed, consistent with the agreed migration scope.
- `tolerance_check_cache` remains an accepted compatibility argument, but
  evaluation identity is now exact even if a nonzero legacy tolerance is supplied.
- Calls made internally by user-provided analytic Jacobians are not visible
  unless routed through the facade. Counters describe observable facade work.
- Legacy root `max_iterations` still maps to native MINPACK evaluation-limit
  options; it must not be interpreted as an exposed accepted-iteration count.

See [PROGRESS.md](../PROGRESS.md) for the next milestone and remaining work.
