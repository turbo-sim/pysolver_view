# Results, histories and observable work

The shared result constructors are in
[`results.py`](../../src/pysolver_view/results.py). Per-domain evaluation and
history modules own run data. These contracts apply to the current frontends;
they replace the completed phase-by-phase migration checklists.

## Authoritative state

`solve()` returns the physical solution vector. `result` is the detached public
record; `backend_result` retains backend-specific data and conventions. The
returned point determines final values, feasibility and success. A later trial
or observation cannot replace that point. Custom drivers return the last
accepted state on cancellation, invalid trials or budget exhaustion. An invalid
initial root residual returns failure at the initial coordinates with `fun=None`.

Root success requires both backend success and the applicable residual test.
For custom roots the target is resolved `residual_tol`; native roots retain the
frontend check at ten times `tolerance`. A small step or nonzero least-squares
stationarity is not a successful root. Optimization acceptance differs by
backend; see [KKT and SLSQP](optimization_kkt.md).

Solver reuse resets counters, caches and histories. Stored and callback arrays
are copied so mutation of a model's return buffer or callback argument does not
rewrite recorded state. Separate solvers sharing a mutable problem are not
thereby safe for concurrent use.

## Events are backend-specific

| Event | Meaning |
| --- | --- |
| Function observation | An ordinary same-point model observation; can include valid rejected trials |
| Gradient observation | A requested derivative at its unperturbed residual/fitness base |
| Custom iteration | Initial state, then each committed accepted move |
| Native SciPy SLSQP major iteration | A native callback event; `accepted=None`, not proof of final line-search acceptance |
| Derivative probe | Perturbed model work; never a progress row or solver iteration |
| Diagnostic request | Explicit assessment work; counted without progress publication or callbacks |

Custom `iteration_history` retains accepted states separately from the selected
observation stream. Native LM/hybrid do not expose accepted-iteration counts:
`nit` and `iteration_count` are `None`. Consecutive native function-call distances
must not be labeled accepted step sizes or internal trust radii.

Optimization finalization appends a final-solution row only when the returned
point differs exactly from the last recorded point. Genuine repeated solver
events remain. Improvement-only plots construct a separate curve and do not
rewrite raw objectives or constraints.

## Counting model work

Frontend `nfev`/`func_count_total` count observable physical calls, including
numerical derivative probes. `njev` counts derivative requests; it is not a
count of model calls. `derivative_func_count` is a subset of total model calls.
For optimization, `diagnostic_func_count` and `diagnostic_grad_count` expose
explicit assessment work; `failed_grad_count` records failed requests.
Inspect the domain's evaluation context for the precise categories rather than
adding subset counters to totals.

Custom root core results additionally distinguish `derivative_nfev`,
`failed_njev`, `ntrials`, `nrejected`, `ninvalid` and linear-algebra work.
`max_iterations` budgets trial attempts, including rejection; `max_nfev` also
charges probes and failed residual calls. Accepted moves alone advance `nit`.
Native root `max_iterations` maps to MINPACK evaluation-limit options instead
of an exposed accepted-move budget.

Exact cache identity prevents a nearby or complex-perturbed point from becoming
the base snapshot. Legacy `tolerance_check_cache` remains accepted but does not
relax that identity. Reports consume cached data when available; explicit KKT
requests may acquire missing values/derivatives and fit multipliers. Repeated
same-point, same-tolerance report components reuse the assessment.

Calls made entirely inside a user-provided analytic derivative remain invisible
to the frontend. Likewise, a residual call may contain many property updates.
Report counters as observable interface work, not a complete measure of model
or property-library cost.

## Callbacks and storage

Custom frontend callbacks receive detached `(x, callback_count)` arguments at
the initial state and accepted moves, independently of `update_on`. Returning
exactly `False` or raising `StopIteration` requests structured cancellation.
Native frontend callbacks retain their observation semantics and do not acquire
this custom cancellation contract. Low-level custom root callbacks receive a
detached diagnostic dictionary instead.

Custom root `history_level="full"` stores coordinates in accepted, trial and
evaluation records; `"summary"` stores scalar diagnostics; `"none"` disables
stored lists. Budgets, numerical results and counters remain active at every
level. Callbacks receive complete accepted-state records regardless of storage.
Summary records cannot reconstruct all earlier coordinates or numerical scaling
metrics. Printing and live plotting must be disabled separately when timing.

See the [root interface](root_solvers.md), [timing protocol](timing.md), and
[validation guide](validation.md) for usage and reproducible checks.
