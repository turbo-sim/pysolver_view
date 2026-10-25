# PySolverView refactor progress

Last updated: **2026-10-25**

This is the working checklist for the solver, KKT, and convergence-view
improvements. Update it as implementation and validation proceed.

**Current position:** Phase 1 result/history fixes and Phase 2 shared KKT
assessment are implemented. The generic custom SLSQP driver is packaged and
available through `library="pysolver"`. Phase 4 now integrates all four custom
root methods through the same backend name. Root benchmarks remain in development.

**Next integration milestone:** Release/consumer preparation, then update
Turbodash's imports and dependency when explicitly requested. Its existing files
remain untouched. See [KKT/SLSQP validation and usage](development/kkt_slsqp_validation.md)
and the [named KKT demo](demos/demo_kkt_conditions/README.md).

## How to maintain this checklist

- Check an implementation task only when its relevant checks pass.
- Update the phase's outcome with what worked, what failed, and any remaining
  limitation. A sentence or two is sufficient.
- Record validation with its date, scope, and result. Earlier test results do
  not establish that later changes pass.
- Record changed decisions rather than silently changing API expectations.
- Keep detailed design in the linked notes; keep this document focused on
  progress and evidence.

## Iteration and evaluation accounting

Finite-difference and complex-step model evaluations are **never solver
iterations**. Keep them in total model/function/residual calls and, optionally,
a separate derivative-probe count. A complete gradient/Jacobian request counts
as one derivative request, not one iteration and not one gradient request per
probe. Record incomplete derivative requests explicitly if they fail.

Only actual solver iteration events may advance an iteration counter or appear
as iteration-history rows. Derivative probes must not trigger iteration
callbacks, change accepted-iterate state, or consume an iteration budget; they
may consume the function-evaluation budget. If a backend does not expose its
iterations, leave that count unavailable and plot evaluation counts under their
own labels. Do not infer iterations from function or gradient calls.

Probe counts are a subset of model-call totals, not extra calls to add to those
totals. Trial/rejection counters remain separate; numerical derivative probes
are not line-search or trust-region trial attempts.

## Migration validation scope

SciPy SLSQP is the agreed proxy check for the existing solver infrastructure.
There is **no requirement to install, run, benchmark, or test the PyGMO backend,
IPOPT, or SNOPT** for this migration. Their wrappers remain supported, and we
will assume they continue to work if SciPy SLSQP works correctly through the
updated infrastructure. This is an accepted project assumption, not a claim
that those backends have been independently validated.

References below to backend, compatibility, consumer, or cross-backend checks
exclude PyGMO, IPOPT, and SNOPT. Their validation is not a completion or release
gate. Tests for custom SLSQP, root solvers, shared KKT mathematics, and the
Turbodash application remain in scope.

Hessian helpers are optional, noncritical functionality for pysolver and are
applicable to IPOPT as well as other second-order methods through suitable
backend interfaces. There is no requirement to check that the existing Hessian
helpers work correctly, add Hessian regressions, or extend their integration
for this migration. Their validation is not a completion or release gate.
Required first-derivative and first-order KKT checks remain in scope.

## Agreed direction

- Numerical convergence behavior may change, including default tuning, option
  mappings, trajectories, iteration/evaluation counts, and recovery sequences.
  Exact reproduction of current PySolverView runs is not required.
- Judge changes by independently verified convergence to the problem's intended
  residual/feasibility/optimality targets, correct diagnostics, and respected
  budgets/controls. Do not relax targets merely to obtain a success flag.
- Use previous runs as references for convergence and performance, not exact
  history snapshots. Investigate lost convergence or material performance
  regressions; different successful paths are acceptable.
- Preserve the common convergence view across SciPy and PyGMO, including
  SLSQP, IPOPT, and SNOPT.
- Preserve existing public interfaces where practical; expose new behavior
  explicitly and document compatibility changes.
- Add `library="pysolver"` for custom drivers; retain native backends.
- Consolidate optimization KKT diagnostics independently of solver internals.
- Keep turbine physics, configuration, and application workflows in Turbodash.
- Add `newton_raphson` and `gauss_newton`, both with configurable backtracking.
  Explicit `backtracking=None` selects full steps.
- Use `levenberg_marquardt`, `powell_hybrid`, `newton_raphson`, and `gauss_newton`
  as canonical root identifiers; retain `lm`, `hybr`, `newton`, and `gn` as aliases.

## 0. Completed groundwork

- [x] Inspect the custom SLSQP driver and private SciPy bridge in Turbodash.
- [x] Develop experimental controlled LM/hybrid root solvers and multidimensional
  mathematical/thermodynamic problems.
- [x] Prepare root convergence, timing, geometry studies, and mathematical notes.
- [x] Review the old PySolverView and newer Turbodash KKT implementations.
- [x] Confirm bound-only feasibility, multiplier-sign, recursive-reporting, and
  mixed-point history problems in the older implementation.
- [x] Confirm that improvement-only plotting modifies stored objective history.
- [x] Write the [overall refactor plan](development/proposed_refactor.md).
- [x] Write the [KKT comparison and design](development/kkt_comparison_and_design.md).
- [x] Write the [feature review](development/pysolverview_feature_review.md).
- [x] Revise the [Newton extension plan](development/root_finder_extension.md)
  around two methods, optional backtracking, and author-name aliases.

**Outcome:** The shared-library direction is feasible. The immediate risks are
result/history correctness, sign and scaling conversions, and compatibility
with the private SciPy bridge. Reviews identify defects; they do not fix them.

## 1. Trustworthy final results and convergence histories

- [x] Add regressions for final result versus last evaluated/reported point.
- [x] Return authoritative structured backend results internally, including
  SciPy's solution and PyGMO's evolved solution/champion.
- [x] Preserve `solve(x0)` returning a solution vector; expose richer result data.
- [x] Separate model evaluation, recording, and presentation responsibilities.
- [x] Build each row from a point-consistent snapshot.
- [x] Prevent diagnostic evaluation from recursively recording progress.
- [x] Omit artificial duplicate final rows; retain labeled final solutions when
  the backend returns a different point and report final diagnostics separately.
- [x] Preserve raw history when computing improvement-only plots.
- [x] Distinguish model/derivative evaluations from known accepted iterations;
  correct axis labels and step interpretations.
- [x] Verify finite-difference and complex-step probes only increase evaluation
  counters, never iteration counters, iteration callbacks, or iteration budgets.
- [x] Verify plotting/reporting does not change returned solutions or raw
  observations, and rejected trials do not become final solutions.
- [x] Check existing public callbacks and native SciPy SLSQP usage; no PyGMO,
  IPOPT, or SNOPT checks are required.

**Outcome:** Implemented. Backend results now own final coordinates; observation
snapshots, derivative probes, callbacks, plotting, and constraint metadata no
longer corrupt one another. Native SLSQP callback events are labeled as major
iterations with unknown acceptance; native LM/hybrid iteration counts remain
unavailable. Solver reuse starts a fresh run.

**Validation / remaining issues:** 92 tests passed, with 10 Hessian cases
excluded. The initial regression baseline failed before implementation
(optimization: 7 failed/5 passed; roots: 17 failed). Both short examples pass.
Non-SciPy wrappers were adapted statically and remain untested by agreement.
This Phase 1 validation preceded the shared KKT replacement, now completed in Phase 2.

**Demo follow-up:** Fixed gradient-first function histories and Nelder-Mead's
iteration budget; corrected demo axis labels and unused configuration. BFGS's
Wolfe demo explicitly uses an unbounded problem. All 14 in-scope scripts and
41 solves completed successfully in headless runs, with no runtime warnings.
IPOPT and SNOPT cases were skipped; only SciPy cases of the mixed comparison
were executed. Generated audit outputs were kept outside the package repository.

## 2. Shared KKT assessment

- [x] Extract the generic Turbodash numerical checker into PySolverView.
- [x] Separate multiplier estimation from residual reconstruction as appropriate.
- [x] Define a stable diagnostic schema, availability states, and multiplier source.
- [x] Adapt inequality/Jacobian signs and equality multiplier signs explicitly.
- [x] Check all finite bounds, including problems without user inequalities.
- [x] Preserve meaningful norm, tolerance, scaling, and multiplier-order semantics
  through compatibility adapters for existing reports.
- [x] Keep native termination, feasibility, KKT pass, and final acceptance distinct.
- [x] Make diagnostic frequency and extra derivative costs explicit.
- [x] Port generic KKT tests and add regressions for the confirmed older defects.
- [x] Verify same-point diagnostics, unavailable assessments, rescaling, redundant
  constraints, and independence from reporting.

**Outcome:** Implemented in `pysolver_view/diagnostics/kkt.py`. Public inputs use
nonpositive inequalities, with explicit residual reconstruction and independently
estimated multipliers. Existing KKT/variable/constraint/multiplier reports remain
available, including names and explicit unavailable diagnostics.

**Validation / remaining issues:** Analytical edge cases, supplied multiplier reconstruction,
positive rescaling, fixed/opposing constraints, and legacy report regressions
pass. Custom accepted-point reporting reuses the driver's assessment. Native
backend success remains separate from optional post-solve KKT assessment.
Root acceptance remains residual-based and does not use an optimization KKT gate.

## 3. Packaged custom SLSQP and Turbodash migration

- [x] Move the generic driver, private bridge, and SciPy license into PySolverView.
- [x] Keep Turbodash YAML compatibility handling and physical adapters in Turbodash.
- [x] Register the custom optimization backend and adapt combined evaluations,
  derivatives, bounds, scaling, callbacks, and option mappings.
- [x] Validate recovery and native merit-search consistency, cancellation, and
  accepted-state returns. Recovery tuning may change without matching old runs.
- [x] Keep the SciPy 1.17.1 compatibility check local to custom SLSQP; provide a
  supported installation path without breaking other backend imports.
- [x] Port generic SLSQP tests and compare low-level/high-level adapter behavior.
- [ ] Update Turbodash's imports and explicit dependency on the required release.
- [ ] Verify turbine/API tests, physical reevaluation, exported histories, and
  application progress/cancellation.
- [ ] Run the independent turbine KKT acceptance matrix with fresh outputs.
- [ ] Remove duplicate implementations only after consumer validation; retain
  temporary compatibility re-exports where needed.

**Outcome:** Generic extraction and public backend are implemented, with copied
source/license and low-level plus high-level tests. Turbodash migration is
intentionally deferred until the user releases and adopts the package. No
Turbodash source files were changed or removed.

**Validation / remaining issues:** Custom recovery, accepted-state returns,
callback cancellation, analytic/numerical derivative accounting, signs/scaling,
and strict KKT acceptance pass. Only custom SLSQP requires SciPy 1.17.1; other
backends retain their dependency range. Fixed variables must be removed before
using custom SLSQP. Wheel construction remains unverified because poetry-core
is unavailable from this environment's configured package source. Turbine
acceptance/consumer checks await the separately authorized application migration.

## 4. Public integration of all four custom root methods

- [x] Promote reusable numerical code from development into the installable package.
- [x] Add root backend selection without shifting existing positional arguments.
- [x] Integrate authoritative results, histories, callbacks, domain errors, and budgets.
- [x] Give numerical derivatives one owner and preserve scaled probe accounting.
- [x] Integrate and test canonical author names and their short aliases in the public root frontend.
- [x] Keep custom and SciPy/MINPACK provenance explicit.
- [x] Update study imports so they exercise packaged code.
- [x] Rerun root unit tests and convergence/step audits, including 100-variable
  dense and thermodynamic problems.
- [x] Measure adapter overhead separately from model, derivative, and numerical work.
- [x] Add one four-solver demo with full author names and solid/dashed overlap styling.
- [x] Add optional, default-enabled Jacobian column scaling to all four methods,
  preserving physical step limits and derivative probes; verify the 500-node heat case.

**Outcome:** Implemented in `pysolver_view/solvers/root/`, with public integration
through `NonlinearSystemSolver(library="pysolver")`. Native SciPy remains the
default. Benchmark problems/runners remain under development; compatibility
imports re-export the packaged implementation. Roots and custom SLSQP share one
`DomainError`. The [usage note](development/root_step_control/notes/10_public_interface.md)
documents options, callback differences, accepted observations and counters.

**Jacobian-scaling follow-up:** `jacobian_scaling=True` is now the default for
all four custom methods. Numerical equilibration updates with fresh Jacobians
and stays fixed during hybrid secant updates. Physical `x_scale`, step caps,
derivative probes, and the residual-root target retain their meanings. Full-rank
Newton/Gauss-Newton directions are mathematically unchanged. The 500-node,
1000-variable radiation demo now converges in all six configurations; controlled
LM uses 12 accepted iterations and 13 model calls without reweighting equations.
Controlled hybrid also converges, but takes more calls than with scaling disabled.

The follow-up check passes 455 package/root tests (10 Hessian cases excluded),
including 27 new scaling cases. Fresh convergence sweeps retain 100/100 eligible
backend roots, 98/101 eligible custom roots, and 51/51 controlled successes.
The three undamped failures, six unsupported configurations, and five inconsistent
controls remain explicitly reported. Historical timing measurements below
predate numerical scaling and were not repeated for this feature.

**Validation / remaining issues:** Combined package/development check: 418 passed,
10 Hessian tests deselected, including 29 public-root regressions. All 100
native/custom convergence cases pass; the custom study retains 98/101 eligible
roots and 51/51 eligible controlled successes, with expected unsupported and
inconsistent cases. All 11,760 benchmark solves and 3,168 direct/public interface
timing solves reach the target. Adapter counts agree with direct core counts.
Full-history interface cost is measurable on cheap problems and is reported
separately, including timing noise. All eight demo solves and an isolated
package-copy smoke check pass without development imports. This is not a wheel
installation check: poetry-core remains unavailable; Phase 7 release checks remain.

## 5. Newton-Raphson and Gauss-Newton

- [x] Add the two canonical methods and `newton` / `gn` aliases.
- [x] Implement shared options with default backtracking and explicit
  `backtracking=None` for undamped execution.
- [x] Distinguish disabled backtracking from disabled step limits and zero retries.
- [x] Implement strict square-system Newton direction/rank handling.
- [x] Reuse QR/SVD least-squares directions for Gauss-Newton.
- [x] Implement full-step commitment, including valid residual increases.
- [x] Implement shared Armijo backtracking, scaled limits, and gradual recovery.
- [x] Test singular, overdetermined, inconsistent, domain-failure, callback,
  rounding, and budget cases.
- [x] Use explicit benchmark configurations rather than parsing underscored names.
- [x] Run multidimensional convergence and timing studies; retain failed cases.
- [x] Update usage/math notes with measured results and limitations.

**Outcome:** Initially implemented and validated under `development/root_step_control/`,
then promoted through Phase 4. All four methods have canonical author
names and dedicated entry functions backed by two shared iteration engines.
Short aliases remain accepted. All 51 eligible protected comparison cases converged;
undamped methods had three failures, retained in the report.

**Validation / remaining issues:** 94 development tests pass. The original
100-case LM/hybrid comparison retains its root outcomes and call counts.
The custom study contains 112 configurations: 98/101 eligible roots, six
unsupported rectangular Newton cases, and five correctly rejected inconsistent
controls. All 11,760 timed solves across 56 configurations reach the target.
Conservative Newton methods beat conservative LM on the measured 100-variable
workloads; Powell hybrid remains faster there. This is not a universal speed
or thermodynamic robustness guarantee. See the
[convergence report](development/root_step_control/results_custom_convergence/REPORT.md)
and [timing report](development/root_step_control/results_time_perf/RESULTS.md).

## 6. Supporting package improvements

Implement these in focused changes alongside the relevant phases:

- [x] Exact solver-point caching and explicit accounting for diagnostic/probe work.
- [ ] Scaling independent of bounds; explicit handling of fixed/unbounded variables.
- [ ] Backend capability checks for constraints, derivatives, callbacks, and controls.
- [ ] Independent optional-dependency detection for PyGMO and its nonfree plugin.
- [ ] Transparent effective options; review tuned IPOPT/SNOPT defaults and offer
  application profiles. Improved convergence behavior is allowed; document the
  new defaults and option meanings rather than retaining old numerical traces.
- [ ] Meaningful population-solver results and termination reasons.
- [ ] Serializable histories/results and comparison plots using evaluations or time.
- [ ] Optional throttled live plotting that reads recorded data.
- [ ] Explicit per-run state and documented reuse/concurrency behavior.
- [ ] Remove import-time host-application configuration through a documented transition.

**Outcome:** Exact caching, per-run reset, authoritative population results, and
selected capability handling are implemented in Phase 1 and the SLSQP adapter.
The remaining supporting items still require focused work.

**Validation / remaining issues:** Avoid expanding this into an unrelated
framework rewrite. Correct observations and results take priority.

## 7. Compatibility, documentation, and release

- [ ] Inventory public imports, constructor arguments, return values, callback
  contracts, and report/history fields used by dependent packages.
- [ ] Add compatibility tests against representative consumers using SciPy
  SLSQP and the custom solvers; exclude PyGMO, IPOPT, and SNOPT execution.
- [ ] Assess API compatibility separately from numerical trajectory changes;
  avoid tests requiring historical iterates or identical call counts.
- [ ] Document intentional numerical corrections and any API/behavior migrations.
- [ ] Validate installed wheels, license inclusion, and imports without development files.
- [ ] Verify optional dependency isolation and supported Python/SciPy combinations.
- [ ] Update tutorials, folder guides, effective-option examples, and release notes.
- [ ] Decide release number from the final compatibility scope.
- [ ] Update dependent constraints/lock files and validate consumer installations.

**Outcome:** Proposed release path is a compatibility-focused `0.7.0`, then
`1.0.0` once the public contracts are settled. No release decision or version
bump has been made. Turbodash's current `^0.6.9` constraint excludes `0.7.0`.

**Validation / remaining issues:** Package renaming is not part of the current
implementation milestones.

## Evidence log

| Review/date | Evidence | Scope / limitation |
| --- | --- | --- |
| Earlier root development review | 66 root checks recorded as passing; convergence/timing studies saved under [root_step_control](development/root_step_control/README.md). | Development implementation only; not rerun when creating this checklist. |
| 2026-10-25 refactor review | Turbodash `tests/test_slsqp.py`: 41 passed. | Existing driver baseline, before extraction. |
| 2026-10-25 KKT review | Turbodash `test_kkt.py` and `test_kkt_tolerances.py`: 69 passed. | Existing checker/driver integration; not the full turbine matrix. |
| 2026-10-25 feature review | Plotting changed stored objectives from `[3, 5, 2]` to `[3, 3, 2]` in an in-memory reproduction. | Confirmed during review; fixed and regression-tested in Phase 1. |
| 2026-10-25 planning | Four design/review notes completed; Newton plan revised to two configurable methods. | Historical planning milestone; see current phase statuses above. |

Append new validation here as work proceeds. For failures, record the failing
scope and follow-up rather than checking the corresponding milestone complete.

Phase 5 started on 2026-10-25 at the user's request for parallel work. Its scope
is development code, tests, notes, and regenerated studies; production promotion
remains a separate phase. Fresh comparisons separate native SciPy versus custom
Levenberg-Marquardt/Powell hybrid from the four-method custom comparison.

Phase 5 validation completed on 2026-10-25: 91 tests passed; all 239 named
functions and 30 classes in the experiment have docstrings; all 21 Python files
pass Black. Fresh results contain 100 backend cases, 112 custom cases, and
2,352 timing batches. Protected trial bounds were independently audited from
the evaluated coordinates. Source changes for this phase are development-only.

## Decision and progress log

| Date | Update |
| --- | --- |
| 2026-10-25 | Created the checklist. Start with trustworthy results/history, then shared KKT, SLSQP extraction, custom root integration, and Newton-family extensions. |
| 2026-10-25 | Adopt two Newton-family methods with optional backtracking; preserve `lm`/`hybr` and add descriptive aliases. |
| 2026-10-25 | Set the custom backend identifier to `library="pysolver"`. This supersedes `pysolverview` in earlier design notes; it does not rename the package. |
| 2026-10-25 | Approved changing current solver convergence behavior. Verified convergence is the acceptance target; exact reproduction of old histories, tuning, or evaluation counts is not required. |
| 2026-10-25 | Exclude PyGMO, IPOPT, and SNOPT validation from migration/release gates. Use SciPy SLSQP as the proxy and assume those wrappers continue to work. |
| 2026-10-25 | Hessian helpers also apply to IPOPT but are noncritical. Checking old helpers or adding Hessian tests/integration is not required for this migration. |
| 2026-10-25 | Finite-difference and complex-step probes count as model/function evaluations, never iterations. Gradient/Jacobian requests have separate counters. |
| 2026-10-25 | Completed Phase 1 with regression tests prepared first, 89 passing in-scope checks, and two runnable examples. PyGMO/IPOPT/SNOPT/Hessian validation excluded as agreed. |

| 2026-10-25 | Implemented shared KKT assessment and packaged custom SLSQP together: 279 tests passed (10 Hessian cases excluded); 15 demo scripts completed 43 successful solves. Added named-variable/constraint analytical KKT demonstration. Turbodash files/imports/dependencies remain untouched by request. Wheel validation awaits poetry-core availability. |

| 2026-10-25 | Adapted Turbodash independent optimization audits to small mathematical problems. Added controlled SLSQP demo: 51 sensitivity runs, 50 independently audited returned states and one expected native domain exception; 20 tolerance/inconsistency audits passed. Seven new study tests plus existing KKT/integration checks: 109 passed. Saved data, tables, and 15 comparison figures. No meanline solve or Turbodash changes. |

| 2026-10-25 | Added optional automatic KKT checks via check_kkt (default True), zero-assessment disabled mode, explicit unavailable acceptance fields, and a paired timing demo on the named problem. Recorded 50 warmed pairs per backend; full in-scope regression run: 291 passed, 10 Hessian cases deselected. Explicit post-solve KKT reports remain available. |

| 2026-10-25 | Corrected KKT timing comparison to full steps with recovery disabled. All four configurations match point trajectories, iterations, and model/Jacobian calls across 50 warmed pairs per backend. Console tables are ASCII-only. Updated measured totals: SciPy 0.48/1.96 ms and custom 0.48/3.35 ms (off/on); diagnostic schedules remain 1 versus 2 checks. |

| 2026-10-25 | Simplified SLSQP sensitivity to two one-at-a-time sweeps: first and maximum fractions 0.1:0.1:1.0, with recovery disabled and immediate restoration. Plasma colors span 0.2-0.8. Reran 63 cases: 43 successes, 62 returned-state audits passed, one native domain exception. Seven study tests passed; regenerated six study figures. |

| 2026-10-25 | Replaced the two-iteration KKT timing problem with constrained Rosenbrock: 21 iterations and 30 matching model evaluations in all four configurations. Fifty warmed pairs per backend verify matching model trajectories and nonzero enabled KKT work; SciPy performs 21 assessments and custom 22. ASCII tables include iteration counts; timing data/plots regenerated. |

| 2026-10-25 | Refined first-step sensitivity after a logarithmic search and local refinement. Only constrained/unconstrained Rosenbrock remain; fractions 1, 0.1, 0.0184, 0.005, 0.001 with growth 1.5. All 12 selected runs converge/KKT/audit pass. Constrained 0.0184 is monotonic at every evaluation; unconstrained 0.001 has monotonic accepted objectives and no above-start trial overshoot. Nine study tests pass. |

| 2026-10-25 | Simplified selected fraction 0.0184 to 0.016 (still monotonic for constrained Rosenbrock). All plot curves have markers at every point. Replaced final KKT scatter with post-solve optimality histories on recorded points, preserving solver accounting. All 12 solves/audits and nine study tests pass; both figures regenerated. |

| 2026-10-25 | Consolidated KKT assessment, custom SLSQP, package integration, demos, studies, tracked plot updates and project notes into topic commits. Fresh combined check: 387 passed, 10 Hessian cases deselected, covering package tests, SLSQP study tests and all 94 root development tests. Canonical root author names and four explicit entry functions are implemented in development; public root integration and installed-wheel validation remain pending. |

| 2026-10-25 | Consolidated KKT demos to named tutorial, timing benchmark, and README. Removed three older scripts and generated logs/reports/cache/empty figures. Added final-only audit, scaled coordinates with transformed bound multiplier checks, explicit report export and headless mode to the main tutorial. Sixteen relevant tests pass; default/combined tutorial commands and timing smoke run verified. |

| 2026-10-25 public roots | Phase 4 complete: all four custom roots packaged; common frontend, shared domain errors, numerical-derivative ownership, and single demo integrated. 418 tests passed, 10 Hessian tests deselected; 212 convergence configurations, 11,760 main timed solves and 3,168 interface timed solves rerun. All eligible controlled cases pass. Isolated package-copy import/solve smoke passes; wheel installation and consumer migration remain open. |

| 2026-10-25 radiative heat demo | Added a 98-unknown Chebyshev collocation problem inspired by Nozzlex, with constant conductivity and distributed radiation. Newton, LM and hybrid compare full/large initial steps with a 0.1 initial limit using the public interface. All six converge and agree with an independent adaptive BVP within 3e-10 in temperature; 19 tests cover derivatives, spatial refinement, energy balance, controls and accounting. Small starts reduce residual excursions; hybrid uses fewer iterations while LM uses more. Solution/history figures use white-faced markers above the lines; ASCII summaries, saved numerical histories and a folder guide explain the comparison. The nonradiating limit also runs successfully. |

| 2026-10-25 LM mesh investigation | Reproduced LM failure after increasing the heat demo to 500 nodes: boundary temperatures approach zero while the summed residual decreases. Independent trust-region step audits agree within 5e-12 relative error; the principal causes are mesh-dependent residual weighting and fixed variable scaling, not an identified damping bug. Native LM with default scaling converges; forcing identity scaling attempts invalid temperatures. Boundary reweighting, exact boundary elimination, quadrature weighting, and fixed Jacobian-column scaling each restore custom convergence and pass the original residual target. Reproducible scripts, saved evidence, and recommendations are in development/radiative_heat_investigation/. Production solvers and the user's 500-node example remain unchanged by this investigation; improvements are proposed, not implemented. |

| 2026-10-25 Jacobian scaling | Implemented default-enabled `jacobian_scaling` for all four custom root methods, separate from physical `x_scale`. Fresh-Jacobian running column norms shape LM/hybrid steps; physical caps and numerical probes remain unchanged. All six 500-node heat runs converge; controlled LM reaches 4.5e-11 in 12 accepted iterations / 13 calls. Controlled hybrid converges in 26 / 30, versus 20 / 22 unscaled. Updated source/usage/math notes and regenerated demo figures/data. 455 tests pass (10 Hessian cases excluded); 212 convergence configurations retain 100/100 backend roots, 98/101 eligible custom roots, and 51/51 controlled successes. Historical investigation runners explicitly disable scaling to preserve their evidence. |

## Source-layout and domain refactor - 25 October 2026

Implemented the structural stages in [refactor.md](refactor.md). The repository,
distribution, import name and public version remain `pysolver_view` / `0.6.9`.
The implementation is now under [src/pysolver_view](src/pysolver_view), with
lightweight problem contracts, domain-specific frontends/backends/numerical
solvers, explicit lazy public exports, and compatibility modules pointing to
canonical objects. Numerical root modules, KKT mathematics and the SLSQP driver
functions retain their pre-refactor ASTs.

Optimization evaluation contexts own caches and counters; scaling, named
constraint gathering and KKT adaptation are separate modules. Native roots have
an evaluation context; the custom adapter receives an explicit run context.
Public histories/results retain detached data. Passive table, text and plot
renderers consume supplied records, while frontend convenience methods retain
explicit diagnostic evaluation and callback responsibilities.

Promoted the reusable mathematical models, suites and optional heated-gas duct
to `roots/problems/`, with one concrete problem per module. Development imports
forward to those classes; runners and instrumentation remain under development.
Numerical/problem tests now live under `tests/roots/`. The public four-solver
demo imports only the installed package.

Selected behavior changes are documented in the
[migration guide](docs/source/source_layout.rst): imports no longer configure
exception hooks, JAX platform/precision or its logger; `configure_jax()` provides
explicit opt-in. The constrained Rosenbrock example keeps NumPy precision and
uses a local 64-bit JAX context for automatic derivatives. CoolProp loads only
on physical-model construction via the `thermodynamics` extra. PyGMO availability
is independent of its nonfree plugin, and the NLOpt adapter accepts common and
legacy option names. Shared mutable problem scaling, general derivative-policy
changes, logger ownership and the historical Lorentz spelling remain separate
follow-ups.

Validation on Windows / Python 3.13.2 / SciPy 1.17.1:

- Before modification: **464 tests and 184 subtests passed**, excluding the 10
  Hessian cases specified as outside the migration scope.
- Final package tests: **475 passed, 166 subtests passed, 10 deselected**.
  Separate development study/instrumentation checks: **30 passed, 18 subtests
  passed**. Total: **505 tests and 184 subtests passed**, without warnings.
- All **212 convergence rows exactly match** the saved pre-refactor source,
  including success, iterations, residual/Jacobian counts and numerical metrics.
  Backend comparison: 100/100 successful; custom comparison: 98/101 eligible
  successful, including 51/51 controlled successes. The three uncontrolled
  failures, six unsupported cases and five inconsistent-system controls retain
  their baseline outcomes. See the new
  [backend report](development/root_step_control/results_refactor_convergence/REPORT.md)
  and [custom report](development/root_step_control/results_refactor_custom_convergence/REPORT.md).
- Repeated the full interface timing study with **3,168 timed solves per source
  version**. All 24 configurations retain counts and successful roots. The median
  ratio of public runtime after/before was 0.947, with variable individual cases.
  A further 2,400 timed solves in alternating processes examined the two largest
  apparent slowdowns; neither reproduced the slowdown. These measurements remain
  subject to workstation noise and do not establish a speedup. See
  [interface timings](development/root_step_control/results_refactor_adapter_perf/RESULTS.md)
  and its `focused_comparison.json`.
- Editable installation works. Poetry-core 2.1.2 built both the sdist and a wheel
  **from that sdist**. Installed the wheel in a fresh environment and ran
  [wheel_smoke.py](tests/packaging/wheel_smoke.py) from outside the checkout with
  `python -I`, without source-path injection. Both the no-JAX/no-CoolProp run and
  the separate thermodynamics-extra run passed. Confirmed installed paths,
  compatibility identities, native/custom solves, promoted models, and SciPy
  license inclusion in both archives. The wheel excludes development and demos.
- All eight solves in the standalone four-method demo passed using the installed
  wheel from outside the checkout. The constrained optimization example also
  converges without JAX.
- API pages regenerated; Sphinx HTML build succeeds **without warnings**.
  Ruff's undefined/unused-name checks, Black checks and `git diff --check` pass.
  `poetry check --lock` passes with the existing Poetry-metadata deprecation
  notices; the metadata style was intentionally retained.

At the user's request, workflow modernization and the extra CI jobs were
reverted. The original jobs, triggers, action versions, Python/Poetry setup,
publishing and documentation deployment behavior are retained. The sole workflow
change is calling `tests/run_tests.py` in the two existing test jobs. Test paths,
enabled suites, pytest defaults and headless plotting are controlled in that
master script; its default runs both package and development checks with the
agreed Hessian exclusion. The user selected the full suite and authorized the
necessary SciPy dependency update: Python >=3.11 selects SciPy 1.17.1, while
Python 3.10 retains its existing range. Wheel and optional-model gates described
above were performed locally, rather than added to the workflows.
Invoking the master runner with no arguments from outside the repository passed
505 tests and 184 subtests, with the existing 10 Hessian exclusions. Confirmed
that workflow contents match the originals exactly except the two runner calls;
the test-discovery configuration added to `pyproject.toml` was removed so test
selection and defaults have one home in the Python script.
PyGMO/IPOPT/SNOPT execution, Hessian validation, downstream application migration
and publication remain outside this change. No release version was bumped.
The user's existing edit to `demos/demo_slsqp_controls/run_sensitivity.py` was
preserved.
