# PySolverView progress

Last updated: **2026-10-25**

Keep this file to completed milestones and outstanding tasks. Implementation
notes and reproduction commands live in [docs/development](docs/development/README.md).

## Completed

- [x] Correct final-result ownership, point-consistent histories, and non-mutating plots.
- [x] Separate iteration, model-call, derivative-probe, and diagnostic accounting.
- [x] Consolidate KKT assessment, multiplier handling, bounds, and acceptance diagnostics.
- [x] Package custom SLSQP with recovery, step controls, cancellation, and its SciPy license.
- [x] Integrate LM, Powell hybrid, Newton-Raphson, and Gauss-Newton through
  `library="pysolver"`, including aliases, backtracking, and Jacobian scaling.
- [x] Reorganize the package under `src/`, retaining compatible public imports.
- [x] Isolate evaluation caches and counters per run; preserve detached results/history.
- [x] Make JAX/CoolProp lazy, separate PyGMO/plugin detection, and remove import-time
  host configuration; provide explicit `configure_jax()`.
- [x] Move reusable models/audits into the package and all tests under `tests/`;
  retain focused study runners and consolidate development notes.
- [x] Validate packaged wheels, compatibility imports, license inclusion, and
  optional thermodynamics models outside the checkout.
- [x] Modernize documentation with Markdown, the shared builder, working tutorials,
  and updated deployment/version handling; remove redundant build wrappers.

## Remaining

### Correctness and usability

- [ ] Move optimization scaling state off shared problem objects; define solver
  reuse/concurrency guarantees and handle fixed/unbounded variables explicitly.
- [ ] Complete backend capability checks and expose effective options consistently;
  clarify population-method termination and externally tuned defaults.
- [ ] Unify first-derivative shape, accounting, and failure conventions; address
  invalid probes near model-domain boundaries.
- [ ] Make logger setup independent of root logging and repeated handler attachment.
- [ ] Add a supported result/history serialization format and comparison plots
  against evaluations or elapsed time.
- [ ] Add optional throttled live plotting from recorded data.

### Consumer validation and release

- [ ] Confirm the Turbodash migration against turbine/API checks, independent KKT
  audits, physical reevaluation, histories/exports, and progress/cancellation.
  Its local imports/dependency are already switched to PySolverView.
- [ ] Finalize the consumer compatibility inventory and release notes, including
  intentional API, numerical, and option/default changes.
- [ ] Verify the declared Python/SciPy support matrix and repeat package/artifact
  checks on the final release candidate.
- [ ] Publish the approved `0.7.0` release, then replace consumers' editable
  dependencies with release constraints and validate clean installations.

PyGMO/IPOPT/SNOPT execution and Hessian validation remain outside the agreed
release gates. Longer-term improvements above are not automatically release blockers.
See [current limitations](docs/development/limitations.md) and
[validation commands](docs/development/validation.md) for details.
