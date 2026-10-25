# Development experiments

[Root step control](root_step_control/README.md) investigates explicit first-step,
maximum-step, and radius-growth controls for Levenberg–Marquardt and Powell-style hybrid root
solvers. It includes standalone implementations, difficult example systems,
regression tests, and reproducible convergence comparisons with SciPy.

[Root-finder extension plan and status](root_finder_extension.md) describes
Newton–Raphson and Gauss–Newton with optional backtracking, now implemented
in development alongside Levenberg–Marquardt and Powell hybrid.

These modules are experiments run from a repository checkout. They do not change
the installed `pysolver_view` solver API.

## Architecture and feature reviews

Migration validation uses SciPy SLSQP as the proxy for existing solver wrappers.
PyGMO, IPOPT, and SNOPT checks are not required; their continued operation is
assumed. Custom solver, shared KKT, root, and Turbodash checks remain in scope.

Hessian helpers also apply to IPOPT but are noncritical. Validation of the
existing helpers is not required for the migration.

- [Proposed solver refactor](proposed_refactor.md): shared numerical code,
  backend integration, migration stages, and validation.
- [KKT comparison and design](kkt_comparison_and_design.md): current PySolverView
  and Turbodash checks, confirmed issues, and a shared diagnostic service.
- [PySolverView feature review](pysolverview_feature_review.md): features to
  retain, repair, simplify, or defer, centered on consistent convergence views.

Finite-difference and complex-step probes count as model/function evaluations,
never solver iterations. Gradient/Jacobian requests are tracked separately.

[Phase 1 result/history validation](result_history_validation.md) records the
initial failing regressions, implemented fixes, and links to short examples.

[Shared KKT and custom SLSQP](kkt_slsqp_validation.md) describes the implemented
package backend, reports, mathematical conventions, validation, and remaining
consumer migration. Run the [named KKT demo](../demos/demo_kkt_conditions/README.md)
to compare native SciPy and custom SLSQP on a known constrained minimum.

[Controlled SLSQP studies](slsqp_controls/README.md) adapts the Turbodash audit
ideas to small mathematical problems. It includes independent KKT/derivative
checks, a tolerance matrix, and a [sensitivity demo](../demos/demo_slsqp_controls/README.md)
for initial fractions, persistent caps, growth, and domain recovery.
