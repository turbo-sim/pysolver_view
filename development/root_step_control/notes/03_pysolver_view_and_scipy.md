# Keeping the controls in pysolver_view and proposing them to SciPy

Recommendation: keep this work in pysolver_view as an explicit experimental
capability, then promote it after testing representative thermodynamic models.
It is useful enough to discuss with SciPy maintainers now, but the present
implementation is not ready to submit as a replacement for SciPy's LM or
hybrid solvers. The reusable idea is control of attempted evaluations; the
current code is one implementation of that idea.

## A suitable role for pysolver_view

pysolver_view is a sensible place to combine model interfaces, scaling,
convergence histories, evaluation failures, and selectable solver backends.
Thermodynamic equations and property-library details should remain in the
application's `NonlinearSystemProblem` subclasses. A generic solver can understand
that an evaluation is undefined without knowing which fluid or equation of state
caused the failure.

Keep the backend distinction visible when integrating the experiments:

| Implementation | What pysolver_view would own | Maintenance consequence |
| --- | --- | --- |
| Existing SciPy `root` adapter | Options, model calls, reports and final residual checks | Retains mature native MINPACK behavior |
| Turbodash's controlled SLSQP | Reverse-communication driver and coordinated private-state edits | Requires an explicit SciPy version compatibility policy |
| Experimental controlled LM | Trust-region loop, damping solve and acceptance policy | Own numerical implementation and its validation |
| Experimental controlled hybrid | Dogleg steps, Broyden updates, refresh and acceptance policy | Own numerical implementation; differs from native HYBRJ |

Use backend names that communicate this distinction, such as
`scipy_minpack` and `experimental_controlled`, alongside the method name. These
are proposed API names, not options currently implemented. Do not silently
replace existing `method="hybr"` behavior when adding the controlled variant to
the public solver class.

A shared reporting contract should distinguish accepted iterates, attempted
solver steps, derivative probes, and rejected domain evaluations. Share scaling
and error-reporting utilities where appropriate, but let each algorithm retain
its own acceptance and state-update logic. Root residual descent, an SQP merit
line search, and an interior-point filter are different mechanisms.

## Evidence needed before routine thermodynamic use

The [current results](../results_convergence/REPORT.md) compare the seven standard systems and
additional 100-variable workloads. The tests cover actual step limits,
accepted-state consistency, domain recovery, rank-deficient models and history
storage modes. A real CoolProp duct workload now supplies property-library
calls and conservation checks. Its single-phase regime and chosen start do not
establish reliability near saturation, critical conditions, phase changes, or
difficult flash calculations.

Before promoting the new backends, I would complete these steps:

1. **Build a thermodynamic regression collection.** Use representative coupled
   flow, energy, pressure and property equations through `NonlinearSystemProblem`.
   Record the fluid, property backend/version, valid state assumptions, equation
   scaling and physically meaningful root criteria. Test multiple initial guesses
   and operating conditions, not only previously successful starts.
2. **Measure expensive and failed work.** Count property calls and time as well
   as residual/Jacobian calls. Record every undefined trial, largest physical
   component changes, recovery attempts, final residuals and selected solution
   branch. A safer trajectory may justify more residual calls.
3. **Isolate each control.** Compare native `factor`, initial radius alone,
   persistent cap alone, growth policy, and domain recovery with otherwise equal
   settings. The current controlled preset changes both maximum radius and
   growth, so their individual effects have not been isolated.
4. **Protect derivative evaluations.** A capped trial does not protect forward
   differences. Test analytic derivatives where available and design explicit
   bound/domain-aware numerical probes. Verify derivatives and failure behavior
   near the actual model's difficult regions. The current experiment stops on an
   invalid derivative probe; it does not recover it.
5. **Check numerical and implementation limits.** Include inconsistent systems,
   nearly singular Jacobians, noisy residuals, scaling extremes and larger
   dimensions. PySolverView maintains hybrid QR factors with rank-one updates
   and uses a QR Gauss–Newton path for LM. Damped/rank-deficient cases still
   require SVD.
   Measure these paths under realistic controls; generous-step timing alone
   does not predict the cost of a cautious thermodynamic solve.
6. **Make integration reproducible.** Keep native backends available, document
   status semantics, test supported dependencies, preserve third-party notices
   when reusing code, and add explicit compatibility tests for the SLSQP private
   bridge. Fix final-point and history semantics before public integration.

Continuation between nearby operating points can complement these controls.
It reduces the distance a solve needs to travel; a maximum step then protects
individual trial evaluations along that path.

The [timing study](../results_time_perf/RESULTS.md) compares the custom implementation
with native SciPy at 20 and 100 variables. Actual property work can dominate
the remaining solver overhead, while inexpensive small systems expose Python
bookkeeping costs. Generous-radius and actively damped runs must be assessed
separately because the latter perform additional SVD work and may require
many more property evaluations.

For another optimization pass, prioritize a validated QR-based damped
least-squares subproblem to avoid QR followed by SVD on most tightly restricted
LM steps. MINPACK's augmented triangular-factor approach is a useful reference,
but adopting it requires independent implementation and correctness tests.
This would be a further change to the custom implementation. Preserve full
histories while validating physical safety;
use summary/disabled storage only when detailed coordinate traces are unnecessary,
since the current timing data do not establish reliable speed gains from less logging.

## A realistic SciPy contribution

The public LM and hybrid root APIs currently document `factor` and `diag`, but
no persistent maximum radius. This is a concrete feature gap with applications
beyond thermodynamics. Live documentation checked on 24 October 2026 identifies
itself as SciPy 1.18.0; the local experiment and native-source review use 1.17.1.
See the official [LM options](https://docs.scipy.org/doc/scipy/reference/optimize.root-lm.html)
and [hybrid options](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html).

My suggested first proposal is **optional explicit initial and maximum scaled
trial radii in native MINPACK-backed root solvers**, with existing behavior
preserved when they are omitted. Agree on the metric, interaction with `diag`
and automatic scaling, and whether the bound is strict or approximate before
implementing it. A small reproducible coupled example should demonstrate both
the native excursion and the requested guarantee.

This requires changes below the public Python options parser. MINPACK owns its
iteration state inside the native call. Its LM parameter calculation accepts
approximately 10% tolerance in the radius equation, so merely capping the
internal radius does not establish a strict cap on actual evaluated steps.
A native implementation must consistently update the step, predicted reduction,
trust-region state and stopping logic. See the tagged
[SciPy MINPACK source](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/optimize/__minpack.c).

Domain-failure recovery could be a subsequent proposal with a precisely defined
callback/error protocol. Do not combine a first radius feature with a mandatory
new monotone acceptance rule, a replacement linear-algebra implementation, and
thermodynamic-specific exceptions. That would make correctness and compatibility
much harder to assess. A supported SLSQP control interface would likewise be a
separate proposal; it could eventually remove reliance on private workspace
layout.

SciPy's contribution guidance recommends discussing new features on its
development forum before implementation is complete. Inclusion also requires
documentation, tests, benchmarks, appropriate style and compatible licensing.
The present experiments can supply motivation and examples, but maintainer
agreement and a narrowly scoped native patch are separate work.
[SciPy contribution guidance](https://docs.scipy.org/doc/scipy/dev/hacking.html).

Because this prototype was developed with AI assistance, any eventual SciPy
submission also needs human ownership: the contributor must understand and
review the implementation, disclose the tools and generated portions, and
submit it themselves. SciPy's current policy excludes autonomous agent PRs and
AI-written maintainer communication except translation or grammar editing.
These notes are technical preparation, not a ready-to-post PR or forum
message. [SciPy AI policy](https://github.com/scipy/scipy/blob/main/doc/source/dev/conduct/ai_policy.rst).
