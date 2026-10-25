# Root step control for thermodynamic models

These notes address the design questions raised after the root-step experiments.
They describe the code currently present and distinguish it from proposed work.
The [folder guide](../README.md) lists the source, tests, runners, and outputs.
For practical details, see the [timing protocol](06_time_performance.md) and
[solver usage guide](07_usage_and_controls.md).
Documentation and local implementation reviewed on 25 October 2026; the numerical
experiment used SciPy 1.17.1.

**Recommendation:** keep explicit step controls in pysolver_view as opt-in
capabilities, and validate them on actual thermodynamic problems before making
the new root implementations production defaults. Start with a physically
meaningful small trial radius, retain a persistent maximum, reject undefined
evaluations, require sufficient residual reduction for accepted root steps, and
restore larger steps only when the local model earns that increase.

| Question | Answer | Detailed note |
| --- | --- | --- |
| Should the first step be small and reduce the residual? | Limit the first trial before evaluation; require the first **accepted** step to reduce the scaled residual norm. Validity and reduction cannot generally be guaranteed before the trial is evaluated. | [Thermodynamic step policy](01_thermodynamic_step_policy.md) |
| Are the new root solvers sophisticated MINPACK wrappers? | No. They are independent Python LM and Powell-style hybrid iteration loops using public linear algebra. Your existing SLSQP driver does control a native SciPy backend. | [What was implemented](02_implementation_and_backend_boundaries.md) |
| Should these controls live in pysolver_view? | Yes, with explicit backend identity, shared diagnostics, and documented differences in supported controls. Keep application-specific thermodynamic validity in the problem adapter. | [Library and SciPy roadmap](03_pysolver_view_and_scipy.md) |
| Is this ready for a SciPy PR? | It supports a feature discussion and a reproducible motivation. The current experimental solvers are not yet a ready replacement or extension of native MINPACK. | [Library and SciPy roadmap](03_pysolver_view_and_scipy.md) |
| Can IPOPT and SNOPT use similar protection? | Yes in principle, but start with their native controls and failure-reporting interfaces. Their step and acceptance rules differ from the root solvers. | [IPOPT and SNOPT](04_ipopt_and_snopt.md) |
| How do LM and hybrid work mathematically? | Both use a scaled least-squares model; LM solves a damped trust-region subproblem, while hybrid combines dogleg steps with Broyden updates. | [Mathematics and step geometry](05_lm_and_hybrid_mathematics.md) |
| How do Newton–Raphson and Gauss–Newton differ? | Newton requires a square nonsingular Jacobian; Gauss–Newton also supports rectangular/rank-deficient models. Both have optional Armijo backtracking and scaled limits. | [Newton-family mathematics](08_newton_and_gauss_newton.md) |

The implementation uses QR-based fast paths, maintained hybrid QR updates, lazy SVD,
and optional scalar or disabled histories. The control policy remains independent
of the storage level. [Implementation details](02_implementation_and_backend_boundaries.md)
explain the numerical methods; [timing results](../results_time_perf/RESULTS.md) compare
the custom solvers with native SciPy.

The [four-method custom convergence report](../results_custom_convergence/REPORT.md)
adds Newton–Raphson and Gauss–Newton. Full-step runs are explicitly identified:
their valid accepted residuals may increase, and an undefined trial ends the run.
The coupled logarithmic domain test is a mathematical recovery test, distinct
from the real-property duct benchmarks.

The timing study separates generous-radius comparisons with native SciPy
from custom runs with active conservative controls. Damped LM may invoke SVD
after QR, while hybrid reuses updated QR factors. Summary histories reduce
stored data, but wall-time effects must be judged from the measured variability.
These are workload-specific measurements, not claims of native MINPACK
equivalence or universal speed.

The next engineering task is to extend the thermodynamic validation
suite with application-specific `NonlinearSystemProblem` subclasses, difficult initial guesses,
validity-aware property evaluations, checked derivatives, and measurements of
failed property calls as well as convergence. The current suite already includes
20- and 100-variable single-phase CoolProp duct systems. Difficult flash/phase
boundary models and multiple operating-point studies remain future work.

The [experiment report](../results_convergence/REPORT.md) contains the repeated seven-problem
multidimensional study and the additional 100-variable comparisons. The suite
distinguishes residual-target success, trial-distance audits and execution cost.
It demonstrates controllable trial distances, not universal robustness for
thermodynamic models. All sources, results and notes needed to reproduce these
studies are stored under this directory.
