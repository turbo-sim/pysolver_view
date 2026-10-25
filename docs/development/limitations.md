# Confirmed limitations and validation scope

These are current implementation boundaries checked against the source. Completed
migration defects and speculative third-party integrations are omitted.

## Shared model state and scaling

`OptimizationSolver.__init__` still writes `problem.problem_scale`. Reusing one
problem with differently configured solvers can therefore change shared state.
Evaluation contexts isolate run caches, but do not make arbitrary model instances
or mutable CoolProp states safe for concurrent use.

Optimization's existing normalization is based on variable bounds. A general
independent variable-scale policy for fixed or unbounded variables has not been
introduced. Custom SLSQP requires fixed variables to be eliminated. Root
`x_scale`, numerical equilibration and application residual normalization have
separate meanings; none automatically establishes physical domain validity.

## Backend coverage

Optimization capability metadata is verified for native/custom SLSQP, while
many other registry entries retain unknown capability fields. Listing a method
does not enforce every constraint, derivative, event or cancellation combination.
For example, the SciPy adapter does not supply Hessians/Hessian products needed
by some listed trust-region methods. Inspect the adapter before choosing those
combinations.

PyGMO, IPOPT and SNOPT adapters remain present, but execution is outside routine
validation. Interface checks and static adaptation do not establish numerical
behavior of their installed native solvers. Native trial-radius or domain-error
controls for those backends have not been added by the root-control work.

Custom SLSQP relies on a private SciPy 1.17.1 bridge. Supporting a new SciPy
version requires bridge review and validation. Python 3.10's supported SciPy
range does not run this custom bridge or the complete suite.

## Derivatives and dense roots

Custom roots use dense linear algebra and do not provide sparse solves, variable
bounds, robust losses or automatic residual weighting. Root stationarity is not
root success. Invalid derivative probes stop with a diagnostic; adaptive
smaller/opposite-sided domain-aware probing is not implemented. A trial-distance
cap does not constrain numerical derivative probes.

The general differentiation helpers and custom root evaluator retain distinct
shape, probe and failure conventions. The private SciPy numerical-difference
dependency is isolated, not eliminated. `LorentzEquations.gradient` also makes
internal numerical model calls that frontend counters cannot observe. The
historical `LorentzEquations` spelling remains supported.

## Host utilities and evidence

Package import avoids global host configuration, but an explicit call to
`reporting.logging.create_logger` still uses root logging configuration and may
accumulate handlers across repeated setup calls.

The CoolProp duct studies exercise a single-phase model and chosen starts.
They do not validate phase boundaries, critical states, nested flashes or a full
turbomachinery application. The radiative heat demo is a separate model.
Performance measurements apply to their recorded machine, settings and storage
policy. Optional-backend execution, Hessian validation and downstream application
migration are not established by the routine test results.

For current checks and reproduction commands, see [validation](validation.md).
