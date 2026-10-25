# Source layout and compatibility

The implementation lives in `src/pysolver_view`. Install the checkout with
`python -m pip install -e .` or `poetry install` before running scripts.
The repository, distribution and Python import remain `pysolver_view`;
the public version remains 0.6.9 pending a separate release decision.

## Canonical modules

* `roots.solver` and `optimization.solver` coordinate the public interfaces.
* Each domain's `problems` package contains its lightweight contract and one
  module per concrete problem. `roots.problems` also contains the reusable
  mathematical benchmarks, suites and optional heated gas duct.
* `backends` translates backend-specific calls and `solvers` contains the
  custom numerical algorithms. Custom SLSQP's bridge and SciPy license live in
  `optimization/solvers/slsqp`.
* Evaluation contexts own per-run caches and physical-call counters.
  `optimization.scaling`, `constraints` and `kkt_adapter` separate
  coordinate conversions, constraint data and diagnostic adaptation.
* `diagnostics.kkt` remains the independent mathematical implementation.
  `reporting` formats supplied records; explicit diagnostic requests in the
  frontend still perform and count the necessary model work.

For example:

```python
from pysolver_view import NonlinearSystemSolver
from pysolver_view.roots.problems import BroydenTridiagonalProblem
from pysolver_view.optimization.problems import OptimizationProblem

problem = BroydenTridiagonalProblem(10)
solver = NonlinearSystemSolver(problem, library="pysolver", method="hybr")
solution = solver.solve(problem.x0)
```

Existing top-level imports and the historical `nonlinear_system`,
`optimization_problems`, `nonlinear_system_problems`,
`numerical_differentiation`, `pysolver_utilities`, `custom_root`,
`custom_optimization`, `optimization_wrappers` and `solvers` paths remain
available. Classes, options, exceptions and entry functions resolve to the same
canonical objects. `pysolver_view.optimization` is now a package that preserves
its public classes, packing helpers and backend dispatch dictionaries.

Constructor arguments, solution vectors, result ownership, history keys,
inequality signs, scaling conventions and callback contracts remain supported.
Native observations and custom accepted iterations retain their distinct
meanings. Numerical root algorithms, KKT mathematics and SLSQP's state machine
were relocated without redesigning their numerical policies.

Tests that patch implementation globals must patch their defining modules:
`optimization.backends.scipy.minimize`, `roots.backends.scipy.root` and
`optimization.solvers.slsqp._scipy_backend`. Compatibility re-exports do not
redirect a function's global namespace.

## Explicit host configuration

Importing the package no longer replaces `sys.excepthook`, selects JAX's CPU
platform, changes JAX precision or adjusts its logger. This is an intentional
behavior change. Applications that want the old JAX policy can opt in before
using JAX:

```python
import pysolver_view as psv
psv.configure_jax(enable_x64=True, platform="cpu")
```

`JAX_AVAILABLE` and `is_jax_available()` query availability without loading
JAX. Explicit access to the compatibility attributes `jax` and `jnp` loads
it when installed and returns `None` otherwise. The constrained Rosenbrock
example uses NumPy for ordinary calls and a local 64-bit JAX context for automatic
derivatives, preserving model precision without changing global configuration.

CoolProp loads only when constructing `HeatedGasDuctProblem`. Install it with
`python -m pip install 'pysolver_view[thermodynamics]'`. Mathematical models,
problem contracts and numerical solvers work without it. Study orchestration,
timing instrumentation and results remain under `development`. The former
development forwarding modules have been removed. All reusable models and the
SLSQP benchmark audit helpers are installed in the package.

PyGMO availability is independent of its nonfree plugin; SNOPT loads that plugin
only when requested. The NLOpt adapter accepts the common `tolerance` and
`max_iterations` names as well as the old `tol` and `max_iter` names.
These adapter contracts have isolated checks; PyGMO, IPOPT and SNOPT execution
is outside this migration's validation scope.

## Validation and packaging

Run `poetry run python tests/run_tests.py` locally and in CI. This master script
owns the test paths, enabled suites, default pytest options, cache location and
headless plotting default. Edit `TEST_SUITES`, `ENABLED_SUITES` and
`PYTEST_OPTIONS` there to control testing. It discovers all checks under `tests/`
by default. Use `--suite studies` to select the benchmark and study checks;
additional arguments are forwarded to pytest.
Hessian helpers retain their implementation and remain outside the agreed
migration validation scope.

Custom SLSQP's private bridge requires SciPy 1.17.1. The dependency metadata and
lockfile select that version on Python 3.11 and newer, including the existing
Python 3.12 CI jobs. Python 3.10 retains the existing SciPy range for native
methods and custom roots. The full test suite requires Python 3.11 or newer.
CoolProp tests remain optional when the thermodynamics extra is absent.

The GitHub test jobs invoke `tests/run_tests.py`; test selection and options
are controlled by that script. Documentation deployment runs
`poetry run python docs/build_docs.py --no-autobuild` and publishes
`docs/_build/html/`. The authored documentation lives in `docs/source/` as
Markdown, with API pages regenerated from docstrings on every build.
These documentation changes do not change package publication or runtime
dependencies.

`python -m build` builds an sdist and then builds its wheel from that sdist.
Install the wheel into a fresh environment with SciPy 1.17.1. From outside the
checkout, run that environment's Python with `-I` against
`tests/packaging/wheel_smoke.py`. This checks installed paths, compatibility
identity, native/custom solves, promoted models and license inclusion without
JAX or CoolProp. Install CoolProp and repeat with `--thermodynamics` to check
the optional physical model separately.

Shared mutable problem scaling, fixed/unbounded scaling policy, logger ownership,
general derivative convention changes and numerical algorithm changes remain
separate follow-up work. The historical `LorentzEquations` spelling remains
supported unchanged.
