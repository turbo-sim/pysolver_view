# Working in pysolver_view

This guide applies to the whole repository. Follow the user's current task and
preserve unrelated working-tree changes. Read `README.md`, the relevant code,
and nearby tests before editing; historical design notes describe proposals as
well as implemented behavior.

## Environment and imports

- The repository, distribution, and Python import are `pysolver_view`.
  Examples generally use `import pysolver_view as psv`.
  `library="pysolver"` selects the custom backend; it is not an import name.
- Use this project's Poetry environment: `poetry install`, then
  `poetry run python ...`. The implementation uses a `src/` layout and must be
  installed, normally in editable mode, before running tests or demos.
- Verify `sys.executable`, `psv.__file__`, and the defining module when imports
  look wrong. A neighboring Turbodash environment can contain an older release.
  Restart existing IPython/IDE kernels after installation or structural changes.
  Do not add `sys.path` workarounds to demos to hide installation problems.
- Dependency constraints live in `pyproject.toml` and `poetry.lock`. The custom
  SLSQP bridge currently requires SciPy 1.17.1 on Python 3.11 or newer. Read its
  version check before changing SciPy; other methods have different requirements.

## Repository structure

- `src/pysolver_view/roots/`: root frontend (`solver.py`), per-run evaluation,
  method aliases, problem contracts/models, backend adapters, and custom solvers.
- `src/pysolver_view/optimization/`: optimization frontend, evaluation, scaling,
  constraints, KKT adaptation, problem models, backends, and custom SLSQP.
- `src/pysolver_view/diagnostics/`: independent KKT mathematics and SLSQP audits.
- `src/pysolver_view/differentiation/`: finite differences, complex step,
  existing Hessian helpers, and optional JAX support.
- `src/pysolver_view/reporting/`: formatting, plots, style, and logging.
  Shared results, histories, exceptions, and validation live at package level.
- `tests/`: checks organized by domain, including studies and packaging checks.
  Add tests here rather than under `development/`.
- `demos/`: short, readable user examples with explanations in nearby READMEs.
- `development/`: reproducible `run_*.py` studies and their generated results.
  Keep study-specific orchestration in the runner that uses it. Put reusable
  algorithms, problems, and audits in the installed package.
- `docs/source/`: Sphinx/API documentation and the source-layout guide.
  `docs/development/`: design notes, numerical explanations, and study guides.
- `PROGRESS.md`: task checklist and historical validation evidence. Update it
  for substantial implementation milestones; label historical measurements.

Use canonical domain modules for new implementation and internal imports.
Legacy modules such as `nonlinear_system.py`, `optimization_wrappers.py`, and
`solvers/root/` forward to canonical objects for compatibility. Keep those
exports pointing to the same classes/functions; do not duplicate implementations.
Patch defining modules in tests, rather than compatibility re-exports.

## Coding and text style

- Use four-space indentation, descriptive `snake_case` functions/variables,
  `PascalCase` classes, and uppercase module constants. Follow nearby code for
  type annotations; avoid unrelated formatting or typing rewrites.
- Use Black's default Python formatting for new or materially edited code.
  The repository contains older formatting: keep changes focused.
- Give new or changed functions and classes clear docstrings, including helpers.
  Explain purpose, input/output shapes, units, coordinate conventions, and side
  effects where relevant. Public numerical APIs generally use NumPy-style
  Parameters/Returns sections; short helpers can use concise docstrings.
- Use ASCII punctuation in code, console tables, labels, comments, and prose.
  Use the actual hyphen-minus `-`, including `Levenberg-Marquardt`,
  `Newton-Raphson`, and `Gauss-Newton`. Avoid typographic dashes and Unicode
  minus signs. Use LaTeX for mathematical symbols in documentation and plots.
- In Markdown, use `$...$` for inline math and `$$...$$` for display math.
  Do not use backslash-parenthesis or backslash-bracket math delimiters.
- Keep optional dependencies lazy. Importing the package must not configure
  the host's exception hook, logging policy, JAX platform, or JAX precision.
  JAX and CoolProp functionality must remain optional.

## Numerical and reporting contracts

- Preserve public argument, import, callback, result, and history contracts
  unless the task explicitly changes them. Numerical trajectories may change;
  correctness and verified convergence matter more than reproducing old iterates.
- Results describe the authoritative returned/accepted point. Record native
  observations, custom accepted iterations, rejected trials, and derivative
  probes according to their actual meanings; do not relabel them as equivalent.
- Finite-difference and complex-step evaluations count as model/function calls,
  never solver iterations. Track gradient/Jacobian and diagnostic work separately.
- Keep evaluation caches and mutable run state in the evaluation/run context.
  Reports and plots consume detached records without mutating history or doing
  hidden model work. Explicit diagnostic requests can evaluate and count work.
- For roots, success requires the residual target. Least-squares stationarity
  or a tiny step above that target is not a successful root. Recognized domain
  failures may permit recovery; do not swallow unrelated application errors or
  fabricate residuals for undefined points.
- Distinguish physical `x_scale` and displacement limits from numerical
  Jacobian equilibration. Enforce limits on actual rounded trial coordinates,
  including rejected trials. Numerical scaling must not change probe distances.
- Respect constraint signs at each interface: public optimization/KKT uses
  inequalities `g(x) <= 0`; the direct SLSQP `Evaluation` uses nonnegative
  inequalities. Convert signs explicitly in adapters, including derivatives
  and multipliers. KKT checks establish first-order conditions, not a general
  proof of a minimum. Preserve optional `check_kkt=False` and unavailable metrics.
- Keep optimizer/backend logic, mathematical diagnostics, and presentation in
  their existing layers. Do not introduce model or solver callbacks into the
  independent KKT mathematics or passive formatters.

## Tests and validation

Run checks through the project's environment:

```bash
# Shared local/CI entry point; selection and defaults live in this script.
poetry run python tests/run_tests.py

# Focus on studies, or run a specific regression directly.
poetry run python tests/run_tests.py --suite studies
poetry run python -m pytest tests/roots/test_custom_root.py -q

# Documentation build, when changing Sphinx/API documentation.
poetry run python docs/build_docs.py --no-autobuild
```

Inspect `tests/run_tests.py` for the current suites and exclusions instead of
adding a second test-discovery configuration. Prefer targeted checks for small
changes and the master runner for changes crossing public interfaces or layers.
Use meaningful convergence, independent residual/KKT, domain, accounting,
mutation, or import tests; avoid brittle historical trajectory requirements.

PyGMO, IPOPT, and SNOPT execution is outside the agreed routine validation scope.
Use native SciPy SLSQP and custom solvers for the normal checks. Existing Hessian
helpers can also serve IPOPT but are noncritical; their validation is excluded
by the master runner unless the current task specifically requires it.
When testing installed packages, use the existing wheel smoke script from
outside the checkout, with isolated Python, rather than injecting source paths.

## Demos, plots, and studies

- New examples should use public interfaces and the installed problem contracts.
  Keep them readable and self-contained, with important parameters exposed near
  the top. Include CLI options and an `if __name__ == "__main__"` entry point
  when running a study or exporting configurable results.
- Use `psv.set_plot_options(...)`. For new solver-comparison figures, use
  `plasma` colors sampled from 0.2 to 0.8 and markers at every recorded point,
  with white faces above the connecting lines. Match existing local conventions
  when extending another plot. Use readable numeric precision and full method
  names; call displacement columns `Step size`.
- Provide headless operation, normally `--no-show`; use `MPLBACKEND=Agg` when
  auditing older examples. Keep raw and accepted histories truthful and explain
  rejected/undefined trials. Do not silently replace a curve by best-so-far data.
- Store generated studies under `results_<case>/` or an explicit output path.
  Avoid overwriting unrelated results or committing generated caches/logs by
  accident. For timing, separate model, derivative, linear algebra, and interface
  costs; control BLAS threads and avoid concurrent performance measurements.

## Versioning and scope

Version management uses `.bumpversion.toml` and `bump-my-version`; commands are
documented at the bottom of `pyproject.toml` and in `README.md`. A real bump
requires a clean tree and creates a commit and tag. Do not bump, tag, push, or
publish unless requested. Keep the SciPy license with the packaged SLSQP bridge.
Do not change neighboring repositories or their environments as part of a local
fix unless the user requests it. Never stage unrelated existing edits or deletions.
