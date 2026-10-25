
# PySolverView

PySolverView provides a common interface and convergence histories for optimization
and root finding, including native SciPy methods and custom solvers with step controls.

## Features

- Print and visualize convergence history
- Streamlined problem definition (object-oriented)
- Improved function and gradient evaluation to avoid unnecesary model evaluations
- Controlled Levenberg-Marquardt, Powell hybrid, Newton-Raphson, and Gauss-Newton
  through `NonlinearSystemSolver(library="pysolver")`; native SciPy remains the default

The [root solver guide](docs/development/root_solvers.md)
explains options, scaling, callbacks, and result accounting. Run the single
[four-solver demo](demos/demo_eqn_solving/demo_custom_root_solvers.py) to compare
accepted convergence histories. Benchmark suites and timing studies remain under
[development studies](docs/development/README.md).

Custom roots enable Jacobian-based numerical scaling by default; disable it
with `options={"jacobian_scaling": False}`. User-defined physical step scales
and limits retain their meaning. The
[radiative heat demo](demos/demo_eqn_solving/README.md) compares controlled and
large-step runs on a 1000-variable collocation problem.

## Documentation

Install the checkout before running tests or demos; the package lives under
`src/pysolver_view/`:

```bash
python -m pip install -e .
# Optional physical benchmark models:
python -m pip install -e '.[thermodynamics]'
```

Install development dependencies with `poetry install --with dev --extras jax`. Add the
`thermodynamics` extra to run the optional CoolProp model checks as well.
Custom SLSQP requires SciPy 1.17.1, which is selected automatically on Python
3.11 and newer. Python 3.10 retains its existing SciPy dependency range for
native methods and custom roots; the full test suite needs Python 3.11 or newer.

```bash
poetry run python tests/run_tests.py
# Run just one of the configured suites:
poetry run python tests/run_tests.py --suite package
poetry run python tests/run_tests.py --suite studies
poetry run python docs/build_docs.py --no-autobuild
```

Documentation is authored in Markdown under `docs/source/`; the builder generates
the API reference and writes HTML to `docs/_build/html/`. Omit `--no-autobuild`
for a live preview. See the [documentation build guide](docs/README.md) for
the source structure, authoring conventions, and additional build options.

Control test paths and default pytest options in
[`tests/run_tests.py`](tests/run_tests.py). By default it runs the complete
`tests/` tree, including study checks, with Hessian validation excluded from the
agreed migration scope. Both existing CI test jobs call this same script.

The [source-layout migration guide](docs/source/source_layout.md) documents
canonical imports, compatibility exports, optional dependencies and the explicit
JAX configuration API. The package name, distribution name and version remain unchanged.

Please refer to the [`documentation`](https://turbo-sim.github.io/pysolver_view/index.html) for:
- Installation instructions
- Tutorials showing how to use the code
- Complete documentation of all functions and classes in PySolverView


## Version bumping

Version bumps use `bump-my-version`, configured in
[.bumpversion.toml](.bumpversion.toml). Install the development tools with
`poetry install --with dev`, then commit your work before bumping:

```bash
# Preview the next patch version without modifying files.
poetry run bump-my-version bump --dry-run --verbose patch

# Choose one increment: patch, minor, or major.
poetry run bump-my-version bump patch

# Push the version commit and release tag from main.
git push origin main --follow-tags
```

The tool updates `pyproject.toml`, `src/pysolver_view/__init__.py`,
and its own current-version setting. `docs/conf.py` reads the package version
automatically. It creates a version commit and an annotated `vX.Y.Z` tag;
pushing the tag triggers the release workflow.
A real bump requires a clean working tree.

## Other info
- Pygmo and pygmo-plugin-nonfree should be installed with Conda
