"""Run with an installed wheel, outside the checkout, using python -I.

The default gate requires an environment without JAX or CoolProp. Install the
thermodynamics extra and rerun with --thermodynamics for the physical-model gate.
"""

import argparse
import importlib.metadata
import importlib.resources
import importlib.util
from pathlib import Path
import sys

import numpy as np

import pysolver_view as psv


def main(thermodynamics=False):
    installed = Path(psv.__file__).resolve()
    assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
    assert "site-packages" in installed.parts, installed
    assert psv.__version__ == importlib.metadata.version("pysolver_view")
    assert "development" not in sys.modules
    if not thermodynamics:
        assert importlib.util.find_spec("CoolProp") is None
        assert importlib.util.find_spec("jax") is None

    from pysolver_view.nonlinear_system import NonlinearSystemProblem
    from pysolver_view.optimization import OptimizationProblem
    from pysolver_view.solvers.slsqp import DomainError, Evaluation, SLSQPOptions
    from pysolver_view.solvers.root.options import StepControlOptions
    from pysolver_view.solvers.root.exceptions import DomainError as RootDomainError
    from pysolver_view.roots.problems import BroydenTridiagonalProblem

    assert NonlinearSystemProblem is psv.NonlinearSystemProblem
    assert OptimizationProblem is psv.OptimizationProblem
    assert DomainError is RootDomainError is psv.DomainError
    assert Evaluation is psv.Evaluation and SLSQPOptions is psv.SLSQPOptions
    assert StepControlOptions is psv.StepControlOptions

    for library, methods in (
        ("scipy", ("lm", "hybr")),
        ("pysolver", ("lm", "hybr", "newton", "gn")),
    ):
        for method in methods:
            problem = BroydenTridiagonalProblem(6)
            solver = psv.NonlinearSystemSolver(
                problem,
                library=library,
                method=method,
                print_convergence=False,
                tolerance=1e-9,
                max_iterations=500,
            )
            point = solver.solve(problem.x0)
            assert solver.success, (library, method, solver.message)
            assert np.linalg.norm(problem.residual(point)) < 1e-8
            assert solver.result.nfev >= solver.result.njev

    for library in ("scipy", "pysolver"):
        solver = psv.OptimizationSolver(
            psv.RosenbrockProblem(2),
            library=library,
            print_convergence=False,
            tolerance=1e-9,
            max_iterations=500,
            extra_options={"optimality_tol": 1e-4} if library == "pysolver" else {},
        )
        point = solver.solve([1.1, 1.1])
        assert solver.success, (library, solver.message)
        np.testing.assert_allclose(point, [1, 1], atol=2e-4)

    license_file = importlib.resources.files(
        "pysolver_view.optimization.solvers.slsqp"
    ).joinpath("SCIPY_LICENSE.txt")
    assert "Redistribution" in license_file.read_text(encoding="utf-8")
    files = importlib.metadata.files("pysolver_view")
    assert not any(
        str(f).startswith(("development/", "demos/", "tests/")) for f in files
    )

    if thermodynamics:
        from pysolver_view.roots.problems import HeatedGasDuctProblem

        problem = HeatedGasDuctProblem(n_cells=3)
        solver = psv.NonlinearSystemSolver(
            problem,
            library="pysolver",
            method="hybr",
            print_convergence=False,
            tolerance=1e-8,
            options={
                "x_scale": problem.x_scale,
                "initial_radius": 100.0,
                "max_radius": 1e6,
            },
        )
        point = solver.solve(problem.x0)
        assert solver.success, solver.message
        assert np.linalg.norm(problem.residual(point)) < 1e-7
    print(
        f"Installed-wheel imports, identities, solves and license passed: {installed}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermodynamics", action="store_true")
    main(parser.parse_args().thermodynamics)
