"""Canonical identities and lightweight imports across the source-layout migration."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "old,new,names",
    [
        ("optimization", "optimization.solver", ["OptimizationSolver"]),
        ("optimization", "optimization.problems.base", ["OptimizationProblem"]),
        ("nonlinear_system", "roots.solver", ["NonlinearSystemSolver"]),
        ("nonlinear_system", "roots.problems.base", ["NonlinearSystemProblem"]),
        (
            "optimization_problems",
            "optimization.problems",
            [
                "RosenbrockProblem",
                "RosenbrockProblemConstrained",
                "HS71Problem",
                "LorentzEquationsOpt",
            ],
        ),
        ("nonlinear_system_problems", "roots.problems", ["LorentzEquations"]),
        (
            "solvers",
            "optimization.solvers.slsqp",
            ["DomainError", "Evaluation", "SLSQPOptions", "minimize_slsqp"],
        ),
        (
            "solvers.slsqp",
            "optimization.solvers.slsqp",
            ["DomainError", "Evaluation", "SLSQPOptions", "minimize_slsqp"],
        ),
        ("solvers.exceptions", "exceptions", ["DomainError"]),
        ("solvers.root.exceptions", "exceptions", ["DomainError"]),
        (
            "solvers.root",
            "roots.solvers",
            [
                "StepControlOptions",
                "NewtonOptions",
                "BacktrackingOptions",
                "StepLimitOptions",
                "solve_root",
                "solve_levenberg_marquardt",
                "solve_powell_hybrid",
                "solve_newton_raphson",
                "solve_gauss_newton",
            ],
        ),
        (
            "solvers.root.options",
            "roots.solvers.options",
            ["StepControlOptions", "NewtonOptions"],
        ),
        ("solvers.root.linear_model", "roots.solvers.linear_model", ["LinearModel"]),
        ("solvers.root.scaling", "roots.solvers.scaling", ["JacobianScaling"]),
        (
            "solvers.root.solver",
            "roots.solvers.solver",
            ["solve_root", "normalize_method"],
        ),
        (
            "optimization",
            "optimization.constraints",
            ["count_constraints", "combine_objective_and_constraints"],
        ),
        (
            "numerical_differentiation",
            "differentiation",
            [
                "approx_gradient",
                "approx_derivative",
                "forward_finite_differences",
                "central_finite_differences",
                "complex_step_derivative",
                "approx_jacobian_hessians",
                "compute_hessians_jax",
            ],
        ),
        (
            "pysolver_utilities",
            "reporting.style",
            ["set_plot_options", "savefig_in_formats"],
        ),
        (
            "pysolver_utilities",
            "validation",
            ["DictionaryValidationError", "validate_keys"],
        ),
        (
            "custom_root",
            "roots.backends.custom",
            ["RootSolverAdapter", "resolve_root_options"],
        ),
        ("custom_optimization", "optimization.backends.custom", ["minimize_pysolver"]),
        (
            "optimization_wrappers",
            "optimization.backends.scipy",
            ["minimize_scipy", "convert_pygmo_to_scipy_bounds"],
        ),
        ("optimization_wrappers", "optimization.backends.pygmo", ["minimize_pygmo"]),
        (
            "optimization_wrappers",
            "optimization.backends.pygmo_nlopt",
            ["minimize_nlopt"],
        ),
    ],
)
def test_compatibility_objects_are_canonical(old, new, names):
    legacy = importlib.import_module("pysolver_view." + old)
    canonical = importlib.import_module("pysolver_view." + new)
    public = importlib.import_module("pysolver_view")
    for name in names:
        assert getattr(legacy, name) is getattr(canonical, name)
        if name in public.__all__:
            assert getattr(public, name) is getattr(canonical, name)


def run_isolated(script):
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "imports",
    [
        "import pysolver_view",
        "from pysolver_view.optimization.problems.base import OptimizationProblem",
        "from pysolver_view.roots.problems.base import NonlinearSystemProblem",
        "from pysolver_view.roots.methods import normalize_method; assert normalize_method('hybr') == 'powell_hybrid'",
        "from pysolver_view.optimization.solvers.slsqp import minimize_slsqp",
        "from pysolver_view.roots.solvers import solve_root",
        "from pysolver_view.roots.problems import BroydenTridiagonalProblem; BroydenTridiagonalProblem(4)",
        "from pysolver_view.roots.problems import HeatedGasDuctProblem",
    ],
)
def test_numerical_and_problem_imports_do_not_load_ui_or_optional_packages(imports):
    run_isolated(
        """
import sys
import importlib.abc
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'jax', 'CoolProp', 'IPython', 'pygmo', 'pygmo_plugins_nonfree'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
"""
        + imports
        + """
assert 'pysolver_view.optimization.solver' not in sys.modules
assert 'pysolver_view.roots.solver' not in sys.modules
assert not any(n.startswith('development') for n in sys.modules)
"""
    )


def test_import_preserves_host_configuration_and_native_metadata_stays_lightweight():
    run_isolated("""
import sys, os, logging
hook = sys.excepthook
platform = os.environ.get('JAX_PLATFORM_NAME')
level = logging.getLogger('jax').level
import pysolver_view as psv
assert sys.excepthook is hook
assert os.environ.get('JAX_PLATFORM_NAME') == platform
assert logging.getLogger('jax').level == level
assert 'jax' not in sys.modules
from pysolver_view.roots.solver import NonlinearSystemSolver
assert 'pysolver_view.roots.solvers.trust_region' not in sys.modules
assert 'pysolver_view.roots.solvers.line_search' not in sys.modules
""")


def test_optional_thermodynamics_has_actionable_error_only_on_construction():
    run_isolated("""
import sys, importlib.abc
class NoCoolProp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'CoolProp':
            raise ModuleNotFoundError('CoolProp deliberately unavailable')
sys.meta_path.insert(0, NoCoolProp())
from pysolver_view.roots.problems import HeatedGasDuctProblem, BroydenTridiagonalProblem
from pysolver_view import solve_root
p = BroydenTridiagonalProblem(4)
assert solve_root(p.residual, p.x0, p.gradient).success
try:
    HeatedGasDuctProblem()
except ImportError as exc:
    assert 'pysolver_view[thermodynamics]' in str(exc)
else:
    raise AssertionError('Missing dependency must be reported on construction')
""")


def test_representative_consumer_subclassing_and_version():
    import pysolver_view as psv
    from pysolver_view.nonlinear_system import NonlinearSystemProblem
    from importlib.metadata import version

    class ConsumerProblem(NonlinearSystemProblem):
        def residual(self, x):
            return [x[0] - 2]

    assert isinstance(ConsumerProblem(), psv.NonlinearSystemProblem)
    solver = psv.NonlinearSystemSolver(ConsumerProblem(), print_convergence=False)
    assert solver.solve([0])[0] == pytest.approx(2)
    assert psv.__version__ == version("pysolver_view")


def test_pygmo_availability_does_not_require_nonfree_plugin():
    run_isolated("""
import sys, types, importlib.abc
class NoNonfree(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pygmo_plugins_nonfree':
            raise AssertionError('The nonfree plugin is only needed by SNOPT')
sys.meta_path.insert(0, NoNonfree())
sys.modules['pygmo'] = types.ModuleType('pygmo')
from pysolver_view.optimization.backends.pygmo import PYGMO_AVAILABLE
assert PYGMO_AVAILABLE
assert 'pygmo_plugins_nonfree' not in sys.modules
""")


def test_constrained_example_keeps_precision_without_global_jax_configuration():
    pytest.importorskip("jax")
    run_isolated("""
import sys
import numpy as np
from pysolver_view.optimization.problems import RosenbrockProblemConstrained
p = RosenbrockProblemConstrained(4)
x = np.full(4, 1.5)
assert p.fitness(x).dtype == np.float64
assert 'jax' not in sys.modules
import jax
before = jax.config.jax_enable_x64
jacobian = p.gradient(x)
assert jacobian.dtype == np.float64
assert jax.config.jax_enable_x64 == before
step = 1e-5
central = np.column_stack([(p.fitness(x + step*d)-p.fitness(x - step*d))/(2*step) for d in np.eye(4)])
np.testing.assert_allclose(jacobian, central, rtol=1e-7, atol=1e-7)
""")
