"""Solver interfaces, diagnostics and numerical tools with lazy explicit exports."""

from importlib import import_module

__version__ = "0.8.1"
URL_GITHUB = "https://github.com/turbo-sim/pysolver_view"
URL_DOCS = "https://turbo-sim.github.io/pysolver_view/"
_EXPORTS = {
    "optimization": (".optimization", None),
    "roots": (".roots", None),
    "solvers": (".solvers", None),
    "diagnostics": (".diagnostics", None),
    "differentiation": (".differentiation", None),
    "reporting": (".reporting", None),
    "SOLVER_OPTIONS": (".roots.methods", "SOLVER_OPTIONS"),
    "OptimizationSolver": (".optimization", "OptimizationSolver"),
    "OptimizationProblem": (".optimization.problems", "OptimizationProblem"),
    "as_optimization_problem": (".optimization.problems", "as_optimization_problem"),
    "count_constraints": (".optimization", "count_constraints"),
    "combine_objective_and_constraints": (
        ".optimization",
        "combine_objective_and_constraints",
    ),
    "OPTIMIZATION_LIBRARIES": (".optimization", "OPTIMIZATION_LIBRARIES"),
    "VALID_LIBRARIES_AND_METHODS": (".optimization", "VALID_LIBRARIES_AND_METHODS"),
    "NonlinearSystemSolver": (".roots", "NonlinearSystemSolver"),
    "NonlinearSystemProblem": (".roots.problems", "NonlinearSystemProblem"),
    "RosenbrockProblem": (".optimization.problems", "RosenbrockProblem"),
    "RosenbrockProblemConstrained": (
        ".optimization.problems",
        "RosenbrockProblemConstrained",
    ),
    "HS71Problem": (".optimization.problems", "HS71Problem"),
    "LorentzEquationsOpt": (".optimization.problems", "LorentzEquationsOpt"),
    "BenchmarkProblem": (".roots.problems", "BenchmarkProblem"),
    "MINPACK_SOURCE": (".roots.problems", "MINPACK_SOURCE"),
    "LorentzEquations": (".roots.problems", "LorentzEquations"),
    "ExtendedRosenbrockProblem": (".roots.problems", "ExtendedRosenbrockProblem"),
    "PowellSingularProblem": (".roots.problems", "PowellSingularProblem"),
    "PowellBadlyScaledProblem": (".roots.problems", "PowellBadlyScaledProblem"),
    "BrownAlmostLinearProblem": (".roots.problems", "BrownAlmostLinearProblem"),
    "BroydenTridiagonalProblem": (".roots.problems", "BroydenTridiagonalProblem"),
    "DiscreteBoundaryValueProblem": (".roots.problems", "DiscreteBoundaryValueProblem"),
    "DiscreteIntegralEquationProblem": (
        ".roots.problems",
        "DiscreteIntegralEquationProblem",
    ),
    "HelicalValleyProblem": (".roots.problems", "HelicalValleyProblem"),
    "CoupledOverdeterminedRosenbrockProblem": (
        ".roots.problems",
        "CoupledOverdeterminedRosenbrockProblem",
    ),
    "LogarithmicDomainProblem": (".roots.problems", "LogarithmicDomainProblem"),
    "HeatedGasDuctProblem": (".roots.problems", "HeatedGasDuctProblem"),
    "get_extension_problems": (".roots.problems", "get_extension_problems"),
    "get_problems": (".roots.problems", "get_problems"),
    "get_large_problems": (".roots.problems", "get_large_problems"),
    "approx_gradient": (".differentiation", "approx_gradient"),
    "forward_finite_differences": (".differentiation", "forward_finite_differences"),
    "central_finite_differences": (".differentiation", "central_finite_differences"),
    "complex_step_derivative": (".differentiation", "complex_step_derivative"),
    "DERIVATIVE_METHODS": (".differentiation", "DERIVATIVE_METHODS"),
    "approx_jacobian_hessians": (".differentiation", "approx_jacobian_hessians"),
    "compute_hessians_jax": (".differentiation", "compute_hessians_jax"),
    "approx_derivative": (".differentiation", "approx_derivative"),
    "set_plot_options": (".pysolver_utilities", "set_plot_options"),
    "print_installed_fonts": (".pysolver_utilities", "print_installed_fonts"),
    "print_rc_parameters": (".pysolver_utilities", "print_rc_parameters"),
    "savefig_in_formats": (".pysolver_utilities", "savefig_in_formats"),
    "create_logger": (".pysolver_utilities", "create_logger"),
    "print_object": (".pysolver_utilities", "print_object"),
    "print_dict": (".pysolver_utilities", "print_dict"),
    "validate_keys": (".pysolver_utilities", "validate_keys"),
    "DictionaryValidationError": (".pysolver_utilities", "DictionaryValidationError"),
    "COLORS_PYTHON": (".pysolver_utilities", "COLORS_PYTHON"),
    "COLORS_MATLAB": (".pysolver_utilities", "COLORS_MATLAB"),
    "SCIPY_SOLVERS": (".optimization_wrappers", "SCIPY_SOLVERS"),
    "PYGMO_SOLVERS": (".optimization_wrappers", "PYGMO_SOLVERS"),
    "GENETIC_SOLVERS": (".optimization_wrappers", "GENETIC_SOLVERS"),
    "GRADIENT_SOLVERS": (".optimization_wrappers", "GRADIENT_SOLVERS"),
    "NLOPT_SOLVERS": (".optimization_wrappers", "NLOPT_SOLVERS"),
    "minimize_scipy": (".optimization_wrappers", "minimize_scipy"),
    "minimize_pygmo": (".optimization_wrappers", "minimize_pygmo"),
    "minimize_nlopt": (".optimization_wrappers", "minimize_nlopt"),
    "convert_pygmo_to_scipy_bounds": (
        ".optimization_wrappers",
        "convert_pygmo_to_scipy_bounds",
    ),
    "minimize": (".optimization_wrappers", "minimize"),
    "PYGMO_AVAILABLE": (".optimization_wrappers", "PYGMO_AVAILABLE"),
    "assess_kkt": (".diagnostics", "assess_kkt"),
    "evaluate_kkt_residuals": (".diagnostics", "evaluate_kkt_residuals"),
    "DomainError": (".exceptions", "DomainError"),
    "Evaluation": (".optimization.solvers.slsqp", "Evaluation"),
    "SLSQPOptions": (".optimization.solvers.slsqp", "SLSQPOptions"),
    "minimize_slsqp": (".optimization.solvers.slsqp", "minimize_slsqp"),
    "StepControlOptions": (".roots.solvers", "StepControlOptions"),
    "NewtonOptions": (".roots.solvers", "NewtonOptions"),
    "BacktrackingOptions": (".roots.solvers", "BacktrackingOptions"),
    "StepLimitOptions": (".roots.solvers", "StepLimitOptions"),
    "solve_root": (".roots.solvers", "solve_root"),
    "solve_levenberg_marquardt": (".roots.solvers", "solve_levenberg_marquardt"),
    "solve_powell_hybrid": (".roots.solvers", "solve_powell_hybrid"),
    "solve_newton_raphson": (".roots.solvers", "solve_newton_raphson"),
    "solve_gauss_newton": (".roots.solvers", "solve_gauss_newton"),
    "METHOD_DISPLAY_NAMES": (".roots.methods", "METHOD_DISPLAY_NAMES"),
    "normalize_method": (".roots.methods", "normalize_method"),
    "RootSolverAdapter": (".roots.backends.custom", "RootSolverAdapter"),
    "resolve_root_options": (".roots.backends.custom", "resolve_root_options"),
    "minimize_pysolver": (".optimization.backends.custom", "minimize_pysolver"),
    "configure_jax": (".differentiation._jax", "configure_jax"),
    "is_jax_available": (".differentiation._jax", "is_jax_available"),
    "numerical_differentiation": (".numerical_differentiation", None),
    "optimization_wrappers": (".optimization_wrappers", None),
    "pysolver_utilities": (".pysolver_utilities", None),
    "nonlinear_system": (".nonlinear_system", None),
    "optimization_problems": (".optimization_problems", None),
    "nonlinear_system_problems": (".nonlinear_system_problems", None),
}
__all__ = list(_EXPORTS) + ["__version__", "URL_GITHUB", "URL_DOCS", "JAX_AVAILABLE"]


def __getattr__(name):
    if name in {"JAX_AVAILABLE", "jax", "jnp"}:
        from .differentiation._jax import is_jax_available, load_jax

        available = is_jax_available()
        if name == "JAX_AVAILABLE":
            return available
        return load_jax()[0 if name == "jax" else 1] if available else None
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = import_module(module, __name__)
    if symbol is not None:
        value = getattr(value, symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
