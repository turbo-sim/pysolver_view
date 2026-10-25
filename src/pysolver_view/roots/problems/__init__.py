"""Reusable root problems; optional dependencies load only when requested."""

from importlib import import_module

_EXPORTS = {
    "NonlinearSystemProblem": (".base", "NonlinearSystemProblem"),
    "BenchmarkProblem": (".base", "BenchmarkProblem"),
    "MINPACK_SOURCE": (".base", "MINPACK_SOURCE"),
    "LorentzEquations": (".lorenz", "LorentzEquations"),
    "ExtendedRosenbrockProblem": (".extended_rosenbrock", "ExtendedRosenbrockProblem"),
    "PowellSingularProblem": (".powell_singular", "PowellSingularProblem"),
    "PowellBadlyScaledProblem": (".powell_badly_scaled", "PowellBadlyScaledProblem"),
    "BrownAlmostLinearProblem": (".brown_almost_linear", "BrownAlmostLinearProblem"),
    "BroydenTridiagonalProblem": (".broyden_tridiagonal", "BroydenTridiagonalProblem"),
    "DiscreteBoundaryValueProblem": (
        ".discrete_boundary_value",
        "DiscreteBoundaryValueProblem",
    ),
    "DiscreteIntegralEquationProblem": (
        ".discrete_integral_equation",
        "DiscreteIntegralEquationProblem",
    ),
    "HelicalValleyProblem": (".helical_valley", "HelicalValleyProblem"),
    "CoupledOverdeterminedRosenbrockProblem": (
        ".coupled_overdetermined_rosenbrock",
        "CoupledOverdeterminedRosenbrockProblem",
    ),
    "LogarithmicDomainProblem": (".logarithmic_domain", "LogarithmicDomainProblem"),
    "HeatedGasDuctProblem": (".heated_gas_duct", "HeatedGasDuctProblem"),
    "get_extension_problems": (".suites", "get_extension_problems"),
    "get_problems": (".suites", "get_problems"),
    "get_large_problems": (".suites", "get_large_problems"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
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
