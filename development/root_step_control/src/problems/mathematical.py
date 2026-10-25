"""Compatibility exports for packaged root problems."""
from importlib import import_module


_EXPORTS = {'ExtendedRosenbrockProblem': ('pysolver_view.roots.problems', 'ExtendedRosenbrockProblem'), 'PowellSingularProblem': ('pysolver_view.roots.problems', 'PowellSingularProblem'), 'PowellBadlyScaledProblem': ('pysolver_view.roots.problems', 'PowellBadlyScaledProblem'), 'BrownAlmostLinearProblem': ('pysolver_view.roots.problems', 'BrownAlmostLinearProblem'), 'BroydenTridiagonalProblem': ('pysolver_view.roots.problems', 'BroydenTridiagonalProblem'), 'DiscreteBoundaryValueProblem': ('pysolver_view.roots.problems', 'DiscreteBoundaryValueProblem'), 'DiscreteIntegralEquationProblem': ('pysolver_view.roots.problems', 'DiscreteIntegralEquationProblem'), 'HelicalValleyProblem': ('pysolver_view.roots.problems', 'HelicalValleyProblem'), 'CoupledOverdeterminedRosenbrockProblem': ('pysolver_view.roots.problems', 'CoupledOverdeterminedRosenbrockProblem'), 'LogarithmicDomainProblem': ('pysolver_view.roots.problems', 'LogarithmicDomainProblem')}
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
