"""Compatibility exports for packaged root problems."""
from importlib import import_module


_EXPORTS = {'BenchmarkProblem': ('pysolver_view.roots.problems', 'BenchmarkProblem'), 'NonlinearSystemProblem': ('pysolver_view.roots.problems', 'NonlinearSystemProblem'), 'MINPACK_SOURCE': ('pysolver_view.roots.problems', 'MINPACK_SOURCE'), '_dimension': ('pysolver_view.roots.problems.base', '_dimension')}
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
