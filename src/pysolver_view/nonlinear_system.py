"""Compatibility exports; implementations live in roots.solver."""

from importlib import import_module

_EXPORTS = {
    "NonlinearSystemSolver": ("pysolver_view.roots.solver", "NonlinearSystemSolver"),
    "NonlinearSystemProblem": ("pysolver_view.roots.solver", "NonlinearSystemProblem"),
    "SOLVER_OPTIONS": ("pysolver_view.roots.solver", "SOLVER_OPTIONS"),
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
