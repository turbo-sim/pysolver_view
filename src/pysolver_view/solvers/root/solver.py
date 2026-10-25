"""Compatibility exports; implementations live in roots.solvers.solver."""

from importlib import import_module

_EXPORTS = {
    "BacktrackingOptions": (
        "pysolver_view.roots.solvers.solver",
        "BacktrackingOptions",
    ),
    "DomainError": ("pysolver_view.roots.solvers.solver", "DomainError"),
    "METHOD_ALIASES": ("pysolver_view.roots.solvers.solver", "METHOD_ALIASES"),
    "METHOD_DISPLAY_NAMES": (
        "pysolver_view.roots.solvers.solver",
        "METHOD_DISPLAY_NAMES",
    ),
    "NewtonOptions": ("pysolver_view.roots.solvers.solver", "NewtonOptions"),
    "StepControlOptions": ("pysolver_view.roots.solvers.solver", "StepControlOptions"),
    "StepLimitOptions": ("pysolver_view.roots.solvers.solver", "StepLimitOptions"),
    "_METHOD_SOLVERS": ("pysolver_view.roots.solvers.solver", "_METHOD_SOLVERS"),
    "normalize_method": ("pysolver_view.roots.solvers.solver", "normalize_method"),
    "solve_gauss_newton": ("pysolver_view.roots.solvers.solver", "solve_gauss_newton"),
    "solve_levenberg_marquardt": (
        "pysolver_view.roots.solvers.solver",
        "solve_levenberg_marquardt",
    ),
    "solve_newton_raphson": (
        "pysolver_view.roots.solvers.solver",
        "solve_newton_raphson",
    ),
    "solve_powell_hybrid": (
        "pysolver_view.roots.solvers.solver",
        "solve_powell_hybrid",
    ),
    "solve_root": ("pysolver_view.roots.solvers.solver", "solve_root"),
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
