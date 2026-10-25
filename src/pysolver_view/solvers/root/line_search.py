"""Compatibility exports; implementations live in roots.solvers.line_search."""

from importlib import import_module

_EXPORTS = {
    "BudgetExhausted": ("pysolver_view.roots.solvers.line_search", "BudgetExhausted"),
    "DomainError": ("pysolver_view.roots.solvers.line_search", "DomainError"),
    "HistoryRecorder": ("pysolver_view.roots.solvers.line_search", "HistoryRecorder"),
    "JacobianScaling": ("pysolver_view.roots.solvers.line_search", "JacobianScaling"),
    "LinearModel": ("pysolver_view.roots.solvers.line_search", "LinearModel"),
    "ModelEvaluator": ("pysolver_view.roots.solvers.line_search", "ModelEvaluator"),
    "NewtonOptions": ("pysolver_view.roots.solvers.line_search", "NewtonOptions"),
    "UnrepresentableProbe": (
        "pysolver_view.roots.solvers.line_search",
        "UnrepresentableProbe",
    ),
    "_gauss_newton_direction": (
        "pysolver_view.roots.solvers.line_search",
        "_gauss_newton_direction",
    ),
    "_least_squares_stationary": (
        "pysolver_view.roots.solvers.line_search",
        "_least_squares_stationary",
    ),
    "_newton_direction": (
        "pysolver_view.roots.solvers.line_search",
        "_newton_direction",
    ),
    "_solve_line_search": (
        "pysolver_view.roots.solvers.line_search",
        "_solve_line_search",
    ),
    "norm": ("pysolver_view.roots.solvers.line_search", "norm"),
    "prepare_inputs": ("pysolver_view.roots.solvers.line_search", "prepare_inputs"),
    "prepare_trial": ("pysolver_view.roots.solvers.line_search", "prepare_trial"),
    "solve_gauss_newton": (
        "pysolver_view.roots.solvers.line_search",
        "solve_gauss_newton",
    ),
    "solve_newton_raphson": (
        "pysolver_view.roots.solvers.line_search",
        "solve_newton_raphson",
    ),
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
