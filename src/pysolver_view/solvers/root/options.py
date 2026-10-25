"""Compatibility exports; implementations live in roots.solvers.options."""

from importlib import import_module

_EXPORTS = {
    "BacktrackingOptions": (
        "pysolver_view.roots.solvers.options",
        "BacktrackingOptions",
    ),
    "Integral": ("pysolver_view.roots.solvers.options", "Integral"),
    "NewtonOptions": ("pysolver_view.roots.solvers.options", "NewtonOptions"),
    "Real": ("pysolver_view.roots.solvers.options", "Real"),
    "StepControlOptions": ("pysolver_view.roots.solvers.options", "StepControlOptions"),
    "StepLimitOptions": ("pysolver_view.roots.solvers.options", "StepLimitOptions"),
    "_integer_at_least": ("pysolver_view.roots.solvers.options", "_integer_at_least"),
    "_positive_finite": ("pysolver_view.roots.solvers.options", "_positive_finite"),
    "_validate_derivatives": (
        "pysolver_view.roots.solvers.options",
        "_validate_derivatives",
    ),
    "_validate_scaling": ("pysolver_view.roots.solvers.options", "_validate_scaling"),
    "dataclass": ("pysolver_view.roots.solvers.options", "dataclass"),
    "field": ("pysolver_view.roots.solvers.options", "field"),
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
