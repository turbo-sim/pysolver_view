"""Compatibility exports for optimization backend adapters."""

from importlib import import_module

_EXPORTS = {
    "SCIPY_SOLVERS": ("pysolver_view.optimization.backends.registry", "SCIPY_SOLVERS"),
    "PYGMO_SOLVERS": ("pysolver_view.optimization.backends.registry", "PYGMO_SOLVERS"),
    "GENETIC_SOLVERS": (
        "pysolver_view.optimization.backends.registry",
        "GENETIC_SOLVERS",
    ),
    "GRADIENT_SOLVERS": (
        "pysolver_view.optimization.backends.registry",
        "GRADIENT_SOLVERS",
    ),
    "NLOPT_SOLVERS": ("pysolver_view.optimization.backends.registry", "NLOPT_SOLVERS"),
    "minimize_scipy": (
        "pysolver_view.optimization.backends.registry",
        "minimize_scipy",
    ),
    "minimize_pygmo": (
        "pysolver_view.optimization.backends.pygmo",
        "minimize_pygmo",
    ),
    "minimize_nlopt": (
        "pysolver_view.optimization.backends.pygmo_nlopt",
        "minimize_nlopt",
    ),
    "convert_pygmo_to_scipy_bounds": (
        "pysolver_view.optimization.backends.scipy",
        "convert_pygmo_to_scipy_bounds",
    ),
    "minimize": ("pysolver_view.optimization.backends.scipy", "minimize"),
    "PYGMO_AVAILABLE": ("pysolver_view.optimization.backends.pygmo", "PYGMO_AVAILABLE"),
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
