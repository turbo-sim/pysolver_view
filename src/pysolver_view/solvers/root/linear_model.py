"""Compatibility exports; implementations live in roots.solvers.linear_model."""

from importlib import import_module

_EXPORTS = {
    "Integral": ("pysolver_view.roots.solvers.linear_model", "Integral"),
    "LinearModel": ("pysolver_view.roots.solvers.linear_model", "LinearModel"),
    "_COUNTERS": ("pysolver_view.roots.solvers.linear_model", "_COUNTERS"),
    "_EPS": ("pysolver_view.roots.solvers.linear_model", "_EPS"),
    "_QR_RCOND_MIN": ("pysolver_view.roots.solvers.linear_model", "_QR_RCOND_MIN"),
    "_feasible": ("pysolver_view.roots.solvers.linear_model", "_feasible"),
    "get_lapack_funcs": (
        "pysolver_view.roots.solvers.linear_model",
        "get_lapack_funcs",
    ),
    "norm": ("pysolver_view.roots.solvers.linear_model", "norm"),
    "qr": ("pysolver_view.roots.solvers.linear_model", "qr"),
    "qr_update": ("pysolver_view.roots.solvers.linear_model", "qr_update"),
    "solve_triangular": (
        "pysolver_view.roots.solvers.linear_model",
        "solve_triangular",
    ),
    "svd": ("pysolver_view.roots.solvers.linear_model", "svd"),
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
