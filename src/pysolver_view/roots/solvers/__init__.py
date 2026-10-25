"""Controlled root solvers, loaded independently of the frontend and UI."""

from importlib import import_module

_EXPORTS = {
    "DomainError": ("pysolver_view.exceptions", "DomainError"),
    **{
        name: (".options", name)
        for name in [
            "StepControlOptions",
            "NewtonOptions",
            "BacktrackingOptions",
            "StepLimitOptions",
        ]
    },
    "METHOD_DISPLAY_NAMES": ("pysolver_view.roots.methods", "METHOD_DISPLAY_NAMES"),
    "solve_root": (".solver", "solve_root"),
    **{
        name: (".trust_region", name)
        for name in ["solve_levenberg_marquardt", "solve_powell_hybrid"]
    },
    **{
        name: (".line_search", name)
        for name in ["solve_newton_raphson", "solve_gauss_newton"]
    },
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
