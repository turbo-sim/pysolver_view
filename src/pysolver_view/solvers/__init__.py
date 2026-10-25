"""Compatibility exports for custom numerical solvers."""

from importlib import import_module

_EXPORTS = {
    "DomainError": ("pysolver_view.optimization.solvers.slsqp", "DomainError"),
    "Evaluation": ("pysolver_view.optimization.solvers.slsqp", "Evaluation"),
    "SLSQPOptions": ("pysolver_view.optimization.solvers.slsqp", "SLSQPOptions"),
    "minimize_slsqp": ("pysolver_view.optimization.solvers.slsqp", "minimize_slsqp"),
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
