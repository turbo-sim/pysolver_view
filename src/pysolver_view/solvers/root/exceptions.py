"""Compatibility exports; implementations live in roots.solvers.exceptions."""

from importlib import import_module

_EXPORTS = {
    "BudgetExhausted": ("pysolver_view.roots.solvers.exceptions", "BudgetExhausted"),
    "DomainError": ("pysolver_view.roots.solvers.exceptions", "DomainError"),
    "UnrepresentableProbe": (
        "pysolver_view.roots.solvers.exceptions",
        "UnrepresentableProbe",
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
