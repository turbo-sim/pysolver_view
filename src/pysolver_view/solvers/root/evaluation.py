"""Compatibility exports; implementations live in roots.solvers.evaluation."""

from importlib import import_module

_EXPORTS = {
    "BudgetExhausted": ("pysolver_view.roots.solvers.evaluation", "BudgetExhausted"),
    "DomainError": ("pysolver_view.roots.solvers.evaluation", "DomainError"),
    "ModelEvaluator": ("pysolver_view.roots.solvers.evaluation", "ModelEvaluator"),
    "UnrepresentableProbe": (
        "pysolver_view.roots.solvers.evaluation",
        "UnrepresentableProbe",
    ),
    "norm": ("pysolver_view.roots.solvers.evaluation", "norm"),
    "prepare_inputs": ("pysolver_view.roots.solvers.evaluation", "prepare_inputs"),
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
