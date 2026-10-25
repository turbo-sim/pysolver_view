"""Optimization interfaces, with lightweight problem and numerical imports."""

from importlib import import_module

_EXPORTS = {
    "OptimizationSolver": (".solver", "OptimizationSolver"),
    "OptimizationProblem": (".problems.base", "OptimizationProblem"),
    "count_constraints": (".constraints", "count_constraints"),
    "combine_objective_and_constraints": (
        ".constraints",
        "combine_objective_and_constraints",
    ),
    "OPTIMIZATION_LIBRARIES": (".backends.registry", "OPTIMIZATION_LIBRARIES"),
    "VALID_LIBRARIES_AND_METHODS": (
        ".backends.registry",
        "VALID_LIBRARIES_AND_METHODS",
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
