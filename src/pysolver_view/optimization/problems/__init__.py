"""Bundled optimization models and lightweight problem contract."""

from importlib import import_module

_EXPORTS = {
    "OptimizationProblem": (".base", "OptimizationProblem"),
    "as_optimization_problem": (".root_adapter", "as_optimization_problem"),
    "StudyProblem": (".slsqp_controls", "StudyProblem"),
    "RosenbrockProblem": (".rosenbrock", "RosenbrockProblem"),
    "RosenbrockProblemConstrained": (
        ".constrained_rosenbrock",
        "RosenbrockProblemConstrained",
    ),
    "HS71Problem": (".hs71", "HS71Problem"),
    "LorentzEquationsOpt": (".lorenz", "LorentzEquationsOpt"),
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
