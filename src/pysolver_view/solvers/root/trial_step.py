"""Compatibility exports; implementations live in roots.solvers.trial_step."""

from importlib import import_module

_EXPORTS = {
    "norm": ("pysolver_view.roots.solvers.trial_step", "norm"),
    "prepare_equilibrated_trial": (
        "pysolver_view.roots.solvers.trial_step",
        "prepare_equilibrated_trial",
    ),
    "prepare_trial": ("pysolver_view.roots.solvers.trial_step", "prepare_trial"),
    "prepare_trust_region_trial": (
        "pysolver_view.roots.solvers.trial_step",
        "prepare_trust_region_trial",
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
