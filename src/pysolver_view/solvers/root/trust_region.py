"""Compatibility exports; implementations live in roots.solvers.trust_region."""

from importlib import import_module

_EXPORTS = {
    "BudgetExhausted": ("pysolver_view.roots.solvers.trust_region", "BudgetExhausted"),
    "DomainError": ("pysolver_view.roots.solvers.trust_region", "DomainError"),
    "HistoryRecorder": ("pysolver_view.roots.solvers.trust_region", "HistoryRecorder"),
    "JacobianScaling": ("pysolver_view.roots.solvers.trust_region", "JacobianScaling"),
    "LinearModel": ("pysolver_view.roots.solvers.trust_region", "LinearModel"),
    "ModelEvaluator": ("pysolver_view.roots.solvers.trust_region", "ModelEvaluator"),
    "StepControlOptions": (
        "pysolver_view.roots.solvers.trust_region",
        "StepControlOptions",
    ),
    "UnrepresentableProbe": (
        "pysolver_view.roots.solvers.trust_region",
        "UnrepresentableProbe",
    ),
    "_solve_trust_region": (
        "pysolver_view.roots.solvers.trust_region",
        "_solve_trust_region",
    ),
    "_update_powell_jacobian": (
        "pysolver_view.roots.solvers.trust_region",
        "_update_powell_jacobian",
    ),
    "norm": ("pysolver_view.roots.solvers.trust_region", "norm"),
    "prepare_equilibrated_trial": (
        "pysolver_view.roots.solvers.trust_region",
        "prepare_equilibrated_trial",
    ),
    "prepare_inputs": ("pysolver_view.roots.solvers.trust_region", "prepare_inputs"),
    "prepare_trust_region_trial": (
        "pysolver_view.roots.solvers.trust_region",
        "prepare_trust_region_trial",
    ),
    "solve_levenberg_marquardt": (
        "pysolver_view.roots.solvers.trust_region",
        "solve_levenberg_marquardt",
    ),
    "solve_powell_hybrid": (
        "pysolver_view.roots.solvers.trust_region",
        "solve_powell_hybrid",
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
