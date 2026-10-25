"""Compatibility exports; implementations live in optimization.solvers.slsqp.driver."""

from importlib import import_module

_EXPORTS = {
    "_CallbackStop": (
        "pysolver_view.optimization.solvers.slsqp.driver",
        "_CallbackStop",
    ),
    "_format_tolerance_pair": (
        "pysolver_view.optimization.solvers.slsqp.driver",
        "_format_tolerance_pair",
    ),
    "_grow_step_fraction": (
        "pysolver_view.optimization.solvers.slsqp.driver",
        "_grow_step_fraction",
    ),
    "_relative_step_fraction": (
        "pysolver_view.optimization.solvers.slsqp.driver",
        "_relative_step_fraction",
    ),
    "minimize_slsqp": (
        "pysolver_view.optimization.solvers.slsqp.driver",
        "minimize_slsqp",
    ),
    "Evaluation": ("pysolver_view.optimization.solvers.slsqp.driver", "Evaluation"),
    "SLSQPOptions": ("pysolver_view.optimization.solvers.slsqp.driver", "SLSQPOptions"),
    "DomainError": ("pysolver_view.optimization.solvers.slsqp.driver", "DomainError"),
    "Backend": ("pysolver_view.optimization.solvers.slsqp.driver", "Backend"),
    "assess_kkt": ("pysolver_view.optimization.solvers.slsqp.driver", "assess_kkt"),
    "MESSAGES": ("pysolver_view.optimization.solvers.slsqp.driver", "MESSAGES"),
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
