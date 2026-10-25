"""Compatibility exports; implementations live in roots.backends.custom."""

from importlib import import_module

_EXPORTS = {
    "RootSolverAdapter": ("pysolver_view.roots.backends.custom", "RootSolverAdapter"),
    "resolve_root_options": (
        "pysolver_view.roots.backends.custom",
        "resolve_root_options",
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
