"""Compatibility exports for presentation and validation helpers."""

from importlib import import_module

_EXPORTS = {
    "set_plot_options": ("pysolver_view.reporting.style", "set_plot_options"),
    "print_installed_fonts": ("pysolver_view.reporting.style", "print_installed_fonts"),
    "print_rc_parameters": ("pysolver_view.reporting.style", "print_rc_parameters"),
    "savefig_in_formats": ("pysolver_view.reporting.style", "savefig_in_formats"),
    "create_logger": ("pysolver_view.reporting.logging", "create_logger"),
    "print_object": ("pysolver_view.reporting.text", "print_object"),
    "print_dict": ("pysolver_view.reporting.text", "print_dict"),
    "validate_keys": ("pysolver_view.validation", "validate_keys"),
    "DictionaryValidationError": (
        "pysolver_view.validation",
        "DictionaryValidationError",
    ),
    "COLORS_PYTHON": ("pysolver_view.reporting.style", "COLORS_PYTHON"),
    "COLORS_MATLAB": ("pysolver_view.reporting.style", "COLORS_MATLAB"),
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
