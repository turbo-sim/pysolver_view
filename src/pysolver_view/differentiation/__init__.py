"""Numerical differentiation with isolated optional dependencies."""

from importlib import import_module

_EXPORTS = {
    "approx_gradient": (".finite_difference", "approx_gradient"),
    "forward_finite_differences": (".finite_difference", "forward_finite_differences"),
    "central_finite_differences": (".finite_difference", "central_finite_differences"),
    "complex_step_derivative": (".finite_difference", "complex_step_derivative"),
    "DERIVATIVE_METHODS": (".finite_difference", "DERIVATIVE_METHODS"),
    "approx_jacobian_hessians": (".hessians", "approx_jacobian_hessians"),
    "compute_hessians_jax": (".hessians", "compute_hessians_jax"),
    "approx_derivative": ("._scipy_numdiff", "approx_derivative"),
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
