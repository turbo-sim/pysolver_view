"""Public root-solver dispatch, method aliases, and configuration re-exports."""

from .exceptions import DomainError
from .options import (
    BacktrackingOptions,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
)
from .solvers import (
    solve_gauss_newton,
    solve_levenberg_marquardt,
    solve_newton_raphson,
    solve_powell_hybrid,
)

METHOD_ALIASES = {
    "levenberg_marquardt": "levenberg_marquardt",
    "lm": "levenberg_marquardt",
    "powell_hybrid": "powell_hybrid",
    "hybr": "powell_hybrid",
    "newton_raphson": "newton_raphson",
    "newton": "newton_raphson",
    "gauss_newton": "gauss_newton",
    "gn": "gauss_newton",
}

METHOD_DISPLAY_NAMES = {
    "levenberg_marquardt": "Levenberg–Marquardt",
    "powell_hybrid": "Powell hybrid",
    "newton_raphson": "Newton–Raphson",
    "gauss_newton": "Gauss–Newton",
}

_METHOD_SOLVERS = {
    "levenberg_marquardt": solve_levenberg_marquardt,
    "powell_hybrid": solve_powell_hybrid,
    "newton_raphson": solve_newton_raphson,
    "gauss_newton": solve_gauss_newton,
}


def normalize_method(method):
    """Resolve one documented root-method spelling to its canonical identifier."""
    if not isinstance(method, str) or method.strip().lower() not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown root method; choose from {', '.join(METHOD_ALIASES)}."
        )
    return METHOD_ALIASES[method.strip().lower()]


def solve_root(
    fun, x0, jac=None, *, method="levenberg_marquardt", options=None, callback=None
):
    """Solve a root problem using an experimental method and explicit step policy.

    levenberg_marquardt and powell_hybrid use StepControlOptions and trust
    regions; lm and hybr remain accepted aliases. newton_raphson and
    gauss_newton use NewtonOptions, with newton and gn as their aliases;
    backtracking is enabled by default and explicit backtracking=None selects
    full valid steps, including residual increases. Newton requires a square
    numerically full-rank Jacobian; the other methods allow m >= n.

    Dictionaries and matching option dataclasses are accepted. Numerical
    derivatives, residual budgets, copied callbacks, history levels and
    residual-based root success are consistent across drivers. Callbacks receive
    full detached initial/committed records and stop on an exact False return.
    Results preserve the last committed point and separately report attempted
    steps, model calls, canonical method, and requested spelling. Undefined
    initial points or derivative probes terminate; programming errors propagate.
    """
    canonical = normalize_method(method)
    driver = _METHOD_SOLVERS[canonical]
    result = driver(fun, x0, jac, options=options, callback=callback)
    result.requested_method = method
    result.method_name = METHOD_DISPLAY_NAMES[canonical]
    return result
