"""Public root-solver dispatch, method aliases, and configuration re-exports."""

from .exceptions import DomainError
from .options import (
    BacktrackingOptions,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
)
from .solvers.line_search import solve_newton
from .solvers.trust_region import solve_trust_region

METHOD_ALIASES = {
    "lm": "lm",
    "levenberg_marquardt": "lm",
    "hybr": "hybr",
    "powell_hybrid": "hybr",
    "newton_raphson": "newton_raphson",
    "newton": "newton_raphson",
    "gauss_newton": "gauss_newton",
    "gn": "gauss_newton",
}

METHOD_DISPLAY_NAMES = {
    "lm": "Levenberg–Marquardt",
    "hybr": "Powell hybrid",
    "newton_raphson": "Newton–Raphson",
    "gauss_newton": "Gauss–Newton",
}


def normalize_method(method):
    """Resolve one documented root-method spelling to its canonical identifier."""
    if not isinstance(method, str) or method.strip().lower() not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown root method; choose from {', '.join(METHOD_ALIASES)}."
        )
    return METHOD_ALIASES[method.strip().lower()]


def solve_root(fun, x0, jac=None, *, method="lm", options=None, callback=None):
    """Solve a root problem using an experimental method and explicit step policy.

    lm/levenberg_marquardt and hybr/powell_hybrid use StepControlOptions and
    trust regions. newton_raphson/newton and gauss_newton/gn use NewtonOptions;
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
    driver = (
        solve_newton
        if canonical in ("newton_raphson", "gauss_newton")
        else solve_trust_region
    )
    result = driver(fun, x0, jac, method=canonical, options=options, callback=callback)
    result.requested_method = method
    result.method_name = METHOD_DISPLAY_NAMES[canonical]
    return result
