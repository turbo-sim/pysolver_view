"""Public root-solver dispatch, method aliases, and configuration re-exports."""

from ..methods import (
    METHOD_ALIASES as METHOD_ALIASES,
    METHOD_DISPLAY_NAMES,
    normalize_method,
)

from .exceptions import DomainError as DomainError
from .options import (
    BacktrackingOptions as BacktrackingOptions,
    NewtonOptions as NewtonOptions,
    StepControlOptions as StepControlOptions,
    StepLimitOptions as StepLimitOptions,
)
from .line_search import solve_gauss_newton, solve_newton_raphson
from .trust_region import (
    solve_levenberg_marquardt,
    solve_powell_hybrid,
)

_METHOD_SOLVERS = {
    "levenberg_marquardt": solve_levenberg_marquardt,
    "powell_hybrid": solve_powell_hybrid,
    "newton_raphson": solve_newton_raphson,
    "gauss_newton": solve_gauss_newton,
}


def solve_root(
    fun,
    x0,
    jac=None,
    *,
    method="levenberg_marquardt",
    options=None,
    callback=None,
    observer=None,
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
    result = driver(fun, x0, jac, options=options, callback=callback, observer=observer)
    result.requested_method = method
    result.method_name = METHOD_DISPLAY_NAMES[canonical]
    return result
