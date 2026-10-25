"""Controlled root solvers using public NumPy and SciPy numerical kernels."""

from .exceptions import DomainError
from .options import (
    BacktrackingOptions,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
)
from .solver import METHOD_DISPLAY_NAMES, solve_root
from .line_search import solve_gauss_newton, solve_newton_raphson
from .trust_region import (
    solve_levenberg_marquardt,
    solve_powell_hybrid,
)

__all__ = [
    "DomainError",
    "StepControlOptions",
    "NewtonOptions",
    "BacktrackingOptions",
    "StepLimitOptions",
    "METHOD_DISPLAY_NAMES",
    "solve_root",
    "solve_levenberg_marquardt",
    "solve_powell_hybrid",
    "solve_newton_raphson",
    "solve_gauss_newton",
]
