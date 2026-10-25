"""Experimental root solvers with trust-region or backtracking step controls."""

from .src import (
    BacktrackingOptions,
    DomainError,
    METHOD_DISPLAY_NAMES,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
    solve_root,
    solve_levenberg_marquardt,
    solve_powell_hybrid,
    solve_newton_raphson,
    solve_gauss_newton,
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
