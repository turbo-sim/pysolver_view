"""Experimental root solvers with trust-region or backtracking step controls."""

from .src.solver import (
    BacktrackingOptions,
    DomainError,
    METHOD_DISPLAY_NAMES,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
    solve_root,
)

__all__ = [
    "DomainError",
    "StepControlOptions",
    "NewtonOptions",
    "BacktrackingOptions",
    "StepLimitOptions",
    "METHOD_DISPLAY_NAMES",
    "solve_root",
]
