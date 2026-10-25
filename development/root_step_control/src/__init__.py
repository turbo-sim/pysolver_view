"""Shared solver implementations and problem definitions for the experiments."""

from .exceptions import DomainError
from .options import (
    BacktrackingOptions,
    NewtonOptions,
    StepControlOptions,
    StepLimitOptions,
)
from .solver import METHOD_DISPLAY_NAMES, solve_root

__all__ = [
    "DomainError",
    "StepControlOptions",
    "NewtonOptions",
    "BacktrackingOptions",
    "StepLimitOptions",
    "METHOD_DISPLAY_NAMES",
    "solve_root",
]
