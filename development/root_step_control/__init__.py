"""Compatibility exports for the root studies; implementations live in the package."""

from pysolver_view.roots.solvers import (
    BacktrackingOptions, DomainError, METHOD_DISPLAY_NAMES, NewtonOptions,
    StepControlOptions, StepLimitOptions, solve_root, solve_levenberg_marquardt,
    solve_powell_hybrid, solve_newton_raphson, solve_gauss_newton,
)
