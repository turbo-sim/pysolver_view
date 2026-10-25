"""Benchmark problem definitions and suites; thermodynamics remains optional."""

from .base import BenchmarkProblem, MINPACK_SOURCE
from .mathematical import (
    ExtendedRosenbrockProblem,
    PowellSingularProblem,
    PowellBadlyScaledProblem,
    BrownAlmostLinearProblem,
    BroydenTridiagonalProblem,
    DiscreteBoundaryValueProblem,
    DiscreteIntegralEquationProblem,
    HelicalValleyProblem,
    CoupledOverdeterminedRosenbrockProblem,
    LogarithmicDomainProblem,
)
from .suites import get_problems, get_large_problems, get_extension_problems

__all__ = [
    "BenchmarkProblem",
    "MINPACK_SOURCE",
    "ExtendedRosenbrockProblem",
    "PowellSingularProblem",
    "PowellBadlyScaledProblem",
    "BrownAlmostLinearProblem",
    "BroydenTridiagonalProblem",
    "DiscreteBoundaryValueProblem",
    "DiscreteIntegralEquationProblem",
    "HelicalValleyProblem",
    "CoupledOverdeterminedRosenbrockProblem",
    "LogarithmicDomainProblem",
    "get_problems",
    "get_large_problems",
    "get_extension_problems",
]
