"""Reproducible problem collections for convergence and timing studies."""

from . import (
    BroydenTridiagonalProblem,
    BrownAlmostLinearProblem,
    CoupledOverdeterminedRosenbrockProblem,
    DiscreteBoundaryValueProblem,
    DiscreteIntegralEquationProblem,
    ExtendedRosenbrockProblem,
    HelicalValleyProblem,
    LogarithmicDomainProblem,
    PowellBadlyScaledProblem,
    PowellSingularProblem,
)


def get_extension_problems():
    """Return thermodynamic, domain-recovery and rectangular Newton-family controls."""
    from .heated_gas_duct import HeatedGasDuctProblem

    return [
        HeatedGasDuctProblem(),
        LogarithmicDomainProblem(),
        CoupledOverdeterminedRosenbrockProblem(),
        CoupledOverdeterminedRosenbrockProblem(inconsistent=True),
    ]


def get_problems():
    """Seven well-known, genuinely multidimensional systems at standard starts."""
    return [
        ExtendedRosenbrockProblem(n=10),
        PowellSingularProblem(),
        PowellBadlyScaledProblem(),
        BrownAlmostLinearProblem(n=10),
        BroydenTridiagonalProblem(n=20),
        DiscreteBoundaryValueProblem(n=20),
        HelicalValleyProblem(),
    ]


def get_large_problems():
    """Three 100-variable cases: banded, dense, and property-heavy coupling.

    CoolProp is only imported when this expanded suite is requested. The duct
    refines the original ten-cell domain at fixed total heat and resistance.
    """
    from .heated_gas_duct import HeatedGasDuctProblem

    return [
        BroydenTridiagonalProblem(n=100),
        DiscreteIntegralEquationProblem(n=100),
        HeatedGasDuctProblem(
            n_cells=50, total_heat_per_mass=1e5, total_loss_coefficient=80
        ),
    ]
