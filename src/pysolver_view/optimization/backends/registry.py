"""Backend dispatch and capability metadata; optional backends load on request."""

SCIPY_SOLVERS = [
    "nelder-mead",
    "powell",
    "cg",
    "bfgs",
    "newton-cg",
    "l-bfgs-b",
    "tnc",
    "cobyla",
    "slsqp",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
]
"""List of valid Scipy solvers"""


PYGMO_SOLVERS = [
    "ipopt",
    "snopt",
    # "worhp",
    "de",
    "sga",
    "pso",
]
"""List of valid Pygmo solvers"""

GENETIC_SOLVERS = [
    "de",
    "sga",
    "pso",
]

GRADIENT_SOLVERS = list(set(PYGMO_SOLVERS + SCIPY_SOLVERS) - set(GENETIC_SOLVERS))

NLOPT_SOLVERS = [
    "auglag",
    "auglag_eq",
    "bobyqa",
    "ccsaq",
    "cobyla",
    "lbfgs",
    "mma",
    "neldermead",
    "newuoa",
    "newuoa_bound",
    "praxis",
    "sbplx",
    "slsqp",
    "tnewton",
    "tnewton_precond",
    "tnewton_precond_restart",
    "tnewton_restart",
    "var1",
    "var2",
]
"""List of valid NLOpt solvers (interfaced through Pygmo)"""


def minimize_pygmo(problem, x0, method, options):
    from .pygmo import minimize_pygmo as run

    return run(problem, x0, method, options)


def minimize_nlopt(problem, x0, method, options):
    from .pygmo_nlopt import minimize_nlopt as run

    return run(problem, x0, method, options)


def minimize_pounce(problem, x0, method, options):
    """Load the optional Pounce adapter only when a Pounce solve is requested."""
    from .pounce import minimize_pounce as run

    return run(problem, x0, method, options)


from .scipy import minimize_scipy
from .custom import minimize_pysolver

OPTIMIZATION_LIBRARIES = {
    "scipy": minimize_scipy,
    "pysolver": minimize_pysolver,
    "pygmo": minimize_pygmo,
    "pygmo_nlopt": minimize_nlopt,
    "pounce": minimize_pounce,
}
VALID_LIBRARIES_AND_METHODS = {
    "scipy": SCIPY_SOLVERS,
    "pysolver": ["slsqp"],
    "pygmo": PYGMO_SOLVERS,
    "pygmo_nlopt": NLOPT_SOLVERS,
    "pounce": ["ipm", "sqp"],
}
CAPABILITIES = {
    (library, method): dict(
        shape=None,
        constraints=None,
        derivatives=None,
        events=None,
        cancellation=None,
        step_controls=None,
        verified=False,
    )
    for library, methods in VALID_LIBRARIES_AND_METHODS.items()
    for method in methods
}
CAPABILITIES.update(
    {
        ("scipy", "slsqp"): dict(
            shape="scalar objective",
            verified=True,
            constraints=True,
            derivatives=True,
            events="major_iteration",
            cancellation=False,
            step_controls=False,
        ),
        ("pysolver", "slsqp"): dict(
            shape="scalar objective; no fixed variables",
            verified=True,
            constraints=True,
            derivatives=True,
            events="accepted",
            cancellation=True,
            step_controls=True,
        ),
        **{
            ("pounce", method): dict(
                shape="scalar objective; dense first derivatives",
                verified=True,
                constraints=True,
                derivatives=True,
                events="native_iteration",
                cancellation=True,
                step_controls=False,
            )
            for method in ("ipm", "sqp")
        },
    }
)
