"""Root backend capabilities; recognizing a method loads no numerical engine."""

from ..methods import METHOD_ALIASES

CAPABILITIES = {
    ("scipy", "levenberg_marquardt"): dict(
        shape="m >= n", events="observation", cancellation=False, step_controls=False
    ),
    ("scipy", "powell_hybrid"): dict(
        shape="m == n", events="observation", cancellation=False, step_controls=False
    ),
    **{
        ("pysolver", name): dict(
            shape="m == n" if name == "newton_raphson" else "m >= n",
            events="accepted",
            cancellation=True,
            step_controls=True,
        )
        for name in set(METHOD_ALIASES.values())
    },
}

for capability in CAPABILITIES.values():
    capability.update(
        constraints=False,
        derivatives=("analytic", "2-point", "3-point", "cs"),
        verified=True,
    )


def solve_backend(context, x0):
    if context.library == "pysolver":
        from .custom import RootRunContext, RootSolverAdapter

        return RootSolverAdapter(RootRunContext.from_solver(context)).solve(x0)
    from .scipy import solve_scipy

    return solve_scipy(context, x0)
