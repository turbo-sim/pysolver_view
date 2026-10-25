"""Native SciPy root adapter with the established evaluation-budget mapping."""

import numpy as np
from scipy.optimize import root


def solve_scipy(context, x0):
    return root(
        context.residual,
        np.array(x0, dtype=float, copy=True),
        jac=context.gradient,
        method=context.method,
        tol=context.tol,
        options=context.options.copy(),
    )
