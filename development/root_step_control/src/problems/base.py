"""Shared metadata for benchmark problems using NonlinearSystemProblem."""

import numpy as np
from pysolver_view.nonlinear_system import NonlinearSystemProblem

MINPACK_SOURCE = "https://www.netlib.org/minpack/ex/file16"


def _dimension(n, *, multiple=1):
    """Validate n as an integer >= 2 divisible by multiple, and return it as a Python
    int.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if n % multiple:
        raise ValueError(f"n must be a multiple of {multiple}")
    return int(n)


class BenchmarkProblem(NonlinearSystemProblem):
    """PSV root-problem base carrying reproducible benchmark metadata."""

    source = MINPACK_SOURCE

    def __init__(self, name, x0, description, *, x_scale=None, solution=None):
        """Store the benchmark name, start vector, scales, description and optional
        reference root.
        """
        self.name = name
        self.x0 = np.asarray(x0, dtype=float)
        self.n = self.x0.size
        self.m = self.n
        self.root_exists = True
        self.x_scale = (
            np.ones(self.n) if x_scale is None else np.asarray(x_scale, dtype=float)
        )
        self.description = description
        self.solution = None if solution is None else np.asarray(solution, dtype=float)
