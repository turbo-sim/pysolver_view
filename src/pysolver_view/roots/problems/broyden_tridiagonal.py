import numpy as np
from .base import BenchmarkProblem, _dimension


class BroydenTridiagonalProblem(BenchmarkProblem):
    """MINPACK problem 13: nearest-neighbor nonlinear coupling.

    f[i] = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1,
    with zero exterior values and standard start x[:] = -1.
    """

    def __init__(self, n=20):
        """Create n nearest-neighbor nonlinear equations at the standard all-minus-one
        start.
        """
        n = _dimension(n)
        super().__init__(
            f"broyden_tridiagonal_n{n}",
            -np.ones(n),
            "Broyden tridiagonal nonlinear equations with neighbor coupling; standard start x=-1.",
        )

    def residual(self, x):
        """Return the n Broyden equations with zero exterior values and nonlinear
        diagonal terms.
        """
        values = (3 - 2 * x) * x + 1
        values[1:] -= x[:-1]
        values[:-1] -= 2 * x[1:]
        return values

    def gradient(self, x):
        """Return the analytic tridiagonal Jacobian as a dense n-by-n array."""
        return (
            np.diag(3 - 4 * x)
            - np.diag(np.ones(self.n - 1), -1)
            - np.diag(2 * np.ones(self.n - 1), 1)
        )
