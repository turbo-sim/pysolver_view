import numpy as np
from .base import BenchmarkProblem, _dimension


class ExtendedRosenbrockProblem(BenchmarkProblem):
    """Block extension of MINPACK problem 1; each two-variable block is coupled.

    f[2k] = 1 - x[2k], f[2k+1] = 10*(x[2k+1] - x[2k]**2).
    Standard start repeats (-1.2, 1); the root is the all-ones vector.
    """

    def __init__(self, n=10):
        """Create an even-dimensional set of Rosenbrock pairs at the standard repeated
        start.
        """
        n = _dimension(n, multiple=2)
        super().__init__(
            f"extended_rosenbrock_n{n}",
            np.tile([-1.2, 1.0], n // 2),
            "Extended Rosenbrock equations, coupled pairs with narrow curved valleys; standard start.",
            solution=np.ones(n),
        )

    def residual(self, x):
        """Return n Rosenbrock equations, interleaving coordinate and curved-valley
        residuals.
        """
        values = np.empty(self.n, dtype=np.result_type(x, float))
        values[::2] = 1 - x[::2]
        values[1::2] = 10 * (x[1::2] - x[::2] ** 2)
        return values

    def gradient(self, x):
        """Return the analytic n-by-n Jacobian with one coupled 2-by-2 block per
        variable pair.
        """
        jac = np.zeros((self.n, self.n), dtype=np.result_type(x, float))
        even = np.arange(0, self.n, 2)
        jac[even, even] = -1
        jac[even + 1, even] = -20 * x[even]
        jac[even + 1, even + 1] = 10
        return jac
