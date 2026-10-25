import numpy as np
from .base import BenchmarkProblem


class PowellSingularProblem(BenchmarkProblem):
    """MINPACK problem 2: four coupled equations, singular Jacobian at zero.

    f = (x0+10*x1, sqrt(5)*(x2-x3), (x1-2*x2)**2,
         sqrt(10)*(x0-x3)**2); standard start (3, -1, 0, 1).
    """

    def __init__(self):
        """Configure the four Powell equations and standard start whose root has a
        singular Jacobian.
        """
        super().__init__(
            "powell_singular_n4",
            [3, -1, 0, 1],
            "Four coupled Powell singular equations; rank-deficient Jacobian at the root; standard start.",
            solution=np.zeros(4),
        )

    def residual(self, x):
        """Return the four coupled linear and squared Powell residuals at x."""
        return np.array(
            [
                x[0] + 10 * x[1],
                np.sqrt(5) * (x[2] - x[3]),
                (x[1] - 2 * x[2]) ** 2,
                np.sqrt(10) * (x[0] - x[3]) ** 2,
            ]
        )

    def gradient(self, x):
        """Return the analytic 4-by-4 Jacobian, including derivatives of the two squared
        equations.
        """
        difference = x[1] - 2 * x[2]
        outer = 2 * np.sqrt(10) * (x[0] - x[3])
        return np.array(
            [
                [1, 10, 0, 0],
                [0, 0, np.sqrt(5), -np.sqrt(5)],
                [0, 2 * difference, -4 * difference, 0],
                [outer, 0, 0, -outer],
            ]
        )
