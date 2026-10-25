import numpy as np
from .base import BenchmarkProblem, _dimension


class BrownAlmostLinearProblem(BenchmarkProblem):
    """MINPACK problem 8: a dense sum coupling and one nonlinear product.

    f[i] = x[i] + sum(x) - (n+1), i < n-1; f[n-1] = prod(x)-1.
    Standard start is x[:] = 0.5; all ones is a root (not necessarily unique).
    """

    def __init__(self, n=10):
        """Configure n sum/product equations at x=0.5 with an all-ones reference root."""
        n = _dimension(n)
        super().__init__(
            f"brown_almost_linear_n{n}",
            np.full(n, 0.5),
            "Brown almost-linear system with dense sum/product coupling; standard start x=0.5.",
            solution=np.ones(n),
        )

    def residual(self, x):
        """Return n-1 dense sum equations and the final nonlinear product equation."""
        values = x + np.sum(x) - (self.n + 1)
        values[-1] = np.prod(x) - 1
        return values

    def gradient(self, x):
        """Return the dense analytic Jacobian; product derivatives remain valid when a
        component is zero.
        """
        jac = np.ones((self.n, self.n), dtype=np.result_type(x, float)) + np.eye(self.n)
        # Product excluding a component is valid when that component is zero.
        jac[-1] = [np.prod(np.delete(x, index)) for index in range(self.n)]
        return jac
