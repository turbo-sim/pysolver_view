import numpy as np
from .base import BenchmarkProblem, _dimension


class DiscreteBoundaryValueProblem(BenchmarkProblem):
    """MINPACK problem 9: coupled finite-difference boundary-value equations.

    h=1/(n+1), t[i]=(i+1)*h;
    f[i]=2*x[i]-x[i-1]-x[i+1]+h**2*(x[i]+t[i]+1)**3/2,
    with zero exterior values. Standard start x[i]=t[i]*(t[i]-1).
    """

    def __init__(self, n=20):
        """Create an n-point interior grid, zero boundary values and the standard
        parabolic initial guess.
        """
        n = _dimension(n)
        self.h = 1 / (n + 1)
        self.t = np.arange(1, n + 1) * self.h
        super().__init__(
            f"discrete_boundary_value_n{n}",
            self.t * (self.t - 1),
            "Nonlinear finite-difference boundary-value system; cubic neighbor-coupled equations and standard parabolic start.",
        )

    def residual(self, x):
        """Return the nonlinear finite-difference boundary-value equations at the
        interior unknowns.
        """
        values = 2 * x + 0.5 * self.h**2 * (x + self.t + 1) ** 3
        values[1:] -= x[:-1]
        values[:-1] -= x[1:]
        return values

    def gradient(self, x):
        """Return the analytic tridiagonal Jacobian including the cubic reaction
        derivative.
        """
        diagonal = 2 + 1.5 * self.h**2 * (x + self.t + 1) ** 2
        return (
            np.diag(diagonal)
            - np.diag(np.ones(self.n - 1), -1)
            - np.diag(np.ones(self.n - 1), 1)
        )
