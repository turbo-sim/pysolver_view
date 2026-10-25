import numpy as np
from .base import BenchmarkProblem, _dimension


class DiscreteIntegralEquationProblem(BenchmarkProblem):
    """MINPACK problem 10: a dense cubic integral-equation discretization.

    With h=1/(n+1), t[i]=(i+1)*h, and
    K[i,j]=min(t[i],t[j])*(1-max(t[i],t[j])),
    f=x+h/2*K@(x+t+1)**3. The standard start is t*(t-1).
    Every residual depends on every variable. No sparsity is supplied to any
    solver; the analytic Jacobian is a fully populated n-by-n array.
    """

    def __init__(self, n=100):
        """Precompute the dense integral kernel and parabolic start on an n-point
        interior grid.
        """
        n = _dimension(n)
        self.h = 1 / (n + 1)
        self.t = np.arange(1, n + 1) * self.h
        self.kernel = np.minimum.outer(self.t, self.t) * (
            1 - np.maximum.outer(self.t, self.t)
        )
        super().__init__(
            f"discrete_integral_equation_n{n}",
            self.t * (self.t - 1),
            "MINPACK discrete integral equation with dense cubic coupling; standard parabolic start.",
        )

    def residual(self, x):
        """Return the n cubic integral-equation residuals, each depending on every
        unknown.
        """
        return x + 0.5 * self.h * (self.kernel @ (x + self.t + 1) ** 3)

    def gradient(self, x):
        """Return the fully dense analytic Jacobian of the discretized cubic integral
        equation.
        """
        return np.eye(self.n) + 1.5 * self.h * self.kernel * (x + self.t + 1) ** 2
