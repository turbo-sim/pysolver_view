import numpy as np
from ...exceptions import DomainError
from .base import BenchmarkProblem


class PowellBadlyScaledProblem(BenchmarkProblem):
    """MINPACK problem 3: f = (1e4*x*y-1, exp(-x)+exp(-y)-1.0001).

    Standard start is (0, 1); coordinate scales (1e-4, 10) are supplied
    explicitly to every solver. The equations themselves are not rescaled.
    """

    def __init__(self):
        """Set the two-equation standard start, disparate coordinate scales and
        reference root.
        """
        super().__init__(
            "powell_badly_scaled_n2",
            [0, 1],
            "Coupled Powell badly scaled equations; standard start; fixed scales (1e-4, 10).",
            x_scale=[1e-4, 10],
            solution=[1.098159329699817e-5, 9.106146739866524],
        )

    def residual(self, x):
        """Evaluate the coupled product/exponential equations, raising DomainError
        before exponential overflow.
        """
        if np.any(-x > np.log(np.finfo(float).max)):
            raise DomainError("Powell exponential would overflow")
        return np.array([1e4 * x[0] * x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - 1.0001])

    def gradient(self, x):
        """Return the analytic 2-by-2 Jacobian of the product and exponential residuals."""
        return np.array([[1e4 * x[1], 1e4 * x[0]], [-np.exp(-x[0]), -np.exp(-x[1])]])
