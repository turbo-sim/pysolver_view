import numpy as np
from ...exceptions import DomainError
from .base import BenchmarkProblem


class HelicalValleyProblem(BenchmarkProblem):
    """MINPACK problem 5: three equations coupling angle, radius, and height.

    f=(10*(z-10*theta), 10*(sqrt(x*x+y*y)-1), z), with the
    piecewise MINPACK angle branch. Standard start is (-1, 0, 0).
    The axis x=y=0 is excluded because the Jacobian is undefined there.
    """

    def __init__(self):
        """Set the three-variable angular/radial/height equations, standard start and
        known root.
        """
        super().__init__(
            "helical_valley_n3",
            [-1, 0, 0],
            "Helical valley: coupled angular/radial/height equations with MINPACK's angle branch; standard start.",
            solution=[1, 0, 0],
        )

    @staticmethod
    def _radius(x):
        """Return the planar radius, raising DomainError on the axis where the angle is
        undefined.
        """
        radius = np.hypot(x[0], x[1])
        if radius == 0:
            raise DomainError("Helical valley angle is undefined on x=y=0")
        return radius

    def residual(self, x):
        """Return the three helical-valley equations using the specified piecewise
        angular branch.
        """
        radius = self._radius(x)
        if x[0] > 0:
            theta = np.arctan(x[1] / x[0]) / (2 * np.pi)
        elif x[0] < 0:
            theta = np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5
        else:
            theta = np.copysign(0.25, x[1])
        return np.array([10 * (x[2] - 10 * theta), 10 * (radius - 1), x[2]])

    def gradient(self, x):
        """Return the analytic 3-by-3 Jacobian away from the excluded planar origin."""
        radius = self._radius(x)
        angular_scale = 100 / (2 * np.pi * radius**2)
        return np.array(
            [
                [angular_scale * x[1], -angular_scale * x[0], 10],
                [10 * x[0] / radius, 10 * x[1] / radius, 0],
                [0, 0, 1],
            ]
        )
