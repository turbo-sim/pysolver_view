import numpy as np
from .base import BenchmarkProblem, _dimension
from .extended_rosenbrock import ExtendedRosenbrockProblem


class CoupledOverdeterminedRosenbrockProblem(BenchmarkProblem):
    """Extend Rosenbrock equations with dense redundant rows and an optional inconsistency.

    The first n rows are the standard extended Rosenbrock equations. Another n
    rows are dense linear combinations of them, so every added residual couples
    all variable pairs without changing the all-ones root. An optional constant
    residual of one creates an inconsistent least-squares problem with a known
    minimum residual norm of one. These are explicitly constructed controls,
    not additional original MINPACK test problems.
    """

    source = (
        "Constructed dense overdetermined extension of MINPACK Rosenbrock equations."
    )

    def __init__(self, n=10, *, inconsistent=False):
        """Configure 2*n coupled equations, optionally adding a nonzero constant row."""
        n = _dimension(n, multiple=2)
        self.square_problem = ExtendedRosenbrockProblem(n)
        self.mixing = 0.2 * np.eye(n) + np.ones((n, n)) / n
        self.inconsistent = bool(inconsistent)
        m = 2 * n + int(self.inconsistent)
        super().__init__(
            f"coupled_rosenbrock_n{n}_m{m}",
            self.square_problem.x0.copy(),
            "Dense redundant combinations of extended Rosenbrock equations; "
            + (
                "constant final residual makes the system inconsistent with minimum norm one."
                if inconsistent
                else "consistent overdetermined system with an all-ones root."
            ),
            solution=None if inconsistent else np.ones(n),
        )
        self.m = m
        self.root_exists = not self.inconsistent

    def residual(self, x):
        """Evaluate the square equations, their dense combinations, and optional constant."""
        values = self.square_problem.residual(x)
        combined = np.concatenate((values, self.mixing @ values))
        if self.inconsistent:
            combined = np.append(combined, 1.0)
        return combined

    def gradient(self, x):
        """Return the analytic rectangular Jacobian, including a zero constant-row derivative."""
        square = self.square_problem.gradient(x)
        combined = np.vstack((square, self.mixing @ square))
        if self.inconsistent:
            combined = np.vstack((combined, np.zeros(self.n)))
        return combined
