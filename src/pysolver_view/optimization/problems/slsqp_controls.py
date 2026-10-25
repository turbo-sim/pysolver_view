"""Small analytical optimization models with explicit domain and call records."""

import numpy as np
from ...exceptions import DomainError
from .base import OptimizationProblem


class StudyProblem(OptimizationProblem):
    """Provide Rosenbrock variants and a coupled square-root domain example."""

    def __init__(self, case):
        """Select a model and keep model attempts separate from audit evaluations."""
        if case not in {
            "rosenbrock",
            "constrained_rosenbrock",
            "square_root",
            "inconsistent",
        }:
            raise ValueError(case)
        self.case = case
        self.variable_names = ["x", "y"]
        self.calls = []

    @property
    def initial(self):
        """Return a valid start whose first unscaled QP direction is aggressive."""
        return np.array([0.36, 0.9] if self.case == "square_root" else [-1.2, 1.0])

    def values(self, x):
        """Evaluate without recording: used for independent post-solve audits."""
        a, b = x
        if self.case == "square_root":
            if np.real(a) <= 0:
                raise DomainError("Square-root property model requires x > 0")
            u, v = np.sqrt(a) - 0.2, b - 0.4
            return np.array([u * u + v * v + (u + v) ** 2])
        objective = 100 * (b - a * a) ** 2 + (1 - a) ** 2
        constraints = [] if self.case == "rosenbrock" else [a + b - 1]
        if self.case == "inconsistent":
            constraints.append(2 - a - b)
        return np.array([objective, *constraints])

    def fitness(self, x):
        """Record every attempted point, including domain failures, exactly once."""
        record = {"x": np.real(x).copy(), "valid": False, "objective": None}
        self.calls.append(record)
        values = self.values(x)
        record.update(valid=True, objective=float(np.real(values[0])))
        return values

    def gradient(self, x):
        """Return exact objective and constraint derivatives in public signs."""
        a, b = x
        if self.case == "square_root":
            u, v = np.sqrt(a) - 0.2, b - 0.4
            return np.array([[(2 * u + v) / np.sqrt(a), 2 * u + 4 * v]])
        rows = [[400 * a * (a * a - b) + 2 * (a - 1), 200 * (b - a * a)]]
        if self.case != "rosenbrock":
            rows.append([1.0, 1.0])
        if self.case == "inconsistent":
            rows.append([-1.0, -1.0])
        return np.array(rows)

    def get_bounds(self):
        """Leave Rosenbrock unbounded; the square-root box does not enforce its domain."""
        return (
            ([-2.0, -2.0], [2.0, 2.0])
            if self.case == "square_root"
            else ([-np.inf] * 2, [np.inf] * 2)
        )

    def get_nec(self):
        """These cases have no equality constraints."""
        return 0

    def get_nic(self):
        """Return the number of nonpositive linear constraints."""
        return (
            2
            if self.case == "inconsistent"
            else int(self.case == "constrained_rosenbrock")
        )
