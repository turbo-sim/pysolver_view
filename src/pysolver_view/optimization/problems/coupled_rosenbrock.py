"""Moderate constrained Rosenbrock model for comparing smooth NLP solvers."""

import numpy as np

from .base import OptimizationProblem


class CoupledRosenbrockProblem(OptimizationProblem):
    """Couple Rosenbrock pairs with linear equalities and nonlinear disks.

    Parameters
    ----------
    blocks : int, optional
        Number of two-variable Rosenbrock blocks; must be a multiple of three.
        The default produces 24 variables, four equalities and 12 inequalities.
    coupling : float, optional
        Nonnegative weight penalizing differences of neighboring first variables.

    Notes
    -----
    Coordinates are interleaved ``[a_0, b_0, a_1, b_1, ...]``. Each group of
    three first variables has mean 0.8, and each pair lies inside a disk with
    squared radius ``1 + 0.08*sin(i)``. Inequalities use the public ``g <= 0``
    convention. All quantities are dimensionless and no variable scaling is
    needed. The point ``a=0.8, b=0.5`` is strictly inside every disk and meets
    every equality. First derivatives are exact.
    """

    def __init__(self, blocks=12, coupling=0.2):
        """Validate dimensions and prepare fixed radii and readable names."""
        if isinstance(blocks, bool) or not isinstance(blocks, int):
            raise TypeError("blocks must be an integer multiple of three.")
        if blocks < 3 or blocks % 3:
            raise ValueError("blocks must be a positive multiple of three.")
        if not np.isfinite(coupling) or coupling < 0:
            raise ValueError("coupling must be finite and nonnegative.")
        self.blocks = blocks
        self.coupling = float(coupling)
        self.radius_squared = 1.0 + 0.08 * np.sin(np.arange(blocks))
        self.variable_names = [
            name for i in range(blocks) for name in (f"a_{i}", f"b_{i}")
        ]

    @property
    def initial(self):
        """Return the shared infeasible start, strictly inside variable bounds."""
        x = np.empty(2 * self.blocks)
        x[0::2] = 0.4 + 0.04 * np.sin(np.arange(self.blocks))
        x[1::2] = 1.2 + 0.04 * np.cos(np.arange(self.blocks))
        return x

    def fitness(self, x):
        """Return objective, group equalities, then nonpositive disk inequalities."""
        x = np.asarray(x)
        a, b = x[0::2], x[1::2]
        differences = np.diff(a)
        objective = np.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2)
        objective += self.coupling * np.sum(differences * differences)
        equalities = np.mean(a.reshape(-1, 3), axis=1) - 0.8
        inequalities = a * a + b * b - self.radius_squared
        return np.r_[objective, equalities, inequalities]

    def gradient(self, x):
        """Return exact rows for objective, equalities and disk inequalities."""
        x = np.asarray(x)
        a, b = x[0::2], x[1::2]
        matrix = np.zeros((1 + self.get_nec() + self.get_nic(), x.size))
        matrix[0, 0::2] = 400 * a * (a * a - b) + 2 * (a - 1)
        matrix[0, 1::2] = 200 * (b - a * a)
        differences = np.diff(a)
        matrix[0, 0:-2:2] -= 2 * self.coupling * differences
        matrix[0, 2::2] += 2 * self.coupling * differences
        for group in range(self.get_nec()):
            matrix[1 + group, 6 * group : 6 * group + 6 : 2] = 1 / 3
        rows = 1 + self.get_nec() + np.arange(self.blocks)
        matrix[rows, 2 * np.arange(self.blocks)] = 2 * a
        matrix[rows, 2 * np.arange(self.blocks) + 1] = 2 * b
        return matrix

    def get_bounds(self):
        """Return finite dimensionless lower and upper coordinate bounds."""
        return np.full(2 * self.blocks, -1.5), np.full(2 * self.blocks, 2.0)

    def get_nec(self):
        """Return one mean equality for every three Rosenbrock blocks."""
        return self.blocks // 3

    def get_nic(self):
        """Return one nonlinear disk inequality for every Rosenbrock block."""
        return self.blocks
