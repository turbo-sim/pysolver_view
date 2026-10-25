import numpy as np
from .base import OptimizationProblem
from ..constraints import combine_objective_and_constraints


class RosenbrockProblem(OptimizationProblem):
    r"""
    Implementation of the Rosenbrock problem.

    The Rosenbrock problem, also known as Rosenbrock's valley or banana function,
    is defined as:

    .. math::

        \begin{align}
        \text{minimize} \quad  & f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right] \\
        \end{align}

    Methods
    -------
    evaluate_problem(x)
        Evaluates the Rosenbrock function and its constraints.
    get_bounds()
        Returns the bounds for the problem.
    get_n_eq()
        Returns the number of equality constraints.
    get_n_ineq()
        Returns the number of inequality constraints.
    """

    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        """Rosenbrock function value"""
        f = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        return combine_objective_and_constraints(f, None, None)

    def gradient(self, x):
        """Rosenbrock function gradient"""
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        grad = np.zeros_like(x)
        grad[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        grad[-1] = 200 * (x[-1] - x[-2] ** 2)
        return grad

    def hessians(self, x, lower_triangular=True):
        """Rosenbrock function gradient"""
        x = np.atleast_1d(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        if lower_triangular:
            H = H[np.tril_indices(len(x))]  # Lower triangular
            H = np.asarray([H])  # Correct Pygmo array shape
        # return H.squeeze()
        return H

    # def hessians(self, x, lower_triangular=True):
    #     return approx_jacobian_hessians(self.fitness, x, lower_triangular=lower_triangular)

    def get_bounds(self):
        return (-10 * np.ones(self.dim), 10 * np.ones(self.dim))

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0
