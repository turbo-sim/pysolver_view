from .base import OptimizationProblem
from ..constraints import combine_objective_and_constraints


class HS71Problem(OptimizationProblem):
    r"""
    Implementation of the Hock Schittkowski problem No.71.

    This class implements the following optimization problem:

    .. math::

        \begin{align}
        \text{minimize} \quad  & f(\mathbf{x}) = x_1x_4(x_1+x_2+x_3) + x_3 \\
        \text{s.t.} \quad      & x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40 \\
                                & 25 - x_1 x_2 x_3 x_4 \le 0 \\
                                & 1 \le x_1, x_2, x_3, x_4 \le 5
        \end{align}


    References
    ----------
    - W. Hock and K. Schittkowski. Test examples for nonlinear programming codes. Lecture Notes in Economics and Mathematical Systems, 187, 1981. `doi: 10.1007/978-3-642-48320-2 <https://doi.org/10.1007/978-3-642-48320-2>`_.


    Methods
    -------
    evaluate_problem(x)`:
        Compute objective, equality, and inequality constraint.
    get_bounds()`:
        Return variable bounds.
    get_n_eq()`:
        Get number of equality constraints.
    get_n_ineq()`:
        Get number of inequality constraints.

    """

    def fitness(self, x):
        # Objective function
        f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        # Equality constraints
        c_eq = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2) - 40.0

        # Inequality constraints
        c_ineq = 25 - x[0] * x[1] * x[2] * x[3]
        # c_ineq = None

        return combine_objective_and_constraints(f, c_eq, c_ineq)

    def get_bounds(self):
        return (4 * [1.0], 4 * [5])

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1
