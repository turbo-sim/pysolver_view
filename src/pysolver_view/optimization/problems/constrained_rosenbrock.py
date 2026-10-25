import numpy as np
from .base import OptimizationProblem
from ...differentiation._scipy_numdiff import approx_derivative
from ...differentiation._jax import load_jax, is_jax_available


class RosenbrockProblemConstrained(OptimizationProblem):
    r"""
    Implementation of the Chained Rosenbrock function with trigonometric-exponential constraints.

    This problem is also referred to as Example 5.1 in the report by Luksan and Vlcek. The optimization problem is described as:

    .. math::

        \begin{align}
        \text{minimize} \quad & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right] \\
        \text{s.t.} \quad & 3x_{k+1}^3 + 2x_{k+2} - 5 + \sin(x_{k+1}-x_{k+2})\sin(x_{k+1}+x_{k+2}) + \\
                            & + 4x_{k+1} - x_k \exp(x_k-x_{k+1}) - 3 = 0, \; \forall k=1,...,n-2 \\
                            & -5 \le x_i \le 5, \forall i=1,...,n
        \end{align}

    References
    ----------
    - Luksan, L., and Jan Vlcek. “Sparse and partially separable test problems for unconstrained and equality constrained optimization.” (1999). `doi: link provided <http://hdl.handle.net/11104/0123965>`_.

    Methods
    -------
    evaluate_problem(x):
        Compute objective, equality, and inequality constraint.
    get_bounds():
        Return variable bounds.
    get_n_eq():
        Get number of equality constraints.
    get_n_ineq():
        Get number of inequality constraints.
    """

    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):

        # Ordinary model calls retain NumPy's precision and complex-step values.
        # Only JAX tracing needs its array namespace.
        xp = load_jax()[1] if type(x).__module__.startswith(("jax", "jaxlib")) else np
        # xp = np
        x = xp.asarray(x)

        # Objective function
        f = [xp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)]

        # Equality constraints
        c_eq = []
        for k in range(self.dim - 2):
            val = (
                3 * x[k + 1] ** 3
                + 2 * x[k + 2]
                - 5
                + xp.sin(x[k + 1] - x[k + 2]) * xp.sin(x[k + 1] + x[k + 2])
                + 4 * x[k + 1]
                - x[k] * xp.exp(x[k] - x[k + 1])
                - 3
            )
            c_eq.append(val)

        fitness = xp.concatenate((xp.asarray(f), xp.asarray(c_eq)))

        return fitness

        # c_ineq = None

        # return combine_objective_and_constraints(f, c_eq, c_ineq)

    def gradient(self, x, method="automatic_differentiation"):
        if method == "automatic_differentiation":
            if not is_jax_available():
                print("JAX not available. Falling back to finite differences.")
                method = "finite_differences"
            else:
                try:
                    jax, jnp = load_jax()
                    with jax.experimental.enable_x64():
                        jac_fn = jax.jacobian(self.fitness)
                        return np.array(jac_fn(jnp.array(x)))
                except Exception as e:
                    print(
                        f"Automatic differentiation failed ({e}). Falling back to finite differences."
                    )
                    method = "finite_differences"

        if method == "finite_differences":
            return approx_derivative(
                self.fitness,
                x,
                method="2-point",
                abs_step=1e-8,
            )

        raise ValueError(
            f"Unsupported method '{method}'. Use 'finite_differences' or 'automatic_differentiation'."
        )

    # def gradient(self, x):
    #     gradient = approx_derivative(
    #         self.fitness,
    #         x,
    #         method="2-point",
    #         abs_step=1e-8,
    #     )
    #     return gradient

    # def hessians(self, x):
    #     H = approx_jacobian_hessians(
    #         self.fitness, x, abs_step=1e-4, lower_triangular=True
    #     )
    #     return H

    def get_bounds(self):
        return ([-10] * self.dim, [+10] * self.dim)

    def get_nec(self):
        return self.dim - 2

    def get_nic(self):
        return 0
