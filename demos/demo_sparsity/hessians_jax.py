"""Investigate how to define Jacobian and Hessian sparsity in Pygmo"""

# Import packages
import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # By default jax uses 32 bit, for scientific computing we need 64 bit precision

import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv


class RosenbrockProblemConstrained(psv.OptimizationProblem):
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

    def get_bounds(self):
        return ([-10] * self.dim, [+10] * self.dim)

    def get_nec(self):
        return self.dim - 2

    def get_nic(self):
        return 0
        
    def fitness(self, x):
        # Ensure inputs are arrays compatible with NumPy or JAX
        x = jnp.asarray(x)  # Works for both NumPy and JAX

        # Objective function
        f = jnp.atleast_1d(jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

        # Equality constraints
        def single_constraint(k):
            return (
                3 * x[k + 1] ** 3
                + 2 * x[k + 2] 
                - 5
                + jnp.sin(x[k + 1] - x[k + 2]) * jnp.sin(x[k + 1] + x[k + 2])
                + 4 * x[k + 1]
                - x[k] * jnp.exp(x[k] - x[k + 1])
                - 3
            )

        # Use list comprehension for constraints (compatible with both libraries)
        c_eq = jnp.array([single_constraint(k) for k in range(self.dim - 2)])

        # Combine objective and constraints
        return jnp.concatenate([f, c_eq])
    
    
    def gradient(self, x):
        """
        Compute the gradient of the fitness function (objective + constraints).
        """
        return jax.jacfwd(self.fitness, argnums=0)(x)
    
    def hessians(self, x):
        # print("Hessian calculated!")
        return psv.compute_hessians_jax(self.fitness, x, lower_triangular=True)

    def hessians_approx(self, x):

        H = psv.approx_jacobian_hessians(
            self.fitness, x, abs_step=1e-4, lower_triangular=True
        )
        return H
    



# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Set the seed for reproducibility
seed = 30
np.random.seed(seed)


# Define problem
# ndim = 25
ndim = 3
# x0 = 0.5 * jnp.ones(ndim)
x0 = np.random.uniform(-5, 5, size=ndim)
problem = RosenbrockProblemConstrained(ndim)
problem_name = type(problem).__name__


# print("Approximate hessians")
# print(problem.hessians_approx(x0))

# print("Exact hessians")
# print(problem.hessians_jax(x0))

# Solve problem with exact Hessian matrix
solver = psv.OptimizationSolver(
    problem,
    library="pygmo",
    method="ipopt",
    print_convergence=True,
    plot_convergence=False,
    update_on="gradient",
    max_iterations=1000,
    extra_options={"hessian_approximation": "exact"},
)
solver.solve(x0)


# Initialize figure
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_title(f"Solving the {problem_name} in {len(x0)} dimensions")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Objective function")
ax.set_yscale("log")
ax2.set_ylabel("Constraint violation")
ax2.set_yscale("log")

ax.plot(
    solver.convergence_history["grad_count"],
    solver.convergence_history["objective_value"],
    label="Objective (Exact Hessian)",
    marker="o",
    markersize=3.5,
    color="black",
)
ax2.plot(
    solver.convergence_history["grad_count"],
    solver.convergence_history["constraint_violation"],
    label="Constraint",
    marker="o",
    markersize=3.5,
    color="black",
    linestyle=":",
)

plt.show()

