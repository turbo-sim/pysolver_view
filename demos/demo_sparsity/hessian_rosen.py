"""Investigate how to define Jacobian and Hessian sparsity in Pygmo"""

# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv

import pygmo as pg
from scipy.optimize import rosen, rosen_der, rosen_hess

class RosenbrockProblem:
    """
    PyGMO-compatible implementation of the Rosenbrock optimization problem.

    Includes the objective function, gradient, Hessian, and gradient sparsity definition.
    """

    def __init__(self, dim):
        """
        Parameters
        ----------
        dim : int
            Number of dimensions for the Rosenbrock function.
        """
        self.dim = dim

    def fitness(self, x):
        """
        Computes the Rosenbrock function value.

        Parameters
        ----------
        x : array-like
            Input variables.

        Returns
        -------
        list
            Objective function value as a one-element list.
        """
        return [rosen(x)]  # Use scipy's implementation for simplicity

    def gradient(self, x):
        """
        Computes the gradient of the Rosenbrock function.

        Parameters
        ----------
        x : array-like
            Input variables.

        Returns
        -------
        array
            Gradient of the Rosenbrock function.
        """
        return rosen_der(x)  # Use scipy's implementation for simplicity

    def hessians(self, x):
        """
        Computes the Hessian of the Rosenbrock function.

        Parameters
        ----------
        x : array-like
            Input variables.

        Returns
        -------
        array
            Lower triangular part of the Hessian, as required by PyGMO.
        """
        H = rosen_hess(x)
        H = H[np.tril_indices(len(x))]
        H = np.asarray([H])
        return H

    def get_bounds(self):
        return (-10 * np.ones(self.dim), 10 * np.ones(self.dim))

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0


# Create the folder to save figures
OUT_DIR = "figures"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Define problem
ndim = 70
x0 = 1.5 * np.ones(ndim)
problem = RosenbrockProblem(ndim)
problem_name = type(problem).__name__


# print(problem.fitness(x0))
# # print(problem.gradient(x0))
# problem.hessians(x0)

# Solve problem with exact Hessian matrix
solver = psv.OptimizationSolver(
    problem,
    library="pygmo",
    method="ipopt",
    display_text=True,
    plot_convergence=False,
    update_on="gradient",
    max_iterations=1000,
    extra_options={"hessian_approximation": "exact"},
)
solver.solve(x0)



# # Initialize figure
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.set_title(f"Solving the {problem_name} in {len(x0)} dimensions")
# ax.set_xlabel("Number of iterations")
# ax.set_ylabel("Objective function")
# ax.set_yscale("log")
# ax2.set_ylabel("Constraint violation")
# ax2.set_yscale("log")



# ax.plot(
#     solver.convergence_history["grad_count"],
#     solver.convergence_history["objective_value"],
#     label="Objective (Exact Hessian)",
#     marker="o",
#     markersize=3.5,
#     color="black",
# )
# ax2.plot(
#     solver.convergence_history["grad_count"],
#     solver.convergence_history["constraint_violation"],
#     label="Constraint",
#     marker="o",
#     markersize=3.5,
#     color="black",
#     linestyle=":",
# )

