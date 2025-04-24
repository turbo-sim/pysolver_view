"""Compare the evolution of the constrained Rosenbrock problem solution when using SNOPT
for different termination tolerances"""

# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv

# Create the folder to save figures
OUT_DIR = "figures"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = psv.create_logger("convergence_history", use_datetime=True)

# Define problem
ndim = 50
# x0 = 1.5*np.ones(ndim)
# x0 = [2, 2, 2, 2]
x0 = [1.1, 4.2, 3.5, 1.5]
problem = psv.HS71Problem()
# problem = psv.RosenbrockProblemConstrained(ndim)
problem_name = type(problem).__name__
                
# Initialize figure
fig, ax = plt.subplots()
ax.set_title(f"Solving the {problem_name} in {len(x0)} dimensions")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Constraint violation")
ax.set_yscale("log")

# Solve for various max previous steps used in the BFGS update
tolerances = [1e-3, 1e-6, 1e-9][::-1]
colors = plt.get_cmap("magma")(np.linspace(0.25, 0.8, len(tolerances)))
for i, tolerance in enumerate(tolerances):
    solver = psv.OptimizationSolver(
        problem,
        library="pygmo",
        method="ipopt",
        display_text=True,
        plot_convergence=False,
        logger=logger,
        update_on="gradient",
        max_iterations=10000,
        tolerance=tolerance,
    )
    solver.solve(x0)
    ax.plot(
        solver.convergence_history["grad_count"],
        solver.convergence_history["constraint_violation"],
        label=f'tol={tolerance:0.0e}',
        marker="o",
        markersize=3.5,
        color=colors[i],
    )
    print(solver.x_final)


# Add legend
ax.legend(loc="upper right", fontsize=10, ncol=1)
fig.tight_layout(pad=1)

# Save figure
psv.savefig_in_formats(fig, os.path.join(OUT_DIR, f"ipopt_tolerance_sensitivity"))

# Keep plots open
plt.show()


