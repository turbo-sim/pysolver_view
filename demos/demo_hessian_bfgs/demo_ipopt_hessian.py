"""Compare the evolution of objective function and constraints for the constrained Rosenbrock
problem when solving using IPOPT with exact Hessian vs approximate BFGS update."""

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
ndim = 20
x0 = 1.5 * np.ones(ndim)
problem = psv.RosenbrockProblem(ndim)
# problem = psv.RosenbrockProblemConstrained(ndim)
problem_name = type(problem).__name__

# Initialize figure
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_title(f"Solving the {problem_name} in {len(x0)} dimensions")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Objective function")
ax.set_yscale("log")
ax2.set_ylabel("Constraint violation")
ax2.set_yscale("log")

# Solve problem with exact Hessian matrix
solver = psv.OptimizationSolver(
    problem,
    library="pygmo",
    method="ipopt",
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    update_on="gradient",
    max_iterations=1000,
    extra_options={"hessian_approximation": "exact"},
)
solver.solve(x0)
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

# Solve problem with BFGS Hessian update
solver = psv.OptimizationSolver(
    problem,
    library="pygmo",
    method="ipopt",
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    update_on="gradient",
    max_iterations=1000,
    extra_options={
        "limited_memory_update_type": "bfgs",
        "limited_memory_max_history": 30,
    },
)
solver.solve(x0)
ax.plot(
    solver.convergence_history["grad_count"],
    solver.convergence_history["objective_value"],
    label="Objective (BFGS update)",
    marker="o",
    markersize=3.5,
    color="blue",
)
ax2.plot(
    solver.convergence_history["grad_count"],
    solver.convergence_history["constraint_violation"],
    label="Constraint",
    marker="o",
    markersize=3.5,
    color="blue",
    linestyle=":",
)

# Add legend
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right",
    fontsize=9,
    ncol=2,
    columnspacing=0.75,
)
fig.tight_layout(pad=1)

# Save figure
psv.savefig_in_formats(
    fig, os.path.join(OUT_DIR, f"IPOPT_HessianComparison_{problem_name}_{ndim}dims")
)

# Keep plots open
plt.show()
