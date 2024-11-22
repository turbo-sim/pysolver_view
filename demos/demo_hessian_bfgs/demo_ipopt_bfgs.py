"""Compare the evolution of the Rosenbrock problem solution when using IPOPT
for different number of previous steps used for the BFGS update"""

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
# problem = pv.RosenbrockProblemConstrained(ndim)
problem_name = type(problem).__name__

# Initialize figure
fig, ax = plt.subplots()
ax.set_title(f"Solving the {problem_name} in {len(x0)} dimensions")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Objective function")
ax.set_yscale("log")

# Solve with exact Hessian
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
    label="Exact Hessian",
    marker="o",
    markersize=2.5,
    color="black",
)

# Solve for various max previous steps used in the BFGS update
bfgs_history = [10, 20, 30, 40, 50]
colors = plt.get_cmap("magma")(np.linspace(0.25, 0.8, len(bfgs_history)))
for i, bfgs_steps in enumerate(bfgs_history):
    solver = psv.OptimizationSolver(
        problem,
        library="pygmo",
        method="ipopt",
        print_convergence=True,
        plot_convergence=False,
        logger=logger,
        update_on="gradient",
        max_iterations=1000,
        tolerance=1e-4,
        extra_options={
            "limited_memory_update_type": "bfgs",
            "limited_memory_max_history": bfgs_steps,
        },
    )
    solver.solve(x0)
    ax.plot(
        solver.convergence_history["grad_count"],
        solver.convergence_history["objective_value"],
        label=f"BFGS with {bfgs_steps}-step history",
        marker="o",
        markersize=2.5,
        color=colors[i],
    )


# Add legend
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_ylim([ax.get_ylim()[0], 10000*ax.get_ylim()[-1]])
fig.tight_layout(pad=1)

# Save figure
psv.savefig_in_formats(
    fig, os.path.join(OUT_DIR, f"IPOPT_BFGS_steps_{problem_name}_{ndim}dims")
)

# Keep plots open
plt.show()
