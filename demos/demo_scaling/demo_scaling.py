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

# Define optimization problem
ndim = 10
x0 = 1.50 * np.ones(ndim)
problem = psv.RosenbrockProblem(ndim)
# problem = psv.RosenbrockProblemConstrained(ndim)

# Solve problem
# library, method = ("scipy", "slsqp")
library, method = ("pygmo", "ipopt")
fig, ax = plt.subplots()
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Objective function")
ax.set_yscale("log")
scales = [None, 10, 50, 100, 500, 1000]
# scales = [None]
cmap = "magma"
colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.75, len(scales)))
for i, scale in enumerate(scales):
    solver = psv.OptimizationSolver(
        problem,
        library="scipy",
        method="slsqp",
        max_iterations=100,
        tolerance=1e-8,
        problem_scale=scale,
        display_text=True,
        plot_convergence=False,
        logger=logger,
        update_on="function",
    )
    solver.solve(x0)

    ax.plot(
        solver.convergence_history["func_count"],
        solver.convergence_history["objective_value"],
        label=f"Scaling: {scale}",
        marker="o",
        markersize=2.5,
        markeredgewidth=1.25,
        linewidth=1.00,
        color=colors[i],
    )

#     print(solver.problem.print_design_variables_report(x_norm=solver.x_final, normalized=True))
#     print(solver.problem.print_design_variables_report(x_norm=solver.x_final, normalized=False))


# Show legend
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(pad=1)

# Show figure
plt.show()
