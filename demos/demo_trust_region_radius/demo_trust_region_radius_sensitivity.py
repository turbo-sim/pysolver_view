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

# Define Rosenbrock problem
ndim = 10
x0 = 1.50 * np.ones(ndim)
problem = psv.RosenbrockProblem(ndim)


# -------------------------------------------------------------------- #
# Analyze sensitivity to trust region radius
# -------------------------------------------------------------------- #
fig, ax = plt.subplots()
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel("Objective function values")
ax.set_yscale("log")
trust_region_radius = [2, 1, 0.5, 0.1, 0.01]
cmap = "magma"
colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.8, len(trust_region_radius)))
for radius, color in zip(trust_region_radius, colors):

    # Create solver
    options = {"initial_tr_radius": radius}
    solver = psv.OptimizationSolver(
        problem,
        method="trust-constr",
        display_text=True,
        plot_convergence=False,
        logger=logger,
        extra_options=options,
        update_on="function",
    )

    # Solve problem
    solver.solve(x0)

    # Plot convergence history
    ax.plot(
        solver.convergence_history["func_count"],
        solver.convergence_history["objective_value"],
        label=rf"$R={radius:0.2f}$",
        marker="o",
        markersize=2.5,
        color=color,
    )

title = "Initial trust region radius"
ax.legend(title=title, title_fontsize=10, loc="lower left", fontsize=10)
fig.tight_layout(pad=1)
psv.savefig_in_formats(fig, os.path.join(OUT_DIR, "sensitivity_initial_trust_region_radius"))

# Keep plots open
plt.show()
