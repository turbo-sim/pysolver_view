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
# Analyze sensitivity to the first Wolfe condition (Armijo)
# -------------------------------------------------------------------- #
fig, ax = plt.subplots()
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel("Objective function values")
ax.set_yscale("log")
wolfe_conditions = [
    (1e-6, 0.90),
    (1e-5, 0.90),
    (1e-4, 0.90),
    (1e-3, 0.90),
    (1e-2, 0.90),
    (1e-1, 0.90),
]
cmap = "magma"
colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.8, len(wolfe_conditions)))
for (c1, c2), color in zip(wolfe_conditions, colors):

    # Create solver
    options = {"c1": c1, "c2": c2}
    solver = psv.OptimizationSolver(
        problem,
        library="scipy",
        method="bfgs",
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
        label=rf"$c_1={c1:0.2e},\, c_2={c2:0.2f}$",
        marker="o",
        markersize=2.5,
        color=color,
    )

title = "BFGS Wolfe conditions"
ax.legend(title=title, title_fontsize=10, loc="lower left", fontsize=10)
fig.tight_layout(pad=1)
psv.savefig_in_formats(fig, os.path.join(OUT_DIR, "sensitivity_wolfe_condition_1"))


# -------------------------------------------------------------------- #
# Analyze sensitivity to the second Wolfe condition (Curvature)
# -------------------------------------------------------------------- #
fig, ax = plt.subplots()
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel("Objective function values")
ax.set_yscale("log")
wolfe_conditions = [(1e-4, 0.9)]
wolfe_conditions = [
    (1e-4, 0.60),
    (1e-4, 0.70),
    (1e-4, 0.80),
    (1e-4, 0.90),
    (1e-4, 0.99),
]
cmap = "magma"
colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.8, len(wolfe_conditions)))
for (c1, c2), color in zip(wolfe_conditions, colors):

    # Create solver
    options = {"c1": c1, "c2": c2}
    solver = psv.OptimizationSolver(
        problem,
        library="scipy",
        method="bfgs",
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
        label=rf"$c_1={c1:0.2e},\, c_2={c2:0.2f}$",
        marker="o",
        markersize=2.5,
        color=color,
    )

title = "BFGS Wolfe conditions"
ax.legend(title=title, title_fontsize=10, loc="lower left", fontsize=10)
fig.tight_layout(pad=1)
psv.savefig_in_formats(fig, os.path.join(OUT_DIR, "sensitivity_wolfe_condition_2"))


# Keep plots open
plt.show()