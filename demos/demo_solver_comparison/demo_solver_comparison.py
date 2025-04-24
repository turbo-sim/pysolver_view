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

ndim = 10

PROBLEMS_AND_X0 = [
    (psv.RosenbrockProblem(ndim), 1.5 * np.ones(ndim)),
    (psv.RosenbrockProblemConstrained(ndim), 1.5 * np.ones(ndim)),
    (psv.HS71Problem(), 1.5 * np.ones(4)),
    # (pv.LorentzEquationsOpt(), [-3, -3, 3]),
]

for problem, x0 in PROBLEMS_AND_X0:
    # Get problem name
    problem_name = type(problem).__name__

    # Initialize figure
    fig, ax = plt.subplots()
    ax.set_title(f"Solving the {problem_name} in {len(x0)} dimensions")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Objective function")
    ax.set_yscale("log")
    # ax2 = ax.twinx()
    # ax2.set_ylabel("Constraint violation")
    # ax2.set_yscale("log")

    # Define solver configurations
    SOLVERS = [
        # ("scipy", "l-bfgs-b"),
        # ("scipy", "bfgs"),
        ("scipy", "trust-constr"),
        ("scipy", "slsqp"),
        # ("pygmo", "snopt"),
        ("pygmo", "ipopt"),
    ]

    cmap = "magma"
    colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.8, len(SOLVERS)))
    for i, (library, method) in enumerate(SOLVERS):
        # Solve problem
        solver = psv.OptimizationSolver(
            problem,
            library=library,
            method=method,
            max_iterations=500,
            tolerance=1e-5,
            display_text=True,
            plot_convergence=False,
            logger=logger,
            update_on="gradient",
        )
        solver.solve(x0)

        # Plot convergence history
        ax.plot(
            solver.convergence_history["grad_count"],
            solver.convergence_history["objective_value"],
            label=method,
            marker="o",
            markersize=2.5,
            markeredgewidth=1.00,
            linewidth=1.00,
            color=colors[i],
        )

        # ax2.plot(
        #     solver.convergence_history["grad_count"],
        #     solver.convergence_history["constraint_violation"],
        #     label=method,
        #     marker="None",
        #     linestyle="--",
        #     markersize=2.5,
        #     markeredgewidth=1.00,
        #     linewidth=1.00,
        #     color=colors[i],
        # )

    # Add legend
    ax.legend(loc="upper right", fontsize=9, ncol=1)
    fig.tight_layout(pad=1)

    # Save figure
    psv.savefig_in_formats(
        fig, os.path.join(OUT_DIR, f"solver_comparison_{problem_name}")
    )

# Keep plots open
plt.show()
