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
# problem = psv.RosenbrockProblem(ndim)
problem = psv.RosenbrockProblemConstrained(ndim)

# Solve problem
library, method = ("scipy", "slsqp")
# library, method = ("pygmo", "ipopt")
solver = psv.OptimizationSolver(
    problem,
    library="scipy",
    method="slsqp",
    max_iterations=100,
    tolerance=1e-8,
    problem_scale=500,
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    update_on="function",
)
solver.solve(x0)
solver.print_optimization_report()

# Show figure
plt.show()
