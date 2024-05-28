# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as pv

# Set options for publication-quality figures
pv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = pv.create_logger("convergence_history", use_datetime=True)

# Solve Rosenbrock problem
x0 = 1.5*np.asarray([1, 1, 1, 1])
problem = pv.RosenbrockProblem(len(x0))
solver = pv.OptimizationSolver(
    problem,
    library="scipy",
    method="l-bfgs-b",
    max_iterations=100,
    tolerance=1e-6,
    print_convergence=True,
    plot_convergence=True,
    logger=logger,
    update_on="gradient",
    plot_scale_objective="log"
)

# Solve the problem
solver.solve(x0)

# Keep plots open
plt.show()

# # Save the figure
# solver.plot_convergence_history(savefile=True)


