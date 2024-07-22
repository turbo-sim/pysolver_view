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
solver_1 = pv.OptimizationSolver(
    problem,
    library="scipy",
    method="nelder-mead",
    max_iterations=100,
    tolerance=1e-6,
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    update_on="function",
    plot_scale_objective="log",
    plot_improvement_only=False,
)
solver_1.solve(x0)
solver_1.plot_convergence_history()

# Solve Rosenbrock problem
x0 = 1.5*np.asarray([1, 1, 1, 1])
problem = pv.RosenbrockProblem(len(x0))
solver_2 = pv.OptimizationSolver(
    problem,
    library="scipy",
    method="nelder-mead",
    max_iterations=100,
    tolerance=1e-6,
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    update_on="function",
    plot_scale_objective="log",
    plot_improvement_only=True,
)
solver_2.solve(x0)
solver_2.plot_convergence_history()

# Keep plots open
plt.show()

