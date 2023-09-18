# Import packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import PySolverView from parent directory
sys.path.insert(0, os.path.abspath('..'))
import PySolverView as psv

# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = psv.create_logger("convergence_history", use_datetime=True)

# Solve Rosenbrock problem
x0 = np.asarray([2, 2, 2, 2])
problem = psv.RosenbrockProblem()
solver = psv.OptimizationSolver(problem, x0, display=True, plot=True, logger=logger)
sol = solver.solve(method="slsqp")

# Solve constrained Rosenbrock problem
x0 = np.asarray([2, 2, 2, 2])
problem = psv.RosenbrockProblemConstrained()
solver = psv.OptimizationSolver(problem, x0, display=True, plot=True, logger=logger)
sol = solver.solve(method="slsqp")

# Solve Hock-Schittkowski problem
x0 = np.asarray([1.5, 5, 5, 1])
problem = psv.HS71Problem()
solver = psv.OptimizationSolver(problem, x0, display=True, plot=True, logger=logger)
sol = solver.solve(method="slsqp")

# Keep plots open
plt.show()

# # Save figure
# solver.plot_convergence_history(savefig=True, use_datetime=False)


