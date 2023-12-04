# Import packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import PySolverView from parent directory
sys.path.insert(0, os.path.abspath(".."))
import pysolver_view as pv

# Set options for publication-quality figures
pv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = pv.create_logger("convergence_history", use_datetime=True)

# Solve Rosenbrock problem
x0 = np.asarray([2, 2, 2, 2])
problem = pv.RosenbrockProblem()
solver = pv.OptimizationSolver(
    problem, x0, method="slsqp", display=True, plot=False, logger=logger
)
solver.solve()
solver.plot_convergence_history()

# Solve constrained Rosenbrock problem
x0 = np.asarray([2, 2, 2, 2])
problem = pv.RosenbrockProblemConstrained()
solver = pv.OptimizationSolver(
    problem, x0, method="slsqp", display=True, plot=False, logger=logger
)
solver.solve()
solver.plot_convergence_history()

# Solve Hock-Schittkowski problem
x0 = np.asarray([1.5, 5, 5, 1])
problem = pv.HS71Problem()
solver = pv.OptimizationSolver(
    problem, x0, method="slsqp", display=True, plot=False, logger=logger
)
solver.solve()
solver.plot_convergence_history()

# Keep plots open
plt.show()

