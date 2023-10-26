# Import packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import package from parent directory
sys.path.insert(0, os.path.abspath('..'))
import pysolver_view as pv

# Set options for publication-quality figures
pv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = pv.create_logger("convergence_history", use_datetime=True)

# Find a stationary point of the Lorentz equations
x0 = np.asarray([1.0, -3.0, 2.0])
problem = pv.LorentzEquations()
solver = pv.NonlinearSystemSolver(problem, x0, display=True, plot=True, logger=logger)
solution = solver.solve(method="hybr")
# solution = solver.solve(method="lm")
# solution = solver.plot_convergence_history()

# Keep plots open
plt.show()

# # Save figure
# solver.plot_convergence_history(savefig=True, use_datetime=False)

