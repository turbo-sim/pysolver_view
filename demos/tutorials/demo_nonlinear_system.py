# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv

# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = psv.create_logger("convergence_history", use_datetime=True)

# Find a stationary point of the Lorentz equations
x0 = np.asarray([1.0, -3.0, 2.0])
problem = psv.LorentzEquations()
solver = psv.NonlinearSystemSolver(
    problem,
    method="hybr",
    print_convergence=True,
    plot_convergence=True,
    logger=logger,
)

# Solve the problem
solver.solve(x0)

# Keep plots open
plt.show()

# # Save the figure
# solver.plot_convergence_history(savefile=True)


