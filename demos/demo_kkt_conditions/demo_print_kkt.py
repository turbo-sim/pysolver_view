# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv


# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = psv.create_logger("convergence_history", use_datetime=True, to_console=False)

# Define optimization problem
# ndim = 20
# x0 = 1.5 * np.ones(ndim)
# problem = psv.RosenbrockProblem(ndim)
# problem = psv.RosenbrockProblemConstrained(ndim)
x0 = np.asarray([2.5, 3, 2, 2])
problem = psv.HS71Problem()

# Solve problem
# library, method = ("scipy", "slsqp")
library, method = ("pygmo", "ipopt")
solver = psv.OptimizationSolver(
    problem,
    library=library,
    method=method,
    max_iterations=100,
    tolerance=1e-6,
    problem_scale=10.0,
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    update_on="gradient",
    # extra_options={"hessian_approximation": "exact"},
    # extra_options={'tol': 1e-9}
)
solver.solve(x0)
solver.print_optimization_report(
    include_design_variables=True,
    include_constraints=True,
    include_kkt_conditions=True,
    include_multipliers=True,
    savefile=True,
)


# TODO: I have to improve the printing and logging behavior
# TODO: currently there are some clashes leading to bad/unintuitive user experience
# TODO: update the documentation

# Show figure
plt.show()
