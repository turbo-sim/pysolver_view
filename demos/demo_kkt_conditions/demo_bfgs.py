"""Compare the evolution of the Rosenbrock problem solution when using SNOPT
for different number of previous steps used for the BFGS update"""

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

# Define problem
ndim = 20
x0 = 1.5*np.ones(ndim)
# x0 = [7, 0]
problem = psv.RosenbrockProblem(ndim)
# problem = psv.SimoneProblem()
# problem = psv.RosenbrockProblemConstrained(ndim)
# problem_name = type(problem).__name__
                
# Solve for various max previous steps used in the BFGS update
solver = psv.OptimizationSolver(
    problem,
    library="pygmo",
    method="ipopt",
    print_convergence=True,
    plot_convergence=False,
    logger=logger,
    # problem_scale=1000,
    update_on="function",
    max_iterations=10000,
    tolerance=1e-8,
    plot_scale_objective="log",
    extra_options={
                #    "tau_min": 0.1, "mu_init": 0.01,
                   "limited_memory_max_history": 100,
                   "limited_memory_update_type": "bfgs", 
                   "limited_memory_max_skipping": 100,
                   "hessian_approximation": "limited-memory",
                   "tol": 1e-9}
)

solver.solve(x0)
solver.plot_convergence_history()



# solver = psv.OptimizationSolver(
#     problem,
#     library="scipy",
#     method="slsqp",
#     print_convergence=False,
#     plot_convergence=False,
#     logger=logger,
#     # problem_scale=100e6,
#     update_on="function",
#     max_iterations=10000,
#     tolerance=1e-8,
#     plot_scale_objective="log",
#     # extra_options={"limited_memory_max_history": 100,
#     #                "limited_memory_update_type": "bfgs", 
#     #                "limited_memory_max_skipping": 100,
#     #                "hessian_approximation": "limited-memory",
#     #                "tol": 1e-9}
# )

# print("Solving problem without scaling")
# solver.solve(x0)
# solver.print_convergence_history()
# solver.plot_convergence_history()


# print("Solving problem with scaling")

# solver = psv.OptimizationSolver(
#     problem,
#     library="scipy",
#     method="slsqp",
#     print_convergence=False,
#     plot_convergence=False,
#     logger=logger,
#     problem_scale=100e6,
#     update_on="function",
#     max_iterations=10000,
#     tolerance=1e-8,
#     plot_scale_objective="log",
#     # extra_options={"limited_memory_max_history": 100,
#     #                "limited_memory_update_type": "bfgs", 
#     #                "limited_memory_max_skipping": 100,
#     #                "hessian_approximation": "limited-memory",
#     #                "tol": 1e-9}
# )
# solver.solve(x0)
# solver.print_convergence_history()
# solver.plot_convergence_history()




# Keep plots open
plt.show()


