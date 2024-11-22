import numpy as np
import pysolver_view as psv

# Define the problem
x0 = 1.5 * np.array([1, 2, 1, 0.5])
problem = psv.RosenbrockProblem(len(x0))
# problem = pv.RosenbrockProblemConstrained(len(x0))
# problem = pv.HS71Problem()

# Get the fitness function handle
f = problem.fitness

# Define step size
# h = None
h = 1e-6
# h = np.abs(x0)*1e-6

# Compute gradient using forward finite differences with scalar step size
grad_forward_fd_scalar = psv.approx_gradient(f, x0, abs_step=h, method="forward_finite_differences")
print(f"Gradient using forward finite differences with step size ({h}):\n{grad_forward_fd_scalar}\n")

# Compute gradient using central finite differences with scalar step size
grad_central_fd_scalar = psv.approx_gradient(f, x0, abs_step=h, method="central_finite_differences")
print(f"Gradient using central finite differences with step size ({h}):\n{grad_central_fd_scalar}\n")

# Compute gradient using complex step method with scalar step size
grad_complex_step_scalar = psv.approx_gradient(f, x0, abs_step=h, method="complex_step")
print(f"Gradient using complex step method with step size ({h}):\n{grad_complex_step_scalar}\n")

