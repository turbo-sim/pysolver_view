import numpy as np
import pysolver_view as pv

# Define the problem
x0 = 1.5 * np.array([1, 2, 1, 0.5])
problem = pv.RosenbrockProblem(len(x0))
f = problem.fitness

# Set lower triangular form to True
LT = True

# Compute the analytic Hessian
H_analytic = problem.hessians(x0, lower_triangular=LT)
print(f"Analytic Hessian at x0 (lower triangular={LT}):\n{H_analytic}\nShape: {H_analytic.shape}\n")

# Compute the Hessian using finite differences
H_finite_diff = pv.approx_jacobian_hessians(f, x0, lower_triangular=LT)
print(f"Finite Difference Hessian at x0 (lower triangular={LT}):\n{H_finite_diff}\nShape: {H_finite_diff.shape}\n")

# Compare the analytic Hessian with the finite difference Hessian
tol = 1e-3
if np.allclose(H_analytic, H_finite_diff, atol=0, rtol=tol):
    print(f"The analytic Hessian and finite difference Hessian match within relative tolerance {tol}.")
else:
    print(f"The analytic Hessian and finite difference Hessian do not match within relative tolerance {tol}.")
    print(np.isclose(H_analytic, H_finite_diff, atol=0, rtol=tol))
