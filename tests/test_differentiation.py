import pytest
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as pv


# Define step sizes
step_sizes = [1e-4, 1e-6, 1e-8]

# Initialize the problems and initial points
x0 = [1.5 * np.array([1, 1, 1, 1])]

PROBLEMS = [
    pv.RosenbrockProblem(len(x0)),
    pv.RosenbrockProblemConstrained(len(x0)),
    pv.HS71Problem(),
]

# Create a combined parameter list for problems and step sizes
PROBLEM_COMBINATIONS = list(itertools.product(PROBLEMS, x0, step_sizes))

# Create a combined parameter list for problems and step sizes
PROBLEM_COMBINATIONS_BIS = list(itertools.product(PROBLEMS, x0, [1e-6]))

# ---------------------------------------------------------------------------- #
# Forward Finite Differences Consistency Test
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("problem, x0, h", PROBLEM_COMBINATIONS)
def test_forward_finite_differences_consistency(problem, x0, h):
    f = problem.fitness
    grad_approx_derivative = pv.approx_derivative(f, x0, abs_step=h, method="2-point")
    grad_approx_gradient = pv.approx_gradient(f, x0, abs_step=h, method="forward_finite_differences")
    grad_forward_fd = pv.forward_finite_differences(f, x0, abs_step=h)
    
    print(grad_approx_derivative)
    print(grad_forward_fd)

    atol = 1e-15
    assert np.allclose(grad_approx_derivative, grad_approx_gradient, atol=atol), (
        f"Gradient mismatch between approx_derivative and approx_gradient for h={h}: "
        f"{grad_approx_derivative} vs {grad_approx_gradient} within absolute tolerance of {atol}."
    )
    assert np.allclose(grad_approx_derivative, grad_forward_fd, atol=atol), (
        f"Gradient mismatch between approx_derivative and forward_finite_differences for h={h}: "
        f"{grad_approx_derivative} vs {grad_forward_fd} within absolute tolerance of {atol}."
    )


# ---------------------------------------------------------------------------- #
# Central Finite Differences Consistency Test
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("problem, x0, h", PROBLEM_COMBINATIONS)
def test_central_finite_differences_consistency(problem, x0, h):
    f = problem.fitness
    grad_approx_derivative = pv.approx_derivative(f, x0, abs_step=h, method="3-point")
    grad_approx_gradient = pv.approx_gradient(f, x0, abs_step=h, method="central_finite_differences")
    grad_central_fd = pv.central_finite_differences(f, x0, abs_step=h)

    atol = 1e-15
    assert np.allclose(grad_approx_derivative, grad_approx_gradient, atol=atol), (
        f"Gradient mismatch between approx_derivative and approx_gradient for h={h}: "
        f"{grad_approx_derivative} vs {grad_approx_gradient} within absolute tolerance of {atol}."
    )
    assert np.allclose(grad_approx_derivative, grad_central_fd, atol=atol), (
        f"Gradient mismatch between approx_derivative and central_finite_differences for h={h}: "
        f"{grad_approx_derivative} vs {grad_central_fd} within absolute tolerance of {atol}."
    )


# ---------------------------------------------------------------------------- #
# Complex Step Consistency Test
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("problem, x0, h", PROBLEM_COMBINATIONS)
def test_complex_step_consistency(problem, x0, h):
    f = problem.fitness
    grad_approx_derivative = pv.approx_derivative(f, x0, abs_step=h, method="cs")
    grad_approx_gradient = pv.approx_gradient(f, x0, abs_step=h, method="complex_step")
    grad_complex_step = pv.complex_step_derivative(f, x0, abs_step=h)

    atol = 1e-15
    assert np.allclose(grad_approx_derivative, grad_approx_gradient, atol=atol), (
        f"Gradient mismatch between approx_derivative and approx_gradient for h={h}: "
        f"{grad_approx_derivative} vs {grad_approx_gradient} within absolute tolerance of {atol}."
    )
    assert np.allclose(grad_approx_derivative, grad_complex_step, atol=atol), (
        f"Gradient mismatch between approx_derivative and grad_complex_step for h={h}: "
        f"{grad_approx_derivative} vs {grad_complex_step} within absolute tolerance of {atol}."
    )


# ---------------------------------------------------------------------------- #
# Comparison gradient approximation methods
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("problem, x0, h", PROBLEM_COMBINATIONS_BIS)
def test_comparison_across_methods(problem, x0, h):
    f = problem.fitness

    grad_forward_fd = pv.forward_finite_differences(f, x0, abs_step=h)
    grad_central_fd = pv.central_finite_differences(f, x0, abs_step=h)
    grad_complex_step = pv.complex_step_derivative(f, x0, abs_step=h)

    print(problem, grad_forward_fd, grad_central_fd)

    atol = 1e-3
    rtol = 1e-3
    assert np.allclose(grad_forward_fd, grad_central_fd, atol=atol, rtol=rtol), (
        f"Gradient mismatch between forward finite differences and central finite differences for h={h}: "
        f"{grad_forward_fd} vs {grad_central_fd} within absolute tolerance of {atol}."
    )
    assert np.allclose(grad_forward_fd, grad_complex_step, atol=atol, rtol=rtol), (
        f"Gradient mismatch between forward finite differences and complex step for h={h}: "
        f"{grad_forward_fd} vs {grad_complex_step} within absolute tolerance of {atol}."
    )
    assert np.allclose(grad_central_fd, grad_complex_step, atol=atol, rtol=rtol), (
        f"Gradient mismatch between central finite differences and complex step for h={h}: "
        f"{grad_central_fd} vs {grad_complex_step} within absolute tolerance of {atol}."
    )



# ---------------------------------------------------------------------------- #
# Test Hessian approximation by finite differences
# ---------------------------------------------------------------------------- #

# Generate 5 initial guesses in 10 dimensions with normally distributed disturbance
np.random.seed(42)  # Seed for reproducible results
ndim = 8
initial_guesses = [np.ones(ndim) + np.random.normal(0, 1, ndim) for _ in range(5)]

# Try calculations for different step sizes
stepsize = [1e-4, 1e-5]

# Create a combined parameter list for initial guesses and step sizes
STEPSIZE_INITIALGUESS_COMBINATIONS = list(itertools.product(stepsize, initial_guesses))

@pytest.mark.parametrize("abs_step, x0", STEPSIZE_INITIALGUESS_COMBINATIONS)
def test_hessian_finite_differences(abs_step, x0):

    # Define the problem
    problem = pv.RosenbrockProblem(len(x0))

    # Set lower triangular form to True
    LT = True

    # Compute the analytic Hessian
    H_analytic = problem.hessians(x0, lower_triangular=LT)

    # Define the fitness function
    f = problem.fitness

    # Compute the Hessian using finite differences
    H_finite_diff = pv.approx_jacobian_hessians(f, x0, abs_step=abs_step, lower_triangular=LT)

    # Compare the analytic Hessian with the finite difference Hessian
    tol = 1e-2
    assert np.allclose(H_analytic, H_finite_diff, atol=tol, rtol=tol), (
        f"Analytic Hessian and finite difference Hessian do not match within tolerance {tol} for step size {abs_step}.\n"
        f"Analytic Hessian:\n{H_analytic}\n"
        f"Finite Difference Hessian:\n{H_finite_diff}\n"
    )


if __name__ == "__main__":

    # Running pytest from Python
    # pytest.main([__file__, "-vv", "-k test_hessian_finite_differences"])
    # pytest.main([__file__])
    pytest.main([__file__, "-v"])
    # pytest.main([__file__, "-vv"])
