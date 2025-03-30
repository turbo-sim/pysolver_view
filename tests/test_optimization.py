import pytest
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv


# Set up logger with unique date-time name
logger = psv.create_logger("convergence_history", use_datetime=True)

# Define solver configurations
UNCONSTRAINED_SOLVERS = [
    ("scipy", "l-bfgs-b"),
    # ("scipy", "bfgs"),
    ("scipy", "trust-constr"),
    ("scipy", "slsqp"),
    ("pygmo", "ipopt"),
    # ("pygmo", "snopt"),
]

CONSTRAINED_SOLVERS = [
    ("scipy", "trust-constr"),
    ("scipy", "slsqp"),
    ("pygmo", "ipopt"),
    # ("pygmo", "snopt"),
]


# ---------------------------------------------------------------------------- #
# Unconstrained Rosenbrock problem
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("library, method", UNCONSTRAINED_SOLVERS)
def test_rosenbrock_problem_unconstrained(library, method):
    # Set up problem and solver
    x0 = 1.5 * np.array([1, 1, 1, 1])
    problem = psv.RosenbrockProblem(len(x0))
    extra_options = {"tol": 1e-6} if method == "ipopt" else {}
    solver = psv.OptimizationSolver(
        problem,
        library=library,
        method=method,
        print_convergence=False,
        plot_convergence=False,
        max_iterations=300,
        tolerance=1e-9,
        update_on="gradient",
        logger=logger,
        extra_options=extra_options
    )

    # Perform the optimization
    solver.solve(x0)

    # Tolerance for tests
    atol = 1e-6

    # Numerical solution
    x_numeric = solver.x_final
    f_numeric = solver.fitness(solver.x_final)

    # Analytic solution
    x_analytic = np.ones(len(x0))
    f_analytic = 0.00

    # Assertions
    assert solver.success, f"Optimization with {library} and {method} did not converge"
    assert np.allclose(x_numeric, x_analytic, atol=atol), (
        f"Optimal solution mismatch for {library}/{method}: "
        f"calculated solution {x_numeric} deviates from expected {x_analytic} "
        f"within absolute tolerance of {atol}."
    )
    assert np.allclose(f_numeric, f_analytic, atol=atol), (
        f"Optimal fitness mismatch for {library}/{method}: "
        f"calculated fitness {f_numeric} deviates from expected {f_analytic} "
        f"within absolute tolerance of {atol}."
    )


# ---------------------------------------------------------------------------- #
# Constrained Rosenbrock problem
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("library, method", CONSTRAINED_SOLVERS)
def test_rosenbrock_problem_constrained(library, method):
    # Set up problem and solver
    x0 = 1.5 * np.array([1, 1, 1, 1])
    problem = psv.RosenbrockProblemConstrained(len(x0))
    solver = psv.OptimizationSolver(
        problem,
        library=library,
        method=method,
        print_convergence=False,
        plot_convergence=False,
        max_iterations=300,
        tolerance=1e-9,
        update_on="gradient",
        logger=logger,
    )

    # Perform the optimization
    solver.solve(x0)

    # Tolerance for tests
    atol = 1e-6

    # Numerical solution
    x_numeric = solver.x_final
    f_numeric = solver.fitness(solver.x_final)

    # Analytic solution
    x_analytic = np.ones(len(x0))
    f_analytic = np.zeros(len(x0) - 1)

    # Assertions
    assert solver.success, f"Optimization with {library} and {method} did not converge"
    assert np.allclose(x_numeric, x_analytic, atol=atol), (
        f"Optimal solution mismatch for {library}/{method}: "
        f"calculated solution {x_numeric} deviates from expected {x_analytic} "
        f"within absolute tolerance of {atol}."
    )
    assert np.allclose(f_numeric, f_analytic, atol=atol), (
        f"Optimal fitness mismatch for {library}/{method}: "
        f"calculated fitness {f_numeric} deviates from expected {f_analytic} "
        f"within absolute tolerance of {atol}."
    )


# ---------------------------------------------------------------------------- #
# Lorentz system stationary points problem
# ---------------------------------------------------------------------------- #

CONSTRAINED_SOLVERS = [
    #    ("scipy", "trust-constr"),
    ("scipy", "slsqp"),
    ("pygmo", "ipopt"),
    # ("pygmo", "snopt"),
]

# Calculate stationary points of the Lorentz system
SIGMA, BETA, RHO = 1.0, 2.0, 3.0
STATIONARY_POINTS = [
    np.asarray([0.00, 0.00, 0.00]),
    np.asarray([+np.sqrt(BETA * (RHO - 1)), +np.sqrt(BETA * (RHO - 1)), RHO - 1]),
    np.asarray([-np.sqrt(BETA * (RHO - 1)), -np.sqrt(BETA * (RHO - 1)), RHO - 1]),
]


# Parametrize a test for the Lorentz stationary points solved as an optimization problem
@pytest.mark.parametrize(
    "library, method, stationary_point",
    [
        (lib, method, point)
        for lib, method in CONSTRAINED_SOLVERS
        for point in STATIONARY_POINTS
    ],
)
def test_lorentz_system(library, method, stationary_point):
    # Set up problem and solver
    problem = psv.LorentzEquationsOpt(sigma=SIGMA, beta=BETA, rho=RHO)
    solver = psv.OptimizationSolver(
        problem,
        library=library,
        method=method,
        print_convergence=True,
        plot_convergence=False,
        max_iterations=300,
        tolerance=1e-12,
        update_on="gradient",
        logger=logger,
    )

    # Solve the system
    print(stationary_point)
    x0 = stationary_point + 0.5 * np.asarray([1.0, 1.0, 1.0])
    solver.solve(x0)

    # Tolerance for tests
    atol = 1e-6

    # Numerical solution
    x_numeric = solver.x_final
    f_numeric = solver.fitness(solver.x_final)

    # Analytic solution
    x_analytic = stationary_point
    f_analytic = np.zeros(1 + 3)

    # Assertions
    assert solver.success, f"Optimization with {library} and {method} did not converge"
    assert np.allclose(x_numeric, x_analytic, atol=atol), (
        f"Optimal solution mismatch for {library}/{method}: "
        f"calculated solution {x_numeric} deviates from expected {x_analytic} "
        f"within absolute tolerance of {atol}."
    )
    assert np.allclose(f_numeric, f_analytic, atol=atol), (
        f"Optimal fitness mismatch for {library}/{method}: "
        f"calculated fitness {f_numeric} deviates from expected {f_analytic} "
        f"within absolute tolerance of {atol}."
    )


# ---------------------------------------------------------------------------- #
# Test the maximum number of iterations setting
# ---------------------------------------------------------------------------- #
SOLVERS_AND_MAXITER = [
    ("scipy", "slsqp", 10),
    ("scipy", "slsqp", 20),
    ("pygmo", "ipopt", 10),
    ("pygmo", "ipopt", 20),
    # ("pygmo", "snopt", 10),
    # ("pygmo", "snopt", 20),
]


@pytest.mark.parametrize("library,method,max_iter", SOLVERS_AND_MAXITER)
def test_max_iter(library, method, max_iter):
    # Set up problem and solver
    x0 = 1.5 * np.array([1, 1, 1, 1])
    extra_options = {"tol": 1e-6} if method == "ipopt" else {}
    problem = psv.RosenbrockProblem(len(x0))
    solver = psv.OptimizationSolver(
        problem,
        library=library,
        method=method,
        print_convergence=False,
        plot_convergence=False,
        max_iterations=max_iter,
        tolerance=1e-16,
        update_on="gradient",
        logger=logger,
        extra_options=extra_options,
    )

    # Perform the optimization
    solver.solve(x0)

    # Check if iterations are within the expected range
    iter_margin = 4  # For some reason the solvers do not end exactly when they should
    assert max_iter - iter_margin <= solver.grad_count <= max_iter + iter_margin, (
        f"Solver {library}/{method} stopped at {solver.grad_count} iterations, which is not within "
        f"±2 iterations of the max limit {max_iter}."
    )


if __name__ == "__main__":

    # Run only the 'test_lorentz_system' function
    # pytest.main([__file__, "-vv", "-k test_rosenbrock_problem_unconstrained"])

    # Running pytest from Python
    pytest.main([__file__])
    # pytest.main([__file__, "-v"])
    # pytest.main([__file__, "-vv"])
