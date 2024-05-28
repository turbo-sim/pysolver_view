import pytest
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as pv


# Set up logger with unique date-time name
logger = pv.create_logger("convergence_history", use_datetime=True)

# Define solver configurations
SOLVERS = ["lm", "hybr"]

# Calculate stationary points of the Lorentz system
SIGMA, BETA, RHO = 1.0, 2.0, 3.0
STATIONARY_POINTS = [
    np.asarray([0.00, 0.00, 0.00]),
    np.asarray([+np.sqrt(BETA * (RHO - 1)), +np.sqrt(BETA * (RHO - 1)), RHO - 1]),
    np.asarray([-np.sqrt(BETA * (RHO - 1)), -np.sqrt(BETA * (RHO - 1)), RHO - 1]),
]


# Using pytest.mark.parametrize to test each solver with each solution
@pytest.mark.parametrize(
    "method, stationary_point", [(m, p) for m in SOLVERS for p in STATIONARY_POINTS]
)
def test_lorentz_problem(method, stationary_point):
    # Find a stationary point of the Lorentz equations
    problem = pv.LorentzEquations(sigma=SIGMA, beta=BETA, rho=RHO)
    solver = pv.NonlinearSystemSolver(
        problem,
        method=method,
        display=True,
        plot=False,
        max_iter=300,
        tol=1e-6,
        update_on="function",
        logger=logger,
    )

    # Solve the system
    x0 = stationary_point + np.asarray([0.5, 0.5, 0.5])
    solver.solve(x0)

    # Tolerance for tests
    atol = 1e-6

    # Numerical solution
    x_numeric = solver.x_final
    f_numeric = solver.f_final

    # Analytic solution
    x_analytic = stationary_point
    f_analytic = np.zeros(3)

    # Assertions
    assert solver.success, f"System solution with method '{method}' did not converge"
    assert np.allclose(x_numeric, x_analytic, atol=atol), (
        f"Solution mismatch for method '{method}': "
        f"calculated solution {x_numeric} deviates from expected {x_analytic} "
        f"within absolute tolerance of {atol}."
    )
    assert np.allclose(f_numeric, f_analytic, atol=atol), (
        f"Solution mismatch for method '{method}': "
        f"calculated fitness {f_numeric} deviates from expected {f_analytic} "
        f"within absolute tolerance of {atol}."
    )


if __name__ == "__main__":
    # Running pytest from Python
    pytest.main([__file__])
    # pytest.main([__file__, "-v"])
    # pytest.main([__file__, "-vv"])
