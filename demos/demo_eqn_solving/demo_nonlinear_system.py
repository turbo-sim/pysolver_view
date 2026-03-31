# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pysolver_view as psv

# Define the problem class
class LorentzEquations(psv.OptimizationProblem):
    r"""Implementation of the Lorentz system of nonlinear equations."""

    def __init__(
        self,
        sigma=1.0,
        beta=2.0,
        rho=3.0,
    ):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def residual(self, vars):
        x, y, z = vars
        eq1 = self.sigma * (y - x)
        eq2 = x * (self.rho - z) - y
        eq3 = x * y - self.beta * z
        return np.array([eq1, eq2, eq3], dtype=float)


    def fitness(self, x):
        """
        Return the packed optimization vector:
        [objective, equality_constraints..., inequality_constraints...]

        Here the problem is treated as:
            minimize   0
            subject to residual(x) = 0
        """
        f = 0.0
        c_eq = self.residual(x)
        c_ineq = None
        return psv.combine_objective_and_constraints(f, c_eq, c_ineq)

    def get_bounds(self):
        """
        Return lower and upper bounds for [x, y, z].
        """
        lb = [-10.0, -10.0, -10.0]
        ub = [10.0, 10.0, 10.0]
        return (lb, ub)

    def get_nec(self):
        """
        Number of equality constraints.
        """
        return 3

    def get_nic(self):
        """
        Number of inequality constraints.
        """
        return 0
    


# Set options for publication-quality figures
psv.set_plot_options(grid=False)

# Set up logger with unique date-time name
logger = psv.create_logger("convergence_history", use_datetime=True)

# Solve the problem as a non-linear system
x0 = np.asarray([1.0, -3.0, 2.0])
problem = LorentzEquations()
solver = psv.NonlinearSystemSolver(
    problem,
    method="hybr",
    print_convergence=True,
    plot_convergence=True,
    logger=None,
)
solver.solve(x0)


# Solve the problem as a optimization problem
solver = psv.OptimizationSolver(
    problem,
    library="scipy",
    method="slsqp",
    max_iterations=100,
    tolerance=1e-6,
    print_convergence=True,
    plot_convergence=True,
    logger=logger,
    problem_scale=10,
    update_on="gradient",
    plot_scale_objective="linear",
    plot_scale_constraints="log",
)
solver.solve(x0)

# Keep plots open
plt.show()


