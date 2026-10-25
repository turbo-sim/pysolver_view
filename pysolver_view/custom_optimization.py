"""Adapt package problem conventions to the controlled SLSQP driver."""

import numpy as np


def minimize_pysolver(problem, x0, method, solver_options):
    """Run controlled SLSQP with one owner for model and derivative accounting.

    Package inequalities are nonpositive; the numerical driver uses nonnegative
    slacks. Conversion happens here for both values and Jacobian rows. Driver
    KKT diagnostics and histories retain the driver's convention in
    ``backend_result``; its final ``fitness`` uses package signs. The facade
    converts KKT diagnostics to public signs in ``result`` and iteration history.
    """
    from .solvers.slsqp import Evaluation, SLSQPOptions, minimize_slsqp

    if method != "slsqp":
        raise ValueError("The pysolver optimization backend currently supports slsqp.")
    configured = dict(solver_options)
    tolerance = configured.pop("tolerance")
    max_iterations = configured.pop("max_iterations")
    options = {
        "ftol": tolerance,
        "maxiter": max_iterations,
        "feasibility_tol": tolerance,
        "optimality_tol": tolerance,
        **configured,
    }
    options = SLSQPOptions(**options)
    bounds = problem.get_bounds()
    bounds = None if bounds is None else np.column_stack(bounds)
    equality_count = problem.get_nec()

    def evaluate(x):
        """Convert one combined package evaluation to the driver's signs."""
        values = problem.fitness(x)
        return Evaluation(
            values[0], values[1 : 1 + equality_count], -values[1 + equality_count :]
        )

    def jacobian(x):
        """Convert rows from the facade's analytic or bounded numerical Jacobian."""
        matrix = problem.gradient(x).reshape(-1, x.size).copy()
        matrix[1 + equality_count :] *= -1
        return matrix

    result = minimize_slsqp(
        evaluate,
        x0,
        bounds,
        jac=jacobian,
        options=options,
        callback=problem.accepted_callback,
    )
    result.effective_options = vars(options).copy()
    if result.fitness is not None:
        result.fitness = result.fitness.copy()
        result.fitness[1 + equality_count :] *= -1
    return result
