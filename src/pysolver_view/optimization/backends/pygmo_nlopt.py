"""NLOpt through PyGMO using the common frontend option names."""

from .pygmo import pg, PygmoProblem, _population_result
from .registry import NLOPT_SOLVERS


def minimize_nlopt(problem, x0, method, options):
    """
    Optimize a given problem using the specified NLOpt optimization method.

    Parameters
    ----------
    problem : OptimizationProblem
        An object representing the problem.
    x0 : array_like
        Initial guess for the decision variables.
    method : str
        The name of the Pygmo solver to use. Must be one of the solvers defined in the NLOPT_SOLVERS list.
    options : dict
        Configuration options for the solver. Options vary depending on the solver used.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The backend solution, including its authoritative coordinates, objective,
        termination status, and any available native diagnostics. Coordinates
        use the solver-space convention supplied by the caller.

    Raises
    ------
    ValueError
        If an invalid solver name is specified.
    """

    if pg is None:
        raise ImportError(
            "NLOpt through PyGMO requires pygmo; install it via conda-forge."
        )
    problem = PygmoProblem(problem)

    # Check if method is valid
    if method not in NLOPT_SOLVERS:
        error_message = (
            f"Invalid solver: '{method}'. \nAvailable options:\n   - "
            + "\n   - ".join(NLOPT_SOLVERS)
            + "."
        )
        raise ValueError(error_message)

    # Define exit flag mapping
    exitflag_mapping = {
        1: "Generic success return value.",
        2: "Optimization stopped because stop value was reached.",
        3: "Optimization stopped because relative or absolute tolerance on function value was reached.",
        4: "Optimization stopped because relative or absolute tolerance on x was reached.",
        5: "Optimization stopped because maximum evaluations limit was reached.",
        6: "Optimization stopped because maximum time limit was reached.",
        -1: "Generic failure code.",
        -2: "Invalid arguments (e.g., lower bounds are bigger than upper bounds, unknown algorithm specified, etc.).",
        -3: "Ran out of memory.",
        -4: "Halted due to roundoff errors limiting progress. Optimization might still return a useful result.",
        -5: "Optimization was forcibly stopped.",
    }

    # Define solver options
    options = options.copy()  # Work with a copy to avoid side effects
    tol = options.pop("tolerance") if "tolerance" in options else options.pop("tol")
    max_iter = (
        options.pop("max_iterations")
        if "max_iterations" in options
        else options.pop("max_iter")
    )
    algorithm = pg.algorithm(pg.nlopt(solver=method))
    handle = algorithm.extract(pg.nlopt)
    handle.maxeval = max_iter
    handle.ftol_rel = tol
    handle.xtol_rel = tol

    # Solve the problem
    problem = pg.problem(problem)
    population = pg.population(problem, size=1)
    population.set_x(0, x0)
    population = algorithm.evolve(population)

    # Optimization output
    exitflag = handle.get_last_opt_result()
    message = exitflag_mapping.get(exitflag, "Unknown Status")
    success = exitflag in (1, 2, 3, 4)

    return _population_result(
        population, success=success, message=message, status=exitflag
    )
