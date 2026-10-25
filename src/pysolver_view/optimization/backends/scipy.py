"""Native SciPy optimization adapter."""

import numpy as np
from scipy.optimize import minimize


def convert_pygmo_to_scipy_bounds(pygmo_bounds):
    """
    Create a function that converts bounds from Pygmo format to SciPy format.

    Parameters
    ----------
    pygmo_method : callable
        A method that, when called, returns a tuple containing two lists: the first list of lower bounds and the second of upper bounds (Pygmo convention).

    Returns
    -------
    callable
        A function that, when called, returns a list of tuples where each tuple contains the lower and upper bound for each decision variable (Scipy convention).

    """

    def converter():
        lower_bounds, upper_bounds = pygmo_bounds()
        scipy_bounds = list(zip(lower_bounds, upper_bounds))
        return scipy_bounds

    return converter


def minimize_scipy(problem, x0, method, solver_options):
    """
    Optimize a given problem using the specified SciPy method.

    Parameters
    ----------
    problem : OptimizationProblem
        An object representing the problem.
    x0 : array_like
        Initial guess for the decision variables.
    method : str
        The name of the Scipy solver to use. Must be one of the solvers defined in the SCIPY_SOLVERS list.
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
        If an invalid solver is specified.
    """

    from .registry import SCIPY_SOLVERS

    # Check if method is valid
    if method not in SCIPY_SOLVERS:
        error_message = (
            f"Invalid solver: '{method}'. \nAvailable options:\n   - "
            + "\n   - ".join(SCIPY_SOLVERS)
            + "."
        )
        raise ValueError(error_message)

    # Define the options dictionary
    solver_options = solver_options.copy()  # Work with a copy to avoid side effects
    tol = solver_options.pop("tolerance")
    max_iter = solver_options.pop("max_iterations")
    default_options = {
        "nelder-mead": {"maxiter": max_iter},
        "bfgs": {"gtol": tol, "maxiter": max_iter},
        "l-bfgs-b": {"ftol": tol, "gtol": tol, "maxiter": max_iter},
        "slsqp": {"ftol": tol, "maxiter": max_iter},
        "trust-constr": {"gtol": tol, "xtol": tol, "maxiter": max_iter},
    }
    default_options = (
        default_options[method] if method in default_options.keys() else {}
    )
    solver_options = solver_options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | solver_options  # Merge defaults with input

    # Convert bounds from Pygmo to Scipy convention
    if problem.get_bounds() is None:
        bounds = None
    else:
        bounds = convert_pygmo_to_scipy_bounds(problem.get_bounds)()
        if not np.any(np.isfinite(np.asarray(bounds, dtype=float))):
            bounds = None

    # Define list of constraint dictionaries
    constr = []
    if problem.get_nec() > 0:
        constr.append({"type": "eq", "fun": problem.c_eq, "jac": problem.c_eq_jac})
    if problem.get_nic() > 0:
        # Convert inequality constraint convention from Pygmo (c<0) to SciPy (c>0)
        c_ineq = lambda x: -problem.c_ineq(x)
        c_ineq_jac = lambda x: -problem.c_ineq_jac(x)
        constr.append({"type": "ineq", "fun": c_ineq, "jac": c_ineq_jac})

    # Only SLSQP iteration events are adapted here. Its callback indicates a
    # native major iteration, not necessarily a line-search-accepted point.
    iteration_callback = (
        getattr(problem, "iteration_callback", None) if method == "slsqp" else None
    )

    # Preserve the returned point and diagnostics, even if a later evaluation
    # or reporting callback observed a different point.
    solution = minimize(
        problem.f,
        x0,
        tol=tol,
        jac=None if method in {"nelder-mead", "powell", "cobyla"} else problem.f_jac,
        constraints=constr,
        bounds=bounds,
        method=method,
        options=combined_options,
        callback=iteration_callback,
    )

    return solution
