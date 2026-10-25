"""Optional PyGMO adapter; population results retain native ownership."""

import copy
import numpy as np
from scipy.optimize import OptimizeResult
from .registry import PYGMO_SOLVERS

try:
    import pygmo as pg
except ImportError:
    pg = None
PYGMO_AVAILABLE = pg is not None


def _population_result(population, *, success, message, status=None, native_info=None):
    """Copy the evolved champion into a result without reevaluating the model.

    Unknown convergence is represented by ``success=None``. Native iteration
    or evaluation counts are omitted when the wrapper does not expose them;
    evaluation observations must not be promoted into inferred iterations.
    """
    fitness = np.asarray(population.champion_f).copy()
    return OptimizeResult(
        x=np.asarray(population.champion_x).copy(),
        fun=float(fitness[0]),
        fitness=fitness,
        success=success,
        message=message,
        status=status,
        native_info=native_info,
    )


def minimize_pygmo(problem, x0, method, options):
    """
    Optimize a given problem using the specified Pygmo optimization method.

    Parameters
    ----------
    problem : OptimizationProblem
        An object representing the problem.
    x0 : array_like
        Initial guess for the decision variables.
    method : str
        The name of the Pygmo solver to use. Must be one of the solvers defined in the PYGMO_SOLVERS list.
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

    # Check if pygmo is available
    if not PYGMO_AVAILABLE:
        raise ImportError(
            f"To use the selected optimizer ({method}), you need to install 'pygmo' via Conda.\n\n"
            "   1. Activate the Conda virtual environment with:\n\n"
            "           conda activate <your environment name>\n\n"
            "   2. Install 'pygmo' by running the following command:\n\n"
            "           conda install --channel conda-forge pygmo\n"
        )

    problem = PygmoProblem(problem)
    if method == "snopt":
        from .pygmo_nonfree import _minimize_pygmo_snopt

        return _minimize_pygmo_snopt(problem, x0, options)
    elif method == "ipopt":
        return _minimize_pygmo_ipopt(problem, x0, options)
    # elif method == "worhp":
    #     return _minimize_pygmo_worhp(problem, x0, options)
    elif method == "de":
        return _minimize_pygmo_de(problem, x0, options)
    elif method == "sga":
        return _minimize_pygmo_sga(problem, x0, options)
    elif method == "pso":
        return _minimize_pygmo_pso(problem, x0, options)
    else:
        error_message = (
            f"Invalid solver: '{method}'. \nAvailable options:\n   - "
            + "\n   - ".join(PYGMO_SOLVERS)
            + "."
        )
        raise ValueError(error_message)


def _minimize_pygmo_de(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to de"""

    # Define solver options
    default_options = {
        "gen": 10,
        "pop_size": 10,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options

    # Solve the problem
    algorithm = pg.algorithm(pg.de(gen=int(combined_options["gen"])))
    algorithm.set_verbosity(50)
    problem = pg.problem(problem)
    population = pg.population(problem, int(combined_options["pop_size"]))
    population = algorithm.evolve(population)

    # print solution
    print("Best solution found:", population.champion_x)
    print("Objective value of the best solution:", population.champion_f)

    return _population_result(
        population,
        success=None,
        message="Search budget completed; convergence has not been independently assessed.",
    )


def _minimize_pygmo_sga(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to sga"""

    # Define solver options
    default_options = {
        "gen": 10,
        "pop_size": 10,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options

    # Set seed
    if "seed" in list(combined_options.keys()):
        pg.set_global_rng_seed(combined_options["seed"])

    # Solve the problem
    algorithm = pg.algorithm(pg.sga(gen=int(combined_options["gen"])))
    algorithm.set_verbosity(0)
    problem = pg.problem(problem)
    population = pg.population(problem, int(combined_options["pop_size"]))
    population = algorithm.evolve(population)

    # # print solution
    # print("Best solution found:", population.champion_x)
    # print("Objective value of the best solution:", population.champion_f)

    return _population_result(
        population,
        success=None,
        message="Search budget completed; convergence has not been independently assessed.",
    )


def _minimize_pygmo_pso(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to PSO"""

    # Define solver options
    default_options = {
        "gen": 10,
        "pop_size": 10,
    }
    options = options.copy()  # Work with a copy to avoid side effects
    combined_options = default_options | options

    # Set seed
    if "seed" in list(combined_options.keys()):
        pg.set_global_rng_seed(combined_options["seed"])

    # Solve the problem
    algorithm = pg.algorithm(pg.pso(gen=int(combined_options["gen"])))
    algorithm.set_verbosity(0)
    problem = pg.problem(problem)
    population = pg.population(problem, int(combined_options["pop_size"]))
    population = algorithm.evolve(population)

    # # print solution
    # print("Best solution found:", population.champion_x)
    # print("Objective value of the best solution:", population.champion_f)

    return _population_result(
        population,
        success=None,
        message="Search budget completed; convergence has not been independently assessed.",
    )


def _minimize_pygmo_ipopt(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to IPOPT"""

    # Define mapping from exit flag to status
    exitflag_mapping = {
        0: "Solve Succeeded",
        1: "Solved To Acceptable Level",
        2: "Infeasible Problem Detected",
        3: "Search Direction Becomes Too Small",
        4: "Diverging Iterates",
        5: "User Requested Stop",
        6: "Feasible Point Found",
        -1: "Maximum Iterations Exceeded",
        -2: "Restoration Failed",
        -3: "Error In Step Computation",
        -4: "Maximum CpuTime Exceeded",
        -5: "Maximum WallTime Exceeded",
        -10: "Not Enough Degrees Of Freedom",
        -11: "Invalid Problem Definition",
        -12: "Invalid Option",
        -13: "Invalid Number Detected",
        -100: "Unrecoverable Exception",
        -101: "NonIpopt Exception Thrown",
        -102: "Insufficient Memory",
        -199: "Internal Error",
    }

    # Define default solver options
    default_options = {
        "print_level": 0,  # No output to the screen
        "sb": "yes",  # Suppress banner
        "hessian_approximation": "limited-memory",  # Options: 'exact', 'limited-memory'
        "limited_memory_update_type": "bfgs",  # Options: 'bfgs', 'sr1'
        "line_search_method": "filter",  # Options: 'filter', 'cg-penalty', 'penalty'
        "limited_memory_max_history": 30,  # Max history for L-BFGS
        "limited_memory_max_skipping": 100,  # TODO Not reseting improves convergence in some cases I testd
        "max_iter": 500,  # Maximum number of iterations.
        "tol": 1e-3,  # Desired convergence tolerance (relative). # TODO Can this be tightened when Hessian is not reseted?
        "dual_inf_tol": 1e6,  # Desired threshold for the dual infeasibility.
        "compl_inf_tol": 1e6,  # Desired threshold for the complementarity conditions
        "constr_viol_tol": 1e-6,  # Desired threshold for the constraint and variable bound violation.
        # "acceptable_iter": 10,  # Number of "acceptable" iterates before triggering termination. If it is set to zero, this heuristic is disabled.
        # "acceptable_tol": 1.0,
        # acceptable_dual_inf_tol:
        # acceptable_constr_viol_tol:
        # acceptable_compl_inf_tol:
        # "acceptable_obj_change_tol": 1e-2,
    }

    # Define IPOPT tolerance settings based on generic input
    options = options.copy()  # Work with a copy to avoid side effects
    tol_value = options.pop("tolerance")
    options["constr_viol_tol"] = tol_value
    # IPOPT will not converge if "tol" is too tight
    # This parameter is considered an "expert setting" not controlled by the generic tolerance input

    # Define IPOPT iteration limit settings based on generic input
    max_iter = int(options.pop("max_iterations"))
    options["max_iter"] = max_iter

    # Combine default and given options
    options = default_options | options

    # Solve the problem
    problem = copy.deepcopy(problem)
    problem = pg.problem(problem)
    algorithm = pg.algorithm(pg.ipopt())
    algorithm_handle = algorithm.extract(pg.ipopt)
    _set_pygmo_options(algorithm_handle, options)
    population = pg.population(problem)
    population.push_back(x0)
    population = algorithm.evolve(population)

    # Optimization output
    exitflag = algorithm_handle.get_last_opt_result()
    message = exitflag_mapping.get(exitflag, "Unknown Status")
    success = exitflag in (0, 1)

    return _population_result(
        population, success=success, message=message, status=exitflag
    )


def _set_pygmo_options(handle, options):
    """
    Sets multiple options on a PyGMO solver handle based on the types of values provided in a dictionary.

    This function supports various PyGMO solvers including IPOPT, WORHP, and SNOPT. It dynamically calls
    the appropriate method to set each option (string, integer, or numeric) based on the type of each value
    in the `options` dictionary.

    Parameters
    ----------
    handle : object
        The PyGMO solver handle. This object should have methods for setting options,
        such as `set_string_option`, `set_integer_option`, and `set_numeric_option`.
    options : dict
        A dictionary where each key is the option name and each value is the option value.
        The function determines the method to use based on the type of the value:
        `str` for `set_string_option`, `int` for `set_integer_option`, and `float` for `set_numeric_option`.

    Raises
    ------
    ValueError
        If the value type is not supported (not an int, float, or str).

    Examples
    --------
    >>> algorithm = pg.algorithm(pg.ipopt())
    >>> handle = algorithm.extract(pg.ipopt)
    >>> options = {
    >>>     "sb": "yes",
    >>>     "print_level": 0,
    >>>     "tol": 1e-6,
    >>>     "max_iter": 1000
    >>> }
    >>> set_pygmo_options(handle, options)
    """
    for option_name, value in options.items():
        if isinstance(value, str):
            handle.set_string_option(option_name, value)
        elif isinstance(value, (int, np.int64)):
            handle.set_integer_option(option_name, value)
        elif isinstance(value, float):
            handle.set_numeric_option(option_name, value)
        else:
            raise ValueError(
                "Unsupported option type: {} for option {}".format(
                    type(value), option_name
                )
            )


class PygmoProblem:
    """Keep callbacks shallow through deepcopy; flatten only PyGMO derivatives."""

    def __init__(self, context):
        self.fitness = lambda x: context.fitness(x)
        self.gradient = lambda x: context.gradient(x).flatten()
        self.get_bounds = lambda: context.get_bounds()
        self.get_nec = lambda: context.get_nec()
        self.get_nic = lambda: context.get_nic()
        if hasattr(context, "hessians"):
            self.hessians = lambda x: context.hessians(x)
