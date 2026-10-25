"""Optional nonfree PyGMO plugin and SNOPT shared-library handling."""

import os
import copy
import platform
import ctypes
from .pygmo import pg, _population_result, _set_pygmo_options


def _minimize_pygmo_snopt(problem, x0, options):
    """Solve optimization problem using Pygmo's wrapper to SNOPT"""

    try:
        import pygmo_plugins_nonfree as ppnf
    except ImportError as exc:
        raise ImportError(
            "SNOPT requires pygmo_plugins_nonfree; install it via conda-forge."
        ) from exc

    # Define default solver options
    default_options = {
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1.0,  # 1.0
        "Minor feasibility tolerance": 1e-6,
        "Iterations limit": 500,
        "Major iterations limit": 1e6,
        "Minor iterations limit": 1e6,
        # "Hessian": "full memory", # TODO check this option
        "Hessian updates": 10,
    }

    # Work with a copy to avoid side effects
    options = options.copy()

    # Define SNOPT tolerance settings based on generic input
    tol_value = options.pop("tolerance")
    options["Major feasibility tolerance"] = tol_value
    options["Minor feasibility tolerance"] = tol_value
    # SNOPT will not converge if optimality tolerance is too tight
    # This parameter is considered an "expert setting" not controlled by the generic tolerance input
    # snopt_options["Major optimality tolerance"] = tol_value

    # Define SNOPT iteration limit settings based on generic input
    max_iter = int(options.pop("max_iterations"))
    options["Iterations limit"] = max_iter
    options["Major iterations limit"] = max_iter
    options["Minor iterations limit"] = 10 * max_iter

    # Combine default and given options
    options = default_options | options

    # Define SNOPT path
    lib = os.getenv("SNOPT_LIB")
    if not lib:
        raise ValueError(
            "SNOPT library path not set in environment variables. "
            "Please set the 'SNOPT_LIB' environment variable to the path of the SNOPT library."
        )

    # Try to load the shared library explicitly
    try:
        if platform.system() == "Windows":
            ctypes.WinDLL(lib)
        else:
            ctypes.CDLL(lib)
        print(f"Successfully loaded SNOPT library: {lib}")
    except OSError as e:
        raise RuntimeError(
            f"Failed to load SNOPT shared library at {lib}. "
            f"This may indicate a license issue or incorrect path.\n{e}"
        )

    # Solve the problem
    problem = copy.deepcopy(problem)
    problem = pg.problem(problem)
    algorithm = pg.algorithm(ppnf.snopt7(library=lib, minor_version=7))
    algorithm_handle = algorithm.extract(ppnf.snopt7)
    _set_pygmo_options(algorithm_handle, options)
    population = pg.population(problem)
    population.push_back(x0)
    population = algorithm.evolve(population)

    # Optimization output
    native_info = algorithm.get_extra_info()
    message = _extract_snopt_message(native_info)
    success = (
        True
        if message == "Finished successfully - optimality conditions satisfied"
        else False
    )

    return _population_result(
        population, success=success, message=message, native_info=native_info
    )


def _extract_snopt_message(info_str):
    # Search for the start of the relevant message
    start = info_str.find("Last optimisation return code: ")
    if start == -1:
        return "Optimization return code not found."

    # Adjust the start position to the actual start of the message
    start += len("Last optimisation return code: ")

    # Find the end of the line
    end = info_str.find("\n", start)
    if end == -1:
        end = len(info_str)

    # Extract and return the message
    return info_str[start:end].strip()
