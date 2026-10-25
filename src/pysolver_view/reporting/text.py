from ..roots.methods import METHOD_DISPLAY_NAMES, normalize_method


def print_object(obj):
    """
    Prints all attributes and methods of an object, sorted alphabetically.

    - Methods are identified as callable and printed with the prefix 'Method: '.
    - Attributes are identified as non-callable and printed with the prefix 'Attribute: '.

    Parameters
    ----------
    obj : object
        The object whose attributes and methods are to be printed.

    Returns
    -------
    None
        This function does not return any value. It prints the attributes and methods of the given object.

    """
    # Retrieve all attributes and methods
    attributes = dir(obj)

    # Sort them alphabetically, case-insensitive
    sorted_attributes = sorted(attributes, key=lambda x: x.lower())

    # Iterate over sorted attributes
    for attr in sorted_attributes:
        # Retrieve the attribute or method from the object
        attribute_or_method = getattr(obj, attr)

        # Check if it is callable (method) or not (attribute)
        if callable(attribute_or_method):
            print(f"Method: {attr}")
        else:
            print(f"Attribute: {attr}")


def print_dict(data, indent=0, return_output=False):
    """
    Recursively prints nested dictionaries with indentation or returns the formatted string.

    Parameters
    ----------
    data : dict
        The dictionary to print.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
    return_output : bool, optional
        If True, returns the formatted string instead of printing it.

    Returns
    -------
    str or None
        The formatted string representation of the dictionary if return_output is True, otherwise None.
    """
    lines = []
    for key, value in data.items():
        line = "    " * indent + str(key) + ": "
        if isinstance(value, dict):
            if value:
                lines.append(line)
                lines.append(print_dict(value, indent + 1, True))
            else:
                lines.append(line + "{}")
        else:
            lines.append(line + str(value))

    output = "\n".join(lines)
    if return_output:
        return output
    else:
        print(output)


def root_progress(record):
    """Format one recorded root observation without model access."""
    return (
        f" {record['func_count']:15d}{record['grad_count']:15d}"
        f"{record['norm_residual']:24.6e}{record['norm_step']:24.6e} "
    )


def optimization_progress(record):
    """Format one recorded optimization observation without diagnostic work."""
    status = (
        f" {record['grad_count']:13d}{record['func_count']:13d}"
        f"{record['objective_value']:+16.3e}{record['constraint_violation']:+18.3e}"
        f"{record['norm_step']:+18.3e}{record['optimality']:+18.3e} "
    )
    return status + (" Final solution" if record["event_kind"] == "final" else "")


def emit(message, *, logger=None, to_console=False, savefile=False, filename=None):
    """Write prepared text to the requested destinations without model access."""
    if logger:
        for line in message.splitlines():
            logger.info(line)
    if to_console:
        print(message)
    if savefile:
        if not filename:
            raise ValueError("Filename must be provided if savefile is True.")
        with open(filename, "w") as stream:
            stream.write(message)


def optimization_header(problem_name, method):
    """Format the optimization header from supplied run data."""
    initial_message = (
        f" Starting optimization process for {problem_name}\n"
        f" Optimization algorithm employed: {method}"
    )
    header = f" {'Grad-eval':>13}{'Func-eval':>13}{'Func-value':>16}{'Infeasibility':>18}{'Norm of step':>18}{'Optimality':>18} "
    separator = "-" * len(header)
    lines_to_output = [
        separator,
        initial_message,
        separator,
        header,
        separator,
    ]

    return header, lines_to_output


def optimization_footer(header, message, success, elapsed_time, result, check_kkt):
    """Format the optimization footer from supplied run data."""
    separator = "-" * len(header)
    exit_message = f" Exit message: {message}"
    success_status = f" Success: {success}"
    time_message = f" Solution time: {elapsed_time:.3f} seconds"
    lines_to_output = [separator, success_status, exit_message, time_message]
    lines_to_output += [
        f" Final objective: {result.fun:+.6e}",
        f" Final constraint violation: {result.constraint_violation:+.6e}",
        (
            f" Final optimality (cached derivatives): {result.optimality:+.6e}"
            if check_kkt
            else " Final optimality: not assessed (KKT disabled)."
        ),
        f" Total model calls: {result.nfev}; " f"gradient evaluations: {result.njev}",
    ]
    if not check_kkt:
        lines_to_output.append(
            " Automatic KKT checks: disabled; optimality not assessed."
        )
    lines_to_output += [separator]

    return lines_to_output


def root_header(problem_name, method, library):
    """Format the root header from supplied run data."""
    method_name = METHOD_DISPLAY_NAMES[normalize_method(method)].replace("\u2013", "-")
    initial_message = (
        f" Solve system of equations for {problem_name}\n"
        f" Root-finding algorithm employed: {method_name} ({library})"
    )
    header = (
        f" {'Func-eval':>15}{'Grad-eval':>15}{'Norm of residual':>24}{'Step size':>24} "
    )
    separator = "-" * len(header)
    lines_to_output = [
        separator,
        initial_message,
        separator,
        header,
        separator,
    ]

    return header, lines_to_output


def root_footer(
    header, message, success, elapsed_time, x_final, include_solution_in_footer
):
    """Format the root footer from supplied run data."""
    separator = "-" * len(header)
    exit_message = f"Exit message: {message}"
    success_status = f"Success: {success}"
    time_message = f"Solution time: {elapsed_time:.3f} seconds"
    solution_header = "Solution:"
    solution_vars = [f"   x{i} = {x:+6e}" for i, x in enumerate(x_final)]
    lines_to_output = [separator, success_status, exit_message, time_message]
    if include_solution_in_footer:
        lines_to_output += [solution_header]
        lines_to_output += solution_vars
    lines_to_output += [separator, ""]

    return lines_to_output
