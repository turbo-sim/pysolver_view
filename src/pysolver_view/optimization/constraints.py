"""Public objective/constraint packing with nonpositive inequalities."""

import copy
import numpy as np
from ..validation import validate_keys


def count_constraints(var):
    """
    Retrieve the number of constraints based on the provided input.

    This function returns the count of constraints based on the nature of the
    input:

    - `None` returns 0
    - Scalar values return 1
    - Array-like structures return their length

    Parameters
    ----------
    var : None, scalar, or array-like (list, tuple, np.ndarray)
        The input representing the constraint(s). This can be `None`, a scalar value,
        or an array-like structure containing multiple constraints.

    Returns
    -------
    int
        The number of constraints:

        - 0 for `None`
        - 1 for scalar values
        - Length of the array-like for array-like inputs

    Examples
    --------
    >>> count_constraints(None)
    0

    >>> count_constraints(5.0)
    1

    >>> count_constraints([1.0, 2.0, 3.0])
    3
    """
    # If constraint is None
    if var is None:
        return 0
    # If constraint is a scalar (assuming it's numeric)
    elif np.isscalar(var):
        return 1
    # If constraint is array-like
    else:
        return len(var)


def combine_objective_and_constraints(f, c_eq=None, c_ineq=None):
    """
    Combine an objective function with its associated equality and inequality constraints.

    This function takes in an objective function value, a set of equality constraints,
    and a set of inequality constraints. It then returns a combined Numpy array of
    these values. The constraints can be given as a list, tuple, numpy array, or as
    individual values.

    Parameters
    ----------
    f : float
        The value of the objective function.
    c_eq : float, list, tuple, np.ndarray, or None
        The equality constraint(s). This can be a single value or a collection of values.
        If `None`, no equality constraints will be added.
    c_ineq : float, list, tuple, np.ndarray, or None
        The inequality constraint(s). This can be a single value or a collection of values.
        If `None`, no inequality constraints will be added.

    Returns
    -------
    np.ndarray
        A numpy array consisting of the objective function value followed by equality and
        inequality constraints.

    Examples
    --------
    >>> combine_objective_and_constraints(1.0, [0.5, 0.6], [0.7, 0.8])
    array([1. , 0.5, 0.6, 0.7, 0.8])

    >>> combine_objective_and_constraints(1.0, 0.5, 0.7)
    array([1. , 0.5, 0.7])
    """

    # Validate objective function value
    if isinstance(f, (list, tuple, np.ndarray)):
        if len(f) != 1:
            raise ValueError(
                "Objective function value 'f' must be a scalar or single-element array."
            )
        f = f[0]  # Unwrap the single element to ensure it's treated as a scalar

    # Add objective function
    combined_list = [f]

    # Add equality constraints
    if c_eq is not None:
        if isinstance(c_eq, (list, tuple, np.ndarray)):
            combined_list.extend(c_eq)
        else:
            combined_list.append(c_eq)

    # Add inequality constraints
    if c_ineq is not None:
        if isinstance(c_ineq, (list, tuple, np.ndarray)):
            combined_list.extend(c_ineq)
        else:
            combined_list.append(c_ineq)

    return np.array(combined_list)


def get_constraint_data(self, x_norm, tol):
    """
    Return constraint evaluation data for the current optimization problem.

    This method returns a list of dictionaries describing the satisfaction
    status of all equality and inequality constraints at a given normalized
    design point.

    If `self.problem.constraint_data` exists and is valid, it is returned
    directly after key validation.

    Parameters
    ----------
    x_norm : array_like
        Normalized design variable vector.
    tol : float
        Tolerance for constraint satisfaction.

    Returns
    -------
    list of dict
        Each dictionary contains:

        - **name** : str
          Constraint identifier.

        - **type** : str
          Either "=" for equality or "<" for inequality constraints.

        - **target** : float
          Desired target value (0.0 for all constraints).

        - **value** : float
          Actual evaluated constraint value.

        - **satisfied** : bool
          Whether the constraint is satisfied (based on `tol`).

        Optionally, each dictionary may also include:

        - **normalized_mismatch** : float
          Mismatch scaled by the normalization factor.

        - **mismatch** : float
          Raw constraint violation (value - target).

        - **normalize** : float
          Normalization factor (if defined).

    Raises
    ------
    TypeError
        If `self.problem.constraint_data` exists but is not a list of dictionaries,
        or if required keys are missing from any entry.
    """
    # Metadata must belong to the same model evaluation as the requested values.
    self._evaluate_fitness(x_norm, diagnostic=True)
    return constraint_data_from_values(
        self.cache["fitness"],
        self.N_eq,
        self.N_ineq,
        tol,
        self.cache["constraint_data"],
        has_metadata=self.cache["has_constraint_data"],
    )


def constraint_data_from_values(
    fitness, equality_count, inequality_count, tol, metadata=None, *, has_metadata=False
):
    """Validate and annotate detached physical metadata using supplied residuals.

    Fitness follows public signs: objective, equalities, then nonpositive
    inequalities. This helper does no model work and preserves physical names,
    values, and targets while computing satisfaction from normalized residuals.
    """
    fitness = np.asarray(fitness)
    if fitness.shape != (1 + equality_count + inequality_count,):
        raise ValueError("Constraint report values do not match declared counts.")
    c_eq = fitness[1 : 1 + equality_count]
    c_ineq = fitness[1 + equality_count :]
    data = copy.deepcopy(metadata)
    if has_metadata:
        if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
            raise TypeError("Expected self.constraint_data to be a list of dicts")
        required = {"name", "type", "target", "value"}
        allowed = required | {
            "satisfied",
            "normalized_mismatch",
            "mismatch",
            "normalize",
        }
        equality_index, inequality_index = 0, 0
        for entry in data:
            validate_keys(entry, required, allowed)
            if entry["type"] == "=":
                if equality_index >= equality_count:
                    raise ValueError("Too many named equality constraints.")
                mismatch = c_eq[equality_index]
                entry["satisfied"] = bool(abs(mismatch) <= tol)
                equality_index += 1
            elif entry["type"] in {"<", ">"}:
                if inequality_index >= inequality_count:
                    raise ValueError("Too many named inequality constraints.")
                mismatch = c_ineq[inequality_index]
                entry["satisfied"] = bool(mismatch <= tol)
                inequality_index += 1
            else:
                raise ValueError("Named constraint type must be '=', '<', or '>'.")
        if (equality_index, inequality_index) != (equality_count, inequality_count):
            raise ValueError("Named constraint counts must match the fitness vector.")
        return data

    data = []
    for kind, residuals in (("equality", c_eq), ("inequality", c_ineq)):
        for index, value in enumerate(residuals):
            equality = kind == "equality"
            data.append(
                dict(
                    name=f"{kind}_constraint_{index}",
                    type="=" if equality else "<",
                    target=0.0,
                    value=value,
                    satisfied=bool(abs(value) <= tol if equality else value <= tol),
                )
            )
    return data
