"""Coordinate transformations used by the legacy problem convenience methods."""

import numpy as np


def get_bounds_normalized(self):
    """
    Return normalized bounds in [0, problem_scale]. If no scaling is applied (i.e., problem_scale is None),
    the physical bounds are returned instead.

    Returns
    -------
    tuple of lists
        Tuple (lb, ub) in normalized space or the original physical bounds if problem_scale is None.
    """
    if self.problem_scale is None:
        return self.get_bounds()

    n = len(self.get_bounds()[0])
    return [0.0] * n, [self.problem_scale] * n


def scale_physical_to_normalized(self, x_phys):
    """
    Convert physical design variable values to normalized values in the range [0, problem_scale].
    If self.problem_scale is None, no scaling is applied.

    The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.
    It automatically handles the case of fixed variables (i.e., upper bound = lower bound) by returning 0.0.

    Parameters
    ----------
    x_phys : array-like
        Physical values of the decision variables.

    Returns
    -------
    np.ndarray
        Normalized values in the range [0, problem_scale] if scaling is applied,
        otherwise the original physical values.
    """
    x_phys = np.asarray(x_phys)
    if self.problem_scale is None:
        return x_phys
    lb, ub = self.get_bounds()
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    x_scaled = []
    for xi, lbi, ubi in zip(x_phys, lb, ub):
        if ubi == lbi:
            x_scaled.append(0.0)
        else:
            x_scaled.append(self.problem_scale * (xi - lbi) / (ubi - lbi))

    return np.array(x_scaled)


def scale_normalized_to_physical(self, x_norm):
    """
    Convert normalized values in the range [0, problem_scale] back to physical variable values.
    If self.problem_scale is None, no scaling is applied.

    The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.

    Parameters
    ----------
    x_norm : array-like
        Normalized values of the decision variables.

    Returns
    -------
    np.ndarray
        Physical values corresponding to the normalized input, or the original values if no scaling.
    """
    x_norm = np.asarray(x_norm)
    if self.problem_scale is None:
        return x_norm
    lb, ub = self.get_bounds()
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    return lb + (ub - lb) * (x_norm / self.problem_scale)


def fitness_normalized_input(self, x_norm):
    """
    Evaluate fitness starting from normalized input.

    Parameters
    ----------
    x_norm : array-like
        Normalized input vector.

    Returns
    -------
    array_like
        Output of `fitness` evaluated on the corresponding physical vector.
    """
    x_phys = self.scale_normalized_to_physical(x_norm)
    return self.fitness(x_phys)


def gradient_normalized_input(self, x_norm):
    """
    Compute the gradient of the objective and constraints with respect to normalized variables.

    This function applies the chain rule to convert the gradient computed in the physical space
    to the corresponding gradient in the normalized space. If `problem_scale` is set to `None`,
    the problem is considered unscaled and the gradient is returned directly.

    Parameters
    ----------
    x_norm : array-like
        Normalized input vector (typically in [0, problem_scale]).

    Returns
    -------
    np.ndarray
        Gradient with respect to the normalized variables. The shape is:
        - (n,) for scalar objective or flat constraint vectors.
        - (m, n) for vector-valued constraints with m rows and n design variables.

    Notes
    -----
    The chain rule is applied as follows:

        x_phys = lb + (ub - lb) * (x_norm / problem_scale)
        ∇f(x_norm) = ∇f(x_phys) * d(x_phys)/d(x_norm)

    where:

        d(x_phys)/d(x_norm) = (ub - lb) / problem_scale    [elementwise]

    This correction ensures that exact user-defined gradients are consistent
    with the scaling applied during optimization.
    """
    x_phys = self.scale_normalized_to_physical(x_norm)
    grad_phys = self.gradient(x_phys)

    if self.problem_scale is None:
        return grad_phys

    lb, ub = self.get_bounds()
    lb, ub = np.asarray(lb), np.asarray(ub)
    scaling_factor = (ub - lb) / self.problem_scale

    # Apply scaling using broadcasting to support both 1D and 2D gradients
    return grad_phys * scaling_factor


def clip_to_bounds(self, x_physical, logger=None):
    """
    Clip physical variable values to lie within specified bounds.

    Parameters
    ----------
    x_physical : array-like
        Input vector in physical space.
    logger : logging.Logger or None, optional
        Logger for outputting warnings. If None, warnings are printed to standard output.

    Returns
    -------
    np.ndarray
        Clipped vector.
    """
    lb, ub = self.get_bounds()
    x_physical = np.atleast_1d(x_physical)
    x_clipped = np.clip(x_physical, lb, ub)

    for i, (orig, new, low, high) in enumerate(zip(x_physical, x_clipped, lb, ub)):
        if orig != new:
            msg = (
                "-" * 80 + "\n"
                f"Warning: optimization variable out of bounds\n"
                f"  Name       : {self.variable_names[i]}\n"
                f"  Original   : {orig:.3e}\n"
                f"  Bounds     : [{low:.3e}, {high:.3e}]\n"
                f"  Clipped to : {new:.3e}\n" + "-" * 80
            )
            if logger:
                for line in msg.splitlines():
                    logger.warning(line)
            else:
                print(msg)

    return x_clipped
