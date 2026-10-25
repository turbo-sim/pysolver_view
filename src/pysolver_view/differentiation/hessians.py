"""Existing Hessian utilities; optional JAX loads on demand."""

import numpy as np
from ._jax import load_jax


def approx_jacobian_hessians(f, x, abs_step=1e-5, lower_triangular=True):
    """
    Calculate the Hessian matrices for each component of a vector-valued function using finite differences.

    Parameters
    ----------
    f : callable
        The function for which to find the Hessian matrices. It must take a
        single argument which is a numpy array and can return a scalar or a numpy array.
    x : numpy array
        The point at which the Hessian matrices are calculated.
    abs_step : float, optional
        The step size for the finite differences, default is 1e-5.
    lower_triangular : bool, optional
        If True, the Hessians are returned in a lower triangular form suitable for Pygmo, default is True.

    Returns
    -------
    Hessians : numpy array
        A tensor where each slice along the first dimension corresponds to the Hessian matrix
        of each component of the function f at x. If f returns a scalar, the first dimension size is 1.
        If lower_triangular is True, the Hessians are returned in a lower triangular form.
    """
    x = np.asarray(x)
    f0 = np.atleast_1d(f(x))
    m = len(f0)
    n = len(x)
    Hessians = np.zeros((m, n, n))

    hh = np.eye(n) * abs_step

    for i in range(n):
        for j in range(i, n):
            f_ij = np.atleast_1d(f(x + hh[i] + hh[j]))
            f_i = np.atleast_1d(f(x + hh[i]))
            f_j = np.atleast_1d(f(x + hh[j]))

            Hessians[:, i, j] = (f_ij - f_i - f_j + f0) / abs_step**2
            if i != j:  # Symmetry
                Hessians[:, j, i] = Hessians[:, i, j]

    # Reformat to lower-triangular form suitable for Pygmo
    if lower_triangular:
        k = (n * (n + 1)) // 2
        Hessians_L = np.zeros((m, k))
        for i in range(m):
            Hessians_L[i] = Hessians[i][np.tril_indices(n)]
        Hessians = Hessians_L

    return Hessians


def compute_hessians_jax(fitness_function, x, lower_triangular=True):
    """
    Compute the Hessians of a vector-valued fitness function.

    Parameters
    ----------
    fitness_function : callable
        The vector-valued function for which to compute the Hessians.
    x : array-like
        The point at which to compute the Hessians.
    lower_triangular : bool, optional
        If True, return the Hessians in lower triangular format. Default is True.

    Returns
    -------
    Hessians : ndarray
        A tensor where each slice along the first dimension corresponds to the Hessian
        of a component of the fitness function. If lower_triangular is True, the Hessians
        are returned in a flattened lower triangular format.
    """
    jax, _ = load_jax()

    # Compute full Hessians with JAX
    H_full = jax.jacfwd(jax.jacrev(fitness_function))(x)

    # Convert to NumPy for post-processing
    H_full = np.array(H_full)

    if lower_triangular:
        # Extract the lower triangular part
        n = H_full.shape[1]
        k = (n * (n + 1)) // 2  # Number of elements in the lower triangular part
        H_lower_triangular = np.zeros((H_full.shape[0], k))

        # Loop over all components of the fitness
        for i in range(H_full.shape[0]):
            H_lower_triangular[i] = H_full[i][np.tril_indices(n)]

        return H_lower_triangular

    return H_full
