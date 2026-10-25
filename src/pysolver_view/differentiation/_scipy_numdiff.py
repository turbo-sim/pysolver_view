"""Compatibility boundary for the existing private SciPy differentiation helper."""

from scipy.optimize._numdiff import approx_derivative

__all__ = ["approx_derivative"]
