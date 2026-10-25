"""Numerical Jacobian equilibration independent of physical displacement limits."""

import numpy as np


class JacobianScaling:
    """Equilibrate columns in caller-scaled coordinates using fresh Jacobians.

    For A = J * x_scale, numerical coordinates use A * factors. The reference
    column norm is the initial median nonzero norm. Subsequent fresh Jacobians
    update running maximum column norms, so an established factor never grows.
    Zero columns retain factor one until a nonzero derivative is observed.

    The geometric median of the middle norms is used for even column counts.
    Logarithms avoid overflow/underflow while measuring column norms. Factors
    are limited to [sqrt(eps), 1/sqrt(eps)] to avoid extreme coordinate ratios.
    This is numerical equilibration, not residual weighting or a physical cap.
    """

    def __init__(self, size, enabled):
        """Start with identity scaling, including before any Jacobian is needed."""
        self.enabled = enabled
        self.factors = np.ones(size)
        self._largest_log_norm = np.full(size, -np.inf)
        self._reference_log_norm = None
        self.updates = 0

    def update(self, physical_jacobian):
        """Update numerical factors from a finite Jacobian without model evaluations."""
        if not self.enabled:
            return self.factors
        peaks = np.max(np.abs(physical_jacobian), axis=0)
        nonzero = peaks > 0
        if not np.any(nonzero):
            return self.factors
        normalized = physical_jacobian[:, nonzero] / peaks[nonzero]
        log_norms = np.full(peaks.size, -np.inf)
        log_norms[nonzero] = np.log(peaks[nonzero]) + 0.5 * np.log(
            np.sum(normalized * normalized, axis=0)
        )
        if self._reference_log_norm is None:
            self._reference_log_norm = float(np.median(log_norms[nonzero]))
        self._largest_log_norm = np.maximum(self._largest_log_norm, log_norms)
        known = np.isfinite(self._largest_log_norm)
        log_limit = -0.5 * np.log(np.finfo(float).eps)
        self.factors[known] = np.exp(
            np.clip(
                self._reference_log_norm - self._largest_log_norm[known],
                -log_limit,
                log_limit,
            )
        )
        self.updates += 1
        return self.factors

    def diagnostics(self):
        """Return detached final scaling data for reproducible result inspection."""
        return dict(
            jacobian_scaling=self.enabled,
            jacobian_scale=self.factors.copy(),
            jacobian_scale_updates=self.updates,
        )
