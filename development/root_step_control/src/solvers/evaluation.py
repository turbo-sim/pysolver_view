"""Validated model evaluations, numerical derivatives, and call accounting."""

import numpy as np
from scipy.linalg import norm

from ..exceptions import BudgetExhausted, DomainError, UnrepresentableProbe


def prepare_inputs(x0, x_scale):
    """Copy a finite initial vector and expand its fixed characteristic scales."""
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite vector.")
    scale = np.asarray(x_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full(x.size, scale.item())
    if scale.shape != x.shape:
        raise ValueError("x_scale must be scalar or have the same shape as x0.")
    return x, scale.copy()


class ModelEvaluator:
    """Count residual/Jacobian requests and record every attempted residual call.

    Model inputs and residual outputs are detached from the application. Only explicit
    DomainError and nonfinite residual values are treated as domain failures;
    application programming errors propagate. Drivers decide whether to recover.
    """

    def __init__(self, fun, jac, scale, options, recorder, *, square=False):
        """Bind the model, fixed scales, residual budget, and history recorder."""
        self.fun = fun
        self.jac = jac
        self.scale = scale
        self.max_nfev = options.max_nfev
        self.finite_diff_step = options.finite_diff_step
        self.recorder = recorder
        self.square = square
        self.nfev = self.njev = 0
        self.residual_size = self.last_norm = None

    def residual(self, point, kind):
        """Validate and count one residual call, including undefined model states."""
        if self.nfev >= self.max_nfev:
            raise BudgetExhausted
        self.nfev += 1
        record = None
        if self.recorder.enabled:
            record = dict(kind=kind, valid=False, residual_norm=None, nfev=self.nfev)
            if self.recorder.coordinates:
                record["x"] = point.copy()
            self.recorder.evaluations.append(record)
        try:
            values = np.asarray(self.fun(point.copy()), dtype=float)
            if values.ndim != 1 or values.size < self.scale.size:
                raise ValueError(
                    "fun must return a vector with at least len(x0) entries."
                )
            if self.residual_size is not None and values.size != self.residual_size:
                raise ValueError("Residual size changed during the solve.")
            if self.square and values.size != self.scale.size:
                raise ValueError(
                    "Newton–Raphson requires a square residual/Jacobian pair."
                )
            if not np.all(np.isfinite(values)):
                raise DomainError("Nonfinite residual at model evaluation.")
            residual_norm = float(norm(values, check_finite=False))
            if residual_norm > np.sqrt(np.finfo(float).max):
                raise DomainError(
                    "Residual is too large for a finite squared-norm merit."
                )
        except DomainError as error:
            if record is not None:
                record["reason"] = str(error)
            raise
        self.residual_size = values.size
        self.last_norm = residual_norm
        if record is not None:
            record.update(valid=True, residual_norm=residual_norm)
        return values.copy()

    def jacobian(self, point, residual):
        """Count one Jacobian request and obtain analytic or forward differences.

        Finite-difference probes pass through residual() and consume its budget.
        Shape errors propagate; drivers classify nonfinite derivative values.
        """
        self.njev += 1
        if self.jac is not None:
            jacobian = np.asarray(self.jac(point.copy()), dtype=float)
        else:
            jacobian = np.empty((residual.size, point.size))
            for column in range(point.size):
                probe = point.copy()
                probe[column] += self.finite_diff_step * self.scale[column]
                distance = probe[column] - point[column]
                if distance == 0 or not np.isfinite(probe[column]):
                    raise UnrepresentableProbe(
                        "Finite-difference probe cannot be represented."
                    )
                jacobian[:, column] = (
                    self.residual(probe, "finite_difference") - residual
                ) / distance
        if jacobian.shape != (residual.size, point.size):
            raise ValueError(f"jac must have shape {(residual.size, point.size)}.")
        return jacobian
