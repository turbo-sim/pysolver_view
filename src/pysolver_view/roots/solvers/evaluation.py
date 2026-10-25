"""Validated model evaluations, numerical derivatives, and call accounting."""

import numpy as np
from scipy.linalg import norm

from .exceptions import BudgetExhausted, DomainError, UnrepresentableProbe


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
        self.derivative_method = options.derivative_method
        self.probe_steps = (
            options.finite_diff_step * scale
            if options.derivative_abs_step is None
            else np.broadcast_to(
                np.asarray(options.derivative_abs_step, dtype=float), scale.shape
            ).copy()
        )
        self.recorder = recorder
        recorder.evaluator = self
        self.square = square
        self.nfev = self.njev = 0
        self.derivative_nfev = self.failed_njev = 0
        self.residual_size = self.last_norm = None

    def counts(self):
        """Return physical call counts, keeping probes separate from iterations."""
        return dict(
            nfev=self.nfev,
            njev=self.njev,
            derivative_nfev=self.derivative_nfev,
            failed_njev=self.failed_njev,
        )

    def observe(self, kind, point, residual, *, valid, reason=None):
        """Send a detached evaluation snapshot without affecting the numerical driver."""
        if self.recorder.observer is not None:
            self.recorder.observer(
                dict(
                    kind=kind,
                    x=point.copy(),
                    fun=None if residual is None else residual.copy(),
                    valid=valid,
                    reason=reason,
                    **self.counts(),
                )
            )

    def residual(self, point, kind):
        """Validate and count one residual call, including undefined model states."""
        if self.nfev >= self.max_nfev:
            raise BudgetExhausted
        self.nfev += 1
        if kind == "finite_difference":
            self.derivative_nfev += 1
        record = None
        if self.recorder.enabled:
            record = dict(kind=kind, valid=False, residual_norm=None, nfev=self.nfev)
            if self.recorder.coordinates:
                record["x"] = point.copy()
            self.recorder.evaluations.append(record)
        try:
            values = np.asarray(
                self.fun(point.copy()),
                dtype=complex if np.iscomplexobj(point) else float,
            )
            if values.ndim != 1 or values.size < self.scale.size:
                raise ValueError(
                    "fun must return a vector with at least len(x0) entries."
                )
            if self.residual_size is not None and values.size != self.residual_size:
                raise ValueError("Residual size changed during the solve.")
            if self.square and values.size != self.scale.size:
                raise ValueError(
                    "Newton-Raphson requires a square residual/Jacobian pair."
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
            self.observe(kind, point, None, valid=False, reason=str(error))
            raise
        self.residual_size = values.size
        self.last_norm = residual_norm
        if record is not None:
            record.update(valid=True, residual_norm=residual_norm)
        self.observe(kind, point, values, valid=True)
        return values.copy()

    def jacobian(self, point, residual):
        """Count one Jacobian request and obtain analytic or numerical derivatives.

        Finite-difference probes pass through residual() and consume its budget.
        Shape errors propagate; drivers classify nonfinite derivative values.
        """
        self.njev += 1
        try:
            jacobian = (
                np.asarray(self.jac(point.copy()), dtype=float)
                if self.jac is not None
                else self.numerical_jacobian(point, residual)
            )
            if jacobian.shape != (residual.size, point.size):
                raise ValueError(f"jac must have shape {(residual.size, point.size)}.")
        except Exception:
            self.failed_njev += 1
            self.observe("jacobian", point, residual, valid=False)
            raise
        valid = bool(np.all(np.isfinite(jacobian)))
        self.failed_njev += not valid
        self.observe("jacobian", point, residual, valid=valid)
        return jacobian

    def numerical_jacobian(self, point, residual):
        """Differentiate through budgeted probes with no hidden base reevaluation.

        Forward and central differences divide by the represented displacement.
        Complex step requires the application to preserve imaginary perturbations.
        All probes, including failures, consume the same residual-call budget.
        """
        jacobian = np.empty((residual.size, point.size))
        for column, step in enumerate(self.probe_steps):
            if self.derivative_method == "cs":
                probe = point.astype(complex)
                probe[column] += 1j * step
                jacobian[:, column] = (
                    self.residual(probe, "finite_difference").imag / step
                )
                continue
            plus = point.copy()
            plus[column] += step
            if plus[column] == point[column] or not np.isfinite(plus[column]):
                raise UnrepresentableProbe(
                    "Finite-difference probe cannot be represented."
                )
            upper = self.residual(plus, "finite_difference")
            if self.derivative_method == "2-point":
                jacobian[:, column] = (upper - residual) / (
                    plus[column] - point[column]
                )
            else:
                minus = point.copy()
                minus[column] -= step
                if minus[column] == point[column] or not np.isfinite(minus[column]):
                    raise UnrepresentableProbe(
                        "Finite-difference probe cannot be represented."
                    )
                lower = self.residual(minus, "finite_difference")
                jacobian[:, column] = (upper - lower) / (plus[column] - minus[column])
        return jacobian
