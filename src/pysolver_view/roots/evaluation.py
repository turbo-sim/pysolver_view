"""Native-root evaluations and derivative probes with explicit call accounting."""

import numpy as np
from .. import numerical_differentiation


class RootEvaluation:
    """Own native-root caches and counters; publish only selected observations."""

    def __init__(
        self, problem, *, derivative_method, derivative_abs_step, update_on, observe
    ):
        self.problem = problem
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
        self.update_on = update_on
        self.observe = observe
        self.residuals_last = None
        self._residual_x = None
        self._residual_value = None
        self.grad_count = 0
        self.failed_grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.derivative_func_count = 0

    def residual(self, x, called_from_jac=False):
        """Evaluate residuals, counting derivative probes without recording them.

        Every actual model evaluation increments ``func_count_tot``. Calls
        made for numerical derivatives also increment ``derivative_func_count``
        but never change ordinary observation state or invoke user callbacks.
        Inputs, returned values, and stored snapshots do not share writable
        arrays with a model or callback.
        """
        point = np.array(x, copy=True)
        self.func_count_tot += 1
        if called_from_jac:
            self.derivative_func_count += 1
        else:
            self.func_count += 1
        value = np.atleast_1d(np.array(self.problem.residual(point.copy()), copy=True))
        if not called_from_jac:
            self.residuals_last = value.copy()
            self._residual_x = point.copy()
            self._residual_value = value.copy()
            if self.update_on == "function":
                self.observe(point, value, event_kind="function")
        return value

    def gradient(self, x):
        """Evaluate one Jacobian request at a point-consistent residual base.

        Numerical differences use residual values at the requested point,
        never the most recent probe or an unrelated observation. A request
        increments ``grad_count`` once, regardless of how many probes it uses;
        failed requests also increment ``failed_grad_count``.
        """
        point = np.array(x, dtype=float, copy=True)
        self.grad_count += 1
        try:
            if self._residual_x is not None and np.array_equal(point, self._residual_x):
                base = self._residual_value.copy()
            else:
                base = self.residual(point, called_from_jac=True)
                self._residual_x = point.copy()
                self._residual_value = base.copy()

            if hasattr(self.problem, "gradient"):
                jacobian = self.problem.gradient(point.copy())
            else:

                def evaluate_probe(probe):
                    """Reuse the known base, evaluating only actual new probes."""
                    if np.array_equal(probe, point):
                        return base.copy()
                    return self.residual(probe, called_from_jac=True)

                jacobian = numerical_differentiation.approx_gradient(
                    evaluate_probe,
                    point,
                    f0=base,
                    method=self.derivative_method,
                    abs_step=self.derivative_abs_step,
                )
            jacobian = np.asarray(jacobian).reshape(base.size, point.size).copy()
        except Exception:
            self.failed_grad_count += 1
            raise

        if self.update_on == "gradient":
            self.observe(point, base, event_kind="gradient")
        return jacobian
