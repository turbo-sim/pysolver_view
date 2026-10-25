"""Dense trust-region subproblems with reusable QR and lazy SVD factors.

The ordinary full-rank Gauss--Newton path uses column-pivoted QR. Hybrid
Broyden updates retain that permutation and update the factors in quadratic
work; a triangular condition estimate guards this no-longer-pivoted update.
An SVD supplies minimum-norm solutions for ill-conditioned models and solves
the damped LM subproblem. No normal-equation matrix is formed.
"""

from numbers import Integral

import numpy as np
from scipy.linalg import get_lapack_funcs, norm, qr, qr_update, solve_triangular, svd

_EPS = np.finfo(float).eps
_QR_RCOND_MIN = np.sqrt(_EPS)
_COUNTERS = (
    "qr_factorizations",
    "qr_updates",
    "svd_factorizations",
    "triangular_solves",
    "condition_checks",
    "factorizations_refreshed",
    "lm_secular_iterations",
)
try:
    # Some older SciPy distributions do not wrap TRCON. Their conservative
    # fallback is the SVD path, not an unguarded triangular solve.
    _TRCON = get_lapack_funcs("trcon", dtype=np.float64)
except ValueError:  # pragma: no cover - depends on the installed SciPy
    _TRCON = None


def _feasible(step, radius):
    """Remove a possible last-bit overshoot without moving the boundary out."""
    length = norm(step, check_finite=False)
    if length > radius:
        step *= np.nextafter(radius, 0.0) / length
    return step


class LinearModel:
    """Factor and update the linear residual model ``f + a @ step``.

    The matrix ``a`` is the Jacobian in fixed scaled coordinates, ``f`` is
    the residual at the accepted base, and ``g = a.T @ f`` is the model gradient.
    The attributes ``q`` and ``r`` are QR factors of ``a[:, perm]``. The cached
    damping value ``_mu`` is normalized by the largest singular value squared.

    ``a`` has at least as many rows as columns. ``stats`` is a shared mutable
    counter dictionary; costs accumulate across model instances. A failed
    ``update`` returns False and the caller must discard this model and obtain
    a freshly evaluated Jacobian. Refreshing QR every ``refresh_interval`` successful
    updates limits accumulated factorization drift; it does not evaluate a
    Jacobian or discard the Broyden approximation.
    """

    def __init__(self, a, f, stats=None, refresh_interval=25, *, compute_gradient=True):
        """Copy and factor the scaled linear residual model f + a @ step.

        The matrix a has shape (m, n), f has length m, and m >= n. An optional shared
        stats dictionary accumulates operation counts across models. refresh_interval
        sets how often successful secant updates rebuild the QR factors.
        Set compute_gradient=False for Newton-family directions that do not use
        the dogleg gradient, avoiding an unnecessary overflowing A.T @ f.
        """
        self.a = np.array(a, dtype=float, copy=True)
        self.f = np.array(f, dtype=float, copy=True)
        if (
            self.a.ndim != 2
            or self.a.shape[1] == 0
            or self.a.shape[0] < self.a.shape[1]
            or self.f.shape != (self.a.shape[0],)
            or not np.all(np.isfinite(self.a))
            or not np.all(np.isfinite(self.f))
        ):
            raise ValueError("Expected a finite m-by-n model and m residuals, m >= n.")
        if (
            isinstance(refresh_interval, (bool, np.bool_))
            or not isinstance(refresh_interval, Integral)
            or refresh_interval < 1
        ):
            raise ValueError("refresh_interval must be a positive integer.")
        self.g = None
        if compute_gradient:
            with np.errstate(over="ignore", invalid="ignore"):
                self.g = self.a.T @ self.f
            if not np.all(np.isfinite(self.g)):
                raise ValueError("The linear-model gradient is nonfinite.")
        self.stats = {} if stats is None else stats
        for key in _COUNTERS:
            self.stats.setdefault(key, 0)
        self.refresh_interval = int(refresh_interval)
        self._updates_since_refresh = 0
        self._gauss_newton = None
        self._svd = None
        self._mu = None
        self._factorize()

    def _factorize(self):
        """Rebuild pivoted economic QR factors of the current matrix and assess
        triangular conditioning.
        """
        self.q, self.r, self.perm = qr(
            self.a, mode="economic", pivoting=True, check_finite=False
        )
        self.stats["qr_factorizations"] += 1
        self._qr_healthy = self._check_condition()

    def _check_condition(self):
        """Return whether the triangular factor passes the reciprocal-condition guard,
        counting the check.
        """
        self.stats["condition_checks"] += 1
        if _TRCON is None or not np.all(np.isfinite(self.r)):
            return False
        self.rcond, info = _TRCON(self.r, norm="1", uplo="U", diag="N")
        # R's diagonal alone cannot detect ill-conditioning after QR updates.
        # TRCON estimates its reciprocal condition without another SVD.
        return info == 0 and np.isfinite(self.rcond) and self.rcond >= _QR_RCOND_MIN

    def _ensure_svd(self):
        """Compute/cache singular values, right singular vectors and projected residuals
        for fallback or damping.
        """
        if self._svd is None:
            u, singular_values, vh = svd(
                self.a, full_matrices=False, check_finite=False
            )
            self._svd = singular_values, vh, u.T @ self.f
            self.stats["svd_factorizations"] += 1
        return self._svd

    @property
    def gauss_newton(self):
        """Return the cached unconstrained least-squares step for the current model.

        Use a guarded triangular QR solve when possible; otherwise use an SVD
        minimum-norm solution with a numerical-rank cutoff. The returned cached array
        is internal state; step() supplies a copy when returning it to the driver.
        """
        if self._gauss_newton is None:
            if self._qr_healthy:
                rhs = -(self.q.T @ self.f)
                ordered = solve_triangular(self.r, rhs, check_finite=False)
                self.stats["triangular_solves"] += 1
                self._gauss_newton = np.empty_like(ordered)
                self._gauss_newton[self.perm] = ordered
            else:
                s, vh, c = self._ensure_svd()
                cutoff = _EPS * max(self.a.shape) * s[0]
                coefficients = np.zeros_like(s)
                np.divide(-c, s, out=coefficients, where=s > cutoff)
                self._gauss_newton = vh.T @ coefficients
            if not np.all(np.isfinite(self._gauss_newton)):
                raise ValueError("The Gauss--Newton step is nonfinite.")
        return self._gauss_newton

    @property
    def numerical_rank(self):
        """Return full rank for guarded QR, otherwise the SVD numerical rank."""
        if self._qr_healthy:
            return self.a.shape[1]
        values, _, _ = self._ensure_svd()
        cutoff = _EPS * max(self.a.shape) * values[0]
        return int(np.count_nonzero(values > cutoff))

    @property
    def newton_raphson(self):
        """Return a strict square full-rank Newton direction without rank truncation.

        A numerically singular model raises LinAlgError rather than silently
        replacing Newton with a pseudoinverse least-squares direction.
        """
        if self.a.shape[0] != self.a.shape[1]:
            raise ValueError("Newton–Raphson requires a square Jacobian.")
        if self.numerical_rank != self.a.shape[1]:
            raise np.linalg.LinAlgError("Numerically singular Newton Jacobian.")
        return self.gauss_newton

    def update(self, correction, direction, f_new):
        """Apply ``a += outer(correction, direction)`` and anchor at ``f_new``.

        A finite rejected trial can still improve the Jacobian approximation;
        its caller supplies the unchanged accepted residual in that case.
        Nonfinite input, gradient overflow, or a poorly conditioned updated QR
        requests an exact-Jacobian refresh by returning False.
        """
        correction = np.asarray(correction, dtype=float)
        direction = np.asarray(direction, dtype=float)
        f_new = np.asarray(f_new, dtype=float)
        if (
            correction.shape != self.f.shape
            or direction.shape != (self.a.shape[1],)
            or f_new.shape != self.f.shape
            or not np.all(np.isfinite(correction))
            or not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(f_new))
        ):
            return False
        with np.errstate(over="ignore", invalid="ignore"):
            updated = self.a + np.outer(correction, direction)
            gradient = updated.T @ f_new
        if not np.all(np.isfinite(updated)) or not np.all(np.isfinite(gradient)):
            return False

        self.a, self.f, self.g = updated, f_new.copy(), gradient
        self._gauss_newton = self._svd = self._mu = None
        self._updates_since_refresh += 1
        try:
            if self._updates_since_refresh >= self.refresh_interval:
                self._factorize()
                self.stats["factorizations_refreshed"] += 1
                self._updates_since_refresh = 0
            else:
                self.q, self.r = qr_update(
                    self.q,
                    self.r,
                    correction.copy(),
                    direction[self.perm],
                    overwrite_qruv=True,
                    check_finite=False,
                )
                self.stats["qr_updates"] += 1
                self._qr_healthy = self._check_condition()
        except (ValueError, np.linalg.LinAlgError):
            return False
        return self._qr_healthy and np.all(np.isfinite(self.q))

    def step(self, method, radius):
        """Select a named subproblem step for experiments using the model directly.

        Solver entry points bind the explicit step functions once before iteration.
        Short method aliases remain supported by this compatibility interface.
        """
        if method in ("levenberg_marquardt", "lm"):
            return self.levenberg_marquardt_step(radius)
        if method in ("powell_hybrid", "hybr"):
            return self.powell_hybrid_step(radius)
        raise ValueError("method must be 'levenberg_marquardt' or 'powell_hybrid'.")

    def levenberg_marquardt_step(self, radius):
        """Return the unconstrained step if feasible, otherwise solve LM damping."""
        return self._bounded_step(radius, self._lm)

    def powell_hybrid_step(self, radius):
        """Return the unconstrained step if feasible, otherwise follow the dogleg."""
        return self._bounded_step(radius, self._dogleg)

    def _bounded_step(self, radius, boundary_step):
        """Validate a radius and invoke its boundary solver only when needed."""
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive.")
        if norm(self.gauss_newton, check_finite=False) <= radius:
            return self.gauss_newton.copy()
        return boundary_step(radius)

    def _lm(self, radius):
        """Find a damped least-squares step on the feasible side of the radius equation.

        Search a bracket for normalized damping using safeguarded Newton iterations.
        Compute lengths in singular coordinates, reuse an existing SVD on retries,
        and multiply by right singular vectors only for the final step.
        """
        # Orthonormal V preserves length, so damping iterations need only
        # singular coordinates. V is applied once, after choosing damping.
        singular_values, vh, c = self._ensure_svd()
        scale = singular_values[0]
        s = singular_values / scale
        # Multiply before dividing: an exactly null singular component then
        # stays zero even if its residual divided by scale would overflow.
        with np.errstate(over="ignore", invalid="ignore"):
            weights = (s * c) / scale
        diagonal = s * s
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            upper = float(norm(weights, check_finite=False) / radius)
        if not np.isfinite(upper) or upper == 0:
            raise ValueError("The LM damping bracket is outside floating-point range.")
        lower = max(0.0, upper - diagonal[0])
        mu = self._mu
        if mu is None or not lower < mu < upper:
            mu = 0.5 * lower + 0.5 * upper
        feasible = weights / (diagonal + upper)
        for _ in range(100):
            self.stats["lm_secular_iterations"] += 1
            denominator = diagonal + mu
            coefficients = weights / denominator
            length = norm(coefficients, check_finite=False)
            if length > radius:
                lower = mu
            else:
                upper, feasible = mu, coefficients
                if length >= radius * (1.0 - 1e-10):
                    break
            if length == 0 or not np.isfinite(length):
                candidate = 0.5 * lower + 0.5 * upper
                if candidate == mu:
                    break
                mu = candidate
                continue
            # A safeguarded Newton step for ||z(mu)|| - radius. Work with
            # normalized coordinates to avoid squaring large coefficients.
            normalized = coefficients / length
            slope = np.sum(normalized * normalized / denominator)
            candidate = mu + (1.0 - radius / length) / slope
            if not np.isfinite(candidate) or not lower < candidate < upper:
                candidate = 0.5 * lower + 0.5 * upper
            if candidate == mu:
                break
            mu = candidate
        self._mu = upper
        return _feasible(-(vh.T @ feasible), radius)

    def _dogleg(self, radius):
        """Return the feasible dogleg point between the gradient and Gauss-Newton directions.

        Use the radial gradient step when the Cauchy point is outside the radius;
        otherwise intersect the Cauchy-to-Gauss-Newton segment with the sphere using a
        scaled quadratic formula that avoids cancellation.
        """
        gradient_norm = norm(self.g, check_finite=False)
        if gradient_norm == 0:
            return np.zeros_like(self.g)
        direction = -self.g / gradient_norm
        curvature_norm = norm(self.a @ direction, check_finite=False)
        if curvature_norm == 0:
            return _feasible(radius * direction, radius)
        # Successive divisions avoid explicitly squaring a large curvature.
        cauchy_length = (gradient_norm / curvature_norm) / curvature_norm
        if cauchy_length >= radius:
            return _feasible(radius * direction, radius)
        cauchy = cauchy_length * direction

        # Intersect the Cauchy-to-GN ray with the sphere in radius units.
        # Scaling before subtraction prevents a huge GN point from overflowing
        # the quadratic coefficients. The positive root uses its cancellation-
        # free expression when the linear coefficient is nonnegative.
        geometry_scale = max(np.max(np.abs(self.gauss_newton)), radius)
        ray = self.gauss_newton / geometry_scale - cauchy / geometry_scale
        ray /= norm(ray, check_finite=False)
        scaled_cauchy = cauchy / radius
        beta = float(scaled_cauchy @ ray)
        slack = max(0.0, 1.0 - float(scaled_cauchy @ scaled_cauchy))
        root = np.sqrt(beta * beta + slack)
        distance = slack / (root + beta) if beta >= 0 else root - beta
        return _feasible(cauchy + (radius * distance) * ray, radius)
