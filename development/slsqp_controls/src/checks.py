"""Independent mathematical checks adapted from Turbodash's audit approach.

This module never calls the production KKT assessor. Supplied multipliers are
reconstructed using fresh model values and analytical or finite-difference rows.
"""

import numpy as np


def reconstruct(problem, x, multipliers, jacobian=None):
    """Compute all four public-sign KKT residuals from original coefficients."""
    values = problem.values(x)
    jac = problem.gradient(x) if jacobian is None else jacobian
    lower, upper = map(np.asarray, problem.get_bounds())
    fl, fu = np.isfinite(lower), np.isfinite(upper)
    mu, zl, zu = (
        np.asarray(multipliers[key]) for key in ("inequality", "lower", "upper")
    )
    lagrangian = jac[0] + jac[1:].T @ mu - zl + zu
    primal = max(
        0.0,
        np.max(values[1:], initial=0),
        np.max(lower[fl] - x[fl], initial=0),
        np.max(x[fu] - upper[fu], initial=0),
    )
    dual = max(0.0, np.max(-np.r_[mu, zl, zu], initial=0))
    complementarity = np.max(
        np.abs(
            np.r_[
                mu * values[1:],
                zl[fl] * (x[fl] - lower[fl]),
                zu[fu] * (upper[fu] - x[fu]),
            ]
        ),
        initial=0,
    )
    stationarity = np.linalg.norm(lagrangian, np.inf)
    return dict(
        stationarity=float(stationarity),
        primal_feasibility=float(primal),
        dual_feasibility=float(dual),
        complementarity=float(complementarity),
        optimality=float(max(stationarity, dual, complementarity)),
    )


def fresh_jacobian(problem, x, step):
    """Independently differentiate with central samples or inward bound samples."""
    lower, upper = map(np.asarray, problem.get_bounds())
    result = np.empty((1 + problem.get_nic(), len(x)))
    base = problem.values(x)
    for j in range(len(x)):
        h = step * max(1.0, abs(x[j]))
        if problem.case == "square_root" and j == 0:
            # A fixed absolute probe near x=0 measures a different local regime.
            h = min(h, x[j] * 1e-3 * (step / 1e-5))
        plus, minus = x.copy(), x.copy()
        plus[j] += h
        minus[j] -= h
        # The physical square-root domain is stricter than the variable box.
        left = minus[j] >= lower[j] and not (
            problem.case == "square_root" and j == 0 and minus[j] <= 0
        )
        right = plus[j] <= upper[j]
        if left and right:
            result[:, j] = (problem.values(plus) - problem.values(minus)) / (2 * h)
        elif right:
            result[:, j] = (problem.values(plus) - base) / h
        elif left:
            result[:, j] = (base - problem.values(minus)) / h
        else:
            raise ValueError("No admissible independent derivative sample.")
    return result


def audit_solution(problem, solver, assessment):
    """Check final metrics, histories, and half-step derivative sensitivity.

    An audit can pass for a correctly rejected optimization. Audit failure means
    inconsistent reporting/math, not merely failure to meet an optimization target.
    """
    if not assessment["available"]:
        return dict(passed=False, reason="No multiplier certificate to audit")
    x = solver.x_final.copy()
    metrics = reconstruct(problem, x, assessment["multipliers"])
    checks = [
        np.isclose(metrics[key], assessment[key], rtol=1e-7, atol=1e-10)
        for key in metrics
    ]
    checks.append(
        np.isclose(problem.values(x)[0], solver.result.fun, rtol=1e-12, atol=1e-12)
    )
    for point, value, violation in zip(
        solver.convergence_history["x"],
        solver.convergence_history["objective_value"],
        solver.convergence_history["constraint_violation"],
    ):
        values = problem.values(point)
        lo, hi = map(np.asarray, problem.get_bounds())
        expected = max(
            0.0,
            np.max(values[1:], initial=0),
            np.max(lo - point, initial=0),
            np.max(point - hi, initial=0),
        )
        checks.extend(
            [
                np.isclose(value, values[0], rtol=1e-10, atol=1e-10),
                np.isclose(violation, expected, rtol=1e-10, atol=1e-10),
            ]
        )
    if solver.library == "pysolver":
        history = solver.backend_result.history
        for record in history:
            certificate = record["kkt"]
            if certificate is not None and certificate["available"]:
                reconstructed = reconstruct(
                    problem, record["x"], certificate["multipliers"]
                )
                checks.extend(
                    np.isclose(
                        reconstructed[key], certificate[key], rtol=1e-7, atol=1e-9
                    )
                    for key in reconstructed
                )
        for previous, current in zip(history, history[1:]):
            checks.append(
                np.isclose(
                    current["step_norm"],
                    np.linalg.norm(current["x"] - previous["x"]),
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
        checks.append(
            solver.result.success
            == (solver.result.solver_success and assessment["passed"])
        )
    derivative_checks = []
    for step in [1e-5, 5e-6]:
        jac = fresh_jacobian(problem, x, step)
        fresh = reconstruct(problem, x, assessment["multipliers"], jac)
        exact = problem.gradient(x)
        error = float(np.max(np.abs(jac - exact)))
        derivative_checks.append(
            dict(
                step=step,
                jacobian_error=error,
                scaled_jacobian_error=error / max(1.0, float(np.max(np.abs(exact)))),
                **fresh
            )
        )
    checks.append(derivative_checks[-1]["scaled_jacobian_error"] < 1e-5)
    return dict(
        passed=bool(all(checks)),
        analytical=metrics,
        fresh_derivatives=derivative_checks,
    )
