"""Assess first-order KKT conditions without evaluating or changing a model.

Public inputs use h(x)=0, g(x)<=0, and lower<=x<=upper. The Lagrangian is
L = f + lambda*h + mu*g - z_lower*(x-lower) + z_upper*(x-upper).
Equality multipliers are unrestricted; inequality and bound multipliers must
be nonnegative. All metrics use the supplied objective, constraint, and variable
units. Positive rescaling changes tolerance meanings even though the fitted
multipliers transform consistently.

``assess_kkt`` estimates sign-feasible multipliers by jointly minimizing the
infinity norm of stationarity and complementarity residuals. It then reconstructs
the residuals with the original coefficients, independently of the LP's success
flag. ``evaluate_kkt_residuals`` performs only that reconstruction for a supplied
complete multiplier set, permitting independent checks of native multipliers.

The minimax estimator is adapted from Turbodash's generic numerical checker.
No active-set threshold, model callback, history recording, or solver state is
needed. A passing first-order check does not prove a minimum or second-order
sufficiency; interpretation also depends on derivative accuracy and constraint
qualifications. ``assess_kkt_nonnegative`` is the private SLSQP sign adapter.
"""

from collections.abc import Mapping

import numpy as np
from scipy.optimize import linprog


def _real_array(value, name):
    """Copy a real-valued input without silently discarding imaginary parts."""
    if np.iscomplexobj(value):
        raise ValueError(f"KKT {name} must be real.")
    return np.array(value, dtype=float, copy=True)


def _prepare_inputs(
    x,
    gradient,
    equality_values,
    equality_jacobian,
    inequality_values,
    inequality_jacobian,
    lower,
    upper,
    feasibility_tol,
    optimality_tol,
):
    """Validate and detach all mathematical inputs shared by both operations."""
    names = (
        "coordinates",
        "gradient",
        "equalities",
        "equality Jacobian",
        "inequalities",
        "inequality Jacobian",
        "lower bounds",
        "upper bounds",
    )
    values = (
        x,
        gradient,
        equality_values,
        equality_jacobian,
        inequality_values,
        inequality_jacobian,
        lower,
        upper,
    )
    x, gradient, h, jh, g, jg, lower, upper = (
        _real_array(v, name) for v, name in zip(values, names)
    )
    if (
        x.ndim != 1
        or not x.size
        or gradient.shape != x.shape
        or lower.shape != x.shape
        or upper.shape != x.shape
    ):
        raise ValueError(
            "KKT coordinates, gradient, and bounds must be equally sized vectors."
        )
    n = x.size
    if h.ndim != 1 or g.ndim != 1 or jh.shape != (h.size, n) or jg.shape != (g.size, n):
        raise ValueError(
            "KKT constraint values and Jacobians have inconsistent shapes."
        )
    if not all(np.all(np.isfinite(a)) for a in (x, gradient, h, g, jh, jg)):
        raise ValueError("KKT values and derivatives must be finite.")
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
        or np.any(lower == np.inf)
        or np.any(upper == -np.inf)
    ):
        raise ValueError("Invalid KKT bounds.")
    for tolerance in (feasibility_tol, optimality_tol):
        if np.ndim(tolerance) != 0 or np.iscomplexobj(tolerance):
            raise ValueError("KKT tolerances must be finite positive scalars.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("KKT tolerances must be finite and positive.")
    return x, gradient, h, jh, g, jg, lower, upper


def _geometry(data):
    """Return public Lagrangian normals and finite constraint slacks."""
    x, _, h, jh, g, jg, lower, upper = data
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    with np.errstate(over="ignore", invalid="ignore"):
        lower_slack = x[finite_lower] - lower[finite_lower]
        upper_slack = upper[finite_upper] - x[finite_upper]
    normals = np.vstack(
        (jh, jg, -np.eye(x.size)[finite_lower], np.eye(x.size)[finite_upper])
    ).T
    slacks = np.r_[np.zeros(h.size), -g, lower_slack, upper_slack]
    return normals, slacks, finite_lower, finite_upper


def _base_result(data, feasibility_tol, optimality_tol, source, method):
    """Compute feasibility independently of whether multiplier fitting succeeds."""
    x, _, h, _, g, _, lower, upper = data
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    equality = float(np.max(np.abs(h), initial=0))
    inequality = float(np.max(g, initial=0))
    with np.errstate(over="ignore", invalid="ignore"):
        bound = max(
            float(np.max(lower[finite_lower] - x[finite_lower], initial=0)),
            float(np.max(x[finite_upper] - upper[finite_upper], initial=0)),
        )
    primal = max(equality, inequality, bound)
    return dict(
        x=x.copy(),
        point=x.copy(),
        method=method,
        multiplier_source=source,
        convention="h=0, g<=0; L=f+lambda*h+mu*g-z_lower*(x-lower)+z_upper*(x-upper)",
        norm="infinity",
        equality_feasibility=equality,
        inequality_feasibility=inequality,
        bound_feasibility=bound,
        primal_feasibility=primal,
        feasible=bool(primal <= feasibility_tol),
        feasibility_tol=float(feasibility_tol),
        optimality_tol=float(optimality_tol),
    )


def _unavailable(base, reason):
    """Represent missing diagnostics as unavailable rather than as zero residuals."""
    return dict(
        base,
        available=False,
        passed=False,
        message=reason,
        reason=reason,
        stationarity=None,
        dual_feasibility=None,
        complementarity=None,
        optimality=None,
        kkt_residual=None,
        kkt_ratio=None,
        lagrangian_gradient=None,
        complementarity_products=None,
        multipliers=None,
    )


def _validate_multipliers(data, multipliers):
    """Require all multiplier groups, including explicit zeros at absent bounds."""
    if not isinstance(multipliers, Mapping):
        raise ValueError(
            "KKT multipliers must map equality, inequality, lower, and upper arrays."
        )
    x, _, h, _, g, _, lower, upper = data
    result = {}
    for name, size in (
        ("equality", h.size),
        ("inequality", g.size),
        ("lower", x.size),
        ("upper", x.size),
    ):
        if name not in multipliers:
            raise ValueError(
                f"Missing KKT {name} multipliers; a complete set is required."
            )
        array = _real_array(multipliers[name], f"{name} multipliers")
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"KKT {name} multipliers must be finite with shape {(size,)}."
            )
        result[name] = array
    if np.any(result["lower"][~np.isfinite(lower)] != 0) or np.any(
        result["upper"][~np.isfinite(upper)] != 0
    ):
        raise ValueError("KKT multipliers for absent bounds must be zero.")
    return result


def _reconstruct(data, multipliers, base):
    """Reconstruct residuals in original units without trusting an optimizer fit."""
    x, gradient, _, jh, g, jg, lower, upper = data
    m = multipliers
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    products = dict(
        inequality=m["inequality"] * g, lower=np.zeros(x.size), upper=np.zeros(x.size)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        lagrangian = (
            gradient
            + jh.T @ m["equality"]
            + jg.T @ m["inequality"]
            - m["lower"]
            + m["upper"]
        )
        products["lower"][finite_lower] = m["lower"][finite_lower] * (
            x[finite_lower] - lower[finite_lower]
        )
        products["upper"][finite_upper] = m["upper"][finite_upper] * (
            upper[finite_upper] - x[finite_upper]
        )
    if (
        not np.all(np.isfinite(lagrangian))
        or not all(np.all(np.isfinite(v)) for v in products.values())
        or not np.isfinite(base["primal_feasibility"])
    ):
        return _unavailable(
            base, "Nonfinite KKT residual after multiplier reconstruction."
        )
    stationarity = float(np.max(np.abs(lagrangian), initial=0))
    complementarity = max(
        float(np.max(np.abs(v), initial=0)) for v in products.values()
    )
    dual = float(np.max(-np.r_[m["inequality"], m["lower"], m["upper"]], initial=0))
    optimality = max(stationarity, complementarity, dual)
    ratio = max(
        base["primal_feasibility"] / base["feasibility_tol"],
        optimality / base["optimality_tol"],
    )
    return dict(
        base,
        available=True,
        passed=bool(ratio <= 1),
        message="",
        reason="",
        stationarity=stationarity,
        dual_feasibility=dual,
        complementarity=complementarity,
        optimality=optimality,
        kkt_residual=max(base["primal_feasibility"], optimality),
        kkt_ratio=ratio,
        lagrangian_gradient=lagrangian,
        complementarity_products=products,
        multipliers={name: value.copy() for name, value in m.items()},
    )


def evaluate_kkt_residuals(
    x,
    gradient,
    equality_values,
    equality_jacobian,
    inequality_values,
    inequality_jacobian,
    lower,
    upper,
    *,
    multipliers,
    feasibility_tol=1e-6,
    optimality_tol=1e-4,
    multiplier_source="supplied",
):
    """Evaluate KKT residuals for a complete, explicitly supplied multiplier set.

    Inputs use inequalities <=0 and the plus-sign public Lagrangian. Supply a
    mapping containing equality, inequality, lower, and upper multiplier arrays;
    bound arrays have one element per variable, zero where the bound is absent.
    Missing bound multipliers are not silently filled from an estimator. Negative
    inequality/bound multipliers are accepted as inputs and fail dual feasibility.
    The source label records provenance and does not alter the numerical checks.
    """
    data = _prepare_inputs(
        x,
        gradient,
        equality_values,
        equality_jacobian,
        inequality_values,
        inequality_jacobian,
        lower,
        upper,
        feasibility_tol,
        optimality_tol,
    )
    base = _base_result(
        data, feasibility_tol, optimality_tol, multiplier_source, "supplied_multipliers"
    )
    return _reconstruct(data, _validate_multipliers(data, multipliers), base)


def assess_kkt(
    x,
    gradient,
    equality_values,
    equality_jacobian,
    inequality_values,
    inequality_jacobian,
    lower,
    upper,
    *,
    feasibility_tol=1e-6,
    optimality_tol=1e-4,
):
    """Estimate sign-feasible multipliers and independently assess first-order KKT.

    All inequalities and finite bounds participate in a linear minimax fit of
    stationarity and complementarity; equality multipliers remain unrestricted.
    Column equilibration and gradient scaling protect small gradients and
    positive constraint rescaling. Returned residuals and tolerances are in the
    supplied coordinates and units, not the equilibrated fitting system.
    Failed numerical fitting returns available=False, passed=False and a reason.
    Invalid inputs raise ValueError. No model is evaluated or history mutated.
    """
    data = _prepare_inputs(
        x,
        gradient,
        equality_values,
        equality_jacobian,
        inequality_values,
        inequality_jacobian,
        lower,
        upper,
        feasibility_tol,
        optimality_tol,
    )
    x, gradient, h, _, g, _, _, _ = data
    base = _base_result(
        data, feasibility_tol, optimality_tol, "estimated", "pointwise_minimax"
    )
    normals, slacks, finite_lower, finite_upper = _geometry(data)
    if not np.all(np.isfinite(slacks)):
        return _unavailable(base, "Constraint slacks exceed finite numerical range.")
    fitted = np.zeros(slacks.size)
    if slacks.size and np.any(gradient):
        column_scale = np.maximum(np.max(np.abs(normals), axis=0), np.abs(slacks))
        retained = column_scale > 0
        if np.any(retained):
            matrix = np.vstack(
                (
                    normals[:, retained] / column_scale[retained],
                    np.diag(slacks[retained] / column_scale[retained]),
                )
            )
            residual_scale = float(np.max(np.abs(gradient)))
            rhs = np.r_[
                -gradient / residual_scale, np.zeros(np.count_nonzero(retained))
            ]
            bounds = [
                (None, None) if i < h.size else (0, None)
                for i in np.flatnonzero(retained)
            ] + [(0, None)]
            inequalities = np.vstack(
                (
                    np.column_stack((matrix, -np.ones(matrix.shape[0]))),
                    np.column_stack((-matrix, -np.ones(matrix.shape[0]))),
                )
            )
            fit = linprog(
                np.r_[np.zeros(matrix.shape[1]), 1.0],
                A_ub=inequalities,
                b_ub=np.r_[rhs, -rhs],
                bounds=bounds,
                method="highs-ds",
                options={
                    "primal_feasibility_tolerance": 1e-10,
                    "dual_feasibility_tolerance": 1e-10,
                },
            )
            if not fit.success:
                return _unavailable(
                    base, f"Multiplier estimation failed: {fit.message}"
                )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                fitted[retained] = fit.x[:-1] * residual_scale / column_scale[retained]
            if not np.all(np.isfinite(fitted)):
                return _unavailable(
                    base, "Estimated multipliers exceed finite numerical range."
                )
    offset = h.size + g.size
    lower_multipliers, upper_multipliers = np.zeros(x.size), np.zeros(x.size)
    lower_count = int(np.count_nonzero(finite_lower))
    lower_multipliers[finite_lower] = fitted[offset : offset + lower_count]
    upper_multipliers[finite_upper] = fitted[offset + lower_count :]
    multipliers = dict(
        equality=fitted[: h.size],
        inequality=fitted[h.size : offset],
        lower=lower_multipliers,
        upper=upper_multipliers,
    )
    return _reconstruct(data, multipliers, base)


def assess_kkt_nonnegative(
    x,
    gradient,
    equality_values,
    equality_jacobian,
    inequality_values,
    inequality_jacobian,
    lower,
    upper,
    *,
    feasibility_tol=1e-6,
    optimality_tol=1e-4,
):
    """Adapt the private SLSQP convention c>=0 and L=f-lambda*h-mu*c.

    Constraint values and Jacobians are negated once. Equality multipliers are
    converted back to the driver's minus-sign convention; bound and inequality
    multipliers are unchanged. Residual norms and acceptance remain identical.
    """
    result = assess_kkt(
        x,
        gradient,
        equality_values,
        equality_jacobian,
        -np.asarray(inequality_values),
        -np.asarray(inequality_jacobian),
        lower,
        upper,
        feasibility_tol=feasibility_tol,
        optimality_tol=optimality_tol,
    )
    result["convention"] = (
        "h=0, c>=0; L=f-lambda*h-mu*c-z_lower*(x-lower)-z_upper*(upper-x)"
    )
    if result["multipliers"] is not None:
        result["multipliers"]["equality"] *= -1
        result["complementarity_products"]["inequality"] *= -1
    return result
