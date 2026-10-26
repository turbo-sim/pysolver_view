"""Pure report formatting from supplied variables, constraints and KKT data."""

import numpy as np

from .layout import table_report, validate_report_width


def variables_report(values, bounds, names, normalized=True, *, width=80):
    """Format physical/normalized variables, wrapping full names to width."""
    width = validate_report_width(width)
    values = np.asarray(values)
    lower, upper = (
        (np.full(values.size, -np.inf), np.full(values.size, np.inf))
        if bounds is None
        else bounds
    )
    names = names or [f"design_variable_{i}" for i in range(len(values))]
    scale = "normalized" if normalized else "physical"
    precision = ".4f" if normalized else ".3e"
    rows = [
        (name, format(lb, precision), format(value, precision), format(ub, precision))
        for name, lb, value, ub in zip(names, lower, values, upper)
    ]
    return table_report(
        f"Optimization variables report ({scale} values)",
        ("Variable name", "Lower", "Value", "Upper"),
        rows,
        (width - 40, 13, 13, 13),
        width,
    )


def kkt_report(kkt_data, tol, *, width=80):
    """Format an explicit KKT diagnostic with one common acceptance tolerance."""
    width = validate_report_width(width)
    entries = [
        (
            "First order optimality",
            "first_order_optimality",
            "first_order_optimality_ok",
            "<=",
        ),
        (
            "Equality feasibility",
            "feasibility_equality",
            "feasibility_equality_ok",
            "<=",
        ),
        (
            "Inequality feasibility",
            "feasibility_inequality",
            "feasibility_inequality_ok",
            "<=",
        ),
        ("Dual feasibility", "feasibility_dual", "feasibility_dual_ok", ">="),
        (
            "Complementary slackness",
            "complementary_slackness",
            "complementary_slackness_ok",
            "<=",
        ),
    ]
    rows = [
        (
            name,
            f"{kkt_data[value]:+.3e}",
            f"{operator} {(-tol if operator == '>=' else tol):+.3e}",
            "yes" if kkt_data[ok] else "no",
        )
        for name, value, ok, operator in entries
    ]
    notes = [
        "Coordinates: solver; stationarity norm: infinity.",
        "Multipliers: independently estimated; first-order conditions only.",
    ]
    if not kkt_data["available"]:
        notes.append("Assessment unavailable: " + kkt_data.get("message", ""))
    return table_report(
        "Karush-Kuhn-Tucker (KKT) conditions",
        ("Condition", "Value", "Target", "Ok?"),
        rows,
        (width - 43, 16, 16, 10),
        width,
        notes,
    )


def final_kkt_report(
    assessment, feasibility_tol, optimality_tol, *, enabled=True, width=80
):
    """Render a cached certificate with separate tolerances and no diagnostics."""
    width = validate_report_width(width)
    assessment = assessment or {}
    available = bool(enabled and assessment.get("available", False))
    entries = [
        ("First order optimality", "optimality", optimality_tol),
        ("Stationarity", "stationarity", optimality_tol),
        ("Equality feasibility", "equality_feasibility", feasibility_tol),
        ("Inequality feasibility", "inequality_feasibility", feasibility_tol),
        ("Bound feasibility", "bound_feasibility", feasibility_tol),
        ("Dual feasibility", "dual_feasibility", optimality_tol),
        ("Complementary slackness", "complementarity", optimality_tol),
    ]
    rows = []
    for name, key, limit in entries:
        value = assessment.get(key)
        finite = value is not None and np.isfinite(value)
        rendered = f"{value:+.3e}" if finite else "unavailable"
        status = ("yes" if value <= limit else "no") if finite else "n/a"
        rows.append((name, rendered, f"<= {limit:+.3e}", status))
    notes = [
        "Coordinates: solver; stationarity norm: infinity.",
        "Source: cached final assessment; no additional diagnostic evaluations.",
    ]
    if not enabled:
        notes.append("Automatic KKT checks disabled; optimality not assessed.")
    elif not available:
        notes.append(
            "Assessment unavailable: "
            + (
                assessment.get("message")
                or "No final derivative certificate available."
            )
        )
    else:
        notes.append(
            "Cached final KKT check: "
            + ("passed." if assessment.get("passed") else "failed.")
        )
    return table_report(
        "Karush-Kuhn-Tucker (KKT) conditions",
        ("Condition", "Value", "Target", "Ok?"),
        rows,
        (width - 43, 16, 16, 10),
        width,
        notes,
    )


def final_multipliers_report(
    assessment, constraint_data, var_names, bounds, *, enabled=True, width=80
):
    """Format cached public multipliers through the named width-bounded table."""
    width = validate_report_width(width)
    assessment = assessment or {}
    available = bool(enabled and assessment.get("available", False))
    multipliers = assessment.get("multipliers") if available else None
    equality_count = sum(entry["type"] == "=" for entry in constraint_data)
    inequality_count = len(constraint_data) - equality_count
    size = len(var_names)
    lower, upper = (
        (np.full(size, -np.inf), np.full(size, np.inf))
        if bounds is None
        else map(np.asarray, bounds)
    )
    finite = np.r_[
        np.ones(inequality_count, dtype=bool), np.isfinite(lower), np.isfinite(upper)
    ]
    if multipliers is None:
        equality = np.full(equality_count, np.nan)
        duals = np.full(inequality_count + 2 * size, np.nan)
    else:
        equality = np.asarray(multipliers["equality"])
        duals = np.r_[
            multipliers["inequality"], multipliers["lower"], multipliers["upper"]
        ]
    message = (
        "Automatic KKT checks disabled; multipliers not assessed."
        if not enabled
        else assessment.get("message") or "No final multiplier certificate available."
    )
    data = dict(
        available=available,
        message=message,
        multipliers_eq=equality,
        multipliers_ineq={
            index: float(duals[index]) for index in np.flatnonzero(finite)
        },
    )
    return multipliers_report(data, constraint_data, var_names, bounds, width=width)


def constraint_report(constraint_data, *, width=80):
    """Format named constraint values and targets without shortening names."""
    width = validate_report_width(width)
    rows = [
        (
            entry.get("name", ""),
            f"{entry.get('value', 0.0):+.2e}",
            f"{entry.get('type', '')} {entry.get('target', 0.0):+.2e}",
            "yes" if entry.get("satisfied", False) else "no",
        )
        for entry in constraint_data
    ]
    return table_report(
        "Optimization constraints report",
        ("Constraint name", "Value", "Target", "Ok?"),
        rows,
        (width - 31, 11, 13, 6),
        width,
    )


def multipliers_report(kkt_data, constraint_data, var_names, bounds, *, width=80):
    """Format signed user-constraint and finite-bound multiplier estimates."""
    width = validate_report_width(width)
    equality = [entry for entry in constraint_data if entry["type"] == "="]
    inequality = [entry for entry in constraint_data if entry["type"] in {"<", ">"}]
    rows = [
        (entry["name"], "equality", f"{kkt_data['multipliers_eq'][index]:+.3e}")
        for index, entry in enumerate(equality)
    ]
    n_vars, n_ineq = len(var_names), len(inequality)
    active = kkt_data["multipliers_ineq"]
    for index in range(n_ineq + 2 * n_vars):
        if index < n_ineq:
            name, kind = inequality[index]["name"], "inequality"
        else:
            variable = (index - n_ineq) % n_vars
            side = 0 if index < n_ineq + n_vars else 1
            if bounds is None or not np.isfinite(bounds[side][variable]):
                continue
            name, kind = var_names[variable], (
                "lower bound" if side == 0 else "upper bound"
            )
        value = f"{active[index]:+.3e}" if index in active else "inactive"
        rows.append((name, kind, value))
    notes = (
        []
        if kkt_data["available"]
        else ["Assessment unavailable: " + kkt_data.get("message", "")]
    )
    return table_report(
        "Lagrange multipliers report (estimated, solver coordinates)",
        ("Multiplier for constraint", "Type", "Value"),
        rows,
        (width - 31, 15, 15),
        width,
        notes,
    )
