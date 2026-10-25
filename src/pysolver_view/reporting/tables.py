"""Pure report formatting from supplied variables, constraints and KKT data."""

import numpy as np


def variables_report(values, bounds, names, normalized=True):
    """
    Generate design variable report as a string.

    Parameters
    ----------
    x_norm : array-like
        Normalized design variables (input to the solver).
    normalized : bool, optional
        Whether to report in normalized or physical values. Default is True.

    Returns
    -------
    str
        Formatted string report.
    """
    values = np.asarray(values)
    lb_raw, ub_raw = (
        (np.full(values.size, -np.inf), np.full(values.size, np.inf))
        if bounds is None
        else bounds
    )
    names = names or [f"design_variable_{i}" for i in range(len(values))]
    scale_type = "normalized" if normalized else "physical"
    max_name_width = 38

    lines = []
    lines.append("")
    lines.append("-" * 80)
    header = f"{' Optimization variables report (' + scale_type + ' values)':<80}"
    lines.append(header)
    lines.append("-" * 80)
    lines.append(f" {'Variable name':<39}{'Lower':>13}{'Value':>13}{'Upper':>13}")
    lines.append("-" * 80)

    for key, lb, val, ub in zip(names, lb_raw, values, ub_raw):
        if len(key) > max_name_width:
            key = "..." + key[-(max_name_width - 3) :]

        if normalized:
            lines.append(f" {key:<39}{lb:>13.4f}{val:>13.4f}{ub:>13.4f}")
        else:
            lines.append(f" {key:<39}{lb:>13.3e}{val:>13.3e}{ub:>13.3e}")

    lines.append("-" * 80)
    return "\n".join(lines)


def kkt_report(kkt_data, tol):
    r"""
    Generate a detailed KKT condition satisfaction report (80-character width).

    This report includes five key Karush-Kuhn-Tucker (KKT) checks:

    .. math::

        \|\nabla L(x, \lambda)\| \leq \mathrm{tol} \quad &\text{(First order optimality)} \\
        \max |c_{\text{eq}}(x)| \leq \mathrm{tol} \quad &\text{(Equality feasibility)} \\
        \max(0, c_{\text{ineq}}(x)) \leq \mathrm{tol} \quad &\text{(Inequality feasibility)} \\
        \min(\lambda_{\text{ineq}}) \geq 0 \quad &\text{(Dual feasibility)} \\
        \max |\lambda_i \cdot c_i| \leq \mathrm{tol} \quad &\text{(Complementary slackness)}

    For each condition, the report shows:

        - Actual computed value
        - Comparison direction and target (tolerance or 0)
        - Satisfaction status

    Parameters
    ----------
    x_norm : array-like
        Normalized decision variable vector to evaluate the KKT conditions at.
    tol : float
        Numerical tolerance used for comparisons in optimality and feasibility checks.

    Returns
    -------
    str
        A formatted report string summarizing KKT condition satisfaction.
    """
    # Compute KKT conditions

    # Build report lines
    sep = "-" * 80
    lines = [
        "",
        sep,
        f" {'Karush-Kuhn-Tucker (KKT) conditions':<36}{'Value':>16}{'Target':>16}{'Ok?':>10}",
        sep,
    ]

    entries = [
        (
            "First order optimality",
            kkt_data["first_order_optimality"],
            f"<= {tol:+.3e}",
            kkt_data["first_order_optimality_ok"],
        ),
        (
            "Equality feasibility",
            kkt_data["feasibility_equality"],
            f"<= {tol:+.3e}",
            kkt_data["feasibility_equality_ok"],
        ),
        (
            "Inequality feasibility",
            kkt_data["feasibility_inequality"],
            f"<= {tol:+.3e}",
            kkt_data["feasibility_inequality_ok"],
        ),
        (
            "Dual feasibility",
            kkt_data["feasibility_dual"],
            f">= {-tol:+.3e}",
            kkt_data["feasibility_dual_ok"],
        ),
        (
            "Complementary slackness",
            kkt_data["complementary_slackness"],
            f"<= {tol:+.3e}",
            kkt_data["complementary_slackness_ok"],
        ),
    ]

    for name, val, target, ok in entries:
        status = "yes" if ok else " no"
        lines.append(f" {name:<36}{val:>+16.3e}{target:>16}{status:>10}")

    lines.append(" Coordinates: solver; stationarity norm: infinity.")
    lines.append(" Multipliers: independently estimated; first-order conditions only.")
    if not kkt_data["available"]:
        lines.append(" Assessment unavailable: " + kkt_data.get("message", ""))
    lines.append(sep)
    return "\n".join(lines)


def final_kkt_report(assessment, feasibility_tol, optimality_tol, *, enabled=True):
    """Render a cached final certificate with its separate acceptance limits.

    Metrics and multipliers use public signs and solver coordinates. Missing
    diagnostics remain unavailable; this formatter never estimates multipliers.
    """
    assessment = assessment or {}
    available = bool(enabled and assessment.get("available", False))
    sep = "-" * 80
    lines = [
        "",
        sep,
        f" {'Karush-Kuhn-Tucker (KKT) conditions':<36}{'Value':>16}{'Target':>16}{'Ok?':>10}",
        sep,
    ]
    rows = [
        ("First order optimality", "optimality", optimality_tol),
        ("Stationarity", "stationarity", optimality_tol),
        ("Equality feasibility", "equality_feasibility", feasibility_tol),
        ("Inequality feasibility", "inequality_feasibility", feasibility_tol),
        ("Bound feasibility", "bound_feasibility", feasibility_tol),
        ("Dual feasibility", "dual_feasibility", optimality_tol),
        ("Complementary slackness", "complementarity", optimality_tol),
    ]
    for name, key, limit in rows:
        value = assessment.get(key)
        finite = value is not None and np.isfinite(value)
        rendered = f"{value:+.3e}" if finite else "unavailable"
        status = ("yes" if value <= limit else "no") if finite else "n/a"
        lines.append(
            f" {name:<36}{rendered:>16}{'<= ' + f'{limit:+.3e}':>16}{status:>10}"
        )
    lines.append(" Coordinates: solver; stationarity norm: infinity.")
    lines.append(
        " Source: cached final assessment; no additional diagnostic evaluations."
    )
    if not enabled:
        lines.append(" Automatic KKT checks disabled; optimality not assessed.")
    elif not available:
        reason = (
            assessment.get("message") or "No final derivative certificate available."
        )
        lines.append(" Assessment unavailable: " + reason)
    else:
        lines.append(
            " Cached final KKT check: "
            + ("passed." if assessment.get("passed") else "failed.")
        )
    lines.append(sep)
    return "\n".join(lines)


def final_multipliers_report(
    assessment, constraint_data, var_names, bounds, *, enabled=True
):
    """Format cached public multipliers through the existing named table."""
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
    return multipliers_report(data, constraint_data, var_names, bounds)


def constraint_report(constraint_data):
    """
    generate a formatted constraint report at x_phys, using get_constraint_data to build/validate the entries.
    """
    max_name_width = 48

    # from .pysolver_utilities import print_dict
    # print("Data dictionary")
    # for item in constraint_data:
    #     print()
    #     print_dict(item)

    lines = []
    lines = []
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{' Optimization constraints report':<80}")
    lines.append("-" * 80)
    lines.append(f"{' Constraint name':<49}{'Value':>11}{'Target':>13}{'Ok?':>6}")
    lines.append("-" * 80)

    for entry in constraint_data:
        name = entry.get("name", "")
        ctype = entry.get("type", "")
        target = entry.get("target", 0.0)
        value = entry.get("value", 0.0)
        satisfied = "yes" if entry.get("satisfied", False) else "no"

        if len(name) > max_name_width:
            name = "..." + name[-(max_name_width - 3) :]

        symbol_target = f"{ctype} {target:+.2e}"
        lines.append(f" {name:<48}{value:>+11.2e}{symbol_target:>13}{satisfied:>6}")

    lines.append("-" * 80)
    return "\n".join(lines)


def multipliers_report(kkt_data, constraint_data, var_names, bounds):
    """
    Generate a report of all Lagrange multipliers:
    - Equalities: always included
    - Inequalities and finite bounds: all independently estimated multipliers.
    - Unavailable estimates: NaN with an explicit reason.

    Returns
    -------
    str
        The formatted multipliers report as a single string.
    """
    # Compute KKT data with given tolerance
    eq_data = [c for c in constraint_data if c["type"] == "="]
    ineq_data = [c for c in constraint_data if c["type"] in {"<", ">"}]

    # Print Lagrange multipliers in tabular format
    sep = "-" * 80
    max_name_width = 48
    lines = [
        "",
        sep,
        " Lagrange multipliers report (estimated, solver coordinates)",
        sep,
        f" {'Multiplier for constraint':<{max_name_width}}{'Type':>15}{'Value':>15}",
        sep,
    ]

    # 1. Equality constraints
    for i, entry in enumerate(eq_data):
        name = entry["name"]
        ctype = "equality"
        lam = kkt_data["multipliers_eq"][i]
        valstr = f"{lam:+.3e}"
        if len(name) > max_name_width:
            name = "..." + name[-(max_name_width - 3) :]
        lines.append(f" {name:<{max_name_width}}{ctype:>15}{valstr:>15}")

    # 2. Inequality + bound constraints
    n_vars = len(var_names)
    n_ineq = len(ineq_data)
    total_ineq = n_ineq + 2 * n_vars
    active = set(kkt_data["multipliers_ineq"].keys())

    for j in range(total_ineq):
        if j < n_ineq:
            name = ineq_data[j]["name"]
            ctype = "inequality"
        elif j < n_ineq + n_vars:
            idx = j - n_ineq
            name = var_names[idx]
            ctype = "lower bound"
        else:
            idx = j - n_ineq - n_vars
            name = var_names[idx]
            ctype = "upper bound"

        if len(name) > max_name_width:
            name = "..." + name[-(max_name_width - 3) :]

        if j >= n_ineq:
            bound = bounds
            index = j - n_ineq if j < n_ineq + n_vars else j - n_ineq - n_vars
            side = 0 if j < n_ineq + n_vars else 1
            if bound is None or not np.isfinite(bound[side][index]):
                continue
        if j in active:
            valstr = f"{kkt_data['multipliers_ineq'][j]:+.3e}"
        else:
            valstr = "inactive"

        lines.append(f" {name:<{max_name_width}}{ctype:>15}{valstr:>15}")

    if not kkt_data["available"]:
        lines.append(" Assessment unavailable: " + kkt_data.get("message", ""))
    lines.append(sep)
    return "\n".join(lines)
