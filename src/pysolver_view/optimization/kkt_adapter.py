"""Translate same-point optimization data into the shared KKT mathematics."""

import copy
import time
import numpy as np
from ..diagnostics.kkt import assess_kkt


def kkt_from_cache(self, x_norm, tol, *, optimality_tol=None):
    """Assess public-sign KKT residuals using only same-point cached arrays.

    The legacy dual-feasibility field remains a signed minimum multiplier.
    Shared diagnostics additionally expose the nonnegative dual residual and
    infinity-norm stationarity. Multiplier indices are user inequalities,
    then all lower-bound slots, then all upper-bound slots.
    """
    point = np.asarray(x_norm, dtype=float)
    if not (
        np.array_equal(point, self.cache["x"])
        and np.array_equal(point, self.cache["x_jac"])
    ):
        raise ValueError(
            "KKT assessment requires values and derivatives at the same point."
        )
    optimality_tol = tol if optimality_tol is None else optimality_tol
    tolerances = (tol, optimality_tol)
    if self._kkt_cache is not None and self._kkt_cache[0] == tolerances:
        return copy.deepcopy(self._kkt_cache[1])
    bounds = self.problem.get_bounds_normalized()
    lower, upper = (
        (np.full(point.size, -np.inf), np.full(point.size, np.inf))
        if bounds is None
        else map(np.asarray, bounds)
    )
    started_check = time.perf_counter()
    assessment = assess_kkt(
        point,
        self.cache["f_jac"],
        self.cache["c_eq"],
        self.cache["c_eq_jac"],
        self.cache["c_ineq"],
        self.cache["c_ineq_jac"],
        lower,
        upper,
        feasibility_tol=tol,
        optimality_tol=optimality_tol,
    )
    self.kkt_time += time.perf_counter() - started_check
    self.kkt_count += 1
    assessment["coordinates"] = "solver"
    assessment["physical_x"] = self.problem.scale_normalized_to_physical(point).copy()
    assessment["problem_scale"] = self.problem.problem_scale
    assessment["derivative_source"] = (
        "analytic" if hasattr(self.problem, "gradient") else self.derivative_method
    )
    inequalities = np.r_[self.cache["c_ineq"], lower - point, point - upper]
    finite = np.r_[
        np.ones(self.N_ineq, dtype=bool), np.isfinite(lower), np.isfinite(upper)
    ]
    if assessment["available"]:
        multipliers = assessment["multipliers"]
        duals = np.r_[
            multipliers["inequality"], multipliers["lower"], multipliers["upper"]
        ]
        signed_dual = float(np.min(duals[finite])) if np.any(finite) else 0.0
        selected = {i: float(duals[i]) for i in np.flatnonzero(finite)}
        products = (duals[finite] * inequalities[finite]).tolist()
        equality_multipliers = multipliers["equality"].copy()
    else:
        signed_dual = float("nan")
        selected = {i: float("nan") for i in np.flatnonzero(finite)}
        products = []
        equality_multipliers = np.full(self.N_eq, np.nan)
    first = assessment["stationarity"]
    first = float("nan") if first is None else first
    complementarity = assessment["complementarity"]
    complementarity = float("nan") if complementarity is None else complementarity
    equality_error = float(np.max(np.abs(self.cache["c_eq"]), initial=0))
    inequality_error = float(np.max(np.maximum(inequalities[finite], 0), initial=0))
    result = {
        **assessment,
        "equality_violation": self.cache["c_eq"].copy(),
        "inequality_violation": inequalities,
        "multipliers_eq": equality_multipliers,
        "multipliers_ineq": selected,
        "slack_products": products,
        "first_order_optimality": first,
        "feasibility_equality": equality_error,
        "feasibility_inequality": inequality_error,
        "feasibility_dual": signed_dual,
        "complementary_slackness": complementarity,
        "first_order_optimality_ok": bool(first <= optimality_tol),
        "feasibility_equality_ok": bool(equality_error <= tol),
        "feasibility_inequality_ok": bool(inequality_error <= tol),
        "feasibility_dual_ok": bool(signed_dual >= -optimality_tol),
        "complementary_slackness_ok": bool(complementarity <= optimality_tol),
    }
    self._kkt_cache = tolerances, copy.deepcopy(result)
    return result


def public_driver_kkt(assessment):
    """Detach driver diagnostics and convert equality multipliers to public signs."""
    if assessment is None:
        return None
    public = copy.deepcopy(assessment)
    if public.get("multipliers") is not None:
        public["multipliers"]["equality"] *= -1
    if public.get("complementarity_products") is not None:
        public["complementarity_products"]["inequality"] *= -1
    public["coordinates"] = "solver"
    public["convention"] = "h=0, g<=0; L=f+lambda*h+mu*g-zL*(x-l)+zU*(x-u)"
    return public
