"""Adapt optional Pounce NLP solvers to package evaluations and observations."""

from copy import deepcopy

import numpy as np
from scipy.optimize import OptimizeResult

from ...exceptions import DomainError


class _PounceCallbacks:
    """Keep native observations separate from model calls and derivative probes.

    Pounce's Python intermediate callback supplies scalar metrics without the
    primal vector. Ordinary observations therefore carry the point most recently
    passed to ``objective``, following Pounce's own ``minimize`` facade. This is
    an observation source, not an independent guarantee of step acceptance.
    Restoration observations have no user-space point or objective attached.
    """

    def __init__(self, problem, dimension, equality_count, inequality_count):
        """Initialize detached per-run values and delayed application errors."""
        self.problem = problem
        self.dimension = dimension
        self.equality_count = equality_count
        self.constraint_count = equality_count + inequality_count
        self.last_objective_x = None
        self.last_objective_fitness = None
        self.application_error = None
        self.history = []
        self.objective_calls = 0
        self.gradient_calls = 0

    def _remember_error(self, error):
        """Retain the first application error that Pounce would otherwise hide."""
        if self.application_error is None:
            self.application_error = error

    def objective(self, x):
        """Read combined fitness once and remember only actual objective points."""
        self.objective_calls += 1
        try:
            values = np.asarray(self.problem.fitness(x)).copy()
            objective_value = float(values[0])
        except DomainError:
            # Raising signals native evaluation failure and permits line-search
            # recovery without inventing values at an undefined physical point.
            raise
        except BaseException as error:
            self._remember_error(error)
            raise
        self.last_objective_x = np.asarray(x, dtype=float).copy()
        self.last_objective_fitness = values.copy()
        return objective_value

    def gradient(self, x):
        """Use the frontend's analytic or accounted numerical objective gradient."""
        self.gradient_calls += 1
        try:
            matrix = np.asarray(self.problem.gradient(x)).reshape(
                1 + self.constraint_count, self.dimension
            )
            return matrix[0].copy()
        except BaseException as error:
            # Derivative probes are not recoverable model-domain trials.
            self._remember_error(error)
            raise

    def constraints(self, x):
        """Return equality rows then nonpositive inequality rows without reversal."""
        try:
            return np.asarray(self.problem.fitness(x))[1:].copy()
        except DomainError:
            raise
        except BaseException as error:
            self._remember_error(error)
            raise

    def jacobianstructure(self):
        """Declare a dense row-major constraint Jacobian in zero-based indices."""
        return (
            np.repeat(np.arange(self.constraint_count), self.dimension),
            np.tile(np.arange(self.dimension), self.constraint_count),
        )

    def jacobian(self, x):
        """Flatten frontend constraint-Jacobian rows in the declared dense order."""
        try:
            matrix = np.asarray(self.problem.gradient(x)).reshape(
                1 + self.constraint_count, self.dimension
            )
            return matrix[1:].ravel().copy()
        except BaseException as error:
            self._remember_error(error)
            raise

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """Publish detached native metrics without performing hidden model work.

        ``obj_value``, ``inf_pr``, and ``inf_du`` retain the native internal
        scaling. The separately stored ``fitness`` uses frontend solver-space
        units. Restoration metrics describe a different feasibility subproblem
        and deliberately have no reconstructed user-space coordinates.
        """
        if self.application_error is not None:
            return False
        regular_point = int(alg_mod) == 0 and self.last_objective_x is not None
        entry = {
            "iteration": int(iter_count),
            "event_kind": ("native_iteration" if int(alg_mod) == 0 else "restoration"),
            "accepted": None,
            "coordinate_source": (
                "last_objective_evaluation" if regular_point else "unavailable"
            ),
            "x": self.last_objective_x.copy() if regular_point else None,
            "fitness": (self.last_objective_fitness.copy() if regular_point else None),
            "native_metrics": {
                "alg_mod": int(alg_mod),
                "iter_count": int(iter_count),
                "obj_value": float(obj_value),
                "inf_pr": float(inf_pr),
                "inf_du": float(inf_du),
                "mu": float(mu),
                "d_norm": float(d_norm),
                "regularization_size": float(regularization_size),
                "alpha_du": float(alpha_du),
                "alpha_pr": float(alpha_pr),
                "ls_trials": int(ls_trials),
            },
        }
        self.history.append(deepcopy(entry))
        callback = getattr(self.problem, "native_callback", None)
        if callback is None:
            return True
        try:
            return callback(deepcopy(entry)) is not False
        except StopIteration:
            return False
        except BaseException as error:
            self._remember_error(error)
            return False


def minimize_pounce(problem, x0, method, solver_options):
    """Solve a smooth constrained NLP through Pounce's cyipopt-style interface.

    Parameters
    ----------
    problem : BackendProblem
        Frontend evaluation callbacks, normalized bounds, constraint counts,
        and optional ``native_callback`` observation sink. Inequalities are
        nonpositive. Derivative evaluations and caching belong to this object.
    x0 : array_like, shape (n,)
        Initial point in the coordinates supplied by the frontend.
    method : {"ipm", "sqp"}
        Select the interior-point or active-set SQP NLP algorithm.
    solver_options : dict
        Generic ``tolerance`` and ``max_iterations`` plus Pounce native options.
        Defaults disable Pounce scaling and bound relaxation because the
        frontend already owns problem scaling and declared-bound diagnostics.
        Only first derivatives are supplied, so exact Lagrangian Hessian
        requests are rejected rather than silently changed by the native solver.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Authoritative native returned point, strict native success, detached
        observation history and diagnostics. ``solver_success`` additionally
        identifies acceptable-level native termination. Multipliers retain
        Pounce's ``grad(f) + J.T @ lambda - z_lower + z_upper`` convention in
        the frontend's solver coordinates; no physical rescaling occurs here.
    """
    if method not in {"ipm", "sqp"}:
        raise ValueError("The pounce backend supports only 'ipm' and 'sqp'.")
    configured = dict(solver_options)
    tolerance = configured.pop("tolerance")
    max_iterations = configured.pop("max_iterations")
    unsupported = sorted(
        {"first_step_fraction", "max_step_fraction", "step_growth_factor"}
        & configured.keys()
    )
    if unsupported:
        raise ValueError(
            "Pounce's Python interface does not expose initial or maximum step "
            "fraction controls equivalent to custom SLSQP; unsupported options: "
            + ", ".join(unsupported)
        )
    algorithm = "interior-point" if method == "ipm" else "active-set-sqp"
    requested_algorithm = configured.pop("algorithm", algorithm)
    if requested_algorithm != algorithm:
        raise ValueError(
            f"method={method!r} requires algorithm={algorithm!r}; "
            f"received {requested_algorithm!r}."
        )
    solver_selection = configured.pop("solver_selection", "nlp")
    if solver_selection != "nlp":
        raise ValueError(
            "The pounce backend uses the general NLP interface; "
            "solver_selection must be 'nlp'."
        )
    if any(
        str(configured.get(name, "")).lower() == "exact"
        for name in ("hessian_approximation", "sqp_hessian")
    ):
        raise ValueError(
            "The pounce wrapper currently supplies first derivatives only and "
            "uses quasi-Newton Hessians; exact Lagrangian Hessians are not "
            "supported."
        )
    options = {
        "algorithm": algorithm,
        "tol": tolerance,
        "max_iter": int(max_iterations),
        "print_level": 0,
        "nlp_scaling_method": "none",
        "bound_relax_factor": 0.0,
        "hessian_approximation": "limited-memory",
    }
    if method == "sqp":
        options.update(
            sqp_hessian="damped-bfgs",
            sqp_tol=tolerance,
            sqp_constr_viol_tol=tolerance,
            sqp_dual_inf_tol=tolerance,
            sqp_max_iter=int(max_iterations),
        )
    options.update(configured)
    try:
        import pounce
    except ImportError as error:
        raise ImportError(
            "Pounce requires the optional pounce-solver dependency; install "
            "'pysolver_view[pounce]' or 'pounce-solver==0.13.0'."
        ) from error

    initial = np.asarray(x0, dtype=float).copy()
    equality_count = int(problem.get_nec())
    inequality_count = int(problem.get_nic())
    constraint_count = equality_count + inequality_count
    callbacks = _PounceCallbacks(
        problem, initial.size, equality_count, inequality_count
    )
    bounds = problem.get_bounds()
    lower, upper = (
        (np.full(initial.size, -np.inf), np.full(initial.size, np.inf))
        if bounds is None
        else (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
    )
    native = pounce.Problem(
        n=initial.size,
        m=constraint_count,
        problem_obj=callbacks,
        lb=lower,
        ub=upper,
        cl=np.r_[np.zeros(equality_count), np.full(inequality_count, -np.inf)],
        cu=np.zeros(constraint_count),
    )
    for name, value in options.items():
        native.add_option(name, value)
    try:
        x, native_info = native.solve(x0=initial)
    except BaseException:
        if callbacks.application_error is not None:
            raise callbacks.application_error
        raise
    if callbacks.application_error is not None:
        raise callbacks.application_error

    point = np.asarray(x, dtype=float).copy()
    info = deepcopy(dict(native_info))
    status = int(info["status"])
    fitness = None
    if callbacks.last_objective_x is not None and np.array_equal(
        callbacks.last_objective_x, point
    ):
        fitness = callbacks.last_objective_fitness.copy()
    constraint_multipliers = np.asarray(
        info.get("mult_g", np.full(constraint_count, np.nan)), dtype=float
    ).copy()
    native_multipliers = {
        "equality": constraint_multipliers[:equality_count].copy(),
        "inequality": constraint_multipliers[equality_count:].copy(),
        "lower": np.asarray(
            info.get("mult_x_L", np.full(initial.size, np.nan)), dtype=float
        ).copy(),
        "upper": np.asarray(
            info.get("mult_x_U", np.full(initial.size, np.nan)), dtype=float
        ).copy(),
    }
    return OptimizeResult(
        x=point,
        fun=float(info.get("obj_val", np.nan)),
        fitness=fitness,
        success=status == 0,
        solver_success=status in (0, 1),
        status=status,
        message=str(info.get("status_msg", status)),
        nit=int(info.get("iter_count", 0)),
        nfev=int(info.get("n_obj_evals", callbacks.objective_calls)),
        njev=int(info.get("n_grad_evals", callbacks.gradient_calls)),
        history=deepcopy(callbacks.history),
        info=info,
        native_multipliers=native_multipliers,
        effective_options=deepcopy(options),
    )
