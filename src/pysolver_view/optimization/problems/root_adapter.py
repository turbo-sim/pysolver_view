"""Present existing residual equations as a zero-objective feasibility problem."""

from copy import deepcopy
from numbers import Integral

import numpy as np

from .base import OptimizationProblem


def as_optimization_problem(problem, *, n_variables, n_equations=None, bounds=None):
    r"""Adapt a nonlinear equation system to equality-constrained optimization.

    Return a new :class:`OptimizationProblem` representing
    ``minimize 0 subject to problem.residual(x) == 0``. Every residual becomes
    one equality constraint, in its original order and units. This lets an
    existing root-finding model use an equality-capable ``OptimizationSolver``
    backend without changing the model class or the root/optimization contracts.
    It does not minimize the sum of squared residuals.

    Parameters
    ----------
    problem : NonlinearSystemProblem or object with a callable residual
        Existing model exposing ``residual(x)`` with shape ``(n_equations,)``.
        A scalar residual is also accepted for a single equation. An optional
        callable ``gradient(x)`` supplies the dense residual Jacobian with
        shape ``(n_equations, n_variables)``. For a single equation, a gradient
        of shape ``(n_variables,)`` is accepted, as is a scalar for one variable.
        The adapter keeps a reference to this model; it does not copy the model
        or assign optimization attributes to it.
    n_variables : int
        Positive number of decision variables. Explicit dimensions avoid a
        preliminary model evaluation and allow conversion before choosing an
        initial guess.
    n_equations : int, optional
        Positive number of residual equations. Defaults to ``n_variables`` for
        square systems. Specify it explicitly for rectangular systems. Shapes
        are checked whenever the residual or analytic Jacobian is requested.
    bounds : tuple of array_like, optional
        ``(lower, upper)`` variable bounds in the original model coordinates,
        following the package's optimization convention. Each side may be a
        scalar or a vector of length ``n_variables``. Omission gives unbounded
        variables. Bounds are copied and must satisfy ``lower <= upper``;
        NaN, positive-infinite lower, and negative-infinite upper bounds are
        rejected. No bounds are inferred from the wrapped model.

    Returns
    -------
    OptimizationProblem
        Wrapper whose ``fitness(x)`` returns ``[0, *problem.residual(x)]``,
        ``get_nec()`` returns ``n_equations``, and ``get_nic()`` returns zero.
        If the model provides an analytic Jacobian, ``gradient(x)`` returns
        that Jacobian with a leading zero objective-gradient row. Otherwise
        the wrapper has no ``gradient`` method, allowing the optimizer's
        existing numerical differentiation and probe accounting to operate.

    Raises
    ------
    TypeError
        If ``problem.residual`` is not callable.
    ValueError
        If dimensions or bounds are invalid, or a later model evaluation has
        an incompatible shape.

    Notes
    -----
    Conversion performs no residual or Jacobian evaluations. Each subsequent
    ``fitness`` request makes exactly one residual call, and each analytic
    ``gradient`` request makes exactly one Jacobian call, without evaluating
    residuals. There is no adapter cache or hidden numerical differentiation.
    Consequently optimizer model calls correspond directly to wrapped residual
    calls, including numerical probes. Work inside the model's own Jacobian is
    still outside the optimizer's observable residual-call accounting.

    Input and returned arrays are detached. Complex perturbations are preserved
    for complex-step differentiation. ``DomainError`` and other model exceptions
    propagate unchanged; the selected optimizer determines whether recovery is
    supported. Mutable internal model state remains owned by the caller.

    No residual weighting or variable rescaling is added. Optional
    ``variable_names`` are copied, and optimizer scaling belongs to the wrapper,
    leaving attributes on the original model unchanged. The wrapper initially
    uses physical coordinates (``problem_scale=None``). If bound-based scaling
    is enabled on ``OptimizationSolver``, its existing scaling rules still apply.

    A zero objective gives no preference between feasible roots. Its zero
    gradient is not evidence that the equations are solved: check the final
    original residual against an explicit root tolerance independently of the
    optimizer's status. Residual norms and equality-feasibility criteria can use
    different norms/tolerances. An inconsistent system remains infeasible.

    Backend restrictions still apply. In particular, SLSQP cannot handle more
    equalities than variables and can fail on dependent equality Jacobian rows.
    The custom SLSQP backend also requires fixed variables to be eliminated.
    The adapter represents rectangular systems but does not remove redundant
    equations or select a numerical solver for them.

    Examples
    --------
    Convert a two-equation model without evaluating it during construction:

    >>> class Equations:
    ...     def residual(self, x):
    ...         return np.array([x[0] ** 2 - 1, x[1] - 2])
    ...     def gradient(self, x):
    ...         return np.array([[2 * x[0], 0], [0, 1]])
    >>> adapted = as_optimization_problem(Equations(), n_variables=2)
    >>> adapted.fitness(np.array([1.0, 2.0]))
    array([0., 0., 0.])

    Pass ``adapted`` to ``OptimizationSolver`` with ``method="slsqp"`` and an
    appropriate backend, then independently evaluate the original residual at
    the returned point. Such a post-solve audit is outside solve call counters.
    """
    if not callable(getattr(problem, "residual", None)):
        raise TypeError("problem must provide a callable residual(x).")
    n_variables = _positive_dimension(n_variables, "n_variables")
    n_equations = _positive_dimension(
        n_variables if n_equations is None else n_equations, "n_equations"
    )
    wrapper_type = (
        _DifferentiableRootEqualityProblem
        if callable(getattr(problem, "gradient", None))
        else _RootEqualityProblem
    )
    return wrapper_type(problem, n_variables, n_equations, bounds)


def _positive_dimension(value, name):
    """Validate an explicit positive dimension without evaluating the model."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _copy_bounds(bounds, n_variables):
    """Copy real per-variable bounds while retaining legitimate infinite limits."""
    if bounds is None:
        return np.full(n_variables, -np.inf), np.full(n_variables, np.inf)
    try:
        lower, upper = bounds
    except (TypeError, ValueError) as error:
        raise ValueError("bounds must be a (lower, upper) pair.") from error
    copied = []
    for name, side in (("lower", lower), ("upper", upper)):
        side = np.asarray(side)
        if np.iscomplexobj(side):
            raise ValueError(f"{name} bounds must be real.")
        if side.ndim == 0:
            side = np.full(n_variables, side, dtype=float)
        elif side.shape == (n_variables,):
            side = np.array(side, dtype=float, copy=True)
        else:
            raise ValueError(
                f"{name} bounds must be scalar or have shape ({n_variables},)."
            )
        if np.any(np.isnan(side)):
            raise ValueError("bounds must not contain NaN.")
        copied.append(side)
    lower, upper = copied
    if (
        np.any(np.isposinf(lower))
        or np.any(np.isneginf(upper))
        or np.any(lower > upper)
    ):
        raise ValueError(
            "bounds must satisfy lower <= upper with admissible infinities."
        )
    return lower, upper


class _RootEqualityProblem(OptimizationProblem):
    """Own optimization metadata while delegating residual work to the root model."""

    def __init__(self, problem, n_variables, n_equations, bounds):
        """Capture declared geometry and detached metadata without model calls."""
        self.root_problem = problem
        self._n_variables = n_variables
        self._n_equations = n_equations
        self._bounds = _copy_bounds(bounds, n_variables)
        self.problem_scale = None
        if hasattr(problem, "variable_names"):
            self.variable_names = deepcopy(problem.variable_names)

    def _point(self, x):
        """Copy physical coordinates while preserving complex derivative probes."""
        point = np.array(x, copy=True)
        if point.shape != (self._n_variables,):
            raise ValueError(f"x must have shape ({self._n_variables},).")
        return point

    def fitness(self, x):
        """Return zero objective and unchanged equality residuals from one model call."""
        residual = np.atleast_1d(np.asarray(self.root_problem.residual(self._point(x))))
        if residual.shape != (self._n_equations,):
            raise ValueError(
                f"residual must have shape ({self._n_equations},); got {residual.shape}."
            )
        return np.concatenate((np.zeros(1, dtype=residual.dtype), residual))

    def get_bounds(self):
        """Return detached lower/upper bounds in the original variable coordinates."""
        return tuple(side.copy() for side in self._bounds)

    def get_nec(self):
        """Declare one equality per original residual component."""
        return self._n_equations

    def get_nic(self):
        """Declare no additional inequality constraints."""
        return 0


class _DifferentiableRootEqualityProblem(_RootEqualityProblem):
    """Expose analytic derivatives only for models that actually provide them."""

    def gradient(self, x):
        """Prepend the zero objective row to one dense residual-Jacobian call."""
        jacobian = np.asarray(self.root_problem.gradient(self._point(x)))
        if self._n_equations == 1:
            jacobian = np.atleast_2d(jacobian)
        shape = (self._n_equations, self._n_variables)
        if jacobian.shape != shape:
            raise ValueError(f"gradient must have shape {shape}; got {jacobian.shape}.")
        return np.vstack(
            (np.zeros((1, self._n_variables), dtype=jacobian.dtype), jacobian)
        )
