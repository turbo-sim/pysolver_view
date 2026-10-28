"""Per-run optimization evaluation and backend callback contracts."""

import copy
import numpy as np
from .. import numerical_differentiation
from ..exceptions import DomainError


class OptimizationEvaluation:
    """Own model work, point caches and derivative accounting for one solve."""

    def __init__(
        self,
        problem,
        *,
        library,
        options,
        derivative_method,
        derivative_abs_step,
        fitness,
    ):
        self.problem = problem
        self.library = library
        self.options = options
        self.derivative_method = derivative_method
        self.derivative_abs_step = derivative_abs_step
        self.N_eq = problem.get_nec()
        self.N_ineq = problem.get_nic()
        self.fitness = fitness
        self.derivative_func_count = 0
        self.diagnostic_func_count = 0
        self.diagnostic_grad_count = 0
        self.failed_grad_count = 0
        self._kkt_cache = None
        self.kkt_count = 0
        self.kkt_time = 0.0
        self.grad_count = 0
        self.func_count = 0
        self.func_count_tot = 0
        self.accepted_constraint_metadata = None
        self.cache = {
            "x": None,
            "f": None,
            "c_eq": None,
            "c_ineq": None,
            "x_jac": None,
            "f_jac": None,
            "c_eq_jac": None,
            "c_ineq_jac": None,
            "fitness": None,
            "gradient": None,
            "constraint_data": None,
            "has_constraint_data": False,
        }

    def _evaluate_fitness(
        self,
        x_norm,
        *,
        derivative=False,
        diagnostic=False,
        use_cache=True,
        cache_result=True,
    ):
        """Obtain a detached same-point value without recording or calling observers.

        Exact keys protect tiny representable steps and complex-step probes.
        ``tolerance_check_cache`` is retained as an input for compatibility but
        is not used to substitute values at a different point.
        """
        point = np.asarray(x_norm).copy()
        if (
            use_cache
            and self.cache["x"] is not None
            and np.array_equal(point, self.cache["x"])
        ):
            return self.cache["fitness"].copy()
        self.func_count_tot += 1
        if derivative:
            self.derivative_func_count += 1
        elif not diagnostic:
            self.func_count += 1
        if diagnostic:
            self.diagnostic_func_count += 1
        values = np.atleast_1d(
            np.asarray(self.problem.fitness_normalized_input(point.copy()))
        ).copy()
        if values.shape != (1 + self.N_eq + self.N_ineq,):
            raise ValueError(
                "Fitness must contain the objective and declared constraints."
            )
        if cache_result:
            self._kkt_cache = None
            self.cache.update(
                x=point,
                f=values[0],
                c_eq=values[1 : 1 + self.N_eq].copy(),
                c_ineq=values[1 + self.N_eq :].copy(),
                fitness=values.copy(),
                constraint_data=copy.deepcopy(
                    getattr(self.problem, "constraint_data", None)
                ),
                has_constraint_data=hasattr(self.problem, "constraint_data"),
            )
        return values

    def record_accepted_metadata(self, x):
        """Detach named metadata only when it belongs to the accepted point."""
        point = np.asarray(x).copy()
        if self.cache["x"] is not None and np.array_equal(point, self.cache["x"]):
            self.accepted_constraint_metadata = dict(
                x=point,
                data=copy.deepcopy(self.cache["constraint_data"]),
                present=self.cache["has_constraint_data"],
            )

    def final_constraint_metadata(self, x):
        """Return detached final metadata without evaluating the model."""
        point = np.asarray(x)
        accepted = self.accepted_constraint_metadata
        if accepted is not None and np.array_equal(point, accepted["x"]):
            return copy.deepcopy(accepted["data"]), bool(accepted["present"])
        if self.cache["x"] is not None and np.array_equal(point, self.cache["x"]):
            return copy.deepcopy(self.cache["constraint_data"]), bool(
                self.cache["has_constraint_data"]
            )
        return None, False

    def _evaluate_gradient(self, x_norm, *, diagnostic=False):
        """Compute/cache a full Jacobian without iteration events or probe callbacks."""
        point = np.asarray(x_norm, dtype=float).copy()
        if self.cache["x_jac"] is not None and np.array_equal(
            point, self.cache["x_jac"]
        ):
            return self.cache["gradient"].copy()
        self.grad_count += 1
        if diagnostic:
            self.diagnostic_grad_count += 1
        try:
            # The unperturbed base is an ordinary model point even when SciPy
            # requests its Jacobian first. Only perturbed samples are probes.
            base = (
                self._evaluate_fitness(point, diagnostic=True)
                if diagnostic
                else self.fitness(point)
            )
            if hasattr(self.problem, "gradient"):
                grad = self.problem.gradient_normalized_input(point.copy())
            else:

                def evaluate_probe(probe):
                    """Count derivative model work without disturbing the base snapshot."""
                    if np.array_equal(probe, point):
                        return base.copy()
                    try:
                        return self._evaluate_fitness(
                            probe,
                            derivative=True,
                            diagnostic=diagnostic,
                            use_cache=False,
                            cache_result=False,
                        )
                    except DomainError as error:
                        if self.library != "pysolver":
                            raise
                        changed = np.flatnonzero(probe != point)
                        raise DomainError(
                            "Derivative probe failed; probes are not retried: "
                            + str(error),
                            metadata={
                                "probe": np.real(probe).tolist(),
                                "column": (
                                    int(changed[0]) if changed.size == 1 else None
                                ),
                                "cause": error.to_dict(),
                            },
                        ) from error

                if self.library in {"pysolver", "pounce"}:
                    methods = {
                        "forward_finite_differences": "2-point",
                        "central_finite_differences": "3-point",
                        "complex_step": "cs",
                    }
                    bounds = self.problem.get_bounds_normalized()
                    grad = numerical_differentiation.approx_derivative(
                        evaluate_probe,
                        point,
                        f0=base,
                        method=methods.get(
                            self.derivative_method, self.derivative_method
                        ),
                        abs_step=(
                            self.derivative_abs_step
                            if self.derivative_abs_step is not None
                            else self.options.get("finite_diff_step", 1e-7)
                        ),
                        bounds=(-np.inf, np.inf) if bounds is None else bounds,
                    )
                else:
                    grad = numerical_differentiation.approx_gradient(
                        evaluate_probe,
                        point,
                        f0=base,
                        method=self.derivative_method,
                        abs_step=self.derivative_abs_step,
                    )
            grad = (
                np.asarray(grad).reshape(1 + self.N_eq + self.N_ineq, point.size).copy()
            )
        except Exception:
            self.failed_grad_count += 1
            raise
        self._kkt_cache = None
        self.cache.update(
            x_jac=point,
            f_jac=grad[0].copy(),
            c_eq_jac=grad[1 : 1 + self.N_eq].copy(),
            c_ineq_jac=grad[1 + self.N_eq :].copy(),
            gradient=grad.copy(),
        )
        return grad


class BackendProblem:
    """
    Generic backend callbacks for objective, constraints, bounds and derivatives.
    This class uses anonymous functions (lambda) to prevent issues with deep copying complex objects,
    (like Coolprop's AbstractState objects) which are not deep-copiable.
    """

    def __init__(self, wrapped_problem):
        self.check_kkt = wrapped_problem.check_kkt
        self.iteration_callback = lambda x: wrapped_problem._record_iteration(x)
        self.accepted_callback = lambda entry: wrapped_problem._record_custom_iteration(
            entry
        )
        self.native_callback = lambda entry: wrapped_problem._record_pounce_iteration(
            entry
        )
        self.fitness = lambda x: wrapped_problem.fitness(x)
        self.gradient = lambda x: wrapped_problem.gradient(x)

        # Directly link bounds and constraint counts from the original problem.
        self.get_bounds = lambda: wrapped_problem.problem.get_bounds_normalized()
        self.get_nec = lambda: wrapped_problem.problem.get_nec()
        self.get_nic = lambda: wrapped_problem.problem.get_nic()

        # If the original problem defines Hessians, provide them as well.
        if hasattr(wrapped_problem.problem, "hessians"):
            self.hessians = lambda x: wrapped_problem.problem.hessians(x)

        # Define anonymous functions for objective and constraints with their Jacobians.
        self.f = lambda x: wrapped_problem.fitness(x)[0]
        self.c_eq = lambda x: wrapped_problem.fitness(x)[1 : 1 + self.get_nec()]
        self.c_ineq = lambda x: wrapped_problem.fitness(x)[1 + self.get_nec() :]

        self.f_jac = lambda x: wrapped_problem.gradient(x)[0, :]
        self.c_eq_jac = lambda x: wrapped_problem.gradient(x)[1 : 1 + self.get_nec(), :]
        self.c_ineq_jac = lambda x: wrapped_problem.gradient(x)[1 + self.get_nec() :, :]
