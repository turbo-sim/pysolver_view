"""Public custom-root contracts: accepted states, physical calls, controls and reuse."""

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pysolver_view as psv
from pysolver_view.solvers.root import DomainError as RootDomainError

METHODS = ("levenberg_marquardt", "powell_hybrid", "newton_raphson", "gauss_newton")


class CountedPolynomial(psv.NonlinearSystemProblem):
    """A coupled two-variable system with observable evaluations and root (1, 2)."""

    def __init__(self):
        """Retain every actual residual input, including complex derivative probes."""
        self.calls = []

    def residual(self, x):
        """Evaluate polynomial equations without discarding complex perturbations."""
        self.calls.append(x.copy())
        return np.array([x[0] ** 2 + x[1] - 3, x[0] + x[1] ** 2 - 5])


class AnalyticPolynomial(CountedPolynomial):
    """The same equations with an independently supplied analytic Jacobian."""

    def gradient(self, x):
        """Return exact first derivatives of both coupled equations."""
        return np.array([[2 * x[0], 1.0], [1.0, 2 * x[1]]])


def make_solver(problem, method, **kwargs):
    """Construct a quiet public custom solver with accepted-iteration observations."""
    return psv.NonlinearSystemSolver(
        problem,
        method=method,
        library="pysolver",
        print_convergence=False,
        update_on=kwargs.pop("update_on", "iteration"),
        **kwargs,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("derivative", ["2-point", "3-point", "cs"])
def test_public_numerical_derivatives_have_one_budgeted_owner(method, derivative):
    """Every numerical probe counts once and cannot become an accepted iteration."""
    problem = CountedPolynomial()
    seen = []
    solver = make_solver(
        problem,
        method,
        tolerance=1e-9,
        derivative_method=derivative,
        derivative_abs_step=1e-5 if derivative != "cs" else 1e-20,
        callback_func=lambda x, count: seen.append((x.copy(), count)),
    )
    x = solver.solve([0.8, 1.5])
    assert solver.success
    np.testing.assert_allclose(x, [1, 2], atol=1e-8)
    result = solver.result
    assert result.nfev == solver.func_count_tot == len(problem.calls)
    assert result.nfev == solver.func_count + solver.derivative_func_count
    assert result.derivative_nfev == result.njev * (4 if derivative == "3-point" else 2)
    assert solver.iteration_count == result.nit == len(seen) - 1
    assert len(solver.iteration_history) == result.nit + 1
    assert solver.convergence_history["iteration"] == list(range(result.nit + 1))
    assert solver.result.library == "pysolver"


@pytest.mark.parametrize("method", METHODS)
def test_options_controls_and_public_results_match_packaged_core(method):
    """Explicit options win over facade defaults and preserve step limits and counts."""
    problem = AnalyticPolynomial()
    options = dict(residual_tol=1e-10, max_iterations=200, max_nfev=500, x_scale=[2, 3])
    if method in METHODS[:2]:
        options.update(initial_radius=0.05, max_radius=0.1)
    else:
        options["backtracking"] = dict(
            step_limits=dict(initial_limit=0.05, max_limit=0.1)
        )
    low = psv.solve_root(
        problem.residual, [0.8, 1.5], problem.gradient, method=method, options=options
    )
    problem.calls.clear()
    solver = make_solver(
        problem, method, tolerance=1.0, max_iterations=1, options=options
    )
    solver.solve([0.8, 1.5])
    assert solver.success
    assert solver.result.residual_tolerance == 1e-10
    for field in ("nfev", "njev", "nit", "ntrials", "ninvalid", "nupdates"):
        assert solver.result[field] == low[field]
    np.testing.assert_array_equal(solver.result.x, low.x)
    for row in solver.result.trials:
        assert np.linalg.norm((row["x"] - row["x_base"]) / [2, 3]) <= 0.1 * (1 + 1e-14)


@pytest.mark.parametrize("stop", ["return", "raise"])
def test_callbacks_only_receive_accepted_states_and_can_cancel(stop):
    """Even function observations never expose trial/probe points to custom callbacks."""
    seen = []

    def callback(x, count):
        """Stop after two accepted moves while trying to mutate the supplied copy."""
        seen.append(x.copy())
        x[:] = 999
        if count == 3:
            if stop == "raise":
                raise StopIteration
            return False

    solver = make_solver(
        CountedPolynomial(),
        "newton",
        update_on="function",
        callback_func=callback,
        options=dict(
            backtracking=dict(step_limits=dict(initial_limit=0.01, max_limit=0.01))
        ),
    )
    solver.solve([0.8, 1.5])
    assert not solver.success and solver.result.status == 5
    assert solver.result.nit == 2 and len(seen) == 3
    np.testing.assert_array_equal(solver.result.x, seen[-1])
    for observed, row in zip(seen, solver.iteration_history):
        np.testing.assert_array_equal(observed, row["x"])


@pytest.mark.parametrize("method", METHODS)
def test_probe_budget_exhaustion_returns_last_accepted_point(method):
    """A partial Jacobian cannot exceed the model budget or create a trial/iteration."""
    problem = CountedPolynomial()
    solver = make_solver(problem, method, options={"max_nfev": 2})
    solver.solve([0.8, 1.5])
    assert not solver.success
    assert solver.result.nfev == len(problem.calls) == 2
    assert solver.result.derivative_nfev == 1
    assert solver.result.njev == solver.result.failed_grad_count == 1
    assert solver.result.nit == solver.result.ntrials == 0
    np.testing.assert_array_equal(solver.result.x, [0.8, 1.5])


def test_domain_recovery_and_undefined_initial_state_use_shared_exception():
    """Property failures recover only for trials; initial failures invent no residual."""
    assert RootDomainError is psv.DomainError

    class PositiveProblem(psv.NonlinearSystemProblem):
        """Coupled logarithmic model with a restricted positive domain."""

        def residual(self, x):
            """Reject undefined states explicitly before taking logarithms."""
            if np.any(x <= 0):
                raise psv.DomainError(
                    "Positive state required", metadata={"source": "model"}
                )
            return np.array([[1.0, 0.2], [0.2, 1.0]]) @ np.log(x)

        def gradient(self, x):
            """Differentiate the coupled logarithmic residual analytically."""
            return np.array([[1.0, 0.2], [0.2, 1.0]]) / x

    solver = make_solver(
        PositiveProblem(), "newton", options=dict(backtracking=dict(step_limits=None))
    )
    solver.solve([10, 10])
    assert solver.success and solver.result.ninvalid > 0
    assert all(np.all(row["x"] > 0) for row in solver.iteration_history)
    solver.solve([-1, -1])
    assert not solver.success and solver.result.nfev == 1
    assert solver.result.fun is None and solver.f_final is None
    assert solver.iteration_history == []


@pytest.mark.parametrize("update_on", ["function", "gradient", "iteration"])
def test_history_modes_preserve_numerics_and_plotting_never_evaluates(update_on):
    """History storage can be disabled without changing work, and reuse resets counts."""
    results = []
    for level in ("full", "summary", "none"):
        problem = CountedPolynomial()
        solver = make_solver(
            problem, "hybr", update_on=update_on, options={"history_level": level}
        )
        solver.solve([0.8, 1.5])
        results.append(solver.result)
        calls = len(problem.calls)
        history = deepcopy(solver.convergence_history)
        fig = solver.plot_convergence_history(showfig=False)
        plt.close(fig)
        assert calls == len(problem.calls)
        np.testing.assert_equal(history, solver.convergence_history)
        if level == "none":
            assert not solver.iteration_history and not solver.convergence_history["x"]
        if level == "summary":
            assert all(x is None for x in solver.convergence_history["x"])
        solver.solve([0.8, 1.5])
        assert solver.result.nfev == calls
    for result in results[1:]:
        np.testing.assert_array_equal(result.x, results[0].x)
        assert (result.nfev, result.njev, result.nit) == (
            results[0].nfev,
            results[0].njev,
            results[0].nit,
        )


def test_native_default_and_author_aliases_remain_native():
    """Adding a keyword-only backend does not change old positional constructor usage."""
    native = psv.NonlinearSystemSolver(
        AnalyticPolynomial(), "hybr", 1e-8, 100, {}, "2-point", 1e-6, False
    )
    named = psv.NonlinearSystemSolver(
        AnalyticPolynomial(), method="powell_hybrid", print_convergence=False
    )
    for solver in (native, named):
        solver.solve([0.8, 1.5])
        assert solver.success and solver.result.library == "scipy"
        assert solver.iteration_count is None
    with pytest.raises(ValueError, match="SciPy"):
        psv.NonlinearSystemSolver(AnalyticPolynomial(), method="newton_raphson")


def test_custom_roots_do_not_use_slsqp_private_version_guard(monkeypatch):
    """Custom roots use public linear algebra regardless of the SLSQP version pin."""
    import scipy

    monkeypatch.setattr(scipy, "__version__", "0.0-version-guard-test")
    for method in ("lm", "hybr", "newton", "gn"):
        solver = make_solver(AnalyticPolynomial(), method)
        solver.solve([0.8, 1.5])
        assert solver.success


def test_public_results_and_histories_are_detached():
    """Mutating returned results cannot rewrite accepted history or backend evidence."""
    solver = make_solver(AnalyticPolynomial(), "lm")
    returned = solver.solve([0.8, 1.5])
    backend = deepcopy(solver.backend_result)
    accepted = deepcopy(solver.iteration_history)
    observations = deepcopy(solver.convergence_history)
    returned[:] = 123
    solver.result.x[:] = 456
    solver.result.history[0]["x"][:] = 789
    np.testing.assert_equal(solver.backend_result, backend)
    np.testing.assert_equal(solver.iteration_history, accepted)
    np.testing.assert_equal(solver.convergence_history, observations)
    np.testing.assert_allclose(solver.x_final, [1, 2], atol=1e-8)
