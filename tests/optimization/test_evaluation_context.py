"""Backend callback ownership remains compatible with non-copyable models."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pysolver_view import OptimizationProblem, OptimizationSolver
from pysolver_view.optimization.evaluation import BackendProblem
from pysolver_view.optimization.backends.pygmo import PygmoProblem


class NoncopyableProblem(OptimizationProblem):
    def __deepcopy__(self, memo):
        raise AssertionError("Model ownership must remain with the caller")

    def fitness(self, x):
        return np.array([(x[0] - 1) ** 2])

    def gradient(self, x):
        return np.array([2 * (x[0] - 1)])

    def get_bounds(self):
        return ([-10.0], [10.0])

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0


def test_only_pygmo_flattens_derivatives_and_callbacks_do_not_copy_model():
    solver = OptimizationSolver(
        NoncopyableProblem(), print_convergence=False, check_kkt=False
    )
    context = deepcopy(BackendProblem(solver))
    assert context.gradient(np.array([2.0])).shape == (1, 1)
    assert deepcopy(PygmoProblem(context)).gradient(np.array([2.0])).shape == (1,)
    assert solver.func_count_tot == 1
    assert solver.grad_count == 1
    assert solver.solve([2.0])[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "options",
    [{"tolerance": 1e-7, "max_iterations": 12}, {"tol": 1e-7, "max_iter": 12}],
)
def test_nlopt_adapter_normalizes_common_and_legacy_options_without_real_backend(
    monkeypatch, options
):
    from pysolver_view.optimization.backends import pygmo_nlopt

    handle = SimpleNamespace(get_last_opt_result=lambda: 4)
    population = SimpleNamespace(
        champion_x=np.array([1.0]),
        champion_f=np.array([0.0]),
        set_x=lambda index, x: None,
    )
    algorithm = SimpleNamespace(extract=lambda kind: handle, evolve=lambda pop: pop)
    fake_pg = SimpleNamespace(
        nlopt=lambda **kw: object(),
        algorithm=lambda uda: algorithm,
        problem=lambda p: p,
        population=lambda p, size: population,
    )
    monkeypatch.setattr(pygmo_nlopt, "pg", fake_pg)
    solver = OptimizationSolver(NoncopyableProblem(), print_convergence=False)
    original = options.copy()
    result = pygmo_nlopt.minimize_nlopt(
        BackendProblem(solver), np.array([2.0]), "slsqp", options
    )
    assert result.success
    assert handle.maxeval == 12
    assert handle.ftol_rel == handle.xtol_rel == 1e-7
    assert options == original
