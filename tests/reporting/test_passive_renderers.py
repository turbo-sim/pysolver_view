"""Reports and plots consume data without new model work or mutable aliases."""

from copy import deepcopy

import numpy as np

from pysolver_view import OptimizationSolver, RosenbrockProblem
from pysolver_view.reporting import tables


def test_passive_tables_and_plot_view_are_detached_from_model_and_history():
    solver = OptimizationSolver(RosenbrockProblem(2), print_convergence=False)
    solver.solve([1.1, 1.1])
    point = solver.x_final_norm.copy()
    kkt = solver.evaluate_kkt_conditions(point, 1e-6)
    constraints = solver.get_constraint_data(point, 1e-6)
    before = deepcopy(solver.convergence_history)
    calls = solver.func_count_tot, solver.grad_count, solver.kkt_count

    def forbidden(*args, **kwargs):
        raise AssertionError("Passive rendering must not evaluate the model")

    solver.problem.fitness = solver.problem.gradient = forbidden
    assert "KKT" in tables.kkt_report(kkt, 1e-6)
    assert "constraints" in tables.constraint_report(constraints)
    assert "multipliers" in tables.multipliers_report(
        kkt,
        constraints,
        solver.problem.variable_names,
        solver.problem.get_bounds_normalized(),
    )
    solver.plot_convergence_history(showfig=False)
    solver._plot_state.convergence_history["x"][0][:] = 999
    np.testing.assert_equal(solver.convergence_history, before)
    assert (solver.func_count_tot, solver.grad_count, solver.kkt_count) == calls
