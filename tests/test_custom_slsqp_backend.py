"""Keep the private SciPy bridge explicit, local, and safe to import."""

import numpy as np
import pytest

from pysolver_view.solvers import _slsqp_backend


def test_unsupported_scipy_version_is_rejected_when_backend_is_created(monkeypatch):
    """An incompatible private layout must never reach the compiled routine."""
    monkeypatch.setattr(_slsqp_backend.scipy, "__version__", "0.0.unsupported")
    with pytest.raises(RuntimeError, match="requires SciPy 1.17.1"):
        _slsqp_backend.Backend(
            np.array([0.0]),
            np.array([-np.inf]),
            np.array([np.inf]),
            0,
            0,
            ftol=1e-9,
            maxiter=10,
        )


def test_private_version_guard_does_not_block_native_scipy(monkeypatch):
    """Other solvers remain available when custom SLSQP is version-incompatible."""
    from pysolver_view import OptimizationSolver, RosenbrockProblem

    monkeypatch.setattr(_slsqp_backend.scipy, "__version__", "0.0.unsupported")
    solver = OptimizationSolver(
        RosenbrockProblem(2),
        library="scipy",
        method="slsqp",
        print_convergence=False,
        plot_convergence=False,
    )
    solver.solve(np.array([1.1, 1.1]))
    assert solver.success
