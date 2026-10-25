import numpy as np
from ...exceptions import DomainError
from .base import BenchmarkProblem, _dimension


class LogarithmicDomainProblem(BenchmarkProblem):
    """Coupled positive-domain logarithmic equations with an analytically known root.

    F(x) = log(x) + C @ (x - 1), C = .05*I + .05*ones/n. The all-ones root
    is unique because the Jacobian is positive definite throughout the positive
    orthant. Starting at x=10 produces a Newton trial with negative coordinates,
    making this an explicit model-domain recovery test. It is a mathematical
    surrogate for fragile property evaluations, not a thermodynamic model.
    """

    source = "Constructed coupled logarithmic domain-recovery control."

    def __init__(self, n=10):
        """Create a dense positive-domain system at a valid but difficult start."""
        n = _dimension(n)
        self.coupling = 0.05 * np.eye(n) + 0.05 * np.ones((n, n)) / n
        super().__init__(
            f"coupled_logarithmic_domain_n{n}",
            np.full(n, 10.0),
            "Coupled logarithmic equations on positive coordinates; full first "
            "Newton trial is undefined. Mathematical domain-recovery surrogate.",
            solution=np.ones(n),
        )

    def _check_domain(self, x):
        """Reject nonpositive or nonfinite model coordinates before logarithm evaluation."""
        if np.any(~np.isfinite(x)) or np.any(x <= 0):
            raise DomainError("Logarithmic model requires finite positive coordinates.")

    def residual(self, x):
        """Return the dense coupled logarithmic residual inside the positive domain."""
        self._check_domain(x)
        return np.log(x) + self.coupling @ (x - 1)

    def gradient(self, x):
        """Return the positive-definite dense Jacobian at a valid model point."""
        self._check_domain(x)
        return np.diag(1 / x) + self.coupling
