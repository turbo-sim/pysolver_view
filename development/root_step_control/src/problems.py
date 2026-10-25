"""Coupled multidimensional root problems using the existing PSV interface.

Residuals, analytic Jacobians, and standard starts follow the MINPACK equation
test collection (VECFCN, VECJAC, INITPT):
https://www.netlib.org/minpack/ex/file16

Extended Rosenbrock repeats the two-variable Rosenbrock system in coupled
pairs; the remaining problems couple all their variables. Every class derives
from ``pysolver_view.nonlinear_system.NonlinearSystemProblem`` and therefore
works with the project's existing ``NonlinearSystemSolver`` as well.

These are mathematical equation systems, not scalar minimization objectives.
Variable scales are fixed characteristic displacements, not residual weights.
"""

import numpy as np

from pysolver_view.nonlinear_system import NonlinearSystemProblem

from .solver import DomainError

MINPACK_SOURCE = "https://www.netlib.org/minpack/ex/file16"


def _dimension(n, *, multiple=1):
    """Validate n as an integer >= 2 divisible by multiple, and return it as a Python
    int.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if n % multiple:
        raise ValueError(f"n must be a multiple of {multiple}")
    return int(n)


class BenchmarkProblem(NonlinearSystemProblem):
    """PSV root-problem base carrying reproducible benchmark metadata."""

    source = MINPACK_SOURCE

    def __init__(self, name, x0, description, *, x_scale=None, solution=None):
        """Store the benchmark name, start vector, scales, description and optional
        reference root.
        """
        self.name = name
        self.x0 = np.asarray(x0, dtype=float)
        self.n = self.x0.size
        self.m = self.n
        self.root_exists = True
        self.x_scale = (
            np.ones(self.n) if x_scale is None else np.asarray(x_scale, dtype=float)
        )
        self.description = description
        self.solution = None if solution is None else np.asarray(solution, dtype=float)


class ExtendedRosenbrockProblem(BenchmarkProblem):
    """Block extension of MINPACK problem 1; each two-variable block is coupled.

    f[2k] = 1 - x[2k], f[2k+1] = 10*(x[2k+1] - x[2k]**2).
    Standard start repeats (-1.2, 1); the root is the all-ones vector.
    """

    def __init__(self, n=10):
        """Create an even-dimensional set of Rosenbrock pairs at the standard repeated
        start.
        """
        n = _dimension(n, multiple=2)
        super().__init__(
            f"extended_rosenbrock_n{n}",
            np.tile([-1.2, 1.0], n // 2),
            "Extended Rosenbrock equations, coupled pairs with narrow curved valleys; standard start.",
            solution=np.ones(n),
        )

    def residual(self, x):
        """Return n Rosenbrock equations, interleaving coordinate and curved-valley
        residuals.
        """
        values = np.empty(self.n, dtype=np.result_type(x, float))
        values[::2] = 1 - x[::2]
        values[1::2] = 10 * (x[1::2] - x[::2] ** 2)
        return values

    def gradient(self, x):
        """Return the analytic n-by-n Jacobian with one coupled 2-by-2 block per
        variable pair.
        """
        jac = np.zeros((self.n, self.n), dtype=np.result_type(x, float))
        even = np.arange(0, self.n, 2)
        jac[even, even] = -1
        jac[even + 1, even] = -20 * x[even]
        jac[even + 1, even + 1] = 10
        return jac


class PowellSingularProblem(BenchmarkProblem):
    """MINPACK problem 2: four coupled equations, singular Jacobian at zero.

    f = (x0+10*x1, sqrt(5)*(x2-x3), (x1-2*x2)**2,
         sqrt(10)*(x0-x3)**2); standard start (3, -1, 0, 1).
    """

    def __init__(self):
        """Configure the four Powell equations and standard start whose root has a
        singular Jacobian.
        """
        super().__init__(
            "powell_singular_n4",
            [3, -1, 0, 1],
            "Four coupled Powell singular equations; rank-deficient Jacobian at the root; standard start.",
            solution=np.zeros(4),
        )

    def residual(self, x):
        """Return the four coupled linear and squared Powell residuals at x."""
        return np.array(
            [
                x[0] + 10 * x[1],
                np.sqrt(5) * (x[2] - x[3]),
                (x[1] - 2 * x[2]) ** 2,
                np.sqrt(10) * (x[0] - x[3]) ** 2,
            ]
        )

    def gradient(self, x):
        """Return the analytic 4-by-4 Jacobian, including derivatives of the two squared
        equations.
        """
        difference = x[1] - 2 * x[2]
        outer = 2 * np.sqrt(10) * (x[0] - x[3])
        return np.array(
            [
                [1, 10, 0, 0],
                [0, 0, np.sqrt(5), -np.sqrt(5)],
                [0, 2 * difference, -4 * difference, 0],
                [outer, 0, 0, -outer],
            ]
        )


class PowellBadlyScaledProblem(BenchmarkProblem):
    """MINPACK problem 3: f = (1e4*x*y-1, exp(-x)+exp(-y)-1.0001).

    Standard start is (0, 1); coordinate scales (1e-4, 10) are supplied
    explicitly to every solver. The equations themselves are not rescaled.
    """

    def __init__(self):
        """Set the two-equation standard start, disparate coordinate scales and
        reference root.
        """
        super().__init__(
            "powell_badly_scaled_n2",
            [0, 1],
            "Coupled Powell badly scaled equations; standard start; fixed scales (1e-4, 10).",
            x_scale=[1e-4, 10],
            solution=[1.098159329699817e-5, 9.106146739866524],
        )

    def residual(self, x):
        """Evaluate the coupled product/exponential equations, raising DomainError
        before exponential overflow.
        """
        if np.any(-x > np.log(np.finfo(float).max)):
            raise DomainError("Powell exponential would overflow")
        return np.array([1e4 * x[0] * x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - 1.0001])

    def gradient(self, x):
        """Return the analytic 2-by-2 Jacobian of the product and exponential residuals."""
        return np.array([[1e4 * x[1], 1e4 * x[0]], [-np.exp(-x[0]), -np.exp(-x[1])]])


class BrownAlmostLinearProblem(BenchmarkProblem):
    """MINPACK problem 8: a dense sum coupling and one nonlinear product.

    f[i] = x[i] + sum(x) - (n+1), i < n-1; f[n-1] = prod(x)-1.
    Standard start is x[:] = 0.5; all ones is a root (not necessarily unique).
    """

    def __init__(self, n=10):
        """Configure n sum/product equations at x=0.5 with an all-ones reference root."""
        n = _dimension(n)
        super().__init__(
            f"brown_almost_linear_n{n}",
            np.full(n, 0.5),
            "Brown almost-linear system with dense sum/product coupling; standard start x=0.5.",
            solution=np.ones(n),
        )

    def residual(self, x):
        """Return n-1 dense sum equations and the final nonlinear product equation."""
        values = x + np.sum(x) - (self.n + 1)
        values[-1] = np.prod(x) - 1
        return values

    def gradient(self, x):
        """Return the dense analytic Jacobian; product derivatives remain valid when a
        component is zero.
        """
        jac = np.ones((self.n, self.n), dtype=np.result_type(x, float)) + np.eye(self.n)
        # Product excluding a component is valid when that component is zero.
        jac[-1] = [np.prod(np.delete(x, index)) for index in range(self.n)]
        return jac


class BroydenTridiagonalProblem(BenchmarkProblem):
    """MINPACK problem 13: nearest-neighbor nonlinear coupling.

    f[i] = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1,
    with zero exterior values and standard start x[:] = -1.
    """

    def __init__(self, n=20):
        """Create n nearest-neighbor nonlinear equations at the standard all-minus-one
        start.
        """
        n = _dimension(n)
        super().__init__(
            f"broyden_tridiagonal_n{n}",
            -np.ones(n),
            "Broyden tridiagonal nonlinear equations with neighbor coupling; standard start x=-1.",
        )

    def residual(self, x):
        """Return the n Broyden equations with zero exterior values and nonlinear
        diagonal terms.
        """
        values = (3 - 2 * x) * x + 1
        values[1:] -= x[:-1]
        values[:-1] -= 2 * x[1:]
        return values

    def gradient(self, x):
        """Return the analytic tridiagonal Jacobian as a dense n-by-n array."""
        return (
            np.diag(3 - 4 * x)
            - np.diag(np.ones(self.n - 1), -1)
            - np.diag(2 * np.ones(self.n - 1), 1)
        )


class DiscreteBoundaryValueProblem(BenchmarkProblem):
    """MINPACK problem 9: coupled finite-difference boundary-value equations.

    h=1/(n+1), t[i]=(i+1)*h;
    f[i]=2*x[i]-x[i-1]-x[i+1]+h**2*(x[i]+t[i]+1)**3/2,
    with zero exterior values. Standard start x[i]=t[i]*(t[i]-1).
    """

    def __init__(self, n=20):
        """Create an n-point interior grid, zero boundary values and the standard
        parabolic initial guess.
        """
        n = _dimension(n)
        self.h = 1 / (n + 1)
        self.t = np.arange(1, n + 1) * self.h
        super().__init__(
            f"discrete_boundary_value_n{n}",
            self.t * (self.t - 1),
            "Nonlinear finite-difference boundary-value system; cubic neighbor-coupled equations and standard parabolic start.",
        )

    def residual(self, x):
        """Return the nonlinear finite-difference boundary-value equations at the
        interior unknowns.
        """
        values = 2 * x + 0.5 * self.h**2 * (x + self.t + 1) ** 3
        values[1:] -= x[:-1]
        values[:-1] -= x[1:]
        return values

    def gradient(self, x):
        """Return the analytic tridiagonal Jacobian including the cubic reaction
        derivative.
        """
        diagonal = 2 + 1.5 * self.h**2 * (x + self.t + 1) ** 2
        return (
            np.diag(diagonal)
            - np.diag(np.ones(self.n - 1), -1)
            - np.diag(np.ones(self.n - 1), 1)
        )


class DiscreteIntegralEquationProblem(BenchmarkProblem):
    """MINPACK problem 10: a dense cubic integral-equation discretization.

    With h=1/(n+1), t[i]=(i+1)*h, and
    K[i,j]=min(t[i],t[j])*(1-max(t[i],t[j])),
    f=x+h/2*K@(x+t+1)**3. The standard start is t*(t-1).
    Every residual depends on every variable. No sparsity is supplied to any
    solver; the analytic Jacobian is a fully populated n-by-n array.
    """

    def __init__(self, n=100):
        """Precompute the dense integral kernel and parabolic start on an n-point
        interior grid.
        """
        n = _dimension(n)
        self.h = 1 / (n + 1)
        self.t = np.arange(1, n + 1) * self.h
        self.kernel = np.minimum.outer(self.t, self.t) * (
            1 - np.maximum.outer(self.t, self.t)
        )
        super().__init__(
            f"discrete_integral_equation_n{n}",
            self.t * (self.t - 1),
            "MINPACK discrete integral equation with dense cubic coupling; standard parabolic start.",
        )

    def residual(self, x):
        """Return the n cubic integral-equation residuals, each depending on every
        unknown.
        """
        return x + 0.5 * self.h * (self.kernel @ (x + self.t + 1) ** 3)

    def gradient(self, x):
        """Return the fully dense analytic Jacobian of the discretized cubic integral
        equation.
        """
        return np.eye(self.n) + 1.5 * self.h * self.kernel * (x + self.t + 1) ** 2


class HelicalValleyProblem(BenchmarkProblem):
    """MINPACK problem 5: three equations coupling angle, radius, and height.

    f=(10*(z-10*theta), 10*(sqrt(x*x+y*y)-1), z), with the
    piecewise MINPACK angle branch. Standard start is (-1, 0, 0).
    The axis x=y=0 is excluded because the Jacobian is undefined there.
    """

    def __init__(self):
        """Set the three-variable angular/radial/height equations, standard start and
        known root.
        """
        super().__init__(
            "helical_valley_n3",
            [-1, 0, 0],
            "Helical valley: coupled angular/radial/height equations with MINPACK's angle branch; standard start.",
            solution=[1, 0, 0],
        )

    @staticmethod
    def _radius(x):
        """Return the planar radius, raising DomainError on the axis where the angle is
        undefined.
        """
        radius = np.hypot(x[0], x[1])
        if radius == 0:
            raise DomainError("Helical valley angle is undefined on x=y=0")
        return radius

    def residual(self, x):
        """Return the three helical-valley equations using the specified piecewise
        angular branch.
        """
        radius = self._radius(x)
        if x[0] > 0:
            theta = np.arctan(x[1] / x[0]) / (2 * np.pi)
        elif x[0] < 0:
            theta = np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5
        else:
            theta = np.copysign(0.25, x[1])
        return np.array([10 * (x[2] - 10 * theta), 10 * (radius - 1), x[2]])

    def gradient(self, x):
        """Return the analytic 3-by-3 Jacobian away from the excluded planar origin."""
        radius = self._radius(x)
        angular_scale = 100 / (2 * np.pi * radius**2)
        return np.array(
            [
                [angular_scale * x[1], -angular_scale * x[0], 10],
                [10 * x[0] / radius, 10 * x[1] / radius, 0],
                [0, 0, 1],
            ]
        )


class CoupledOverdeterminedRosenbrockProblem(BenchmarkProblem):
    """Extend Rosenbrock equations with dense redundant rows and an optional inconsistency.

    The first n rows are the standard extended Rosenbrock equations. Another n
    rows are dense linear combinations of them, so every added residual couples
    all variable pairs without changing the all-ones root. An optional constant
    residual of one creates an inconsistent least-squares problem with a known
    minimum residual norm of one. These are explicitly constructed controls,
    not additional original MINPACK test problems.
    """

    source = (
        "Constructed dense overdetermined extension of MINPACK Rosenbrock equations."
    )

    def __init__(self, n=10, *, inconsistent=False):
        """Configure 2*n coupled equations, optionally adding a nonzero constant row."""
        n = _dimension(n, multiple=2)
        self.square_problem = ExtendedRosenbrockProblem(n)
        self.mixing = 0.2 * np.eye(n) + np.ones((n, n)) / n
        self.inconsistent = bool(inconsistent)
        m = 2 * n + int(self.inconsistent)
        super().__init__(
            f"coupled_rosenbrock_n{n}_m{m}",
            self.square_problem.x0.copy(),
            "Dense redundant combinations of extended Rosenbrock equations; "
            + (
                "constant final residual makes the system inconsistent with minimum norm one."
                if inconsistent
                else "consistent overdetermined system with an all-ones root."
            ),
            solution=None if inconsistent else np.ones(n),
        )
        self.m = m
        self.root_exists = not self.inconsistent

    def residual(self, x):
        """Evaluate the square equations, their dense combinations, and optional constant."""
        values = self.square_problem.residual(x)
        combined = np.concatenate((values, self.mixing @ values))
        if self.inconsistent:
            combined = np.append(combined, 1.0)
        return combined

    def gradient(self, x):
        """Return the analytic rectangular Jacobian, including a zero constant-row derivative."""
        square = self.square_problem.gradient(x)
        combined = np.vstack((square, self.mixing @ square))
        if self.inconsistent:
            combined = np.vstack((combined, np.zeros(self.n)))
        return combined


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


def get_extension_problems():
    """Return thermodynamic, domain-recovery and rectangular Newton-family controls."""
    from .problem_thermodynamics import HeatedGasDuctProblem

    return [
        HeatedGasDuctProblem(),
        LogarithmicDomainProblem(),
        CoupledOverdeterminedRosenbrockProblem(),
        CoupledOverdeterminedRosenbrockProblem(inconsistent=True),
    ]


def get_problems():
    """Seven well-known, genuinely multidimensional systems at standard starts."""
    return [
        ExtendedRosenbrockProblem(n=10),
        PowellSingularProblem(),
        PowellBadlyScaledProblem(),
        BrownAlmostLinearProblem(n=10),
        BroydenTridiagonalProblem(n=20),
        DiscreteBoundaryValueProblem(n=20),
        HelicalValleyProblem(),
    ]


def get_large_problems():
    """Three 100-variable cases: banded, dense, and property-heavy coupling.

    CoolProp is only imported when this expanded suite is requested. The duct
    refines the original ten-cell domain at fixed total heat and resistance.
    """
    from .problem_thermodynamics import HeatedGasDuctProblem

    return [
        BroydenTridiagonalProblem(n=100),
        DiscreteIntegralEquationProblem(n=100),
        HeatedGasDuctProblem(
            n_cells=50, total_heat_per_mass=1e5, total_loss_coefficient=80
        ),
    ]
