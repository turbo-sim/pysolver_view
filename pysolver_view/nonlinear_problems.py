import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize._numdiff import approx_derivative


class NonlinearSystemProblem(ABC):
    """
    Abstract base class for root-finding problems.

    Derived root-finding problem objects must implement the following method:

    - `get_values`: Evaluate the system of equations for a given set of decision variables.

    Additionally, specific problem classes can define the `get_jacobian` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    get_values(x)
        Evaluate the system of equations for a given set of decision variables.

    Examples
    --------
    Here's an example of how to derive from `RootFindingProblem`::

        class MyRootFindingProblem(RootFindingProblem):
            def get_values(self, x):
                # Implement evaluation logic here
                pass
    """

    @abstractmethod
    def get_values(self, x):
        """
        Evaluate the system of equations for given decision variables.

        Parameters
        ----------
        x : array-like
            Vector of decision variables.

        Returns
        -------
        array_like
            Vector containing the values of the system of equations for the given decision variables.
        """
        pass


class LorentzEquations(NonlinearSystemProblem):
    r"""
    Implementation of the Lorentz System of Nonlinear Equations.

    This class implements the following system of algebraic nonlinear equations:

    .. math::

        \begin{align}
        \dot{x} &= \sigma(y - x) = 0\\
        \dot{y} &= x(\rho - z) - y = 0\\
        \dot{z} &= xy - \beta z = 0
        \end{align}

    Where:

    - :math:`\sigma` is related to the Prandtl number
    - :math:`\rho` is related to the Rayleigh number
    - :math:`\beta` is a geometric factor

    References
    ----------
    - Edward N. Lorenz. "Deterministic Nonperiodic Flow". Journal of the Atmospheric Sciences, 20(2):130-141, 1963.
    
    Methods
    -------
    get_values(vars)`:
        Evaluate the Lorentz system at a given state.

    Attributes
    ----------
    sigma : float
        The Prandtl number.
    beta : float
        The geometric factor.
    rho : float
        The Rayleigh number.
    """

    def __init__(self, sigma=1.0, beta=2.0, rho=3.0):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def get_values(self, vars):
        x, y, z = vars
        eq1 = self.sigma * (y - x)
        eq2 = x * (self.rho - z) - y
        eq3 = x * y - self.beta * z
        return np.array([eq1, eq2, eq3])
    
    # def get_jacobian(self, x):
    #     return approx_derivative(self.get_values, x, method="cs")