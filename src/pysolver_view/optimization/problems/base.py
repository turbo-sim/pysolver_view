"""Optimization problem contract, independent of solver and presentation code."""

from abc import ABC, abstractmethod
from .. import scaling


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems.

    Derived optimization problem objects must implement the following methods:

    - `fitness`: Evaluate the objective function and constraints for a given set of decision variables.
    - `get_bounds`: Get the bounds for each decision variable.
    - `get_neq`: Return the number of equality constraints associated with the problem.
    - `get_nineq`: Return the number of inequality constraints associated with the problem.

    Additionally, specific problem classes can define the `gradient` method to compute the Jacobians. If this method is not present in the derived class, the solver will revert to using forward finite differences for Jacobian calculations.

    Methods
    -------
    fitness(x)
        Evaluate the objective function and constraints for a given set of decision variables.
    get_bounds()
        Get the bounds for each decision variable.
    get_neq()
        Return the number of equality constraints associated with the problem.
    get_nineq()
        Return the number of inequality constraints associated with the problem.

    Parameters
    ----------
    problem_scale : float, optional
        Scaling factor for normalization. Default is 1.0.
    variable_names : list of str, optional
        Names of the decision variables. Used for reporting and debugging.

    """

    @abstractmethod
    def fitness(self, x):
        """
        Evaluate the objective function and constraints for given decision variables.

        Parameters
        ----------
        x : array-like
            Vector of independent variables (i.e., degrees of freedom).

        Returns
        -------
        array_like
            Vector containing the objective function, equality constraints, and inequality constraints.
        """
        pass

    @abstractmethod
    def get_bounds(self):
        """
        Get the bounds for each decision variable (Pygmo format)

        Returns
        -------
        bounds : tuple of lists
            A tuple of two items where the first item is the list of lower bounds and the second
            item of the list of upper bounds for the vector of decision variables. For example,
            ([-2 -1], [2, 1]) indicates that the first decision variable has bounds between
            -2 and 2, and the second has bounds between -1 and 1.
        """
        pass

    @abstractmethod
    def get_nec(self):
        """
        Return the number of equality constraints associated with the problem.

        Returns
        -------
        neq : int
            Number of equality constraints.
        """
        pass

    @abstractmethod
    def get_nic(self):
        """
        Return the number of inequality constraints associated with the problem.

        Returns
        -------
        nineq : int
            Number of inequality constraints.
        """
        pass

    def get_bounds_normalized(self):
        """
        Return normalized bounds in [0, problem_scale]. If no scaling is applied (i.e., problem_scale is None),
        the physical bounds are returned instead.

        Returns
        -------
        tuple of lists
            Tuple (lb, ub) in normalized space or the original physical bounds if problem_scale is None.
        """
        return scaling.get_bounds_normalized(self)

    def scale_physical_to_normalized(self, x_phys):
        """
        Convert physical design variable values to normalized values in the range [0, problem_scale].
        If self.problem_scale is None, no scaling is applied.

        The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.
        It automatically handles the case of fixed variables (i.e., upper bound = lower bound) by returning 0.0.

        Parameters
        ----------
        x_phys : array-like
            Physical values of the decision variables.

        Returns
        -------
        np.ndarray
            Normalized values in the range [0, problem_scale] if scaling is applied,
            otherwise the original physical values.
        """
        return scaling.scale_physical_to_normalized(self, x_phys)

    def scale_normalized_to_physical(self, x_norm):
        """
        Convert normalized values in the range [0, problem_scale] back to physical variable values.
        If self.problem_scale is None, no scaling is applied.

        The method uses the bounds returned by `self.get_bounds()` and the internal `self.problem_scale`.

        Parameters
        ----------
        x_norm : array-like
            Normalized values of the decision variables.

        Returns
        -------
        np.ndarray
            Physical values corresponding to the normalized input, or the original values if no scaling.
        """
        return scaling.scale_normalized_to_physical(self, x_norm)

    def fitness_normalized_input(self, x_norm):
        """
        Evaluate fitness starting from normalized input.

        Parameters
        ----------
        x_norm : array-like
            Normalized input vector.

        Returns
        -------
        array_like
            Output of `fitness` evaluated on the corresponding physical vector.
        """
        return scaling.fitness_normalized_input(self, x_norm)

    def gradient_normalized_input(self, x_norm):
        """
        Compute the gradient of the objective and constraints with respect to normalized variables.

        This function applies the chain rule to convert the gradient computed in the physical space
        to the corresponding gradient in the normalized space. If `problem_scale` is set to `None`,
        the problem is considered unscaled and the gradient is returned directly.

        Parameters
        ----------
        x_norm : array-like
            Normalized input vector (typically in [0, problem_scale]).

        Returns
        -------
        np.ndarray
            Gradient with respect to the normalized variables. The shape is:
            - (n,) for scalar objective or flat constraint vectors.
            - (m, n) for vector-valued constraints with m rows and n design variables.

        Notes
        -----
        The chain rule is applied as follows:

            x_phys = lb + (ub - lb) * (x_norm / problem_scale)
            ∇f(x_norm) = ∇f(x_phys) * d(x_phys)/d(x_norm)

        where:

            d(x_phys)/d(x_norm) = (ub - lb) / problem_scale    [elementwise]

        This correction ensures that exact user-defined gradients are consistent
        with the scaling applied during optimization.
        """
        return scaling.gradient_normalized_input(self, x_norm)

    def clip_to_bounds(self, x_physical, logger=None):
        """
        Clip physical variable values to lie within specified bounds.

        Parameters
        ----------
        x_physical : array-like
            Input vector in physical space.
        logger : logging.Logger or None, optional
            Logger for outputting warnings. If None, warnings are printed to standard output.

        Returns
        -------
        np.ndarray
            Clipped vector.
        """
        return scaling.clip_to_bounds(self, x_physical, logger=logger)
