.. _optimization-tutorial:

Tutorial: Optimization problems
=======================================

In this tutorial, we will explore how to utilize the :code:`OptimizationSolver` and :code:`OptimizationSolver` classes to solve optimization problems.



Example 1: Unconstrained optimization
------------------------------------------

Problem definition
^^^^^^^^^^^^^^^^^^^^^^^^
The general Rosenbrock problem, also known as Rosenbrock's banana function, in `n` dimensions is defined as:

.. math::
    
    \begin{align}
    \text{minimize} \quad  & f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right] \\
    \end{align}


Problem implementation
^^^^^^^^^^^^^^^^^^^^^^^^

In order to solve this problem with :code:`OptimizationSolver` we have to define a problem class with the following methods:

- :code:`get_values(x)`
- :code:`get_bounds()`
- :code:`get_n_eq()`
- :code:`get_n_ineq()`

As the Rosenbrock problem is a very common test proplem, it is already implemented in the package.
The code snipped below shows the definition of the :code:`RosenbrockProblem` class:

.. code-block:: python

    class RosenbrockProblem(OptimizationProblem):
        
        def __init__(self):
            self.f = None
            self.c_eq = None
            self.c_ineq = None

        def get_values(self, x):

            # Objective function
            self.f = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

            # Equality constraints
            self.c_eq = []
                
            # No inequality constraints given for this problem
            self.c_ineq = []

            # Combine objective function and constraints
            objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

            return objective_and_constraints

        def get_bounds(self):
            return None

        def get_n_eq(self):
            return self.get_number_of_constraints(self.c_eq)

        def get_n_ineq(self):
            return self.get_number_of_constraints(self.c_ineq)


Note that the definition of the :code:`RosenbrockProblem` class is based on the abstract class :code:`OptimizationProblem`.
As a result, the helper functions :code:`merge_objective_and_constraints` and :code:`get_number_of_constraints` are automatically inherited from  :code:`OptimizationProblem`.

Problem solution
^^^^^^^^^^^^^^^^^^^^^^^^

Now that we understand how to implement optimization problems, let move to how to solve them using the :code:`OptimizationSolver` class.

1. **Import Necessary Packages**

   Let's start by importing the packages we need:

   .. code-block:: python

       import os
       import sys
       import numpy as np
       import matplotlib.pyplot as plt

2. **Importing PySolverView**

   To access the PySolverView package located in the parent directory, modify the system path and import the package:

   .. code-block:: python

       sys.path.insert(0, os.path.abspath('..'))
       import PySolverView as psv

3. **Setting Up Plot Options**

   Customize your plots with PySolverView's built-in functionalities for high-quality figures:

   .. code-block:: python

       psv.set_plot_options(grid=False)

4. **Logger Initialization**

   For efficient debugging and analysis, initialize a logger for the optimization process:

   .. code-block:: python

       logger = psv.create_logger("convergence_history", use_datetime=True)

5. **Rosenbrock Problem Solution**

   Set your initial guess, define the problem, and solve:

   .. code-block:: python

       x0 = np.asarray([2, 2, 2, 2]) # Rosenbrock's problem in 4 dimensions
       problem = psv.RosenbrockProblem()
       solver = psv.OptimizationSolver(problem, x0, display=True, plot=True, logger=logger)
       sol = solver.solve(method="slsqp")
       plt.show()  # This will keep the plot window open




After running this code the optimization progress and the final solution will be printed to the console:


.. code-block:: none

    --------------------------------------------------------------------------------
    Starting optimization process for RosenbrockProblem
    --------------------------------------------------------------------------------
        Grad-eval    Func-eval      Func-value     Infeasibility      Norm of step
    --------------------------------------------------------------------------------
                1            1      +1.203e+03        +0.000e+00        +0.000e+00
                2            5      +3.548e+02        +0.000e+00        +2.370e+00 
                3            9      +1.951e+02        +0.000e+00        +4.791e-01 
                4           12      +6.925e+01        +0.000e+00        +1.757e+00 
                5           15      +6.818e+01        +0.000e+00        +2.098e-01 
                6           18      +2.765e+01        +0.000e+00        +4.361e-01 
                7           19      +1.554e+01        +0.000e+00        +7.722e-01 
                8           20      +6.164e+00        +0.000e+00        +3.628e-01 
                9           21      +4.523e+00        +0.000e+00        +3.587e-02 
               10           22      +3.921e+00        +0.000e+00        +4.263e-02 
               11           23      +3.916e+00        +0.000e+00        +4.066e-03 
               12           24      +3.910e+00        +0.000e+00        +1.330e-02 
               13           25      +3.895e+00        +0.000e+00        +4.860e-02 
               14           26      +3.838e+00        +0.000e+00        +1.981e-01 
               15           28      +3.815e+00        +0.000e+00        +1.231e-01 
               16           29      +3.791e+00        +0.000e+00        +6.925e-02 
               17           30      +3.766e+00        +0.000e+00        +3.089e-02 
               18           31      +3.761e+00        +0.000e+00        +5.157e-02 
               19           32      +3.755e+00        +0.000e+00        +1.051e-02 
               20           33      +3.735e+00        +0.000e+00        +8.775e-02 
               21           34      +3.723e+00        +0.000e+00        +6.190e-02 
               22           35      +3.720e+00        +0.000e+00        +1.211e-01 
               23           36      +3.708e+00        +0.000e+00        +1.412e-02 
               24           37      +3.705e+00        +0.000e+00        +5.378e-02 
               25           38      +3.704e+00        +0.000e+00        +2.787e-02 
               26           39      +3.702e+00        +0.000e+00        +2.835e-02 
               27           40      +3.702e+00        +0.000e+00        +2.237e-02 
               28           41      +3.701e+00        +0.000e+00        +1.806e-02 
               29           42      +3.701e+00        +0.000e+00        +1.381e-02 
               30           43      +3.701e+00        +0.000e+00        +1.937e-03 
               31           44      +3.701e+00        +0.000e+00        +2.428e-04 
               32           45      +3.701e+00        +0.000e+00        +2.030e-05 
               33           46      +3.701e+00        +0.000e+00        +7.200e-05 
    --------------------------------------------------------------------------------
    Exit message: Optimization terminated successfully
    Sucess: True
    Solution:
    f  = +3.701429e+00
    x0 = -7.756551e-01
    x1 = +6.130867e-01
    x2 = +3.820567e-01
    x3 = +1.459662e-01
    --------------------------------------------------------------------------------


In addition, the script will also plot the optimization progress as illustrated in the figure below

.. figure:: figures/convergence_history_RosenbrockProblem.svg
   :align: center
   :width: 80%

   Optimization Convergence History for the unconstrained Rosenbrock's problem



Congratulations! You've now successfully set up and solved the Rosenbrock problem using PySolverView.
For the complete implementation of this example, please refer to the :code:`demo_optimization.py` script located in the demos directory.





Example 2: Constrained optimization
------------------------------------------


Problem definition
^^^^^^^^^^^^^^^^^^^^^^^^
In this example we will extend the Rosenbrock problem to include trigonometric-exponential constraints:

.. math::

    \begin{align}
    \text{minimize} \quad & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right] \\
    \text{s.t.} \quad & 3x_{k+1}^3 + 2x_{k+2} - 5 + \sin(x_{k+1}-x_{k+2})\sin(x_{k+1}+x_{k+2}) + \\
                        & + 4x_{k+1} - x_k \exp(x_k-x_{k+1}) - 3 = 0, \; \forall k=1,...,n-2 \\
                        & -5 \le x_i \le 5, \forall i=1,...,n
    \end{align}



Problem implementation
^^^^^^^^^^^^^^^^^^^^^^^^

This problem is also implemented in the package, so we will only have to import it in our script.
The code snipped below shows the definition of the :code:`RosenbrockProblemConstrained` class:

.. code-block:: python

    class RosenbrockProblemConstrained(OptimizationProblem):
        def __init__(self):
            self.f = None
            self.c_eq = None
            self.c_ineq = None

        def get_values(self, x):

            # Objective function
            self.x = x
            self.f = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

            # Equality constraints
            self.c_eq = []
            for k in range(len(x) - 2):
                val = (3 * x[k+1]**3 + 2 * x[k+2] - 5 +
                    np.sin(x[k+1] - x[k+2]) * np.sin(x[k+1] + x[k+2]) +
                    4 * x[k+1] - x[k] * np.exp(x[k] - x[k+1]) - 3)
                self.c_eq.append(val)

            # No inequality constraints given for this problem
            self.c_ineq = []

            # Combine objective function and constraints
            objective_and_constraints = self.merge_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

            return objective_and_constraints
        
        def get_bounds(self):
            bounds = [(-5, 5) for _ in range(len(self.x))]
            return bounds

        def get_n_eq(self):
            return self.get_number_of_constraints(self.c_eq)

        def get_n_ineq(self):
            return self.get_number_of_constraints(self.c_ineq)


Problem solution
^^^^^^^^^^^^^^^^^^^^^^^^

Let's write a script to solve this problem with the :code:`OptimizationSolver` class.

1. **Import Necessary Packages**

   Start by importing the packages we need:

   .. code-block:: python

       import os
       import sys
       import numpy as np
       import matplotlib.pyplot as plt

2. **Importing PySolverView**

   To access the PySolverView package located in the parent directory, modify the system path and import the package:

   .. code-block:: python

       sys.path.insert(0, os.path.abspath('..'))
       import PySolverView as psv

3. **Setting Up Plot Options**

   Customize your plots with PySolverView's built-in functionalities for high-quality figures:

   .. code-block:: python

       psv.set_plot_options(grid=False)

4. **Logger Initialization**

   For efficient debugging and analysis, initialize a logger for the optimization process:

   .. code-block:: python

       logger = psv.create_logger("convergence_history", use_datetime=True)

5. **Rosenbrock Problem Solution**

   Set your initial guess, define the problem, and solve:

   .. code-block:: python

       x0 = np.asarray([2, 2, 2, 2]) # Rosenbrock's problem in 4 dimensions
       problem = psv.RosenbrockProblemConstrained()
       solver = psv.OptimizationSolver(problem, x0, display=True, plot=True, logger=logger)
       sol = solver.solve(method="slsqp")
       plt.show()  # This will keep the plot window open


After running the code above, the optimization progress and the final solution will be printed to the console:


.. code-block:: none

    --------------------------------------------------------------------------------
    Starting optimization process for RosenbrockProblemConstrained
    --------------------------------------------------------------------------------
        Grad-eval    Func-eval      Func-value     Infeasibility      Norm of step
    --------------------------------------------------------------------------------
                1            1      +1.203e+03        +2.600e+01        +0.000e+00
                2            4      +4.460e+02        +2.333e+01        +8.617e-01 
                3            6      +3.132e+02        +1.422e+01        +1.319e+00 
                4            7      +2.043e+01        +2.677e+00        +2.781e+00 
                5            8      +1.638e+00        +2.323e-01        +2.564e-01 
                6            9      +9.424e-01        +3.911e-03        +1.130e-01 
                7           10      +1.105e-02        +2.097e-03        +7.363e-02 
                8           11      +6.271e-06        +3.231e-05        +8.599e-03 
                9           12      +1.503e-11        +3.540e-08        +2.092e-04 
    --------------------------------------------------------------------------------
    Exit message: Optimization terminated successfully
    Sucess: True
    Solution:
    f  = +1.213596e-13
    x0 = +1.000000e+00
    x1 = +1.000000e+00
    x2 = +1.000000e+00
    x3 = +1.000000e+00
    --------------------------------------------------------------------------------


In addition, the script will also plot the optimization progress as illustrated in the figure below

.. figure:: figures/convergence_history_RosenbrockProblemConstrained.svg
   :align: center
   :width: 80%

   Optimization Convergence History for the constrained Rosenbrock's problem


Congratulations! You've now successfully set up and solved the constrained Rosenbrock problem using PySolverView.
For the complete implementation of this example, please refer to the :code:`demo_optimization.py` script located in the demos directory.


