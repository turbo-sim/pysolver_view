.. _nonlinear-system-tutorial:

Tutorial: Nonlinear systems
=======================================

In this tutorial, we will demonstrate how to make use of the :code:`NonlinearSystemSolver` and :code:`NonlinearSystemProblem` classes to solver nonlinear systems of equations


Example 1: Stationary point of the Lorentz equations
------------------------------------------------------

Problem definition
^^^^^^^^^^^^^^^^^^^^^^^^

The Lorenz equations are a set of three differential equations that describe the deterministic chaotic behavior of atmospheric convection and are often used to model the butterfly effect in chaos theory:

.. math::

    \begin{align}
    \dot{x} &= \sigma(y - x)\\
    \dot{y} &= x(\rho - z) - y\\
    \dot{z} &= xy - \beta z
    \end{align}

Where:

- :math:`\sigma` is related to the Prandtl number
- :math:`\rho` is related to the Rayleigh number
- :math:`\beta` is a geometric factor


This set of ordinary differential equations has an stationary point at the origin:

.. math::
    (x, y, z) = (0, 0, 0)

We can solve for this stationary point by setting the time derivatives to zero:

.. math::

    \begin{align}
    \dot{x} &= \sigma(y - x) = 0\\
    \dot{y} &= x(\rho - z) - y = 0\\
    \dot{z} &= xy - \beta z = 0
    \end{align}



Problem implementation
^^^^^^^^^^^^^^^^^^^^^^^^

In order to solve this problem with :code:`NonlinearSystemSolver` we have to define a problem class with the following methods:

- :code:`get_values(x)`

Lorentz system of equations is already implemented in the package and its definition is shown in the code snipped below:

.. code-block:: python

    class LorentzEquations(NonlinearSystemProblem):

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




Problem solution
^^^^^^^^^^^^^^^^^^^^^^^^

Now that we understand how to implement nonlinear problems, let move to how to solve them using the :code:`NonlinearSystemSolver` class.


1. **Import Necessary Packages**

   As always, we need to start by importing the required packages:

   .. code-block:: python

       import os
       import sys
       import numpy as np
       import matplotlib.pyplot as plt

2. **Importing PySolverView**

   PySolverView will help us tackle the Lorenz equations. First, ensure it's accessible:

   .. code-block:: python

       sys.path.insert(0, os.path.abspath('..'))
       import PySolverView as psv

3. **Setting Up Plot Options**

   For a clearer visualization, let's customize our plots. PySolverView makes this easy:

   .. code-block:: python

       psv.set_plot_options(grid=False)

4. **Logger Initialization**

   Let's set up a logger to store the convergence history:

   .. code-block:: python

       logger = psv.create_logger("convergence_history", use_datetime=True)

5. **Finding Stationary Point of the Lorenz Equations**

   Now, let's dive into the main event. We'll initialize our system, define the Lorenz equations problem, and solve for a stationary point:

   .. code-block:: python

       x0 = np.asarray([1.0, 3.0, 5.0]) # Initial guess for the Lorenz system
       problem = psv.LorentzEquations()
       solver = psv.NonlinearSystemSolver(problem, x0, display=True, plot=True, logger=logger)
       solution = solver.solve(method="lm")
       plt.show()  # Keep the visualization accessible



After running this code the optimization progress and the final solution will be printed to the console:


.. code-block:: none

    --------------------------------------------------------------------------------------------
    Solve system of equations for LorentzEquations
    --------------------------------------------------------------------------------------------
                Iteration             F-count         Norm of residual             Norm of step
    --------------------------------------------------------------------------------------------
                    1                    4             9.000000e+00             0.000000e+00
                    2                    9             9.000000e+00             0.000000e+00 
                    3                   13             3.750000e+00             2.915476e+00 
                    4                   18             2.488774e+00             6.029936e-01 
                    5                   22             5.174037e-01             1.256585e+00 
                    6                   26             3.762520e-01             1.032404e+00 
                    7                   30             1.156797e-02             1.908510e-01 
                    8                   34             3.108463e-05             7.890109e-03 
                    9                   38             2.983275e-08             1.647580e-05 
                    10                  42             2.980232e-08             5.853237e-11 
                    11                  46             2.980232e-08             9.067156e-22 
                    12                  50             2.980232e-08             1.051394e-37 
                    13                  54             2.980232e-08             2.152362e-53 
                    14                  58             2.980232e-08             7.330962e-69 
                    15                  62             2.980232e-08             2.199038e-84 
                    16                  66             2.980232e-08            4.571950e-100 
    --------------------------------------------------------------------------------------------
    Exit message: The cosine of the angle between func(x) and any column of the
    Jacobian is at most 0.000000 in absolute value
    Success: True
    Solution:
    x0 = +0.000000e+00
    x1 = +0.000000e+00
    x2 = +0.000000e+00
    --------------------------------------------------------------------------------------------

In addition, the script will also plot the convergence progress as illustrated in the figure below

.. figure:: figures/convergence_history_LorentzEquations.svg
   :align: center
   :width: 80%

   Solution convergence history for the Lorentz equations system



Congratulations! You've now successfully set up and solved the Lorentz equations problem using PySolverView.
For the complete implementation of this example, please refer to the :code:`demo_nonlinear_system.py` script located in the demos directory.


