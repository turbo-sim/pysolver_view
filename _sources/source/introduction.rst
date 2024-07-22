==============================
Getting started
==============================


.. _overview:

Overview
========

PySolverView: An Interface for SciPy Solvers
------------------------------------------------------

:mod:`PySolverView` is a wrapper for the equation and optimization solvers of :mod:`scipy`.


:mod:`scipy.optimize.root`, is a solver for nonlinear systems of equations:

.. math::
    F(x) = 0

Where :math:`F: \mathbb{R}^n \rightarrow \mathbb{R}^n` is a vector-valued function of the vector :math:`x`.


:mod:`scipy.optimize.minimize` is a solver for optimization problems with the following structure:

.. math::

    \begin{align}
    \text{minimize} \quad & f(\mathbf{x}) \; \text{with} \; \mathbf{x} \in \mathbb{R}^n \\
    \text{s.t.} \quad & c_{\mathrm{eq}}(\mathbf{x}) = 0 \\
                      & c_{\mathrm{in}}(\mathbf{x}) \geq 0 \\
                      & \mathbf{x}_l \leq \mathbf{x} \leq \mathbf{x}_u
    \end{align}


Where:

- :math:`\mathbf{x}` represents the vector of decision variables (i.e., degree of freedom).
- :math:`f(\mathbf{x})` is the objective function to minimize.
- :math:`c_{\mathrm{eq}}(\mathbf{x})` are the problem's equality constraints.
- :math:`c_{\mathrm{in}}(\mathbf{x})` are the problem's inequality constraints.
- :math:`\mathbf{x}_l` and :math:`\mathbf{x}_u` are the lower and upper bounds on the decision variables, respectively.




Main Features of PySolverView
--------------------------------
PySolverView is a wrapper for the solvers of the Scipy package. Using this interface brings some advantages with respect to the vanilla Scipy experience:

1. **Monitoring Capabilities:**  
    * Display convergence progress.
    * Write convergence history to log file.
    * Real-time plotting of solution progression.

2. **Problem Definition:**  
    * Leverage templated object to streamlined problem definition.
    * Evaluate objective function and constraints simultaneously to avoid redundant calculations.

3. **Efficient Evaluations with Caching Mechanism:**  
    PySolverView integrates a caching mechanism. In cases where objective functions, equality constraints, and inequality constraints all require evaluations, the mechanism checks for unchanged independent variable seta and uses of previous calculations to avoid redundant recalculations.



.. _installation:

Installation
=====================

If you use `Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_, the following `Bash <https://gitforwindows.org/>`_ command can be executed to set up a new virtual environment with all necessary dependencies:

.. code-block:: bash

    conda env create --file environment.yaml

Executing the above will result in the creation of the `pysolver_env` virtual environment. The environment will then have all packages listed in the YAML file installed.

To initialize the virtual environment, use the command below:

.. code-block:: bash

    conda activate pysolver_env

In scenarios where additional packages are needed:

.. code-block:: bash

    conda install <name of the package>

Alternatively, one can also append the desired package names to the `environment.yaml` file and update the environment:

.. code-block:: bash

    conda env update --file environment.yaml --prune


