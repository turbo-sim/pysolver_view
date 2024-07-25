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

User Installation Guide
========================

This guide will walk you through the process of installing `pysolver_view` via `pip`. To isolate the installation and avoid conflicts with other Python packages, it is recommended to create a dedicated Conda virtual environment.

1. Ensure conda is installed:

   Check if conda is installed in your terminal:

   .. code-block:: bash

      conda list

   If installed packages do not appear, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

2. Open a terminal or command prompt and create a new virtual environment named ``pysolver_env``:

   .. code-block:: bash

      conda create --name pysolver_env python=3.11

3. Activate the newly created virtual environment:

   .. code-block:: bash

      conda activate pysolver_env

4. Install the package using pip within the activated virtual environment:

   .. code-block:: bash

      pip install pysolver_view

5. Verify the installation by running some of the examples in the [`demos`](../../demos) directory


.. note::
   By default, ``pysolver_view`` can use the optimization solvers available in the ``scipy`` package. However, a wider range of solvers are available through the ``pygmo`` wrapper, including `IPOPT <https://coin-or.github.io/Ipopt/>`_ and `SNOPT <https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html>`_.
   
   .. code-block:: bash

      conda install -c conda-forge pygmo
      conda install -c conda-forge pygmo_plugins_nonfree


Developer Installation Guide
============================

This installation guide is intended for developers who wish to contribute to or modify the Turboflow source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

1. **Fork the repository:**

   Navigate to the `project's GitHub page <https://github.com/turbo-sim/pysolver_view>` and click the "Fork" button in the upper right corner of the repository page to create a copy of the repository under your own GitHub account.


2. **Clone the forked repository:**

   Open your terminal and run the following command, replacing `<your-username>` with your GitHub username:

   .. code-block:: bash

      git clone https://github.com/<your-username>/<repository-name>.git

   Navigate into the cloned repository:

   .. code-block:: bash

      cd <repository-name>

3. **Create a dedicated Conda virtual environment for development**:

   Check that conda is installed:

   .. code-block:: bash

      conda list

   If conda is not installed, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.
   
   Create dedicated virtual environment for the package:

   .. code-block:: bash

      conda env create --file environment.yaml

4. **Activate the newly created virtual environment**:

   .. code-block:: bash

      conda activate pysolver_env

5. **Install Poetry to manage dependencies**:

   .. code-block:: bash

      conda install poetry

   Poetry is a powerful dependency manager that offers separation of user and developer dependencies, ensuring that only the necessary packages are installed based on the user's intent. Additionally, it simplifies the process of adding, updating, and removing dependencies, making it easier to maintain the project's requirements.

6. **Use Poetry to install the required dependencies for TurboFlow development**:

   .. code-block:: bash

      poetry install

