# Getting started

(overview)=

## Overview

### PySolverView: An Interface for SciPy Solvers

`pysolver_view` is a wrapper for the equation and optimization solvers of `scipy`.

`scipy.optimize.root`, is a solver for nonlinear systems of equations:

$$
F(x) = 0
$$

Where $F: \mathbb{R}^n \rightarrow \mathbb{R}^n$ is a vector-valued function of the vector $x$.

`scipy.optimize.minimize` is a solver for optimization problems with the following structure:

$$
\begin{aligned}
\text{minimize} \quad & f(\mathbf{x}) \; \text{with} \; \mathbf{x} \in \mathbb{R}^n \\
\text{s.t.} \quad & c_{\mathrm{eq}}(\mathbf{x}) = 0 \\
                  & c_{\mathrm{in}}(\mathbf{x}) \geq 0 \\
                  & \mathbf{x}_l \leq \mathbf{x} \leq \mathbf{x}_u
\end{aligned}
$$

Where:

- $\mathbf{x}$ represents the vector of decision variables (i.e., degree of freedom).
- $f(\mathbf{x})$ is the objective function to minimize.
- $c_{\mathrm{eq}}(\mathbf{x})$ are the problem's equality constraints.
- $c_{\mathrm{in}}(\mathbf{x})$ are the problem's inequality constraints.
- $\mathbf{x}_l$ and $\mathbf{x}_u$ are the lower and upper bounds on the decision variables, respectively.

The expression above follows SciPy's inequality convention. PySolverView problem
classes return inequalities as $c_{\mathrm{in}}(\mathbf{x}) \leq 0$; its SciPy
adapter converts the sign when calling SciPy.

### Main Features of PySolverView

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

(installation)=

## User Installation Guide

This guide will walk you through the process of installing `pysolver_view` via `pip`. To isolate the installation and avoid conflicts with other Python packages, it is recommended to create a dedicated Conda virtual environment.

1. Ensure conda is installed:

   Check if conda is installed in your terminal:

   ```bash
   conda list
   ```

   If installed packages do not appear, [install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Open a terminal or command prompt and create a new virtual environment named `pysolver_env`:

   ```bash
   conda create --name pysolver_env python=3.11
   ```

3. Activate the newly created virtual environment:

   ```bash
   conda activate pysolver_env
   ```

4. Install the package using pip within the activated virtual environment:

   ```bash
   pip install pysolver_view
   ```

5. Verify the installation by running some of the examples in the [demos](https://github.com/turbo-sim/pysolver_view/tree/main/demos) directory

:::{note}
By default, `pysolver_view` can use the optimization solvers available in the `scipy` package. However, a wider range of solvers are available through the `pygmo` wrapper, including [IPOPT](https://coin-or.github.io/Ipopt/) and [SNOPT](https://ccom.ucsd.edu/~optimizers/docs/snopt/introduction.html).

```bash
conda install -c conda-forge pygmo
conda install -c conda-forge pygmo_plugins_nonfree
```
:::

## Developer Installation Guide

This installation guide is intended for developers who wish to contribute to or modify the `pysolver_view` source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

1. **Fork the repository:**

   Navigate to the [project's GitHub page](https://github.com/turbo-sim/pysolver_view) and click the "Fork" button in the upper right corner of the repository page to create a copy of the repository under your own GitHub account.

2. **Clone the forked repository:**

   Open your terminal and run the following command, replacing `<your-username>` with your GitHub username:

   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   ```

   Navigate into the cloned repository:

   ```bash
   cd <repository-name>
   ```

3. **Create a dedicated Conda virtual environment for development**:

   Check that conda is installed:

   ```bash
   conda list
   ```

   If conda is not installed, [install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

   Create dedicated virtual environment for the package:

   ```bash
   conda env create --file environment.yaml
   ```

4. **Activate the newly created virtual environment**:

   ```bash
   conda activate pysolver_env
   ```

5. **Use Poetry to install the required dependencies along with the package for local development**:

   ```bash
   poetry install --with dev
   ```

## Building the documentation

From the repository root, install the development tools and start the live preview:

```bash
poetry install --with dev
poetry run python docs/build_docs.py
```

Use `--no-autobuild` for a single build in `docs/_build/html/`. Authored pages
live in `docs/source/` as Markdown; API pages are regenerated from docstrings.
See the [documentation build guide](https://github.com/turbo-sim/pysolver_view/blob/main/docs/README.md)
for the complete workflow.
