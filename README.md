
# PySolverView

PySolverView provides a common interface and convergence histories for optimization
and root finding, including native SciPy methods and custom solvers with step controls.

## Features

- Print and visualize convergence history
- Streamlined problem definition (object-oriented)
- Improved function and gradient evaluation to avoid unnecesary model evaluations
- Controlled Levenberg–Marquardt, Powell hybrid, Newton–Raphson, and Gauss–Newton
  through `NonlinearSystemSolver(library="pysolver")`; native SciPy remains the default

The [root solver guide](development/root_step_control/notes/10_public_interface.md)
explains options, scaling, callbacks, and result accounting. Run the single
[four-solver demo](demos/demo_eqn_solving/demo_custom_root_solvers.py) to compare
accepted convergence histories. Benchmark suites and timing studies remain under
[`development/root_step_control/`](development/root_step_control/README.md).

Custom roots enable Jacobian-based numerical scaling by default; disable it
with `options={"jacobian_scaling": False}`. User-defined physical step scales
and limits retain their meaning. The
[radiative heat demo](demos/demo_eqn_solving/README.md) compares controlled and
large-step runs on a 1000-variable collocation problem.

## Documentation

Please refer to the [`documentation`](https://turbo-sim.github.io/pysolver_view/index.html) for:
- Installation instructions
- Tutorials showing how to use the code
- Complete documentation of all functions and classes in PySolverView


## Other info
- Pygmo and pygmo-plugin-nonfree should be installed with Conda
