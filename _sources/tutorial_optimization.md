(optimization-tutorial)=

# Tutorial: Optimization problems

In this tutorial, we will explore how to utilize the `OptimizationSolver` and `OptimizationProblem` classes to solve optimization problems.

## Controlled SLSQP callbacks

Use `OptimizationSolver(library="pysolver", method="slsqp")` for controlled
SLSQP through the main frontend. The frontend owns evaluation accounting,
convergence printing, plotting, and result assembly. Configure `logger` and
`print_convergence` there to get a single consistent report.

The usual `callback_functions` receive `(normalized_x, callback_count)`. For
integrations that also need recovery counters and complete KKT diagnostics,
provide a `backend_callback`:

```python
records = []
solver = psv.OptimizationSolver(
    problem,
    library="pysolver",
    method="slsqp",
    problem_scale=1.0,
    update_on="iteration",
    backend_callback=records.append,
)
solver.solve(x0)
```

This hook receives a detached dictionary at the initial and each accepted point.
Rejected trials, derivative probes, and final synchronization do not invoke it.
Records have the same coordinates and signs as `solver.backend_result.history`:
`x` is normalized, equalities are zero, and inequalities in `fitness` are
nonnegative. Equality KKT multiplier signs therefore differ from the frontend's
public `solver.result.kkt` convention. Raw `nfev` and `njev` are backend request
counts; use `solver.result.nfev` for total model calls including derivative work.
The callback performs no additional model or KKT evaluations by itself.

Return `False` or raise `StopIteration` from either callback form to cancel
controlled SLSQP at its current accepted point. Other callback errors propagate.
`backend_callback` is available only with `library="pysolver"`; native SciPy
callbacks do not certify accepted points.

## Pounce interior-point and active-set SQP

Install the optional, pinned Pounce solver in the project environment:

```bash
poetry install --extras pounce
# For an installed release: pip install 'pysolver_view[pounce]'
```

The same problem can be solved with either NLP algorithm:

```python
solver = psv.OptimizationSolver(
    problem,
    library="pounce",
    method="ipm",  # "sqp" selects active-set SQP
    tolerance=1e-8,
    max_iterations=300,
    update_on="iteration",
)
solver.solve(x0)
```

Both methods use the frontend's combined objective/constraint evaluations and
analytic or bounded numerical first derivatives. Public inequalities stay
$g(x)\leq0$; the adapter supplies their lower/upper bounds directly. The initial
implementation declares a dense Jacobian and uses quasi-Newton Hessians:
limited-memory for IPM and damped BFGS for SQP. Exact Hessians, sparse structure,
and cross-solve working-set reuse are not yet exposed by this adapter.

Generic tolerance and iteration limits map to `tol`/`max_iter`, and on SQP also
to `sqp_tol`, `sqp_constr_viol_tol`, `sqp_dual_inf_tol`, and `sqp_max_iter`.
Additional Pounce options go in `extra_options`. Internal NLP scaling and bound
relaxation default to off, preserving the frontend's coordinates and declared
bounds. The backend keeps native termination and all diagnostic arrays in
`backend_result.info`. The public result adds `native_info`, `native_multipliers`
(in solver coordinates), and `effective_options`. Strict native success and
original-model feasibility are required; with `check_kkt=True`, an independent
first-order assessment of the returned point must also pass at
`50*tolerance`. An acceptable-level native exit remains distinguishable through
`solver_success` and is not promoted to strict success. This assessment does
not prove a minimum for a nonconvex model.

Pounce's iteration hook supplies scalar metrics without a current-iterate
accessor. The adapter follows Pounce's own `minimize` facade by retaining the
last objective-evaluation coordinates. `iteration_history` records these as
`event_kind="native_iteration"`, `accepted=None`, and
`coordinate_source="last_objective_evaluation"`. `native_metrics` preserves
Pounce's internal objective, residuals, barrier parameter, step information, and
line-search counts separately from original-model values. A restoration hook
describes a feasibility subproblem and has no original-model point: it appears
only in `backend_result.history` and `backend_callback`, with `x=None`.
The native returned solution always owns the final result.

`callback_functions` receive copied regular native coordinates and callback
counts independently of `update_on`; `backend_callback` receives complete
detached records, including restoration. Return `False` or raise `StopIteration`
to cancel. Other application exceptions propagate, rather than being hidden
by Pounce's native evaluation-failure handling. `check_kkt=False` disables
automatic multiplier fitting, while explicitly requested KKT reports still work.

There is no Python setting equivalent to custom SLSQP's
`first_step_fraction=0.20`. Pounce's line-search reduction factors and
fraction-to-boundary settings have different meanings; the adapter rejects
`first_step_fraction`, `max_step_fraction`, and `step_growth_factor`. A genuine
first-step cap would require an upstream Pounce interface change.

## Solving equations as equality constraints

Use `as_optimization_problem` to solve an existing root problem through an
equality-capable optimizer. For residual equations $F(x)=0$, the adapter creates

$$
\min_x 0 \qquad \text{subject to } F(x)=0.
$$

It returns an `OptimizationProblem` wrapper without changing the original
problem or evaluating it during conversion. The packed fitness is
`[0, *residual(x)]`. If the root problem supplies `gradient(x)`, its Jacobian is
reused with a zero objective-gradient row; otherwise the optimization frontend
uses its configured numerical differentiation.

```python
import numpy as np
import pysolver_view as psv

equations = psv.ExtendedRosenbrockProblem(n=2)
x0 = equations.x0
problem = psv.as_optimization_problem(equations, n_variables=x0.size)
solver = psv.OptimizationSolver(
    problem,
    library="pysolver",
    method="slsqp",
    tolerance=1e-8,
    max_iterations=200,
    extra_options={
        "first_step_fraction": 0.2,
        "max_step_fraction": 1.0,
        "step_growth_factor": 1.5,
    },
    print_convergence=False,
)
x = solver.solve(x0)
residual_norm = np.linalg.norm(equations.residual(x))
root_found = residual_norm <= 1e-8
print(solver.success, root_found, residual_norm)
```

`n_equations` defaults to `n_variables` for square systems. Supply it explicitly
for rectangular systems. Bounds default to unbounded variables; optional
`bounds=(lower, upper)` follows the optimization problem convention. The adapter
does not automatically rescale equations, infer physical bounds or apply the
root solver's `x_scale`. The original residual units and domain-error behavior
remain part of the model contract.

The selected backend must support the constraint count and rank. In particular,
SLSQP cannot handle more equality constraints than variables, and dependent
constraint rows can prevent its subproblem from being solved.

Keep optimizer success and root acceptance separate. With a zero objective,
stationarity alone provides no evidence that the equations hold. Check the
returned point against the original residual tolerance. A maximum constraint
violation and the residual two-norm are different metrics; the explicit
post-solve evaluation above is additional model work outside solver counters.

## Example 1: Unconstrained optimization

### Problem definition

The general Rosenbrock problem, also known as Rosenbrock's banana function, in `n` dimensions is defined as:

$$
\begin{aligned}
\text{minimize} \quad  & f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right] \\
\end{aligned}
$$

### Problem implementation

In order to solve this problem with `OptimizationSolver` we have to define a problem class with the following methods:

- `fitness(x)`
- `get_bounds()`
- `get_nec()`
- `get_nic()`

As the Rosenbrock problem is a very common test proplem, it is already implemented in the package.
The code snipped below shows the definition of the `RosenbrockProblem` class:

```python
import numpy as np
from pysolver_view import OptimizationProblem, combine_objective_and_constraints

class RosenbrockProblem(OptimizationProblem):
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        f = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
        return combine_objective_and_constraints(f, None, None)

    def get_bounds(self):
        return (-np.inf * np.ones(self.dim), np.inf * np.ones(self.dim))

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0
```

Note that the definition of the `RosenbrockProblem` class is based on the abstract class `OptimizationProblem`.
The `combine_objective_and_constraints` helper packs the objective and constraints
into a single vector. The built-in `psv.RosenbrockProblem` also provides analytical
derivatives and finite bounds of $-10 \le x_i \le 10$.

### Problem solution

Now that we understand how to implement optimization problems, let move to how to solve them using the `OptimizationSolver` class.

1. **Import Necessary Packages**

   Let's start by importing the packages we need:

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. **Importing PySolverView**

   Install the package as described in the [installation guide](introduction.md), then import it:

   ```python
   import pysolver_view as psv
   ```

3. **Setting Up Plot Options**

   Customize your plots with PySolverView's built-in functionalities for high-quality figures:

   ```python
   psv.set_plot_options(grid=False)
   ```

4. **Logger Initialization**

   For efficient debugging and analysis, initialize a logger for the optimization process:

   ```python
   logger = psv.create_logger("convergence_history", use_datetime=True)
   ```

5. **Rosenbrock Problem Solution**

   Set your initial guess, define the problem, and solve:

   ```python
   x0 = np.asarray([2, 2, 2, 2]) # Rosenbrock's problem in 4 dimensions
   problem = psv.RosenbrockProblem(dim=len(x0))
   solver = psv.OptimizationSolver(
       problem, method="slsqp", print_convergence=True,
       plot_convergence=True, logger=logger,
   )
   sol = solver.solve(x0)
   plt.show()  # This will keep the plot window open
   ```

The logger records convergence progress. The following output is retained from the
original example; iteration counts and report formatting can differ by version:

```text
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
```

In addition, the script will also plot the optimization progress as illustrated in the figure below

```{figure} figures/convergence_history_RosenbrockProblem.svg
:align: center
:width: 80%

Optimization Convergence History for the unconstrained Rosenbrock's problem
```

Congratulations! You've now successfully set up and solved the Rosenbrock problem using PySolverView.
For more examples, see the [optimization demos](https://github.com/turbo-sim/pysolver_view/tree/main/demos/tutorials).

## Example 2: Constrained optimization

### Problem definition

In this example we will extend the Rosenbrock problem to include trigonometric-exponential constraints:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right] \\
\text{s.t.} \quad & 3x_{k+1}^3 + 2x_{k+2} - 5 + \sin(x_{k+1}-x_{k+2})\sin(x_{k+1}+x_{k+2}) + \\
                    & + 4x_{k+1} - x_k \exp(x_k-x_{k+1}) - 3 = 0, \; \forall k=1,...,n-2 \\
                    & -5 \le x_i \le 5, \forall i=1,...,n
\end{aligned}
$$

### Problem implementation

This problem is also implemented in the package, so we will only have to import it in our script.
The simplified class below illustrates the problem contract with the bounds above.
The built-in `psv.RosenbrockProblemConstrained` uses bounds of $-10 \le x_i \le 10$
and also provides derivatives:

```python
import numpy as np
from pysolver_view import OptimizationProblem, combine_objective_and_constraints

class RosenbrockProblemConstrained(OptimizationProblem):
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        f = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
        c_eq = []
        for k in range(self.dim - 2):
            value = (3 * x[k+1]**3 + 2 * x[k+2] - 5
                     + np.sin(x[k+1] - x[k+2]) * np.sin(x[k+1] + x[k+2])
                     + 4 * x[k+1] - x[k] * np.exp(x[k] - x[k+1]) - 3)
            c_eq.append(value)
        return combine_objective_and_constraints(f, c_eq, None)

    def get_bounds(self):
        return (-5 * np.ones(self.dim), 5 * np.ones(self.dim))

    def get_nec(self):
        return self.dim - 2

    def get_nic(self):
        return 0
```

### Problem solution

Let's write a script to solve this problem with the `OptimizationSolver` class.

1. **Import Necessary Packages**

   Start by importing the packages we need:

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   ```

2. **Importing PySolverView**

   Install the package as described in the [installation guide](introduction.md), then import it:

   ```python
   import pysolver_view as psv
   ```

3. **Setting Up Plot Options**

   Customize your plots with PySolverView's built-in functionalities for high-quality figures:

   ```python
   psv.set_plot_options(grid=False)
   ```

4. **Logger Initialization**

   For efficient debugging and analysis, initialize a logger for the optimization process:

   ```python
   logger = psv.create_logger("convergence_history", use_datetime=True)
   ```

5. **Rosenbrock Problem Solution**

   Set your initial guess, define the problem, and solve:

   ```python
   x0 = np.asarray([2, 2, 2, 2]) # Rosenbrock's problem in 4 dimensions
   problem = psv.RosenbrockProblemConstrained(dim=len(x0))
   solver = psv.OptimizationSolver(
       problem, method="slsqp", print_convergence=True,
       plot_convergence=True, logger=logger,
   )
   sol = solver.solve(x0)
   plt.show()  # This will keep the plot window open
   ```

The following output is retained from the original example; iteration counts and
report formatting can differ by version:

```text
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
```

In addition, the script will also plot the optimization progress as illustrated in the figure below

```{figure} figures/convergence_history_RosenbrockProblemConstrained.svg
:align: center
:width: 80%

Optimization Convergence History for the constrained Rosenbrock's problem
```

Congratulations! You've now successfully set up and solved the constrained Rosenbrock problem using PySolverView.
For more examples, see the [optimization demos](https://github.com/turbo-sim/pysolver_view/tree/main/demos/tutorials).

## Final reports and convergence plots

After a solve, use the frontend to save its convergence history and render the
final design, constraints, KKT conditions, and Lagrange multipliers:

```python
solver.plot_convergence_history(
    savefile=True, filename="convergence_history", output_dir="output", showfig=False
)
solver.print_convergence_history(
    savefile=True, filename="convergence_history.txt", output_dir="output",
    to_console=False,
)
solver.print_optimization_report(
    include_design_variables=True,
    include_constraints=True,
    include_kkt_conditions=True,
    include_multipliers=True,
    width=120,
    savefile=True,
    filename="optimization_report.txt",
    output_dir="output",
)
```

Use `width=80`, `width=100`, or `width=120` to control the maximum number of
characters per final-report line. The default is 80; the minimum is 60.
Additional width expands the name columns, while numeric columns stay aligned.
Long variable and constraint names wrap onto continuation lines rather than
being shortened with `...`. Notes and diagnostic messages wrap as well.
The same width applies to console, logger, and saved output. Individual
`make_variables_report`, `make_constraint_report`, `make_kkt_optimality_report`,
and `make_lagrange_multipliers_report` methods also accept `width`.
This controls final report tables; the live convergence table retains its
separate column layout.

Convergence plots use detached recorded observations and save PNG and SVG.
For `library="pysolver"`, leaving both `x` and `tol` unspecified in the final
report uses the authoritative saved solution and cached public KKT certificate.
It retains the solve's separate `feasibility_tol` and `optimality_tol`, including
for rejected solutions. This rendering performs no new model evaluations,
derivative probes, or multiplier fitting. Named physical constraints and bounds
are detached with the result, so later diagnostic evaluations cannot replace
its final design metadata. Disabled or unavailable KKT metrics remain explicitly
unassessed rather than triggering another probe.

Supplying `x` (physical coordinates) or `tol` requests an explicit diagnostic
assessment instead. That path may perform additional counted model/derivative
work and uses a single `tol` for feasibility and optimality. Other optimization
backends retain their diagnostic report behavior. Convergence history is saved
separately; it is not repeated in `print_optimization_report`.
