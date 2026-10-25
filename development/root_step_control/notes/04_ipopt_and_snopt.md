# Applying the approach to IPOPT and SNOPT

The same principles apply: choose sensible coordinates, restrict risky trials
before calling the property package, communicate genuine evaluation failures,
and recover through the solver's own globalization mechanism. The implementation
must follow each backend's supported interfaces. I recommend native adapters
for IPOPT and SNOPT before considering a modified solver implementation.

This is a researched design note, not an implemented or tested IPOPT/SNOPT
adapter. Sources were checked on 24 October 2026. The cited SNOPT guide is version
7.7; the current cyipopt documentation identifies itself as 1.8.0.dev0. Confirm
behavior against the installed solver and Python binding before promising a
particular control.

## IPOPT: recovery exists, but the controls have different meanings

IPOPT's objective and constraint callbacks, `eval_f` and `eval_g`, return a
success indicator. Its `intermediate_callback` is an iteration-level monitoring
and termination interface; returning false there stops optimization. It is not
a public suspended-trial interface for changing the pending step. These roles
are distinct in the [TNLP API](https://coin-or.github.io/Ipopt/classIpopt_1_1TNLP.html).

In cyipopt, raise `CyIpoptEvaluationError` for a recognized undefined objective
or constraint evaluation. The documented response is to attempt recovery by
shortening the step. The same exception in gradient, Jacobian, or Hessian
callbacks leads to an invalid-number exit; it is not supported in structure or
intermediate callbacks. Ordinary Python exceptions are not a substitute for
this protocol. An invalid initial point also needs separate handling: the
existence of a trial-recovery mechanism does not make an invalid start usable.
See the [cyipopt exception contract](https://cyipopt.readthedocs.io/latest/reference.html#cyipopt.CyIpoptEvaluationError).

The main options relevant to fragile property evaluations are:

| Option | Meaning for this application |
| --- | --- |
| `alpha_red_factor` | Reduction factor during backtracking, default 0.5. It does not make the first attempted step small. |
| `tau_min` | Lower bound on the fraction-to-the-boundary parameter. This concerns distance to variable/slack bounds, not a general scaled displacement cap. |
| `bound_relax_factor` | Set to zero when relaxing declared bounds would permit invalid states. This removes bound relaxation, not every possible domain failure. |
| `honor_original_bounds` | Controls projection of the final point. It does not protect intermediate evaluations. |
| `accept_every_trial_step` | Keep disabled for this use case; enabling it bypasses the normal line-search acceptance safeguards. |

These distinctions follow from the [IPOPT options reference](https://coin-or.github.io/Ipopt/OPTIONS.html).

IPOPT normally uses a filter line search. Reducing a selected root residual norm
at every accepted iteration is not its general acceptance rule. If equations
are represented as nonlinear equality constraints, the algorithm can trade
objective and feasibility progress. Adding an independent monotonic-residual
test would therefore change the method, rather than merely configure it.
The documented default is `line_search_method="filter"` in the
[options reference](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_line_search_method).

My recommendation is to expose evaluation-error recovery, explicit bounds,
scaling, and reporting first. A precise first-step radius or persistent
$\|p/s\|_2$ bound would need additional backend work or a validated admissibility
guard. Such a guard must know the accepted base and distinguish line-search
trials, correction steps, restoration, and derivative calls. Observing the last
function call is insufficient. Do not clip coordinates inside a callback or
return property values from a different point.

## SNOPT: a native step limiter is already relevant

SNOPT's `Major step limit` is particularly close to the requested protection.
For a search direction $p$, its documented restriction uses

$$
\bar\beta = \frac{r(1+\|x\|)}{\|p\|},\qquad
\alpha_1=\min(1,\bar\beta,\beta),
$$

where $\beta$ is the applicable line-search upper limit. The first line-search
evaluation can consequently be shortened before evaluating the model. The
manual conditions this mechanism on obtaining a feasible solution or
subproblem. This is a relative, state-dependent restriction in the solver's
coordinates, not the experiment's fixed physical or scaled radius. It is also
not a separate setting for the first iteration alone. See the
[native Major step limit description](https://web.stanford.edu/group/SOL/software/snoptHelp/The_SPECS_file/Description_of_optional_parameters/snopt_Specs_Description_Major_step_limit.htm).

The `snOptA` function callback can return `Status=-1` when functions or
derivatives are undefined. During line search, SNOPT then shortens the step and
tries again. `Status<=-2` requests termination. The callback must not alter the
input coordinates. The guide also explains that bounds and linear constraints
help restrict function evaluations, subject to feasibility tolerances; they do
not establish validity of a general nonlinear thermodynamic state.
See [SNOPT 7.7, sections 3.5-3.6](https://ccom.ucsd.edu/reports/UCSD-CCoM-19-01.pdf).

Other controls serve different purposes: `Linesearch tolerance` controls search
accuracy; `Function precision` describes function accuracy; `Difference interval`
and `Central difference interval` govern numerical derivatives. The current
online option list also describes `Backoff factor` for an undefined line-search
evaluation. Check availability in the installed version. None of these names
alone implies a maximum physical trial distance. See the
[SNOPT option reference](https://ccom.ucsd.edu/~optimizers/docs/snopt/options.html).

Start by exposing `Major step limit` and the callback failure protocol in a
pysolver_view adapter. The documentation suggests trying values such as 0.1 or
0.01 for rapidly varying functions, but these are experiments to calibrate
after scaling, not guaranteed safe thermodynamic limits. A schedule that starts
conservatively and grows after successful iterations would require a verified
mechanism for changing the live solver setting. Do not assume that any Python
monitoring callback supports that mutation or that restarting preserves all
solver state.

## What should be shared in pysolver_view

Share the model-facing contract: recognized domain failures, fixed variable and
residual scaling, optional cheap validity checks, derivative safeguards, and
evaluation histories. Keep the numerical acceptance rule backend-specific.
SLSQP and SNOPT have merit functions; IPOPT has its own filter/barrier logic;
the experimental root solvers use residual-norm reduction. A single universal
acceptance wrapper would conceal these differences.

Neither a major-step limiter nor successful backtracking proves that all
finite-difference probes are safe. Supply reliable derivatives where possible;
otherwise separately test derivative perturbations near property-domain and
phase boundaries. Also test that a rejected property call leaves any mutable
thermodynamic state usable for the next evaluation.

For both future adapters, record actual model calls and audit their locations
before claiming a hard cap. Keep claims narrow: a verified native line-search
limit, a detected domain rejection, and a certified bound on every residual
evaluation are different capabilities. The present root-step experiment only
establishes the last-accepted-base trial bound for its own solver loops.
