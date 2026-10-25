# Step policy for fragile thermodynamic evaluations

## Start small, but distinguish a trial from an accepted step

Your intuition is sound: a large initial Newton-like correction is undesirable
when evaluating its thermodynamic states can fail or become very expensive.
Choose a conservative first trial **and** retain a maximum for later trials.
An initial restriction alone does not prevent the solver from taking a large
step a few iterations later.

The first trial cannot generally be guaranteed to reduce the residual before
the model is evaluated. Instead, require the first **accepted** step, and every
subsequent accepted root step, to meet both requirements:

1. The thermodynamic evaluation is valid for the intended model and regime.
2. The scaled residual norm decreases sufficiently relative to the predicted
   decrease, rather than merely by an arbitrarily small amount.

If a trial fails either condition, retain the last accepted point, reduce the
radius, and solve the smaller trust-region subproblem. Reducing the radius can
change the LM or dogleg direction; it is not necessarily a scalar shortening of
the previous proposal. Bound the retries and report failure if no acceptable
step is found. Small steps cannot guarantee a root exists or eliminate
nonzero-residual stationary points, inaccurate derivatives, or discontinuities.

This distinction also explains why accepted-iterate plots are insufficient:
an invalid or enormous rejected trial may already have incurred the expensive
property calculation. Inspect all trial evaluations as well.

## Scale equations and variable distances separately

For physical equations $F_i(x)=0$, choose fixed meaningful residual reference
values $q_i>0$ and pass the dimensionless residuals

$$
r_i(x)=\frac{F_i(x)}{q_i},\qquad
\phi(x)=\frac12\|r(x)\|_2^2.
$$

A raw norm mixing mass flow, pressure and energy units can give a misleading
acceptance criterion. Smaller $\|r\|_2$ also does not imply every individual
equation improved. Check individual physical residual tolerances at the final
solution when those tolerances matter. Keep residual weights fixed during a
solve so a reported reduction is not merely a change in weighting.

Variable scales $s_i$ serve a different purpose:

$$
z_i=\frac{p_i}{s_i},\qquad
\|z\|_2\leq\Delta\leq\Delta_{\max},\qquad
|p_i|\leq s_i\Delta_{\max}.
$$

The implemented `x_scale` sets $s_i$; it does **not** normalize residuals. Return
normalized residuals and their consistent Jacobian from the problem object.
The radius is a joint Euclidean limit, so all variables cannot simultaneously
take their individual component maxima.

Default `jacobian_scaling=True` adds numerical column equilibration without
changing these physical limits. If $S=\operatorname{diag}(s)$ and $C$ contains
the numerical factors, the solver factors $A=J_rSC$ and uses numerical steps
$v=C^{-1}z$. LM and hybrid first restrict $\|v\|_2\leq\Delta$, then shorten
the candidate along its ray if needed to also satisfy $\|z\|_2\leq\Delta$.
Both norms are checked after coordinate rounding. Thus automatic numerical
scaling does not make a specified temperature or pressure cap less strict.
It changes the numerical direction and can improve column balance, but does
not choose physical scales, normalize residuals, or guarantee domain validity.
Set `jacobian_scaling=False` to disable it. The
[scaling policy](07_usage_and_controls.md#jacobian-scaling-and-physical-limits)
describes how factors are updated from fresh Jacobians.

As an illustrative calculation, suppose two solver coordinates are temperature
and $y=\log(p/p_{\mathrm{ref}})$, with scales $s_T=10\,\mathrm{K}$ and $s_y=0.05$.
An initial radius of 0.2 limits the first trial to $|\Delta T|\leq2\,\mathrm{K}$
and $|\Delta y|\leq0.01$, approximately a 1% pressure change. A maximum radius
of 1 limits these to 10 K and a pressure increase of at most $e^{0.05}-1$,
approximately 5.13%. These numbers illustrate how to interpret the settings;
they are not fluid-independent safe values.

```python
from pysolver_view import StepControlOptions

# Example only: coordinates are T and log(p / p_ref).
options = StepControlOptions(
    x_scale=[10.0, 0.05],
    initial_radius=0.2,
    max_radius=1.0,
    growth_factor=1.5,
    shrink_factor=0.25,
)
```

A logarithmic pressure coordinate enforces positivity for representable values
and makes its step limit relative. It still needs numerical range checks and
does not establish phase validity. Transformations require corresponding
Jacobian chain rules. Likewise, a positive pressure or temperature alone does
not establish that a requested property calculation is meaningful.

## Require validity before accepting residual improvement

Start from a fully evaluable state. Check cheap, known restrictions before an
expensive property call, then evaluate the actual physical model. A smaller
residual cannot justify an invalid or incorrectly interpreted state. For
example, CoolProp documents ambiguous pressure/temperature states near
saturation and possible incorrect results when an inconsistent phase is
imposed. Its generalized homogeneous-state derivatives have a single-phase
scope. These are reasons to make the phase/model formulation explicit, rather
than relying on small steps alone. [CoolProp high-level interface](https://coolprop.org/coolprop/HighLevelAPI.html#vapor-liquid-and-saturation-states),
[phase handling](https://coolprop.org/coolprop/HighLevelAPI.html#imposing-the-phase-optional),
[derivative scope](https://coolprop.org/coolprop/HighLevelAPI.html#partial-derivatives).

Translate recognized, recoverable property failures into `DomainError`. Avoid
returning invented large residuals or clipping the coordinates inside
`residual(x)`: those actions change the model the solver thinks it evaluated.
Do not catch every exception, since programming errors need a visible failure.
The current solver also rejects nonfinite residuals as undefined evaluations.

An important implementation boundary is **external state**. The solver retains
its last accepted coordinates and residuals, but it cannot roll back mutable
fluid objects, imposed phases, warm-start caches, or other application state.
The problem adapter must fully establish the requested state on every call or
restore/recreate its own state after a failure. A failed property update must
not contaminate the next residual or derivative evaluation.

## Restore larger steps based on evidence

Use actual versus predicted decrease:

$$
\rho_k=\frac{\phi(x_k)-\phi(x_k+p_k)}
{-r_k^T A_kv_k-\frac12\|A_kv_k\|_2^2},
\qquad A_k\approx J_r(x_k)SC_k,\qquad v_k=C_k^{-1}S^{-1}p_k.
$$

Here $p_k$ is the actual rounded physical displacement after enforcing both
limits. $A_k$ is the current numerical Jacobian model; hybrid can use a Broyden
approximation between full Jacobian evaluations, keeping $C_k$ fixed during
those updates. With scaling disabled, $C_k=I$ and $v_k=z_k$.

With a positive denominator, the current implementation accepts a valid trial
only if the numerator is positive and $\rho_k\geq\eta$, with default
$\eta=0.1$. This already ensures the first accepted step reduces the norm of
the residual supplied to the solver. A stationary least-squares point is not
reported as a root unless the residual tolerance is met.

The implemented update policy is:

| Outcome | Radius action |
| --- | --- |
| Undefined trial, or $\rho_k<0.25$ | Multiply by `shrink_factor` |
| $\rho_k\geq0.75$ and either the physical or numerical step norm uses at least 80% of the radius | Multiply by `growth_factor`, capped at `max_radius` |
| Otherwise | Retain the radius |

Thus even a weak accepted step can lead to a smaller next radius. Growth is not
automatic after every successful iteration. A factor such as 1.5 is a useful
setting to evaluate in a thermodynamic test suite, not a guaranteed optimum.
Setting growth to 1 prevents regrowth even after recovery and can make progress
unnecessarily slow. An optional requirement for several successful steps before
regrowth after a property failure could further reduce repeated failures; that
additional recovery policy is **not implemented**.

## Protect derivatives and test the physical model

Finite-difference probes can fail before the first solver trial is even formed.
Their distance is independent of the trust radius. The current implementation
uses `finite_diff_step * x_scale[j]` and stops on an invalid probe; it does not
try a smaller or opposite-sided probe. Prefer verified analytic derivatives
when available for the intended regime. Domain-aware finite differences need
their own implementation, accuracy checks and budgets. Reducing the probe
indefinitely can amplify property noise and cancellation.
Jacobian equilibration does not change probe distances or add model calls;
an explicit `derivative_abs_step` still selects physical probe distances.

The PySolverView suite includes real CoolProp evaluations in coupled
20- and 100-variable heated-duct models. Its benchmark compares both generous and conservative trial
controls and checks physical equation errors. This supplies a reproducible
single-phase cost measurement; it does not establish reliability near saturation,
across phase changes, or for expensive nested flash calculations.

For the next validation stage, add application-specific thermodynamic systems, multiple
initial guesses, and continuation between operating points. Record property
calls and nested flash work, invalid evaluations, residual and Jacobian costs,
largest physical state changes, final equation errors, and wall time. A single
residual call may perform many property evaluations, so `nfev` alone is not a
complete cost measure.

| Available now | Proposed next additions |
| --- | --- |
| Fixed initial/max radii, conditional growth, explicit failure recovery, histories | Problem-provided cheap domain checks or physical distance limits |
| Default numerical Jacobian equilibration independent of physical `x_scale` | Problem-specific validation of scales and behavior near domain boundaries |
| Separate derivative-probe setting and failure diagnostic | Domain-aware derivative probes and checks against property noise |
| Retained accepted solver state | Application-managed thermodynamic state restoration |
| Standard mathematical suite and single-phase CoolProp duct workloads | Application thermodynamics, difficult starts and phase-boundary comparisons |

The generic defaults `initial_radius=1` and `max_radius=100` are not a
thermodynamic protection preset. Select scales and limits for the actual
variables and operating regime, and test them against realistic failures.
