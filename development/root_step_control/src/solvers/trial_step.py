"""Construct trial coordinates and audit their actual scaled displacement."""

import numpy as np
from scipy.linalg import norm


def prepare_trial(x, scale, direction, fraction, step_limit):
    """Construct a trial and audit its actual rounded scaled displacement.

    Keep the requested direction while reserving coordinate ulps when rounding
    would exceed a finite envelope. Return coordinates, physical/scaled steps,
    scaled norm and effective fraction. ValueError identifies numerical geometry
    failure before calling the model; it does not hide application exceptions.
    """
    for attempt in range(4):
        with np.errstate(over="ignore", invalid="ignore"):
            trial = x + scale * (fraction * direction)
            step = trial - x
            scaled_step = step / scale
        if not np.all(np.isfinite(trial)) or not np.all(np.isfinite(scaled_step)):
            raise ValueError("Trial coordinates overflowed; revise variable scaling.")
        length = float(norm(scaled_step, check_finite=False))
        if step_limit is None or length <= step_limit * (1 + 8 * np.finfo(float).eps):
            return trial, step, scaled_step, length, fraction
        if attempt == 3:
            break
        coordinate_ulp = np.spacing(np.maximum(np.abs(x), np.abs(trial)))
        reserve = 2 * norm(coordinate_ulp / scale, check_finite=False)
        fraction *= max(0.0, step_limit - reserve) / length
    raise ValueError("No representable trial within the requested step limit.")


def prepare_trust_region_trial(x, scale, scaled_step, radius):
    """Apply the trust-region radius before evaluation, including coordinate rounding.

    Preserve the driver's subproblem-first clipping order; line-search fractions
    use prepare_trial() so their rounded directional decrease stays consistent.
    """
    # Bound before evaluation, including rounding in the subproblem solve.
    size = norm(scaled_step)
    if size > radius:
        scaled_step *= np.nextafter(radius, 0.0) / size
    with np.errstate(over="ignore", invalid="ignore"):
        trial_x = x + scale * scaled_step
    if not np.all(np.isfinite(trial_x)):
        raise ValueError("Trial coordinates overflowed; revise x_scale or radius.")
    step = trial_x - x
    scaled_step = step / scale
    scaled_norm = float(norm(scaled_step))
    # Coordinate rounding can exceed the radius slightly. Shrink again;
    # if the smallest representable move exceeds it, report stagnation.
    for _ in range(3):
        if scaled_norm <= radius * (1 + 8 * np.finfo(float).eps):
            break
        # Leave room for coordinate addition, whose ulp depends on x,
        # not on the step. Merely nudging radius by one ulp can repeat
        # the identical out-of-radius point when |x/scale| is large.
        coordinate_ulp = np.spacing(np.maximum(np.abs(x), np.abs(trial_x)))
        reserve = 2 * norm(coordinate_ulp / scale)
        scaled_step *= max(0.0, radius - reserve) / scaled_norm
        trial_x = x + scale * scaled_step
        step = trial_x - x
        scaled_step = step / scale
        scaled_norm = float(norm(scaled_step))
    if scaled_norm > radius * (1 + 8 * np.finfo(float).eps):
        raise ValueError("No representable trial within the requested radius.")
    return trial_x, step, scaled_step, scaled_norm
