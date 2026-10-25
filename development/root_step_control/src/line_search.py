"""Representable scaled trial geometry used by both Newton-family methods."""

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
