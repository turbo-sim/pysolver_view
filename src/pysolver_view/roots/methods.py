"""Canonical root method names shared by native and custom adapters."""

# Historical public name for the native SciPy method identifiers.
SOLVER_OPTIONS = ["hybr", "lm"]

METHOD_ALIASES = {
    "levenberg_marquardt": "levenberg_marquardt",
    "lm": "levenberg_marquardt",
    "powell_hybrid": "powell_hybrid",
    "hybr": "powell_hybrid",
    "newton_raphson": "newton_raphson",
    "newton": "newton_raphson",
    "gauss_newton": "gauss_newton",
    "gn": "gauss_newton",
}

METHOD_DISPLAY_NAMES = {
    "levenberg_marquardt": "Levenberg–Marquardt",
    "powell_hybrid": "Powell hybrid",
    "newton_raphson": "Newton–Raphson",
    "gauss_newton": "Gauss–Newton",
}


def normalize_method(method):
    """Resolve one documented root-method spelling to its canonical identifier."""
    if not isinstance(method, str) or method.strip().lower() not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown root method; choose from {', '.join(METHOD_ALIASES)}."
        )
    return METHOD_ALIASES[method.strip().lower()]
