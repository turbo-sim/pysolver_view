"""Lazy optional JAX access; host configuration is explicit."""

from importlib.util import find_spec


def is_jax_available():
    """Return whether JAX can be found without importing or configuring it."""
    return find_spec("jax") is not None


def load_jax():
    """Load JAX only for a caller that explicitly needs automatic differentiation."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "Install pysolver_view[jax] to use automatic differentiation."
        ) from exc
    return jax, jnp


def configure_jax(*, enable_x64=True, platform=None):
    """Opt in to JAX configuration before using JAX (platform=None keeps its default)."""
    jax, _ = load_jax()
    jax.config.update("jax_enable_x64", enable_x64)
    if platform is not None:
        jax.config.update("jax_platform_name", platform)
