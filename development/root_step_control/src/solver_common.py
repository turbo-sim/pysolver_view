"""Exceptions shared by the experimental trust-region and line-search drivers."""


class DomainError(ValueError):
    """Identify an explicitly recognized undefined model state.

    Drivers may shorten a trial after this exception. An undefined initial point
    or derivative probe terminates the solve; unexpected errors propagate.
    """


class BudgetExhausted(Exception):
    """Signal an internal residual-call budget limit, not a model-domain failure."""
