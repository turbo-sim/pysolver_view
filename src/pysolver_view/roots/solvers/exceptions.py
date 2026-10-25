"""Exceptions shared by the experimental trust-region and line-search drivers."""

from ...exceptions import DomainError as DomainError


class BudgetExhausted(Exception):
    """Signal an internal residual-call budget limit, not a model-domain failure."""


class UnrepresentableProbe(Exception):
    """Identify a numerical derivative step that floating point cannot represent."""
