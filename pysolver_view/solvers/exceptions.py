"""Shared recognized model-domain failures for custom optimization and root solvers."""

from copy import deepcopy


class DomainError(Exception):
    """Identify an undefined physical model evaluation that may permit a retry.

    Raise this only when the model recognizes a domain failure at the supplied
    point. Metadata records its cause for reporting; it does not supply substitute
    objective or constraint values. Recovery means trying another point, and can
    still fail if no valid trial step is found within the limits. Derivative
    probes are not retried.
    """

    def __init__(self, message, *, metadata=None):
        """Store a recognized domain failure and detached application metadata."""
        super().__init__(message)
        self.metadata = {} if metadata is None else deepcopy(metadata)

    def to_dict(self):
        """Return detached diagnostic details without exposing internal metadata."""
        return {**self.metadata, "message": str(self), "recoverable": True}
