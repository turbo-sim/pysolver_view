"""Public observation storage; numerical trial histories remain domain-local."""

from copy import deepcopy


def append_observation(history, record, *, level="full", coordinates=("x",)):
    """Append a detached record without inventing iterations or evaluating models."""
    if level == "none":
        return
    for key in history:
        value = (
            None if level == "summary" and key in coordinates else deepcopy(record[key])
        )
        history[key].append(value)
