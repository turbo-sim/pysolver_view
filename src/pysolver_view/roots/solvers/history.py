"""Consistent history storage, detached callbacks, and final root results."""

from copy import deepcopy

from scipy.optimize import OptimizeResult


class HistoryRecorder:
    """Store observations without deciding which trial the algorithm accepts."""

    def __init__(self, level, callback, observer=None):
        """Select coordinate storage and retain the optional accepted-point callback."""
        self.level = level
        self.callback = callback
        self.observer = observer
        self.evaluator = None
        self.enabled = level != "none"
        self.coordinates = level == "full"
        self.history = []
        self.trials = []
        self.evaluations = []

    def accepted(self, x, fun, **diagnostics):
        """Record a committed state and return whether its copied callback stops."""
        if not self.enabled and self.callback is None:
            return False
        if self.coordinates:
            diagnostics["x"] = x.copy()
        if self.enabled:
            self.history.append(diagnostics)
        if self.callback is not None:
            copied = deepcopy(diagnostics)
            if "x" not in copied:
                copied["x"] = x.copy()
            copied["fun"] = fun.copy()
            copied.update(self.evaluator.counts())
            return self.callback(copied) is False
        return False

    def trial(self, x_base, x, diagnostics):
        """Store a trial's diagnostics and optionally copy both endpoint vectors."""
        if self.enabled:
            if self.coordinates:
                diagnostics.update(x_base=x_base.copy(), x=x.copy())
            self.trials.append(diagnostics)

    def result(self, *, x, fun, status, message, linear_algebra, **diagnostics):
        """Build a result from the last accepted state, retaining method diagnostics."""
        return OptimizeResult(
            x=x.copy(),
            fun=None if fun is None else fun.copy(),
            success=status == 0,
            status=status,
            message=message,
            history=self.history,
            trials=self.trials,
            evaluations=self.evaluations,
            history_level=self.level,
            derivative_nfev=self.evaluator.derivative_nfev,
            failed_njev=self.evaluator.failed_njev,
            linear_algebra=dict(linear_algebra),
            **diagnostics,
        )
