"""Solver-independent numerical diagnostics for constrained optimization."""

from .kkt import assess_kkt, evaluate_kkt_residuals

__all__ = ["assess_kkt", "evaluate_kkt_residuals"]
