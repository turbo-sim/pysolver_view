"""Width limits preserve full identifiers, numeric columns, and diagnostics."""

from copy import deepcopy

import numpy as np
import pytest

from pysolver_view import OptimizationSolver
from pysolver_view.reporting import tables

NAME = "stages.0.blade_geometry.stator.trailing_edge_thickness_to_opening"
CONSTRAINT = "stages_performance.0.geometry_interspaces.before_rotor.flaring_angle"


def table_rows(report):
    """Read only body lines between the column-heading and final separators."""
    lines = report.splitlines()
    separators = [i for i, line in enumerate(lines) if line and set(line) == {"-"}]
    return lines[separators[2] + 1 : separators[-1]]


@pytest.mark.parametrize("width", [60, 80, 100, 120, 200])
@pytest.mark.parametrize("normalized", [False, True])
def test_variable_names_wrap_completely_and_numeric_columns_stay_aligned(
    width, normalized
):
    """A wider name column reduces wrapping without dropping any characters."""
    report = tables.variables_report(
        [0.025], ([0.02], [0.1]), [NAME], normalized, width=width
    )
    assert max(map(len, report.splitlines())) <= width
    assert "..." not in report
    rows = table_rows(report)
    name_width = width - 40
    assert "".join(line[1 : 1 + name_width].strip() for line in rows) == NAME
    if width >= 120:
        assert NAME in report and len(rows) == 1
    numbers = rows[0][1 + name_width :]
    for index in range(3):
        assert len(numbers[index * 13 : (index + 1) * 13]) == 13
        assert float(numbers[index * 13 : (index + 1) * 13]) == pytest.approx(
            [0.02, 0.025, 0.1][index]
        )


@pytest.mark.parametrize("width", [80, 100, 120])
def test_constraint_and_multiplier_names_are_complete_with_consistent_widths(width):
    """Both physical constraints and finite bound multipliers keep their names."""
    entries = [dict(name=CONSTRAINT, type="<", value=10.0, target=10.0, satisfied=True)]
    kkt = dict(
        available=True, multipliers_eq=[], multipliers_ineq={0: 0.5, 1: 0.0, 2: 0.25}
    )
    before = deepcopy(entries), deepcopy(kkt)
    reports = [
        tables.constraint_report(entries, width=width),
        tables.multipliers_report(kkt, entries, [NAME], ([0.0], [1.0]), width=width),
    ]
    for report in reports:
        assert max(map(len, report.splitlines())) <= width
        assert all(
            len(line) == width
            for line in report.splitlines()
            if line and set(line) == {"-"}
        )
        assert "..." not in report
        rows = table_rows(report)
        name_width = width - 31
        assert "".join(line[1 : 1 + name_width].strip() for line in rows).startswith(
            CONSTRAINT
        )
        if width >= 120:
            assert CONSTRAINT in report
    assert entries == before[0] and kkt == before[1]


@pytest.mark.parametrize("width", [60, 80, 100, 120])
def test_unavailable_kkt_notes_and_names_stay_bounded(width):
    """Long unavailable reasons wrap honestly instead of overflowing or truncating."""
    reason = "Unavailable multiplier assessment because " + "diagnostic_identifier_" * 8
    assessment = dict(available=False, message=reason)
    report = tables.final_kkt_report(assessment, 1e-6, 1e-4, width=width)
    assert max(map(len, report.splitlines())) <= width
    assert "unavailable" in report and "n/a" in report
    assert "passed." not in report
    assert reason.replace(" ", "") in "".join(report.split())
    assert "..." not in report


def test_very_large_fixed_point_values_cannot_overflow_the_requested_width():
    """Even atypical diagnostic values remain complete in aligned continuation cells."""
    report = tables.variables_report([1e100], ([0.0], [np.inf]), [NAME], width=80)
    assert max(map(len, report.splitlines())) <= 80
    assert "inf" in report


@pytest.mark.parametrize("width", [None, True, False, 0, 59, 80.0, "120"])
def test_invalid_width_is_rejected_before_any_diagnostic_model_access(width):
    """Bad presentation options cannot accidentally trigger a diagnostic evaluation."""
    solver = OptimizationSolver.__new__(OptimizationSolver)
    for operation in (
        lambda: solver.print_optimization_report(width=width),
        lambda: solver.make_variables_report([0.0], width=width),
        lambda: solver.make_constraint_report([0.0], 1e-6, width=width),
        lambda: solver.make_kkt_optimality_report([0.0], 1e-6, width=width),
        lambda: solver.make_lagrange_multipliers_report([0.0], 1e-6, width=width),
    ):
        with pytest.raises(ValueError, match="width"):
            operation()
