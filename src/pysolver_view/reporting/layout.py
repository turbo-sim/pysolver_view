"""Width-bounded plain-text tables that preserve complete cell contents."""

from itertools import zip_longest
from numbers import Integral
import textwrap


def validate_report_width(width):
    """Return a character width large enough for the report's numeric columns."""
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 60:
        raise ValueError("Report width must be an integer of at least 60 characters.")
    return int(width)


def wrap_text(text, width):
    """Wrap notes and titles without discarding long words or identifiers."""
    return textwrap.wrap(
        str(text), width=width, break_long_words=True, break_on_hyphens=False
    ) or [""]


def table_report(title, headers, rows, columns, width, notes=()):
    """Render aligned cells with complete wrapped names and bounded line widths.

    The first column is left aligned; remaining columns are right aligned.
    Continuation lines keep their original column, including numeric cells
    whose unusual values exceed the normal scientific-notation field width.
    Only supplied text is formatted; there is no model or diagnostic access.
    """
    width = validate_report_width(width)
    if sum(columns) + 1 != width:
        raise ValueError("Report columns must fill the requested width.")

    def render(cells):
        """Keep each wrapped cell aligned with the corresponding column."""
        wrapped = [wrap_text(cell, size) for cell, size in zip(cells, columns)]
        for parts in zip_longest(*wrapped, fillvalue=""):
            yield (
                " "
                + "".join(
                    part.ljust(size) if index == 0 else part.rjust(size)
                    for index, (part, size) in enumerate(zip(parts, columns))
                )
            ).rstrip()

    separator = "-" * width
    lines = ["", separator]
    lines.extend(" " + line for line in wrap_text(title, width - 1))
    lines.append(separator)
    lines.extend(render(headers))
    lines.append(separator)
    for row in rows:
        lines.extend(render(row))
    for note in notes:
        lines.extend(" " + line for line in wrap_text(note, width - 1))
    lines.append(separator)
    return "\n".join(lines)
