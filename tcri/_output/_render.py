"""The consoles, plain-text rendering, and the table and panel builders.

Tables and panels are results, so they print to stdout; bars and log lines are diagnostics, on
stderr (:mod:`._progress`, :mod:`._logging`). Every console is made when it writes, never with
``file=``, so ``contextlib.redirect_stdout`` and pytest's capture see the output in a script or a
test. In a notebook Rich writes through IPython's display instead, where ``print=False`` is the
switch.

Rich's default styles are used throughout; tcri defines no theme. A flagged state is set in bold
and said in words, so it reads the same with color off.
"""
from __future__ import annotations

import io
import math
import numbers

#: The most rows a printed table shows; the footer says where the rest is.
MAX_ROWS = 20


#: The width tables are laid out at in a notebook. Notebook output is a ``<pre>`` block that
#: scrolls sideways rather than wrapping, so a wide table prints whole instead of in pieces.
NOTEBOOK_WIDTH = 2000

#: The width a notebook shows without scrolling (Rich's default for Jupyter); a short frame
#: wider than this is printed transposed there.
NOTEBOOK_VIEW = 115


def print_out(renderable) -> None:
    """Print ``renderable`` to stdout; in a notebook, at :data:`NOTEBOOK_WIDTH`."""
    from rich.console import Console

    console = Console()
    if console.is_jupyter:
        console = Console(width=NOTEBOOK_WIDTH)
    console.print(renderable)


def plain_text(renderable, width: int | None = None) -> str:
    """``renderable`` as plain text, without color, for ``__str__`` and files. ``width`` defaults
    to what Rich uses for a stream that is not a terminal (``COLUMNS``, else 80), so this text
    matches what ``print_out`` writes to a file."""
    from rich.console import Console

    console = Console(file=io.StringIO(), width=width, color_system=None, force_terminal=False,
                      force_jupyter=False, highlight=False)
    console.print(renderable)
    return console.file.getvalue()


def cell(value) -> str:
    """One value as table text: integers with thousands separators, floats to 4 significant
    digits, a missing value as ``nan``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isnan(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)


def duration(seconds) -> str:
    """Seconds as ``m:ss``, or ``h:mm:ss`` from an hour."""
    seconds = int(round(float(seconds)))
    hours, rest = divmod(seconds, 3600)
    return (f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours
            else f"{rest // 60}:{rest % 60:02d}")


class _FrameTable:
    """A frame as a table, laid out for the width it is printed at.

    A value is never wrapped or cut: every column is as wide as its widest value or header. Only
    a single value wider than the console itself is folded onto further lines. A frame of at
    most :data:`_TRANSPOSE_ROWS` rows that is wider than the view is printed transposed, one
    column per row, so a wide one-row frame (a contrast) reads as a list of fields. A longer
    frame prints as one table in a notebook, where output scrolls sideways; in a terminal or a
    file, where a long line wraps, it is printed in consecutive pieces of columns that each fit,
    each headed with the columns it holds and repeating the frame's label columns, the
    non-numeric columns it starts with.
    """

    def __init__(self, frame, cols, max_rows, heading, footer):
        self._shown = frame[cols].head(max_rows)
        self._n_rows, self._cols = len(frame), cols
        self._heading, self._footer = heading, footer

    def _width(self, c) -> int:
        from rich.cells import cell_len

        return max([cell_len(str(c))] + [cell_len(cell(v)) for v in self._shown[c]])

    def _table(self, cols, max_width):
        from rich import box
        from rich.table import Table

        table = Table(box=box.SIMPLE_HEAD)
        for c in cols:
            numeric = self._shown[c].dtype.kind in "iuf"
            justify = "right" if numeric else "left"
            if self._width(c) + 4 > max_width:     # one value wider than the console: fold it
                table.add_column(str(c), justify=justify, overflow="fold")
            else:
                table.add_column(str(c), justify=justify, min_width=self._width(c),
                                 no_wrap=True)
        for row in self._shown[cols].itertuples(index=False):
            table.add_row(*(cell(v) for v in row))
        return table

    def _transposed(self):
        from rich import box
        from rich.table import Table

        table = Table(box=box.SIMPLE_HEAD)
        table.add_column("field", overflow="fold")
        for i in range(len(self._shown)):
            table.add_column(str(i), justify="right", overflow="fold")
        for c in self._cols:
            table.add_row(str(c), *(cell(v) for v in self._shown[c]))
        return table

    def _labels(self):
        """The leading non-numeric columns, which name a row. A text column after the first
        number (a formatted interval, say) is a value."""
        labels = []
        for c in self._cols:
            if self._shown[c].dtype.kind in "iufb":
                break
            labels.append(c)
        return labels

    def _pieces(self, max_width: int):
        """The value columns in consecutive groups that each fit ``max_width`` next to the
        label columns."""
        labels = self._labels()
        values = [c for c in self._cols if c not in labels]
        base = sum(self._width(c) + 3 for c in labels) + 1
        pieces, piece, used = [], [], base
        for c in values:
            need = self._width(c) + 3
            if piece and used + need > max_width:
                pieces.append(piece)
                piece, used = [], base
            piece.append(c)
            used += need
        if piece or not pieces:
            pieces.append(piece)
        return pieces

    def layout(self, max_width: int, notebook: bool):
        """``("table",)``, ``("transposed",)`` or ``("pieces", [value columns per piece])``."""
        natural = sum(self._width(c) + 3 for c in self._cols) + 1
        view = min(max_width, NOTEBOOK_VIEW) if notebook else max_width
        if self._n_rows <= _TRANSPOSE_ROWS and natural > view:
            return ("transposed",)
        if natural <= max_width or notebook:
            return ("table",)
        return ("pieces", self._pieces(max_width))

    def __rich_console__(self, console, options):
        from rich.text import Text

        yield self._heading
        plan = self.layout(options.max_width, console.is_jupyter)
        if plan[0] == "table":
            yield self._table(self._cols, options.max_width)
        elif plan[0] == "transposed":
            yield self._transposed()
        else:
            labels, n, start = self._labels(), len(self._cols), len(self._labels())
            for i, piece in enumerate(plan[1]):
                first = 1 if i == 0 else start + 1
                last = start + len(piece)
                held = f"columns {first}–{last} of {n}"
                if i and labels:
                    held += f", with {', '.join(map(str, labels))} repeated"
                yield Text(held, style="dim")
                yield self._table(labels + piece, options.max_width)
                start = last
        yield from self._footer


#: The most rows a frame can have to be printed transposed when it is too wide.
_TRANSPOSE_ROWS = 5


def frame_table(frame, *, title, columns=None, max_rows: int = MAX_ROWS, rest: str | None = None,
                notes=()):
    """``frame`` as a Rich table under ``title``.

    Parameters
    ----------
    frame
        The DataFrame to show.
    title
        The table's title, a string or a ``rich.text.Text``.
    columns
        The columns to show, in order; every column when ``None``. Absent names are skipped.
    max_rows
        Rows shown; a frame with more says how many more, and ``rest`` says where they are.
    rest
        Where the full frame is, for the footer: a ``tcri.get`` call, or a returned dict's key.
    notes
        Lines printed under the table, such as a contrast.

    Returns
    -------
    A Rich renderable: the title, the table and the footer lines. A named index is printed as
    leading columns. A frame too wide for the console is printed transposed (at
    most five rows) or in blocks of columns.
    """
    from rich.text import Text

    if any(n is not None for n in frame.index.names):
        frame = frame.reset_index()           # a named index carries the rows' identity
    cols = [c for c in (columns or list(frame.columns)) if c in frame.columns]
    # the title is a line of its own, so a narrow table does not wrap it
    heading = title if isinstance(title, Text) else Text(str(title))
    footer = [Text(str(n)) for n in notes]
    hidden = len(frame) - max_rows
    if hidden > 0:
        where = f": {rest}" if rest else ""
        footer.append(Text(f"{hidden:,} more rows{where}"))
    return _FrameTable(frame, cols, max_rows, heading, footer)


def panel(title, rows):
    """A titled panel of ``(label, value)`` rows; a row whose value is ``None`` is left out.

    ``value`` is a string or a ``rich.text.Text``; a multi-line value spans several lines.
    """
    from rich.panel import Panel
    from rich.table import Table

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", overflow="fold")
    grid.add_column(overflow="fold")
    for label, value in rows:
        if value is not None:
            grid.add_row(label, value)
    return Panel(grid, title=title, title_align="left", expand=False)
