"""Private: everything tcri prints, and the scoped quieting of what its dependencies print.

- :mod:`._warn`     — :func:`warn`, the one way tcri issues a warning: attributed to the
  caller's line rather than to a line inside tcri.
- :mod:`._logging`  — the handler on the ``tcri`` logger: prints with Rich when the application
  has not configured logging, and hands records to the application's handlers when it has.
- :mod:`._quiet`    — :func:`quiet_third_party`, the scoped context that drops the named scvi and
  Lightning lines and restores the caller's warning filters and logger filters on exit.
- :mod:`._render`   — the stdout and stderr consoles, plain-text rendering, and the table and
  panel builders every printed result goes through.
- :mod:`._progress` — the Rich bars: a status line, a bar over a set, and a bar over passes.

Nothing here changes global state at import or leaves anything changed after a call: no warning
filter, no logger level outside the ``tcri`` logger, no replaced stream, no global hook. This is
a lower layer; it imports nothing from the rest of tcri.
"""
from ._logging import install_handler
from ._progress import current_passes, pass_progress, set_progress, status_progress
from ._quiet import quiet_third_party
from ._render import cell, duration, frame_table, panel, plain_text, print_out
from ._warn import warn

__all__ = ["cell", "current_passes", "duration", "frame_table", "install_handler", "panel", "pass_progress",
           "plain_text", "print_out", "quiet_third_party", "set_progress", "status_progress",
           "warn"]
