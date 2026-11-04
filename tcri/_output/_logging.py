"""The handler on the ``tcri`` logger.

Every tcri module logs on ``logging.getLogger(__name__)``, a child of ``tcri``, and its records
propagate up to ``tcri``. ``tcri`` does not propagate further; its one handler,
:class:`_Router`, decides where a record goes:

- the root logger has no handlers (a bare notebook or script, where nothing would print): the
  record is printed by Rich's ``RichHandler`` on stderr, with the level as its only column;
- the root logger has handlers (the application configured logging): the record is handed to
  them, if it is at or above the root logger's level, or if the ``tcri`` logger is no longer at
  tcri's default level, which means the user set it. ``logging.basicConfig(level="WARNING")``
  therefore keeps tcri's INFO notices out, and ``logging.getLogger("tcri").setLevel("INFO")``
  lets them through.

The default level is ``logging.INFO - 1``, a value nobody sets by name, so that an explicit
``setLevel("INFO")`` can be told apart from it.

A root handler that is also attached to the ``tcri`` logger is skipped: pytest's ``caplog``
attaches its handler to the root logger and to every logger that does not propagate, and passing
the record to it again would capture each message twice.

An application that routes the ``tcri`` logger itself keeps its routing: when the logger
propagates, or carries a handler of the application's (one that is not also a root handler),
the router does nothing, and ``import tcri`` adds no router to a ``tcri`` logger that already
has handlers.

Only the ``tcri`` logger is touched: its level, its ``propagate`` flag and its one handler. The
root logger and every other logger keep the application's configuration.
"""
from __future__ import annotations

import logging

#: tcri's default level for its own logger: INFO, and distinguishable from a user's INFO.
DEFAULT_LEVEL = logging.INFO - 1


class _Router(logging.Handler):
    """Print with Rich when nothing is configured, otherwise hand off to the root's handlers."""

    def __init__(self):
        super().__init__()
        self._rich = None

    def _rich_handler(self):
        from rich.console import Console
        from rich.logging import RichHandler

        if self._rich is None:
            self._rich = RichHandler(show_time=False, show_path=False)
        # a console made for each record, so its color and notebook detection follow the stream
        # stderr is now, after any redirect
        self._rich.console = Console(stderr=True)
        return self._rich

    def emit(self, record):
        own = logging.getLogger("tcri")
        root = logging.getLogger()
        if own.propagate:
            return                                   # the application routes it
        if any(h is not self and h not in root.handlers for h in own.handlers):
            return                                   # the application's own handler has it
        if not root.handlers:
            self._rich_handler().handle(record)
            return
        if own.level == DEFAULT_LEVEL and record.levelno < root.getEffectiveLevel():
            return
        attached = set(own.handlers)
        for handler in root.handlers:
            if handler in attached or record.levelno < handler.level:
                continue
            handler.handle(record)


def install_handler() -> None:
    """Give the ``tcri`` logger its handler, once. Called by ``import tcri``.

    A ``tcri`` logger the application configured before importing tcri (any handler on it) is
    left as it is; a level set on it alone is kept.
    """
    log = logging.getLogger("tcri")
    if log.handlers:
        return
    log.addHandler(_Router())
    if log.level == logging.NOTSET:
        log.setLevel(DEFAULT_LEVEL)
    log.propagate = False
