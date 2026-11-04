"""The Rich bars, on stderr.

Each bar refreshes from the code that advances it, on the calling thread (``auto_refresh=False``),
and never redirects ``sys.stdout`` or ``sys.stderr``. In a terminal it is a live line; in a
notebook a live widget (ipywidgets); in a file or a pipe Rich writes only its final state.

- :func:`status_progress` — one status line with no total, for a fit whose end is not known in
  advance.
- :func:`set_progress` — a bar over a known number of items, for a set of null fits.
- :func:`pass_progress` — a bar over passes of cell batches, drawn by the outer call of
  ``perturb.gene_importance`` and advanced by its parent pass and every reference pass;
  :func:`current_passes` gives a pass the open bar, or ``None``.

Each yields ``None`` when disabled, so the code that advances a bar checks for it.
"""
from __future__ import annotations

import contextlib
import contextvars


def _progress(*columns):
    from rich.console import Console
    from rich.progress import Progress

    return Progress(*columns, console=Console(stderr=True), auto_refresh=False,
                    redirect_stdout=False, redirect_stderr=False, transient=False)


class _Status:
    def __init__(self, progress, task):
        self._progress, self._task = progress, task

    def update(self, text: str) -> None:
        self._progress.update(self._task, description=text)
        self._progress.refresh()


@contextlib.contextmanager
def status_progress(enabled: bool, text: str):
    """A status line showing ``text`` and the elapsed time; yields its ``update`` holder."""
    if not enabled:
        yield None
        return
    from rich.progress import TextColumn, TimeElapsedColumn

    progress = _progress(TextColumn("{task.description}"), TimeElapsedColumn())
    progress.start()
    try:
        status = _Status(progress, progress.add_task(text, total=None))
        progress.refresh()
        yield status
    finally:
        progress.refresh()
        progress.stop()


class _Set:
    def __init__(self, progress, task, label):
        self._progress, self._task, self._label = progress, task, label

    def describe(self, text: str) -> None:
        self._progress.update(self._task, description=f"{self._label} · {text}")
        self._progress.refresh()

    def advance(self) -> None:
        self._progress.advance(self._task)
        self._progress.refresh()


@contextlib.contextmanager
def set_progress(enabled: bool, label: str, total: int):
    """A bar over ``total`` items, labeled ``label``; yields its ``advance``/``describe`` holder."""
    if not enabled:
        yield None
        return
    from rich.progress import BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

    progress = _progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                         TimeElapsedColumn())
    progress.start()
    try:
        bar = _Set(progress, progress.add_task(label, total=total), label)
        progress.refresh()
        yield bar
    finally:
        progress.refresh()
        progress.stop()


class _Passes:
    """Passes of cell batches: ``next_pass`` names the pass, ``batches`` sizes it, ``advance``
    steps one batch. The total is the number of passes times the batches in a pass."""

    def __init__(self, progress, label):
        self._progress, self._label = progress, label
        self._task = progress.add_task(label, total=None)
        self._n_passes, self._done, self._sized = 1, 0, False

    def set_passes(self, n: int) -> None:
        self._n_passes = max(int(n), 1)

    def next_pass(self, name: str) -> None:
        self._done += 1
        self._progress.update(self._task, description=(
            f"{self._label} {self._done}/{self._n_passes} · {name}"))
        self._progress.refresh()

    def batches(self, n: int) -> None:
        if not self._sized:
            self._progress.update(self._task, total=self._n_passes * max(int(n), 1))
            self._sized = True

    def advance(self) -> None:
        self._progress.advance(self._task)
        self._progress.refresh()


_PASSES: contextvars.ContextVar = contextvars.ContextVar("tcri_passes", default=None)


@contextlib.contextmanager
def pass_progress(enabled: bool, label: str):
    """Open the bar over passes for the outer call; an inner call reuses the open one."""
    if not enabled or _PASSES.get() is not None:
        yield _PASSES.get()
        return
    from rich.progress import BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn

    progress = _progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                         TimeElapsedColumn())
    progress.start()
    passes = _Passes(progress, label)
    token = _PASSES.set(passes)
    try:
        yield passes
    finally:
        _PASSES.reset(token)
        progress.refresh()
        progress.stop()


def current_passes():
    """The open bar over passes, or ``None``."""
    return _PASSES.get()
