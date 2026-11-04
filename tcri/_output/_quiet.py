"""``quiet_third_party``: drop the named scvi and Lightning lines, for the span of one call.

Each line is named, so anything else scvi or Lightning says still shows, including warnings about
Trainer arguments the caller passed and Lightning's "Detected KeyboardInterrupt" and "Received
SIGTERM" notices. ``tests/test_console_output.py`` fails when one of these is reworded upstream,
so the list is updated deliberately.

Warnings are dropped by ``warnings.catch_warnings`` plus one ignore filter per line. That swaps
the process-wide filter list for the span (``sys.flags.context_aware_warnings`` is off on standard
builds), and leaving it resets Python's record of which warnings were already shown, so a
warning shown before can show again after the call.

Log lines are dropped by a ``logging.Filter`` attached to the loggers that create them and removed
on exit. Logger levels are not touched. Lightning logs its info lines through
``lightning_utilities.core.rank_zero.log``, which each Lightning package reassigns when it is
imported, so that logger is read when the span opens.
"""
from __future__ import annotations

import contextlib
import logging
import re
import warnings

#: Warning messages dropped, as the regular expressions ``filterwarnings`` matches at the start.
WARNINGS = (
    # scvi, choosing the accelerator (TrainRunner and model load)
    r"`accelerator` has been automatically set to `cpu` although 'mps' exists",
    r"`accelerator` has been set to `mps`",
    # Lightning, about settings tcri makes
    r"GPU available but not used",
    r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    r"The '(train|val)_dataloader' does not have many workers",
    r"The number of training batches \(\d+\) is smaller than the logging interval",
    # Lightning, under Slurm: tcri hides SLURM_NTASKS from it for the span of a fit, and it then
    # suggests launching with srun
    r"The `srun` command is available on your system but is not used",
)

#: Log messages dropped, by their start.
LOG_MESSAGES = (
    "GPU available: ",
    "TPU available: ",
    "💡 Tip: For seamless cloud logging",
    "Seed set to ",
    "`Trainer.fit` stopped: ",
)

#: The prefix Lightning puts on a message when a rank is set (RANK, LOCAL_RANK, SLURM_PROCID).
_RANK = re.compile(r"^\[rank: \d+\] ")

#: Loggers with a fixed name that create the lines above.
_FIXED_LOGGERS = ("lightning.fabric.utilities.seed", "scvi.data._download")


class _DropNamed(logging.Filter):
    def filter(self, record):
        message = _RANK.sub("", record.getMessage())
        if record.name == "scvi.data._download":
            return not (message.startswith("File ") and message.endswith("already downloaded"))
        return not message.startswith(LOG_MESSAGES)


def _emitting_loggers():
    names = list(_FIXED_LOGGERS)
    try:
        from lightning_utilities.core import rank_zero

        names.append(rank_zero.log.name)
    except Exception:
        pass
    return [logging.getLogger(n) for n in dict.fromkeys(names)]


@contextlib.contextmanager
def quiet_third_party():
    """Drop the named lines for the span of the ``with`` block, then restore everything."""
    loggers = _emitting_loggers()
    dropped = _DropNamed()
    for log in loggers:
        log.addFilter(dropped)
    try:
        with warnings.catch_warnings():
            for message in WARNINGS:
                warnings.filterwarnings("ignore", message=message)
            yield
    finally:
        for log in loggers:
            log.removeFilter(dropped)
