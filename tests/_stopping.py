"""The early-stopping rule applied to a recorded series, for tests that check where a fit stopped."""
from __future__ import annotations

import math

import lightning.pytorch as pl

from tcri.model._callbacks import ramp_is_complete

MONITOR = "objective_validation_percell"


class GatedSeries(pl.Callback):
    """Records the monitored value at every check the stopping rule sees: after the ramp."""

    def __init__(self):
        super().__init__()
        self.values: list[float] = []

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not ramp_is_complete(pl_module):
            return
        value = trainer.callback_metrics.get(MONITOR)
        if value is not None:
            self.values.append(float(value))


def stop_index(values, min_delta: float, patience: int, *, step_floor: int = 0,
               steps_per_check: int = 1):
    """Index of the check at which the rule stops on ``values``, or ``None`` if it never does.

    A check is an improvement only if it is below the best improvement so far by more than
    ``min_delta``; ``patience`` checks in a row without one stop the fit, but only at a check at
    least ``step_floor`` optimizer steps after the first, with ``steps_per_check`` steps between
    checks. This is Lightning's ``EarlyStopping`` in ``mode="min"`` with the step floor of
    ``RampGatedEarlyStopping``, applied to the gated series.
    """
    best, wait = math.inf, 0
    for i, value in enumerate(values):
        if value < best - min_delta:
            best, wait = value, 0
        else:
            wait += 1
            if wait >= patience and i * steps_per_check >= step_floor:
                return i
    return None
