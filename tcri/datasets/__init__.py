"""Synthetic datasets with known ground truth (``tcri.datasets``).

:func:`simulate_tcri` generates a single TCR+RNA sample whose **mutual information is
known in closed form**, which is what makes statistical recovery testing possible.

:func:`simulate_cohort` assembles those into the shape most analyses actually have —
patients as replicates, a treatment axis *within* each patient, and a response label
between them — with clones paired across timepoints so the paired metrics have something
to measure.
"""
from ._simulate import (
    mi_from_joint_oracle,
    simulate_cohort,
    simulate_from_fit_params,
    simulate_tcri,
    temperature_scale,
)

__all__ = ["simulate_tcri", "simulate_cohort", "mi_from_joint_oracle",
           "simulate_from_fit_params", "temperature_scale"]
