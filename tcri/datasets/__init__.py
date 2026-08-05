"""Synthetic datasets with known ground truth (``tcri.datasets``).

:func:`simulate_tcri` generates a TCR+RNA dataset whose **mutual information is
known in closed form**, which is what makes statistical recovery testing possible.
"""
from ._simulate import (
    mi_from_joint_oracle,
    simulate_from_fit_params,
    simulate_tcri,
    temperature_scale,
)

__all__ = ["simulate_tcri", "mi_from_joint_oracle", "simulate_from_fit_params",
           "temperature_scale"]
