"""``tcri.diag`` — diagnostics: posterior-predictive checks (all return DataFrames) + the
two relocated training plots. Read-only checks on a finalized model.
"""
from ._ppc import (
    joint_distribution_ppc,
    phenotype_calibration,
    reconstruction_ppc,
    permutation_null,
)
from ._training import loss, archetypes

__all__ = [
    "joint_distribution_ppc",
    "phenotype_calibration",
    "reconstruction_ppc",
    "permutation_null",
    "loss",
    "archetypes",
]
