"""``tcri.diag`` — diagnostics: four checks that return what they computed and, with
``plot=True``, draw their figure from it, plus the two training plots. Read-only checks on a
finalized model.
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
