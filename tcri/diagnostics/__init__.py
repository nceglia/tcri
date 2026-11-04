"""``tcri.diag`` — diagnostics: four checks and the two training plots. Each check draws its
figure (``plot=True``), prints a table (``print=True``) and returns what it computed
(``return_dict=True``); turning all three off raises. ``loss`` and ``archetypes`` only draw.
Read-only checks on a finalized model.
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
