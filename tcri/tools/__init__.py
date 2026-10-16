"""``tcri.tools`` (``tl``) — the joint-distribution engine + the four metric twins.

Every tool computes once and stores its result under a namespaced ``uns`` key with a ``params``
provenance block. ``pl`` reads that store; it never recomputes. Mutual information and the two
entropies store ``{replicate, item, stats}`` and return the replicate summary, one row per
replicate; the other tools return what they store.

``splitby`` produces the between-split contrast as part of the metric, stored under ``stats``.
The replicate unit is the group: the test runs on one value per group, so items are never
counted as replicates.
"""
from ._joint import joint_distribution
from ._entropy import clonotypic_entropy, phenotypic_entropy
from ._mutual_information import mutual_information
from ._flux import phenotypic_flux
from ._delta import (delta_clonotypic_entropy,
                     delta_phenotypic_entropy)

__all__ = [
    "joint_distribution",
    "clonotypic_entropy",
    "phenotypic_entropy",
    "mutual_information",
    "phenotypic_flux",
    "delta_clonotypic_entropy",
    "delta_phenotypic_entropy",
]
