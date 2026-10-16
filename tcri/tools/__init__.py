"""``tcri.tools`` (``tl``) — the joint-distribution engine + the four metric twins.

Every tool computes once, stores its result under a namespaced ``uns`` key with a ``params``
provenance block, and returns ``None``; with ``inplace=False`` it stores nothing and returns the
whole result. :mod:`tcri.get` reads the store, and ``pl`` renders it; neither recomputes. Mutual
information and the two entropies store ``{replicate, item, stats}``, read with
``tcri.get.replicates``, ``items``, ``stats`` and ``draws``; the other tools store ``table`` and
``result``, read with ``tcri.get.result``.

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
