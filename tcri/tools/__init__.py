"""``tcri.tools`` (``tl``) — the joint-distribution engine + the four metric twins.

Every tool computes once, stores its result under a namespaced ``uns`` key with a ``params``
provenance block, and returns the same object. ``pl`` reads that cache; it never recomputes.

``splitby`` produces the between-split contrast as part of the metric, returned under
``stats``. The replicate unit is the group, so item rows are collapsed to one value per group
before the test.
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
