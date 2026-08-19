"""``tcri.tools`` (``tl``) — the joint-distribution engine + the four metric twins.

Every tool computes once, stores its result under a namespaced ``uns`` key with a ``params``
provenance block, and returns the same object. ``pl`` reads that cache; it never recomputes.

``compare_groups`` is no longer here. It was a separate user-facing step only because there
was nowhere to put statistics -- so a contrast meant remembering to call a second function on
the right frame, and getting the replicate unit right yourself. Now ``splitby`` produces the
contrast as part of the metric (``stats``), and the contrast math lives in ``tcri/_stats/_compare.py`` as an
internal helper with one caller.
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
