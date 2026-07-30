"""``tcri.tools`` (``tl``) — the joint-distribution engine + the four metric twins +
``compare_groups``, all engine-backed (PR6). ``tl`` points here (repointed from the old
``tcri.metrics`` in PR6); the metrics default to **bits (log2)** and read the unified
``joint_distribution``.
"""
from ._joint import joint_distribution
from ._entropy import clonotypic_entropy, phenotypic_entropy
from ._mutual_information import mutual_information
from ._flux import phenotypic_flux
from ._compare import compare_groups

__all__ = [
    "joint_distribution",
    "clonotypic_entropy",
    "phenotypic_entropy",
    "mutual_information",
    "phenotypic_flux",
    "compare_groups",
]
