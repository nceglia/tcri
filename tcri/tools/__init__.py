"""``tcri.tools`` (``tl``) — metrics + the joint-distribution engine.

PR5 seeds this package with the unified ``joint_distribution`` engine (re-exported
top-level as ``tcri.joint_distribution``). The four metric twins
(``clonotypic_entropy``/``phenotypic_entropy``/``mutual_information``/``phenotypic_flux``)
and ``compare_groups`` migrate here from ``tcri.metrics`` in Phase 6, at which point
``tl`` is repointed from ``metrics`` to ``tools``.
"""
from ._joint import joint_distribution

__all__ = ["joint_distribution"]
