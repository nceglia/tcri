"""``tcri.pl`` — plotting. The ``tl``<->``pl`` metric twins, the twins of the ``pp`` repertoire
summaries, and the shared colors.

Every twin is a **cache renderer**: it reads the result its ``tl`` or ``pp`` function stored in
``uns`` (through :mod:`tcri.get`) and draws it. None of them computes a metric, so a figure
cannot disagree with the frame the caller is holding. Run the ``tl`` or ``pp`` twin first.
"""
from ._entropy import clonotypic_entropy, phenotypic_entropy
from ._mutual_information import mutual_information
from ._flux import phenotypic_flux
from ._delta import (delta_clonotypic_entropy,
                     delta_phenotypic_entropy)
from ._perturbation import gene_importance
from ._repertoire import clone_persistence, clonotype_sharing, repertoire_summary
from ._colors import NA_COLOR, resolve_colors, tcri_colors

__all__ = [
    "clonotypic_entropy",
    "phenotypic_entropy",
    "mutual_information",
    "phenotypic_flux",
    "delta_clonotypic_entropy",
    "delta_phenotypic_entropy",
    "gene_importance",
    "repertoire_summary",
    "clonotype_sharing",
    "clone_persistence",
    "resolve_colors",
    "tcri_colors",
    "NA_COLOR",
]
