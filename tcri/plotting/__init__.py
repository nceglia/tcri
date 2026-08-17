"""``tcri.pl`` — plotting. The four ``tl``<->``pl`` metric twins plus the shared colours.

Every twin is a **cache renderer**: it reads the result ``tl`` stored in ``uns`` (through
:mod:`tcri.get`) and draws it. None of them computes a metric, so a figure cannot disagree
with the frame the caller is holding. Run the ``tl`` twin first.
"""
from ._entropy import clonotypic_entropy, phenotypic_entropy
from ._mutual_information import mutual_information
from ._flux import phenotypic_flux
from ._colors import NA_COLOR, resolve_colors, tcri_colors

__all__ = [
    "clonotypic_entropy",
    "phenotypic_entropy",
    "mutual_information",
    "phenotypic_flux",
    "resolve_colors",
    "tcri_colors",
    "NA_COLOR",
]
