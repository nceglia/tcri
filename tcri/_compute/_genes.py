"""Private helpers for the TCR segment genes in the expression matrix.

This module imports numpy and pandas and nothing from tcri, so that ``preprocessing`` and
``model`` can both import it.

- :data:`_TCR_SEGMENT`: the one definition of a TCR V, D or J segment gene symbol.
- :data:`_INVARIANT_SEGMENTS`: the segment genes of invariant or semi-invariant T cells, which
  ``pp.exclude_tcr_genes`` leaves in by default.
- :func:`_segment_genes`: which symbols are segment genes outside a ``keep`` list.
- :func:`_looks_like_ensembl`: whether gene names are mostly Ensembl gene IDs.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: The V, D and J segment genes of the four TCR loci, as human or mouse symbols. The digit or
#: ``P`` after the segment letter keeps genes such as TRADD out and lets the γ joining genes
#: TRGJP, TRGJP1 and TRGJP2 in; constant genes (TRAC, TRBC1, TRGC1, TRDC, ...) never match.
_TCR_SEGMENT = re.compile(r"^TR[ABGD][VDJ](?:\d|P)", re.IGNORECASE)

#: Segment genes shared by many distinct clones of one phenotype, so they mark the phenotype
#: rather than a clone: MAIT (TRAV1-2, TRAJ33; mouse Trav1, Traj33), iNKT (TRAV10, TRAJ18;
#: mouse Trav11, Traj18) and Vγ9Vδ2 γδ T cells (TRGV9, TRDV2). Matching ignores case, so the
#: mouse Traj33 and Traj18 are covered by the human entries.
_INVARIANT_SEGMENTS = ("TRAV1-2", "TRAJ33", "TRAV10", "TRAJ18", "TRGV9", "TRDV2", "Trav1",
                       "Trav11")

_ENSEMBL_GENE = re.compile(r"^ENS[A-Z]*G\d{11}(?:\.\d+)?$")


def _segment_genes(symbols, keep) -> np.ndarray:
    """True for each symbol that is a TCR segment gene and not in ``keep``, ignoring case."""
    kept = {str(gene).upper() for gene in keep}
    symbols = pd.Index(symbols).astype(str)
    return np.fromiter((bool(_TCR_SEGMENT.match(s)) and s.upper() not in kept for s in symbols),
                       dtype=bool, count=len(symbols))


def _looks_like_ensembl(names) -> bool:
    """Whether more than half of ``names`` are Ensembl gene IDs, such as ENSG00000211751."""
    names = pd.Index(names).astype(str)
    return sum(bool(_ENSEMBL_GENE.match(n)) for n in names) * 2 > len(names)
