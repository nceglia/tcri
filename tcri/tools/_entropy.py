"""``tl.clonotypic_entropy`` and ``tl.phenotypic_entropy`` — normalized Shannon entropies
in **bits** (§7.2/§7.3), engine-backed.

- ``clonotypic_entropy``: per phenotype φ, H[P(c|φ)] over the **supported** clones (spread of
  a phenotype across clones). Absent/zero-support clones are excluded before normalizing (no
  ε-clip fabricating uniform mass). Normalizer = log2(#supported clones).
- ``phenotypic_entropy``: per clone c, H[P(φ|c)] (plasticity vs commitment). A clone with zero
  posterior mass returns **NaN** (not reindexed-to-zeros → spurious H=1). Normalizer = log2(P).

``n_samples=0`` is the deterministic plug-in; ``n_samples>0`` returns the posterior-mean +
sd + HDI of the entropy (plug-in ≥ posterior-mean for entropy — documented as distinct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from ._common import (build_result, build_stats, joint_draws,
                      metric_table, resolve_groupby,
                      validate_splitby)

__all__ = ["clonotypic_entropy", "phenotypic_entropy"]


def _clonotypic_one(J, cols, *, normalized, n_clones_ref=None):
    """{phenotype: H[P(c|φ)] bits} over supported clones. ``n_clones_ref`` fixes the
    normalizer log2(C) for cross-group comparability (else the #supported clones)."""
    out = {}
    for j, ph in enumerate(cols):
        col = np.asarray(J[:, j], dtype=np.float64)
        supp = col > 0
        s = col[supp].sum()
        if supp.sum() == 0 or s <= 0:
            out[ph] = np.nan
            continue
        v = col[supp] / s
        H = float(-np.sum(v * np.log2(v)))
        if normalized:
            C = int(n_clones_ref) if n_clones_ref else int(supp.sum())
            if C > 1:
                H /= np.log2(C)
        out[ph] = H
    return out


def _phenotypic_one(clone_ids, J, cols, *, normalized):
    """{clone: H[P(φ|c)] bits}; zero-mass clone → NaN."""
    P = len(cols)
    out = {}
    for i, c in enumerate(clone_ids):
        row = np.asarray(J[i], dtype=np.float64)
        s = row.sum()
        if s <= 0:
            out[c] = np.nan
            continue
        p = row / s
        # 0*log0 := 0. Mask first rather than np.where(p>0, p*log2(p), 0): numpy
        # evaluates BOTH branches, so log2(0) still fires divide-by-zero warnings.
        nz = p[p > 0]
        H = float(-np.sum(nz * np.log2(nz)))
        if normalized and P > 1:
            H /= np.log2(P)
        out[c] = H
    return out


def _entropy_metric(adata, *, kind, covariate, groupby, splitby, n_samples, temperature,
                    clones, weighted, normalized, random_state, n_clones_ref=None, device=None):
    """Shared body for both entropies.

    The two differ only in their ITEM AXIS: clonotypic entropy H(c|phi) is one value per
    PHENOTYPE, phenotypic entropy H(phi|c) is one value per CLONE. Everything else -- the group
    loop, the draw handling, the reduction -- is identical, which is why it lives here.
    """
    item_col = "phenotype" if kind == "clonotypic" else "clonotype"

    def _one(clone_ids, J, cols):
        if kind == "clonotypic":
            return _clonotypic_one(J, cols, normalized=normalized, n_clones_ref=n_clones_ref)
        return _phenotypic_one(clone_ids, J, cols, normalized=normalized)

    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)

    def _compute(clone_subset):
        draws, cols = joint_draws(
            adata, covariate, n_samples=n_samples, weighted=weighted, device=device,
            temperature=temperature, clones=clone_subset, random_state=random_state,
        )
        return [_one(ids, J, cols) for ids, J in draws]

    table = metric_table(adata, covariate=covariate, groupby=gkey, splitby=splitby,
                         clones=clones, item_col=item_col, compute=_compute)
    result = build_result(table)
    stats = build_stats(result, groupby=gkey, splitby=splitby)

    payload = {"table": table, "result": result, "stats": stats}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload


@tl_result(key=K.CLONOTYPIC_ENTROPY, version=1, schema=schemas.ClonotypicEntropy)
def clonotypic_entropy(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, weighted=False, normalized=True,
                       n_clones_ref=None, random_state=None, device=None,
                       key_added=None, inplace=True):
    """H[P(c|φ)] per phenotype (bits). ``n_clones_ref`` fixes the normalizer for cross-group
    comparability (else per-group #supported clones). See module docstring."""
    return _entropy_metric(adata, kind="clonotypic", covariate=covariate, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, weighted=weighted, normalized=normalized,
                           random_state=random_state, n_clones_ref=n_clones_ref,
                           device=device)


@tl_result(key=K.PHENOTYPIC_ENTROPY, version=1, schema=schemas.PhenotypicEntropy)
def phenotypic_entropy(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, weighted=False, normalized=True,
                       random_state=None, device=None, key_added=None, inplace=True):
    """H[P(φ|c)] per clone (bits) — computed once, cached, returned. See module docstring."""
    return _entropy_metric(adata, kind="phenotypic", covariate=covariate, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, weighted=weighted, normalized=normalized,
                           random_state=random_state, device=device)
