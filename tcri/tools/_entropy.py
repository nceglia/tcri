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

from ._common import grouped_series, is_precomputed_joint, joint_draws, summarize

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


def _entropy_metric(adata_or_jd, *, kind, covariate, groupby, splitby, n_samples, temperature,
                    clones, weighted, normalized, random_state, n_clones_ref=None, device=None):
    item_name = "phenotype" if kind == "clonotypic" else "clonotype"
    value = f"{kind}_entropy"

    def _one(clone_ids, J, cols):
        if kind == "clonotypic":
            return _clonotypic_one(J, cols, normalized=normalized, n_clones_ref=n_clones_ref)
        return _phenotypic_one(clone_ids, J, cols, normalized=normalized)

    if groupby is not None:
        if is_precomputed_joint(adata_or_jd):
            raise ValueError("groupby requires an AnnData, not a precomputed joint (§7.9).")

        def _compute(cl):
            draws, cols = joint_draws(adata_or_jd, covariate, n_samples=n_samples, weighted=weighted, device=device,
                                      temperature=temperature, clones=cl, random_state=random_state)
            per = [_one(ids, J, cols) for ids, J in draws]
            keys = list(per[0].keys())
            point = {k: float(np.nanmean([p[k] for p in per])) for k in keys}
            drawsd = {k: [p[k] for p in per] for k in keys} if (n_samples and int(n_samples) > 0) else None
            return point, drawsd
        return grouped_series(adata_or_jd, groupby=groupby, splitby=splitby,
                              item_name=item_name, value=value, compute=_compute)

    if is_precomputed_joint(adata_or_jd):
        if n_samples and int(n_samples) > 0:
            raise ValueError("precomputed-joint fast path is valid only at n_samples=0 (§7.9).")
        one = _one(list(adata_or_jd.index), adata_or_jd.values, list(adata_or_jd.columns))
        return pd.Series(one, name=value)

    draws, cols = joint_draws(adata_or_jd, covariate, n_samples=n_samples, weighted=weighted, device=device,
                              temperature=temperature, clones=clones, random_state=random_state)
    per = [_one(ids, J, cols) for ids, J in draws]
    keys = list(per[0].keys())
    if n_samples and int(n_samples) > 0:
        return pd.DataFrame({k: summarize([p[k] for p in per]) for k in keys}).T
    return pd.Series(per[0], name=value)


def clonotypic_entropy(adata_or_jd, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, weighted=False, normalized=True,
                       n_clones_ref=None, random_state=None, device=None):
    """Clonotypic entropy ``H[P(c|φ)]`` — one value per phenotype (bits).

    How clonally diverse each phenotype is. Support-only: clones with zero mass in a
    phenotype column are dropped before normalizing, and an empty column returns ``NaN``
    (never a spurious ``1.0``).

    Parameters
    ----------
    adata_or_jd : AnnData | pandas.DataFrame
        Fitted AnnData or a precomputed joint from :func:`joint_distribution`.
    covariate : str | None
        Covariate value to compute at.
    normalized : bool
        Divide by ``log2`` of the number of supported clones so values land in ``[0, 1]``.
    n_clones_ref : int | None
        Fix the normalizer across groups for comparability (else the per-group count of
        supported clones is used).
    n_samples : int
        ``0`` → point estimate; ``>0`` → posterior mean/sd/HDI.
    groupby, splitby : str | None
        Compute per group and carry a split label, as in :func:`mutual_information`.

    Returns
    -------
    pandas.Series | pandas.DataFrame
        One entry per phenotype (or per group×phenotype with ``groupby``).
    """
    return _entropy_metric(adata_or_jd, kind="clonotypic", covariate=covariate, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, weighted=weighted, normalized=normalized,
                           random_state=random_state, n_clones_ref=n_clones_ref,
                           device=device)


def phenotypic_entropy(adata_or_jd, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, weighted=False, normalized=True,
                       random_state=None, device=None):
    """Phenotypic entropy ``H[P(φ|c)]`` — one value per clone (bits).

    How phenotypically plastic (vs committed) each clone is. All ``P`` phenotypes enter the
    sum with ``0·log0 := 0`` and the normalizer is ``log2(P)``; a clone with zero mass
    returns ``NaN``.

    Parameters
    ----------
    adata_or_jd : AnnData | pandas.DataFrame
        Fitted AnnData or a precomputed joint from :func:`joint_distribution`.
    covariate : str | None
        Covariate value to compute at.
    normalized : bool
        Divide by ``log2(P)`` so values land in ``[0, 1]``.
    n_samples : int
        ``0`` → point estimate; ``>0`` → posterior mean/sd/HDI.
    groupby, splitby : str | None
        Compute per group and carry a split label, as in :func:`mutual_information`.

    Returns
    -------
    pandas.Series | pandas.DataFrame
        One entry per clonotype (or per group×clonotype with ``groupby``).
    """
    return _entropy_metric(adata_or_jd, kind="phenotypic", covariate=covariate, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, weighted=weighted, normalized=normalized,
                           random_state=random_state, device=device)
