"""``tl.clonotypic_entropy`` and ``tl.phenotypic_entropy`` — normalized Shannon entropies
in **bits**, engine-backed, per replicate and per item.

- ``clonotypic_entropy``: per phenotype φ, H[P(c|φ)] over the **supported** clones (spread of
  a phenotype across clones). Absent/zero-support clones are excluded before normalizing (no
  ε-clip fabricating uniform mass). Normalizer = log2(#supported clones). Per replicate,
  H(C|P) = Σ_φ P(φ) H[P(c|φ)], weighted by each phenotype's share of the replicate's joint and
  normalized by log2 of the clones the replicate has at the covariate (or ``n_clones_ref``).
- ``phenotypic_entropy``: per clone c, H[P(φ|c)] (plasticity vs commitment). A clone with zero
  posterior mass returns **NaN** (not reindexed-to-zeros → spurious H=1). Normalizer = log2(P).
  Per replicate, H(P|C) = Σ_c P(c) H[P(φ|c)], weighted by each clone's share of the joint:
  ``1/C`` at ``weighted=False``, its cell share at ``weighted=True``.

In bits, I(C;P) = H(P) − H(P|C) = H(C) − H(C|P). A normalized replicate value whose normalizer
is 0 raises. ``n_samples=0`` is the deterministic plug-in; ``n_samples>0`` returns the
posterior-mean + sd + HDI of the entropy (the plug-in value is ≥ the posterior mean for entropy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from .._compute._tables import (ZeroNormalizer, build_result, build_stats, joint_draws,
                                metric_frames, resolve_groupby, validate_splitby)

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


def _row_entropies(M):
    """Shannon entropy in bits of each row of ``M``, a matrix of distributions; 0·log 0 := 0."""
    logs = np.zeros_like(M)
    np.log2(M, out=logs, where=M > 0)
    return -(M * logs).sum(axis=1)


def _phenotypic_replicate(J, cols, *, normalized):
    """{"value": H(P|C)} of one replicate's joint: each clone's H[P(φ|c)] weighted by its share
    of the joint. NaN for an empty joint; a zero-mass clone has zero weight."""
    J = np.asarray(J, dtype=np.float64)
    total = J.sum()
    if total <= 0:
        return {"value": np.nan}
    mass = J.sum(axis=1)
    held = mass > 0
    value = float(np.sum(mass[held] / total * _row_entropies(J[held] / mass[held, None])))
    if normalized:
        if len(cols) <= 1:
            raise ZeroNormalizer(
                f"there is {len(cols)} phenotype, so normalized phenotypic entropy divides by "
                f"log2({len(cols)}) = 0. Pass normalized=False for bits.")
        value /= np.log2(len(cols))
    return {"value": value}


def _clonotypic_replicate(J, cols, *, normalized, n_clones_ref=None):
    """{"value": H(C|P)} of one replicate's joint: each phenotype's H[P(c|φ)] over its supported
    clones, weighted by its share of the joint. NaN for an empty joint; a zero-mass phenotype has
    zero weight."""
    J = np.asarray(J, dtype=np.float64)
    total = J.sum()
    if total <= 0:
        return {"value": np.nan}
    mass = J.sum(axis=0)
    held = mass > 0
    value = float(np.sum(mass[held] / total * _row_entropies((J[:, held] / mass[held]).T)))
    if normalized:
        C = int(n_clones_ref) if n_clones_ref else int((J.sum(axis=1) > 0).sum())
        if C <= 1:
            raise ZeroNormalizer(
                f"it has {C} clone, so normalized clonotypic entropy divides by log2({C}) = 0. "
                f"Pass normalized=False for bits, or leave this replicate out with clones=.")
        value /= np.log2(C)
    return {"value": value}


def _entropy_metric(adata, *, kind, covariate, groupby, splitby, n_samples, temperature,
                    clones, exclude_pools, weighted, normalized, random_state, n_clones_ref=None,
                    device=None, fit=None):
    """Shared body for both entropies.

    The two differ only in their ITEM AXIS: clonotypic entropy H(c|phi) is one value per
    PHENOTYPE, phenotypic entropy H(phi|c) is one value per CLONE, and in the replicate value that
    weights them. Everything else -- the group loop, the draw handling, the reduction -- is
    identical, which is why it lives here.
    """
    item_col = "phenotype" if kind == "clonotypic" else "clonotype"

    def _items(clone_ids, J, cols):
        if kind == "clonotypic":
            return _clonotypic_one(J, cols, normalized=normalized, n_clones_ref=n_clones_ref)
        return _phenotypic_one(clone_ids, J, cols, normalized=normalized)

    def _replicate(J, cols):
        if kind == "clonotypic":
            return _clonotypic_replicate(J, cols, normalized=normalized,
                                         n_clones_ref=n_clones_ref)
        return _phenotypic_replicate(J, cols, normalized=normalized)

    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)

    def _compute(clone_subset):
        draws, cols = joint_draws(
            adata, covariate, n_samples=n_samples, weighted=weighted, device=device,
            temperature=temperature, clones=clone_subset, random_state=random_state, fit=fit,
        )
        return [(_replicate(J, cols), {k: {"value": v} for k, v in _items(ids, J, cols).items()})
                for ids, J in draws]

    rep_draws, item_draws = metric_frames(
        adata, covariate=covariate, groupby=gkey, splitby=splitby, clones=clones,
        exclude_pools=exclude_pools, item_col=item_col, compute=_compute,
        metric=f"{kind}_entropy", fit=fit)
    rep_summary = build_result(rep_draws)
    payload = {"replicate": {"draws": rep_draws, "summary": rep_summary},
               "item": {"draws": item_draws, "summary": build_result(item_draws)},
               "stats": build_stats(rep_summary, groupby=gkey, splitby=splitby)}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload


@tl_result(key=K.CLONOTYPIC_ENTROPY, version=4, schema=schemas.ClonotypicEntropy,
           default_null="phenotype")
def clonotypic_entropy(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, exclude_pools=True, weighted=False,
                       normalized=True, n_clones_ref=None, random_state=None, device=None,
                       fit=None):
    """H(C|P) per replicate and H[P(c|φ)] per phenotype (bits). ``n_clones_ref`` fixes the
    normalizer for cross-group comparability (else per-group #supported clones). See module
    docstring.

    Returns the replicate summary, one row per replicate, and stores ``{replicate, item,
    stats}``, each level holding ``draws`` and ``summary``, under ``uns[key_added or
    'tcri_clonotypic_entropy']``. The item level has one row per replicate and phenotype, read
    with ``tcri.get.items``. With ``inplace=False`` nothing is stored and the whole result is
    returned.

    A replicate with no joint at ``covariate`` has a NaN replicate row and one NaN item row per
    phenotype. With ``normalized=True``, a replicate with one clone raises ``ValueError`` naming
    it; ``normalized=False`` reports its bits.

    ``null_model`` names the permutation reference: ``"auto"`` is the phenotype null, ``None``
    reports the bare value, and any fit on this object can be named instead. ``fit`` selects
    which fit the number is computed ON, which is a different question -- see the module
    docstring of :mod:`tcri.null`.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the entropy counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out."""
    return _entropy_metric(adata, kind="clonotypic", covariate=covariate, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, exclude_pools=exclude_pools, weighted=weighted,
                           normalized=normalized, random_state=random_state,
                           n_clones_ref=n_clones_ref, device=device, fit=fit)


@tl_result(key=K.PHENOTYPIC_ENTROPY, version=4, schema=schemas.PhenotypicEntropy,
           default_null="phenotype")
def phenotypic_entropy(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, exclude_pools=True, weighted=False,
                       normalized=True, random_state=None, device=None, fit=None):
    """H(P|C) per replicate and H[P(φ|c)] per clone (bits). See module docstring.

    Returns the replicate summary, one row per replicate, and stores ``{replicate, item,
    stats}``, each level holding ``draws`` and ``summary``, under ``uns[key_added or
    'tcri_phenotypic_entropy']``. The item level has one row per replicate and clone, read with
    ``tcri.get.items``. With ``inplace=False`` nothing is stored and the whole result is
    returned.

    A replicate with no joint at ``covariate`` has a NaN replicate row and no item rows. With
    ``normalized=True`` and one phenotype, the call raises ``ValueError``; ``normalized=False``
    reports bits.

    ``null_model`` names the permutation reference (``"auto"`` is the phenotype null); ``fit``
    selects which fit the number is computed on.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the entropy counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out."""
    return _entropy_metric(adata, kind="phenotypic", covariate=covariate, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, exclude_pools=exclude_pools, weighted=weighted,
                           normalized=normalized, random_state=random_state, device=device,
                           fit=fit)
