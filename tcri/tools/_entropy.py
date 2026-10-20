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

**A change** (``cov_from``, ``cov_to``) is ``cov_to`` − ``cov_from``, taken within a draw, both
levels drawn from one shared sample. Per replicate it is the difference of the two replicate
values, each counting every clone the replicate has at its level and normalized at its level
(or both by ``n_clones_ref``). Per item it covers the persistent clones, those at both levels:
phenotypic entropy has one item per persistent clone, and clonotypic entropy recomputes each
phenotype on the persistent clones alone, so both sides share one clone set and one normalizer.
Every change row carries its endpoints, ``value_from`` and ``value_to``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from .._compute._tables import (ZeroNormalizer, at_level, build_result, build_stats,
                                change_draws, check_covariates, joint_draws, metric_frames,
                                resolve_groupby, resolve_seed, rows_of, validate_splitby)

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


def _entropy_metric(adata, *, kind, covariate, cov_from, cov_to, groupby, splitby, n_samples,
                    temperature, clones, exclude_pools, weighted, normalized, random_state,
                    n_clones_ref=None, device=None, fit=None):
    """Shared body for both entropies.

    The two differ only in their ITEM AXIS: clonotypic entropy H(c|phi) is one value per
    PHENOTYPE, phenotypic entropy H(phi|c) is one value per CLONE, and in the replicate value that
    weights them. Everything else -- the group loop, the draw handling, the reduction -- is
    identical, which is why it lives here.
    """
    item_col = "phenotype" if kind == "clonotypic" else "clonotype"
    metric = f"{kind}_entropy"
    change = check_covariates(adata, covariate=covariate, cov_from=cov_from, cov_to=cov_to,
                              metric=metric) == "change"

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

    seed = resolve_seed(random_state) if change else None

    def _compute_change(clone_subset, persistent):
        draws_from, draws_to, cols = change_draws(
            adata, cov_from, cov_to, clones=clone_subset, seed=seed, n_samples=n_samples,
            weighted=weighted, temperature=temperature, device=device, fit=fit)
        per = []
        for (ids_f, Jf), (ids_t, Jt) in zip(draws_from, draws_to):
            h_from = at_level(cov_from, _replicate, Jf, cols)["value"]
            h_to = at_level(cov_to, _replicate, Jt, cols)["value"]
            # the persistent clones' rows, in one order on both sides: a column sum depends on
            # the order its rows are added in
            i_from = _items(persistent, rows_of(ids_f, Jf, persistent), cols)
            i_to = _items(persistent, rows_of(ids_t, Jt, persistent), cols)
            items = {k: {"value": i_to[k] - i_from[k], "value_from": i_from[k],
                         "value_to": i_to[k]} for k in i_from}
            per.append(({"value": h_to - h_from, "value_from": h_from, "value_to": h_to},
                        items))
        return per

    rep_draws, item_draws = metric_frames(
        adata, covariate=covariate, cov_from=cov_from, cov_to=cov_to, groupby=gkey,
        splitby=splitby, clones=clones, exclude_pools=exclude_pools, item_col=item_col,
        compute=_compute_change if change else _compute, metric=metric, fit=fit)
    ends = ("value_from", "value_to") if change else ()
    rep_summary = build_result(rep_draws, extra_values=ends, one_mask=change)
    payload = {"replicate": {"draws": rep_draws, "summary": rep_summary},
               "item": {"draws": item_draws,
                        "summary": build_result(item_draws, extra_values=ends,
                                                one_mask=change)},
               "stats": build_stats(rep_summary, groupby=gkey, splitby=splitby,
                                    change=({"cov_from": cov_from, "cov_to": cov_to}
                                            if change else None))}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload


#: The reference of each mode: the phenotype null at one covariate level, the covariate null on a
#: change, whose movement between levels is what the change measures.
DEFAULT_NULL = {"covariate": "phenotype", "change": "covariate"}


@tl_result(key=K.CLONOTYPIC_ENTROPY, version=4, schema=schemas.ClonotypicEntropy,
           values=("value", "value_from", "value_to"), default_null=DEFAULT_NULL)
def clonotypic_entropy(adata, *, covariate=None, cov_from=None, cov_to=None, groupby=None,
                       splitby=None, n_samples=0, temperature=1.0, clones=None,
                       exclude_pools=True, weighted=False, normalized=True, n_clones_ref=None,
                       random_state=None, device=None, fit=None):
    """H(C|P) per replicate and H[P(c|φ)] per phenotype (bits). ``n_clones_ref`` fixes the
    normalizer for cross-group comparability (else per-group #supported clones). See module
    docstring.

    Stores ``{replicate, item, stats}``, each level holding ``draws`` and ``summary``, under
    ``uns[key_added or 'tcri_clonotypic_entropy']`` and returns ``None``; with
    ``inplace=False`` nothing is stored and the whole result is returned. The replicate level has
    one row per replicate, read with ``tcri.get.replicates``; the item level one row per
    replicate and phenotype, read with ``tcri.get.items``.

    Pass ``covariate`` for the value at one level, or ``cov_from`` with ``cov_to`` for the change
    from one to the other (see the module docstring); on a change each phenotype's item value is
    recomputed on the replicate's persistent clones. ``stats`` then also holds the paired test of
    the change, one row per split level with ``splitby``, or one over every replicate without.

    A replicate with no joint at a level has a NaN replicate row and one NaN item row per
    phenotype. With ``normalized=True``, a replicate with one clone raises ``ValueError`` naming
    it and the level; ``normalized=False`` reports its bits.

    ``null_model`` names the permutation reference: ``"auto"`` is the phenotype null at one
    level and the covariate null on a change, ``None`` reports the bare value, and any fit on
    this object can be named instead. ``fit`` selects
    which fit the number is computed ON, which is a different question -- see the module
    docstring of :mod:`tcri.null`.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the entropy counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out."""
    return _entropy_metric(adata, kind="clonotypic", covariate=covariate, cov_from=cov_from,
                           cov_to=cov_to, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, exclude_pools=exclude_pools, weighted=weighted,
                           normalized=normalized, random_state=random_state,
                           n_clones_ref=n_clones_ref, device=device, fit=fit)


@tl_result(key=K.PHENOTYPIC_ENTROPY, version=4, schema=schemas.PhenotypicEntropy,
           values=("value", "value_from", "value_to"), default_null=DEFAULT_NULL)
def phenotypic_entropy(adata, *, covariate=None, cov_from=None, cov_to=None, groupby=None,
                       splitby=None, n_samples=0, temperature=1.0, clones=None,
                       exclude_pools=True, weighted=False, normalized=True, random_state=None,
                       device=None, fit=None):
    """H(P|C) per replicate and H[P(φ|c)] per clone (bits). See module docstring.

    Stores ``{replicate, item, stats}``, each level holding ``draws`` and ``summary``, under
    ``uns[key_added or 'tcri_phenotypic_entropy']`` and returns ``None``; with
    ``inplace=False`` nothing is stored and the whole result is returned. The replicate level has
    one row per replicate, read with ``tcri.get.replicates``; the item level one row per
    replicate and clone, read with ``tcri.get.items``.

    Pass ``covariate`` for the value at one level, or ``cov_from`` with ``cov_to`` for the change
    from one to the other (see the module docstring); on a change the item rows are the
    replicate's persistent clones. ``stats`` then also holds the paired test of the change, one
    row per split level with ``splitby``, or one over every replicate without.

    A replicate with no joint at a level has a NaN replicate row and no item rows. With
    ``normalized=True`` and one phenotype, the call raises ``ValueError``; ``normalized=False``
    reports bits.

    ``null_model`` names the permutation reference (``"auto"`` is the phenotype null at one
    level and the covariate null on a change); ``fit`` selects which fit the number is computed
    on.

    ``exclude_pools`` leaves out the ``pooled@{group}`` labels ``tcri.pp.pool_rare_clones``
    writes. A pool is a mixture of many rare clones, not a clone, so by default it is left out
    and the entropy counts exactly the clones ``pool_rare_clones`` kept, with ``clones``
    selecting among them. ``False`` counts each pool as one clone. A column with no pooling
    step has nothing to leave out."""
    return _entropy_metric(adata, kind="phenotypic", covariate=covariate, cov_from=cov_from,
                           cov_to=cov_to, groupby=groupby,
                           splitby=splitby, n_samples=n_samples, temperature=temperature,
                           clones=clones, exclude_pools=exclude_pools, weighted=weighted,
                           normalized=normalized, random_state=random_state, device=device,
                           fit=fit)
