"""Shared reduction helpers for the ``tl`` metrics.

Each metric pulls the clone×phenotype joint from the engine (:func:`joint_distribution`,
``use_logits=True``), reduces per draw, and — for ``n_samples>0`` — summarizes the draw
distribution (mean / sd / HDI). ``groupby`` is implemented here by **restricting clones
per group** (§7.1: full-space clone masks + ``clones=``, never slicing the AnnData), which
relies on clones being disjoint across groups (a TCR clone never spans two patients).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._stats import hdi
from ._joint import joint_distribution


def is_precomputed_joint(x) -> bool:
    """A precomputed joint (fast path, §7.9) is a plain DataFrame, not an AnnData."""
    return isinstance(x, pd.DataFrame)


def reject_stacked_covariate_joint(jd) -> None:
    """The precomputed-joint fast path must refuse a table that still carries covariate blocks.

    ``joint_distribution(covariate=None)`` returns a (covariate, clonotype)-indexed frame. Feeding
    that straight to a scalar metric reaches the precomputed-joint fast path, which never calls
    :func:`joint_draws` -- so the covariate guard is bypassed and the old stacked value comes
    back, with each (covariate, clone) pair counted as a distinct clone. Measured: 0.1240077
    stacked against 0.1452629 for the same table marginalised.

    The caller almost certainly wants either one covariate level or an explicit reduction, and
    both are one line. Guessing which is the ambiguity this whole change exists to refuse.
    """
    idx = getattr(jd, "index", None)
    if idx is not None and getattr(idx, "nlevels", 1) > 1 and "covariate" in (idx.names or []):
        raise ValueError(
            "this joint still carries a 'covariate' index level, so each (covariate, clone) "
            "pair would be scored as a distinct clone -- inflating H(c).\n"
            "\n"
            "Pass a single-covariate joint, e.g. tl.joint_distribution(adata, "
            "covariate='<level>'), or reduce the covariate level yourself first, e.g. "
            "jd.groupby(level='clonotype').sum()."
        )


def joint_draws(adata, covariate, *, n_samples, weighted, temperature, clones, random_state,
                use_logits=True, device=None):
    """Return ``(draws, phenotype_cols)`` where ``draws`` is a list of ``(clone_ids, [C, P])``
    per posterior draw (length 1 for ``n_samples=0``).

    Consumes the engine's raw blocks directly. Going through
    :func:`~tcri.tools._joint.joint_distribution` would flatten ``[S, n_rows, P]``
    into a MultiIndex DataFrame only for this function to ``groupby('sample_id')``
    and unpack it straight back to arrays — measured at ~2x the engine core itself.
    Ordering matches the DataFrame path exactly (blocks in covariate order, clones in
    block order, then the ``clones=`` filter applied as a stable reorder).

    ``covariate`` is REQUIRED here. See the guard below.
    """
    from ._joint import _engine_blocks

    # covariate=None used to stack the per-covariate blocks row-wise, so a clone present in k
    # covariate levels contributed k ROWS and the row axis of the joint became the
    # (covariate, clone) pair rather than the clone. H(c) was then the entropy over pseudo-
    # clones. Measured on a 10-clone / 2-covariate fixture:
    #
    #     C=20 P=4  (many clones)   min  +0.0%   average +13.8%
    #     C=6  P=8  (few clones)    min +12.4%   average +15.4%
    #
    # `min` divides by min(H(c), H(phi)). When clones OUTNUMBER phenotypes it selects H(phi),
    # which row-splitting does not touch, so the default looks unaffected; when they do not, it
    # selects H(c) and moves by ~12%. So this is not "invisible at the default" -- that was a
    # property of one fixture. `average` always moves, since it averages both entropies.
    #
    # The three metrics that reduce to a scalar also disagreed on what covariate=None means
    # (one collapsed to a per-phenotype index, one kept a (covariate, clonotype) index, one
    # stacked). Rather than pick a unification -- which is a question about the estimand, not
    # about the code -- covariate is now required wherever the result must be REDUCED.
    #
    # joint_distribution(covariate=None) is deliberately still allowed: it LABELS the blocks
    # with a covariate index level instead of collapsing them, so no ambiguity arises there,
    # and it remains the way to get every covariate in one object.
    if covariate is None:
        raise ValueError(
            "covariate is required for scalar metrics.\n"
            "\n"
            "covariate=None previously stacked every covariate level into one table, treating "
            "each (covariate, clone) pair as a distinct clone. That inflates H(c) and changes "
            "NMI by 12-15% on measured fixtures, under both normalize_mode settings.\n"
            "\n"
            "Pass an explicit covariate level, or use tl.joint_distribution(covariate=None) to "
            "get every level as a labelled (covariate, clonotype) table and reduce it the way "
            "your analysis intends."
        )

    blocks, _n_draws, clonotype_cats, _cov_cats, cols = _engine_blocks(
        adata,
        covariate=covariate,
        groupby=None,
        n_samples=n_samples,
        use_logits=use_logits,
        weighted=weighted,
        temperature=temperature,
        random_state=random_state,
        device=device,
    )

    # Row labels in DataFrame-concat order (per covariate block, per clone). When
    # covariate is None the DataFrame carries a leading `covariate` index level, so
    # its labels are (covariate, clonotype) TUPLES — reproduce that exactly, since
    # callers key on whatever this returns.
    all_cov = covariate is None
    clone_names, ids = [], []
    for m, clone_idx, _J in blocks:
        for i in clone_idx:
            c = clonotype_cats[i]
            clone_names.append(c)
            ids.append((_cov_cats[m], c) if all_cov else c)

    keep = None
    if clones is not None:
        clones = list(clones)
        rank = {c: i for i, c in enumerate(clones)}
        sel = [j for j, c in enumerate(clone_names) if c in rank]
        # stable sort by requested order — mirrors the MultiIndex argsort(kind="stable"),
        # which ranks on the clonotype level only
        keep = sorted(sel, key=lambda j: rank[clone_names[j]])
        ids = [ids[j] for j in keep]

    n_draws_out = blocks[0][2].shape[0] if blocks else 1
    draws = []
    for s in range(n_draws_out):
        arr = np.concatenate([J[s] for _m, _ci, J in blocks], axis=0) if blocks else np.empty((0, len(cols)))
        arr = arr.astype(float, copy=False)
        if keep is not None:
            arr = arr[keep]
        draws.append((list(ids), arr))
    return draws, list(cols)


def summarize(values, *, hdi_prob=0.94) -> dict:
    """Summarize a 1-D array of per-draw metric values → mean / sd / hdi_low / hdi_high."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": np.nan, "sd": np.nan, "hdi_low": np.nan, "hdi_high": np.nan}
    if v.size == 1:
        # A single draw has no spread to report. Returning sd=0 and a zero-width HDI states
        # certainty that was never measured -- at n_samples=1 the interval read
        # [0.289341, 0.289341]. mean is still the draw; the rest is undefined, and NaN says so.
        return {"mean": float(v[0]), "sd": np.nan, "hdi_low": np.nan, "hdi_high": np.nan}
    lo, hi = hdi(v, prob=hdi_prob)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "hdi_low": lo, "hdi_high": hi}


def clone_col(adata):
    from .. import _keys as K
    return adata.uns[K.METADATA]["clone_col"]


def _validate_group_clones(obs, groupby, cc):
    """The metric ``groupby`` restricts the engine by clone id (``clones=``), which is only
    correct when clones are **disjoint across groups** (a clone's cells all live in one group).
    Raise loudly if a clone id spans groups — otherwise a group's estimate would silently
    absorb that clone's cells from other groups (§7.1 groupby↔covariate semantics)."""
    seen = {}
    for g in obs[groupby].dropna().unique().tolist():
        for c in obs.loc[obs[groupby] == g, cc].dropna().unique():
            if c in seen and seen[c] != g:
                raise ValueError(
                    f"groupby={groupby!r}: clonotype {c!r} spans groups {seen[c]!r} and {g!r}. "
                    f"The metric groupby restricts by clone id (clones=), which requires clones "
                    f"to be disjoint across groups (e.g. patient-specific `trb_unique`). Use a "
                    f"clone-disjoint groupby, or pre-filter with `clones=`."
                )
            seen[c] = g


def grouped_scalar(adata, *, groupby, splitby, value, compute, hdi_prob=0.94,
                   restrict_to=None):
    """Loop over ``adata.obs[groupby]`` values, restrict to each group's clones, compute a
    scalar-valued metric per group, and tidy into a DataFrame with the ``splitby`` label.

    ``compute(clones) -> (point, draws_or_None)``: ``point`` is the ``n_samples=0`` scalar (or
    the draw mean), ``draws`` is the per-draw vector (or ``None`` at ``n_samples=0``).
    """
    cc = clone_col(adata)
    obs = adata.obs
    _validate_group_clones(obs, groupby, cc)

    # D9: cells with no group label are dropped by `.dropna()`. Silently excluding part of the
    # data from every reported number is not a default anyone opted into -- measured 20% of
    # cells dropped with no warning on a fixture with a partially-populated column.
    n_missing = int(obs[groupby].isna().sum())
    if n_missing:
        warnings.warn(
            f"groupby={groupby!r}: {n_missing} of {len(obs)} cells "
            f"({100 * n_missing / max(len(obs), 1):.1f}%) have no group label and are excluded "
            f"from every row of this result.",
            UserWarning, stacklevel=3,
        )

    rows = []
    for g in obs[groupby].dropna().unique().tolist():
        gmask = obs[groupby] == g
        clones_g = obs.loc[gmask, cc].dropna().unique().tolist()
        # D3: the caller's `clones=` was shadowed by the group's clone list, so
        # `groupby=... , clones=[...]` returned a frame identical to the unrestricted call --
        # the restriction was accepted and discarded. Intersect instead, preserving the
        # caller's order so the engine's stable reorder is unchanged.
        if restrict_to is not None:
            allowed = set(clones_g)
            clones_g = [c for c in restrict_to if c in allowed]
            if not clones_g:
                continue
        point, draws = compute(clones_g)
        row = {groupby: g, value: point}
        if splitby is not None and splitby in obs.columns:
            row[splitby] = obs.loc[gmask, splitby].iloc[0]
        if draws is not None:
            row.update(summarize(draws, hdi_prob=hdi_prob))
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_series(adata, *, groupby, splitby, item_name, value, compute, hdi_prob=0.94,
                   restrict_to=None):
    """Like :func:`grouped_scalar` but the metric is a per-item (phenotype/clone) map.

    ``compute(clones) -> (point, draws_or_None)``: ``point`` is ``{item: value}``;
    ``draws`` is ``{item: [per-draw values]}`` (or ``None`` at ``n_samples=0``). Tidies to
    one row per (group, item).
    """
    cc = clone_col(adata)
    obs = adata.obs
    _validate_group_clones(obs, groupby, cc)

    # D9: cells with no group label are dropped by `.dropna()`. Silently excluding part of the
    # data from every reported number is not a default anyone opted into -- measured 20% of
    # cells dropped with no warning on a fixture with a partially-populated column.
    n_missing = int(obs[groupby].isna().sum())
    if n_missing:
        warnings.warn(
            f"groupby={groupby!r}: {n_missing} of {len(obs)} cells "
            f"({100 * n_missing / max(len(obs), 1):.1f}%) have no group label and are excluded "
            f"from every row of this result.",
            UserWarning, stacklevel=3,
        )

    rows = []
    for g in obs[groupby].dropna().unique().tolist():
        gmask = obs[groupby] == g
        clones_g = obs.loc[gmask, cc].dropna().unique().tolist()
        # D3: the caller's `clones=` was shadowed by the group's clone list, so
        # `groupby=... , clones=[...]` returned a frame identical to the unrestricted call --
        # the restriction was accepted and discarded. Intersect instead, preserving the
        # caller's order so the engine's stable reorder is unchanged.
        if restrict_to is not None:
            allowed = set(clones_g)
            clones_g = [c for c in restrict_to if c in allowed]
            if not clones_g:
                continue
        point, draws = compute(clones_g)
        split_val = obs.loc[gmask, splitby].iloc[0] if (splitby and splitby in obs.columns) else None
        for item, val in point.items():
            row = {groupby: g, item_name: item, value: val}
            if split_val is not None:
                row[splitby] = split_val
            if draws is not None and item in draws:
                row.update(summarize(draws[item], hdi_prob=hdi_prob))
            rows.append(row)
    return pd.DataFrame(rows)
