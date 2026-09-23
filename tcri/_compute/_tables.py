"""Shared reduction helpers for the ``tl`` metrics.

Each metric pulls the clone×phenotype joint from the engine (:func:`joint_distribution`,
``use_logits=True``), reduces per draw, and — for ``n_samples>0`` — summarizes the draw
distribution (mean / sd / HDI). ``groupby`` is implemented here by **restricting clones
per group** (full-space clone masks + ``clones=``, never slicing the AnnData), which
relies on clones being disjoint across groups (a TCR clone never spans two patients).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._state import keys as K
from .._stats import hdi
# NOTE: ``tools._joint`` is imported lazily inside the functions that need it, never at module
# level. ``_compute`` is the lower layer — every tools/* metric imports *down* into this module —
# so a module-level import back up into ``tools`` inverts the layering and makes the package
# import-order dependent. It happened to resolve here only because ``tools/_joint`` does not
# itself reach back into ``_tables``; that is a property of today's code, not a guarantee.


# `is_precomputed_joint` and `reject_stacked_covariate_joint` lived here to support metrics
# accepting a bare DataFrame instead of an AnnData (the §7.9 "precomputed joint" path). Both are
# gone: the path was declared in the first contract freeze (7599959), implemented because it was
# declared, and had exactly one caller in the repo -- a test scoring one table four ways. With
# every tl taking an AnnData, a stacked joint cannot arrive at a metric, so there is nothing left
# to guard against.


def joint_draws(adata, covariate, *, n_samples, weighted, temperature, clones, random_state,
                use_logits=True, device=None, fit=None):
    """Return ``(draws, phenotype_cols)`` where ``draws`` is a list of ``(clone_ids, [C, P])``
    per posterior draw (length 1 for ``n_samples=0``).

    Consumes the engine's raw blocks directly. Going through
    :func:`~tcri.tools._joint.joint_distribution` would flatten ``[S, n_rows, P]``
    into a MultiIndex DataFrame only for this function to ``groupby('sample_id')``
    and unpack it straight back to arrays.
    Ordering matches the DataFrame path exactly (blocks in covariate order, clones in
    block order, then the ``clones=`` filter applied as a stable reorder).

    ``covariate`` is required: a reduced metric has no defined meaning over stacked covariate
    levels, because the row axis would become the (covariate, clone) pair rather than the clone.
    """
    from ..tools._joint import _engine_blocks  # lazy: see module note

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
            "A scalar metric reduces one joint. Stacking every covariate level into one table "
            "treats each (covariate, clone) pair as a distinct clone, which inflates H(c) and "
            "moves the normalized value under both normalize_mode settings.\n"
            "\n"
            "Pass an explicit covariate level, or use tl.joint_distribution(covariate=None) to "
            "get every level as a labelled (covariate, clonotype) table and reduce it the way "
            "your analysis intends."
        )

    blocks, _n_draws, clonotype_cats, _cov_cats, cols = _engine_blocks(
        adata,
        covariate=covariate,
        n_samples=n_samples,
        use_logits=use_logits,
        weighted=weighted,
        temperature=temperature,
        random_state=random_state,
        device=device,
        fit=fit,
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
    from .._state import keys as K
    return adata.uns[K.METADATA]["clone_col"]


def fit_clone_labels(adata, fit=None):
    """Per-cell clone id **as the fit saw it**, indexed by ``obs_names``.

    ``CLONOTYPE_CATEGORIES[ct_to_c[ct_array]]``, with ``pd.NA`` wherever ``ct_to_c < 0`` (the
    unassigned rows today's callers drop). On the main fit this equals ``obs[clone_col]`` cell
    for cell; under a clonotype null it does not, and THAT is the point. `metric_table` restricts
    the engine by clone id while the engine's rows come from the permuted codes, so a clone list
    built from ``obs`` selects the wrong null rows whenever a permutation stratum spans more than
    one group -- silently, because the parent's ``obs`` passes every disjointness check.
    """
    ct_array = np.asarray(adata.uns[K.fit_key(K.CT_ARRAY, fit)])
    if len(ct_array) != adata.n_obs:
        raise ValueError(
            f"fit_clone_labels received an AnnData whose per-cell registration array "
            f"(uns[{K.fit_key(K.CT_ARRAY, fit)!r}], len {len(ct_array)}) does not match "
            f"adata.n_obs ({adata.n_obs}). This happens on a filtered/sliced AnnData: the "
            f"full-space uns arrays misalign against the subset obsm/obs. Re-run "
            f"model.to_anndata(...) on the filtered object."
        )
    ct_to_c = np.asarray(adata.uns[K.fit_key(K.CT_TO_C, fit)])
    cats = np.asarray(adata.uns[K.CLONOTYPE_CATEGORIES], dtype=object)
    codes = ct_to_c[ct_array]
    labels = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], None)
    return pd.Series(labels, index=adata.obs_names, dtype=object).where(pd.notna(labels), pd.NA)


def clones_at(adata, covariate, *, fit=None, group_clones=None):
    """Clone ids with cells at ``covariate`` **in the fit's own substrate**.

    Returned in ascending clone-code order, which is the substrate's order and not ``obs``
    first-appearance order. No metric value depends on it -- the entropies sum a column,
    `joint_draws` reorders its ids by the requested ``clones=`` list, and `build_result` groups.
    """
    meta = adata.uns[K.METADATA]
    cov_cats = [str(c) for c in adata.uns[K.COVARIATE_CATEGORIES]]
    if str(covariate) not in cov_cats:
        return []
    m = cov_cats.index(str(covariate))
    ct_to_c = np.asarray(adata.uns[K.fit_key(K.CT_TO_C, fit)])
    ct_to_cov = np.asarray(adata.uns[K.fit_key(K.CT_TO_COV, fit)])
    codes = ct_to_c[ct_to_cov == m]
    codes = np.unique(codes[codes >= 0])
    cats = list(adata.uns[K.CLONOTYPE_CATEGORIES])
    at = [cats[c] for c in codes]
    if group_clones is None:
        return at
    allowed = set(group_clones)
    return [c for c in at if c in allowed]


def _refit_hint(adata, fit, groupby):
    """The sentence that tells a caller how to make a null's strata match their ``groupby``.

    Strata are fixed at fit time while ``groupby`` is chosen at metric time, so a caller passing
    a ``groupby`` finer than the strata gets a legal fit and an illegal comparison: the clone
    list comes from the permuted map while the group mask comes from ``obs``, and a clone that
    the shuffle spread across two groups now belongs to both. The fix is a refit, not a re-plot,
    so the message names the refit.
    """
    if fit is None:
        return ""
    settings = adata.uns.get(K.fit_key(K.FIT_SETTINGS, fit))
    if settings is None:
        return ""
    # h5ad stores a list of strings as a numpy ARRAY, so the idiomatic `or []` raises "the truth
    # value of an array with more than one element is ambiguous" on any object that has been to
    # disk -- which is every object a reference is read from in practice. The same spelling bit
    # `keys.fits()` in 0.12; written against the round-tripped shape both times.
    raw = settings.get("strata")
    strata = [] if raw is None else [str(c) for c in raw]
    if not strata:
        return f" This is fit {fit!r}."
    if str(settings.get("kind")) == "condition":
        return (f" This is fit {fit!r}, a condition null, whose strata are fixed at {strata} and "
                f"are not yours to choose. Use a groupby no finer than they are.")
    extended = strata + ([str(groupby)] if str(groupby) not in strata else [])
    return (f" This is fit {fit!r}, permuted within {strata}, which is coarser than "
            f"groupby={groupby!r}. Refit it with within={extended!r}, or use a coarser groupby.")


def _validate_group_clones(labels, groups, groupby, hint=""):
    """The metric ``groupby`` restricts the engine by clone id (``clones=``), which is only
    correct when clones are **disjoint across groups** (a clone's cells all live in one group).
    Raise loudly if a clone id spans groups — otherwise a group's estimate would silently
    absorb that clone's cells from other groups.

    Takes the two RESOLVED Series — clone labels (from :func:`fit_clone_labels` on the fit being
    measured, or ``obs[clone_col]`` for a caller that means the raw column) and the group labels
    — rather than an ``obs`` and two column names, because a fit's clone labels are not a column
    of ``obs`` at all.
    """
    seen = {}
    for g in groups.dropna().unique().tolist():
        for c in labels[(groups == g).to_numpy()].dropna().unique():
            if c in seen and seen[c] != g:
                raise ValueError(
                    f"groupby={groupby!r}: clonotype {c!r} spans groups {seen[c]!r} and {g!r}. "
                    f"The metric groupby restricts by clone id (clones=), which requires clones "
                    f"to be disjoint across groups (e.g. patient-specific `trb_unique`). Use a "
                    f"clone-disjoint groupby, or pre-filter with `clones=`.{hint}"
                )
            seen[c] = g


def resolve_groupby(adata, groupby):
    """``groupby`` if given, else the column registered as ``replicate`` at setup.

    Returns ``(effective, was_resolved)`` so the caller can record the effective value in
    provenance via ``with_resolved_params`` -- otherwise the cached params say ``None`` and
    every reader of them sees a placeholder instead of the column actually used.
    """
    if groupby is not None:
        return groupby, False
    meta = adata.uns.get(K.METADATA) or {}
    replicate = meta.get(K.Config.REPLICATE)
    return replicate, replicate is not None


def validate_splitby(obs, groupby, splitby):
    """``splitby`` labels groups for a contrast, so it needs groups, and the label must be a
    property OF the group.

    Two conditions are enforced: ``splitby`` requires ``groupby``, and ``splitby`` must be
    constant within each group. Without the second, a group's split label would be whichever
    one its first cell happens to carry.
    """
    if splitby is None:
        return
    if groupby is None:
        raise ValueError(
            "splitby requires groupby. splitby labels groups so they can be compared; with no "
            "groups there are no replicates and nothing to contrast. Pass groupby=<the "
            "independent unit>, or register it once via setup_anndata(replicate=...)."
        )
    if splitby not in obs.columns:
        raise ValueError(f"splitby={splitby!r} is not a column of adata.obs")
    spans = obs.groupby(groupby, observed=True)[splitby].nunique(dropna=True)
    bad = spans[spans > 1]
    if len(bad):
        raise ValueError(
            f"splitby={splitby!r} is not constant within groupby={groupby!r}: "
            f"{list(bad.index[:5])} span multiple {splitby!r} values. A split label must be a "
            f"property of the group, or the contrast is between overlapping sets."
        )


def across_groups(values):
    """Between-replicate summary: mean, sd and a percentile CI over GROUPS.

    Distinct from the per-group posterior HDI, and deliberately named differently. They answer
    different questions -- ``hdi_*`` is how sure the model is about one group given the
    Dirichlet draws; ``ci_*`` is how much the quantity varies between patients. Reporting them
    under the same column names would make them indistinguishable on sight.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"value": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_groups": 0}
    if n == 1:
        # one replicate carries no between-group spread; NaN says so rather than 0.0
        return {"value": float(v[0]), "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_groups": 1}
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"value": float(v.mean()), "sd": float(v.std(ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi), "n_groups": n}


def build_result(table, *, value="value", extra_values=()):
    """Reduce ``table`` (per draw) to ``result`` (per group[, item]).

    ``result`` is built FROM ``table`` rather than computed alongside it, so the two cannot
    drift -- which is the same reason ``pl`` now reads the cache instead of recomputing.

    Reduces over ``draw`` ONLY, and the grouping keys are therefore *every other column*
    rather than a named list: any label a metric puts in ``table`` -- ``cov_from``/``cov_to``
    for the flux, an item column, a split label -- identifies a row here.

    Items are KEPT: a swarm plot needs one point per clone, and collapsing them here would make
    the per-item view unreachable from the cached result.
    """
    val_cols = (value, *extra_values)

    # An empty table has NO columns at all, so `keys` comes back empty and the scalar branch
    # below indexes table[value] -- which does not exist -- raising KeyError('value') four
    # frames deep in pandas. That is a real path, not a defensive check: phenotypic_flux
    # produces nothing when no replicate has clones at BOTH covariate levels, which is exactly
    # what happens if the covariate is constant within replicate (a between-group label like
    # treatment arm rather than a within-patient axis). The metric is right to find nothing;
    # crashing on it hides why.
    if table is None or not len(table):
        # The columns the non-empty shape would have had, not just `value`. An empty frame
        # that is missing its declared extras is indistinguishable from a frame computed
        # without them, which is how an empty `phenotypic_flux` -- a real and documented
        # outcome -- would read as "no reference was computed" rather than "no rows".
        return pd.DataFrame(columns=list(val_cols))

    keys = [c for c in table.columns if c not in ("draw", *val_cols)]

    def _extra(chunk):
        # the endpoints of a delta: their posterior means, carried so the paired view is
        # renderable from the delta result alone and therefore matched by construction
        return {c: float(np.nanmean(chunk[c].to_numpy(dtype=float))) if len(chunk) else np.nan
                for c in extra_values}

    if not keys:
        agg = summarize(table[value].to_numpy())
        row = {value: agg.pop("mean"), **agg, **_extra(table)}
        return pd.DataFrame([row])
    rows = []
    for label, chunk in table.groupby(keys, observed=True, dropna=False):
        label = label if isinstance(label, tuple) else (label,)
        row = dict(zip(keys, label))
        agg = summarize(chunk[value].to_numpy())
        row[value] = agg.pop("mean")
        row.update(agg)
        row.update(_extra(chunk))
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_to_replicates(result, *, groupby, splitby=None, value="value", keep=()):
    """One row per replicate: the item axis averaged away.

    THE pseudoreplication step, and it is a shared function rather than a line inside
    ``build_stats`` because the plotting layer calls this same collapse. Marks and p-value
    therefore share one replicate unit: they cannot disagree about it if they come from the
    same collapse.

    ``keep`` names further label columns to preserve -- the plotting layer passes whatever it
    is about to put on x and hue, since collapsing away an axis it is drawing would silently
    drop the split.

    ``value`` may be a LIST, and when it is, every listed column is collapsed under ONE mask:
    a row is dropped if ANY of them is non-finite. Averaging each column over its own mask
    means the value and its reference rest on different replicate sets, and then
    ``mean(excess)`` stops equalling ``mean(value) - mean(null_value)``. A single-column call
    returns a frame with that one value column.
    """
    if result is None or not len(result) or groupby is None or groupby not in result.columns:
        return result
    values = [value] if isinstance(value, str) else list(value)
    values = [v for v in values if v in result.columns]
    if not values:
        return result
    keys = [c for c in (groupby, splitby, *keep) if c and c in result.columns]
    keys = list(dict.fromkeys(keys))
    if len(values) > 1:
        finite = np.isfinite(result[values].to_numpy(dtype=float)).all(axis=1)
        result = result.loc[finite]
        if not len(result):
            return result.loc[:, keys + values]
    return (result.groupby(keys, observed=True, dropna=False)[values]
            .mean().reset_index())


def build_stats(result, *, groupby, splitby, value="value"):
    """The between-split contrast, or ``None`` when ``splitby`` is not set.

    ``value`` may be a LIST of quantities -- ``("value", "excess")`` once a reference has been
    computed. The frame then carries one row per (contrast, quantity) and a ``quantity``
    column saying which is which. Nothing switches automatically on the presence of a
    reference: the plot selects, so the marks and the star it carries are always the same
    quantity.

    The replicate unit is the GROUP. When the metric has an item axis, ``result`` holds one row
    per (group, item) -- so the item rows are averaged to one value per group FIRST, and the
    contrast is over groups. That is what makes pseudoreplication structurally impossible here:
    15 clones from 2 patients contribute n=2, not n=15, because there are only 2 group rows to
    compare.

    The contrast math itself is NOT here -- it is ``_compare.compare_groups``, so the package
    has one implementation of "Mann-Whitney two levels and star the p". This function owns the
    part that is specific to a metric result: collapsing items to groups first, and attaching
    the between-replicate spread of each arm.
    """
    if splitby is None or groupby is None or result is None or not len(result):
        return None
    from .._stats import compare_groups

    quantities = [value] if isinstance(value, str) else list(value)
    quantities = [q for q in quantities if q in result.columns]
    if not quantities:
        return None

    # THE pseudoreplication step. Everything after this sees one number per group. Shared with
    # the plotting layer so the marks cannot describe a different unit from the p-value.
    #
    # ONE collapse over EVERY quantity, not one per quantity: a per-column mask would drop a
    # different non-finite set for `value` than for `excess`, so `n_a`/`n_b` would differ
    # between the two rows of one stats frame and each row would describe a different set of
    # replicates. The rows are then directly comparable, which is what lets a plot switch
    # quantity and keep its own star honest.
    per_group = collapse_to_replicates(result, groupby=groupby, splitby=splitby,
                                       value=quantities)

    rows = []
    for quantity in quantities:
        contrasts = compare_groups(per_group, value=quantity, splitby=splitby)
        if contrasts is None or not len(contrasts):
            continue
        for _, c in contrasts.iterrows():
            a, b = c["group_a"], c["group_b"]
            row = {splitby: f"{a} vs {b}", "level_a": a, "level_b": b,
                   "replicate_unit": groupby, "quantity": quantity}
            for suffix, level in (("a", a), ("b", b)):
                arm = across_groups(per_group.loc[per_group[splitby] == level, quantity])
                row[f"mean_{suffix}"] = arm["value"]
                row[f"sd_{suffix}"] = arm["sd"]
                row[f"ci_low_{suffix}"] = arm["ci_low"]
                row[f"ci_high_{suffix}"] = arm["ci_high"]
                row[f"n_{suffix}"] = arm["n_groups"]
            row["delta"] = float(c["delta"])
            row["stat"] = float(c["U"])
            row["p"] = float(c["p"])
            row["stars"] = c["stars"]
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def metric_table(adata, *, covariate, groupby, splitby, clones, item_col, compute,
                 extra_labels=None, fit=None):
    """Build the long ``table`` every metric shares: one row per (covariate, group, item, draw).

    ``compute(clone_subset)`` returns one entry per draw. With an item axis that entry is a
    ``{item: value}`` mapping; without one (mutual_information) it is a scalar.

    The group loop lives here rather than in each metric so the four metrics cannot diverge on
    what ``groupby`` means. Clone restriction is intersected with the group's clones, never
    shadowed: ``groupby=...`` together with ``clones=[...]`` keeps only the clones in both.
    """
    obs = adata.obs
    clone_labels = fit_clone_labels(adata, fit)
    # a None label is not a label: `phenotypic_flux` has no single covariate, and carrying
    # `covariate=None` through added an all-NaN column to every row of its result
    base = {"covariate": covariate} if covariate is not None else {}
    if extra_labels:
        base.update({k: v for k, v in extra_labels.items() if v is not None})

    def _emit(rows, label_row, per_draw):
        for draw, payload in enumerate(per_draw):
            if item_col is None:
                # a draw's payload is a scalar, or a mapping of value columns when the metric
                # carries more than one. Mutual information is the only metric with no item
                # axis, so it is the only one that reaches this branch with a mapping -- but
                # the two branches now handle a payload the same way, which is the point.
                rows.append({**label_row, "draw": draw,
                             **(payload if isinstance(payload, dict) else {"value": payload})})
            else:
                for item, value in payload.items():
                    row = {**label_row, item_col: item, "draw": draw}
                    # an item's payload is a scalar, or a mapping of value columns when the
                    # metric carries more than one (the delta and its two endpoints)
                    row.update(value if isinstance(value, dict) else {"value": value})
                    rows.append(row)

    rows = []
    if groupby is None:
        _emit(rows, dict(base), compute(clones))
        return pd.DataFrame(rows)

    _validate_group_clones(clone_labels, obs[groupby], groupby,
                           hint=_refit_hint(adata, fit, groupby))
    n_missing = int(obs[groupby].isna().sum())
    if n_missing:
        warnings.warn(
            f"groupby={groupby!r}: {n_missing} of {len(obs)} cells "
            f"({100 * n_missing / max(len(obs), 1):.1f}%) have no group label and are excluded "
            f"from every row of this result.",
            UserWarning, stacklevel=3,
        )

    for g in obs[groupby].dropna().unique().tolist():
        gmask = obs[groupby] == g
        group_clones = clone_labels[gmask.to_numpy()].dropna().unique().tolist()
        if clones is not None:
            allowed = set(group_clones)
            group_clones = [c for c in clones if c in allowed]
            if not group_clones:
                continue
        label_row = {**base, groupby: g}
        if splitby is not None:
            label_row[splitby] = obs.loc[gmask, splitby].iloc[0]
        _emit(rows, label_row, compute(group_clones))
    return pd.DataFrame(rows)
