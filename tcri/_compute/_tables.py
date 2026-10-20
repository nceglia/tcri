"""Shared reduction helpers for the ``tl`` metrics.

Each metric pulls the clone×phenotype joint from the engine (:func:`joint_distribution`,
``use_logits=True``), reduces per draw, and — for ``n_samples>0`` — summarizes the draw
distribution (mean / sd / HDI). ``groupby`` is implemented here by **restricting clones
per group** (full-space clone masks + ``clones=``, never slicing the AnnData), which
relies on clones being disjoint across groups (a TCR clone never spans two patients).
"""
from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import pandas as pd

from .._state import keys as K
from .._stats import hdi
from ._repertoire import _clonotype_sharing, _pool_labels
# NOTE: ``tools._joint`` is imported lazily inside the functions that need it, never at module
# level. ``_compute`` is the lower layer — every tools/* metric imports *down* into this module —
# so a module-level import back up into ``tools`` inverts the layering and makes the package
# import-order dependent. It happened to resolve here only because ``tools/_joint`` does not
# itself reach back into ``_tables``; that is a property of today's code, not a guarantee.


def joint_draws(adata, covariate, *, n_samples, weighted, temperature, clones, random_state,
                use_logits=True, device=None, fit=None):
    """Return ``(draws, phenotype_cols)`` where ``draws`` is a list of ``(clone_ids, [C, P])``
    per posterior draw (length 1 for ``n_samples=0``).

    Consumes the engine's raw blocks directly. Going through
    :func:`~tcri.tools._joint.joint_distribution` would flatten ``[S, n_rows, P]``
    into a MultiIndex DataFrame only for this function to ``groupby('draw')``
    and unpack it straight back to arrays.
    Ordering matches the DataFrame path exactly (blocks in covariate order, clones in
    block order, then the ``clones=`` filter applied as a stable reorder).

    ``covariate`` is required: a reduced metric has no defined meaning over stacked covariate
    levels, because the row axis would become the (covariate, clone) pair rather than the clone.
    """
    from ..tools._joint import _engine_blocks  # lazy: see module note

    # Stacking the per-covariate blocks row-wise gives a clone present in k covariate levels
    # k ROWS, so the row axis of the joint becomes the (covariate, clone) pair rather than the
    # clone and H(c) is the entropy over pseudo-clones.
    #
    # `min` divides by min(H(c), H(phi)). When clones OUTNUMBER phenotypes it selects H(phi),
    # which row-splitting does not touch, so the default can look unaffected; when they do not,
    # it selects H(c) and moves. `average` always moves, since it averages both entropies.
    #
    # The three metrics that reduce to a scalar also have no shared meaning for covariate=None
    # (a per-phenotype index, a (covariate, clonotype) index and a stacked table are each
    # defensible). Picking one is a question about the estimand, not about the code, so
    # covariate is required wherever the result must be REDUCED.
    #
    # joint_distribution(covariate=None) is deliberately allowed: it LABELS the blocks
    # with a covariate index level instead of collapsing them, so no ambiguity arises there,
    # and it remains the way to get every covariate in one object.
    if covariate is None:
        raise ValueError(
            "covariate is required for scalar metrics.\n"
            "\n"
            "A scalar metric reduces one joint. Stacking every covariate level into one table "
            "treats each (covariate, clone) pair as a distinct clone, which inflates H(c) and "
            "the clone axis it is measured on.\n"
            "\n"
            "Pass an explicit covariate level, or use tl.joint_distribution(covariate=None) to "
            "get every level as a labeled (covariate, clonotype) table and reduce it the way "
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
        # A single draw has no spread to report. Returning sd=0 and a zero-width HDI would state
        # certainty that was never measured. mean is still the draw; the rest is undefined, and
        # NaN says so.
        return {"mean": float(v[0]), "sd": np.nan, "hdi_low": np.nan, "hdi_high": np.nan}
    lo, hi = hdi(v, prob=hdi_prob)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "hdi_low": lo, "hdi_high": hi}


def clone_col(adata):
    from .._state import keys as K
    return adata.uns[K.METADATA]["clone_col"]


def _clone_codes(adata, fit):
    """The fit's stored clone code for each clone x covariate row (``ct_to_c``).

    Raises ``ValueError`` when a code is negative. Such a row was built from cells without a
    clonotype, and read through the category list it would name the last clone. The model also
    fitted that row under the last clone, so masking it would not make the rest of the substrate
    right; the fit has to be redone.
    """
    key = K.fit_key(K.CT_TO_C, fit)
    ct_to_c = np.asarray(adata.uns[key])
    if (ct_to_c < 0).any():
        raise ValueError(
            f"uns[{key!r}] holds a negative clone code: this fit has a clone x covariate row for "
            f"cells without a clonotype, and that row cannot be read as a clone. Redo the fit: "
            f"keep only the cells that have a clonotype, run TCRIModel.setup_anndata, train, and "
            f"write the substrate again with to_anndata()."
        )
    return ct_to_c


def fit_clone_labels(adata, fit=None):
    """Per-cell clone id **as the fit saw it**, indexed by ``obs_names``.

    ``CLONOTYPE_CATEGORIES[ct_to_c[ct_array]]``. On the main fit this equals ``obs[clone_col]``
    cell for cell; under a clonotype null it does not, and THAT is the point. `metric_frames`
    restricts the engine by clone id while the engine's rows come from the permuted codes, so a
    clone list built from ``obs`` selects the wrong null rows whenever a permutation stratum spans
    more than one group -- silently, because the parent's ``obs`` passes every disjointness check.

    Raises ``ValueError`` when the fit stores a negative clone code (see :func:`_clone_codes`).
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
    ct_to_c = _clone_codes(adata, fit)
    cats = np.asarray(adata.uns[K.CLONOTYPE_CATEGORIES], dtype=object)
    return pd.Series(cats[ct_to_c[ct_array]], index=adata.obs_names, dtype=object)


def clones_at(adata, covariate, *, fit=None, group_clones=None):
    """Clone ids with cells at ``covariate`` **in the fit's own substrate**.

    Returned in ascending clone-code order, which is the substrate's order and not ``obs``
    first-appearance order. No metric value depends on it -- the entropies sum a column,
    `joint_draws` reorders its ids by the requested ``clones=`` list, and `build_result` groups.

    Raises ``ValueError`` when the fit stores a negative clone code (see :func:`_clone_codes`).
    """
    ct_to_c = _clone_codes(adata, fit)
    # the engine's rule for a level: equal to a registered category, so 0 finds a level
    # registered as 0.0 here exactly as it does in the joint
    cov_cats = list(adata.uns[K.COVARIATE_CATEGORIES])
    if covariate not in cov_cats:
        return []
    m = cov_cats.index(covariate)
    ct_to_cov = np.asarray(adata.uns[K.fit_key(K.CT_TO_COV, fit)])
    codes = np.unique(ct_to_c[ct_to_cov == m])
    cats = list(adata.uns[K.CLONOTYPE_CATEGORIES])
    at = [cats[c] for c in codes]
    if group_clones is None:
        return at
    allowed = set(group_clones)
    return [c for c in at if c in allowed]


def _resolve_clones(adata, clones, exclude_pools, *, column=None):
    """The clones a tool counts: ``clones`` when it is given, else every registered clone,
    minus the pools of the registered clonotype column when ``exclude_pools`` is true.

    Every tool resolves its clones here, so no two tools can disagree about what a clone is. A
    pool is a ``pooled@{group}`` label written by ``tcri.pp.pool_rare_clones``: a mixture of
    many rare clones, not a clone. The derivation record lists the pools of the column it
    wrote, so a column with no pooling step has nothing to exclude. The registered labels and
    the pools are shared by every fit of an object, so the answer does not depend on the fit.
    An id in ``clones`` that names no registered clone is kept here and matches nothing in any
    tool.

    Parameters
    ----------
    adata
        The object carrying the derivation record.
    clones
        The caller's ``clones=``: clone ids, or ``None`` for every registered clone.
    exclude_pools
        Whether the pools are left out.
    column
        The registered clonotype column. ``None`` reads the one ``to_anndata`` records in
        ``uns[K.METADATA]``; an object without that record names no registered column, and
        nothing is excluded. A tool that holds the model passes the column the model
        registered when the record is absent.

    Returns
    -------
    list or None
        ``None`` when ``clones`` is ``None`` and nothing is excluded, which every tool reads as
        every registered clone. Otherwise the clone ids to count, in the order of ``clones``
        when it is given and in the registered category order when not.
    """
    if column is None:
        column = (adata.uns.get(K.METADATA) or {}).get(K.CLONE_COL)
    pools = set(_pool_labels(adata, column)) if exclude_pools and column is not None else set()
    if clones is not None:
        return [c for c in clones if c not in pools]
    if not pools:
        return None
    # the categories `to_anndata` records, or the same categories read off the column on an
    # object it has not written
    registered = (adata.uns[K.CLONOTYPE_CATEGORIES] if K.CLONOTYPE_CATEGORIES in adata.uns
                  else adata.obs[column].astype("category").cat.categories)
    return [c for c in registered if c not in pools]


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
    # disk -- which is every object a reference is read from in practice. Written against the
    # round-tripped shape, not the in-memory one.
    raw = settings.get("strata")
    strata = [] if raw is None else [str(c) for c in raw]
    if not strata:
        return f" This is fit {fit!r}."
    if str(settings.get("kind")) == "covariate":
        return (f" This is fit {fit!r}, a covariate null, whose strata are fixed at {strata} and "
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

    Which clonotypes span groups is decided by :func:`._repertoire._clonotype_sharing`, tcri's
    one rule for it. The error names the first shared clonotype in order of appearance and the
    first two groups it appears in.
    """
    shared = _clonotype_sharing(labels, groups)
    if shared:
        c, spans = next(iter(shared.items()))
        raise ValueError(
            f"groupby={groupby!r}: clonotype {c!r} spans groups {spans[0]!r} and {spans[1]!r}. "
            f"The metric groupby restricts by clone id (clones=), which requires clones "
            f"to be disjoint across groups (e.g. the patient-specific ids "
            f"`tcri.pp.pool_rare_clones` writes). Use a clone-disjoint groupby, or pre-filter "
            f"with `clones=`.{hint}"
        )


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

    It raises unless ``groupby`` is set, ``splitby`` is a column of ``obs``, and ``splitby`` is
    constant within each group. Without the last, a group's split label would be whichever one
    its first cell happens to carry.
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


def build_result(table, *, value="value", extra_values=(), one_mask=False):
    """Reduce ``table`` (per draw) to ``result`` (per group[, item]).

    ``result`` is built FROM ``table`` rather than computed alongside it, so the two cannot
    drift -- which is the same reason ``pl`` now reads the cache instead of recomputing.

    Reduces over ``draw`` ONLY, and the grouping keys are therefore *every other column*
    rather than a named list: any label a metric puts in ``table`` -- ``cov_from``/``cov_to``
    for the flux, an item column, a split label -- identifies a row here.

    Items are KEPT: a swarm plot needs one point per clone, and collapsing them here would make
    the per-item view unreachable from the cached result.

    ``one_mask=True`` reduces ``value`` and every ``extra_values`` column over one set of draws,
    the draws where all of them are finite. A change and its two endpoints are reduced that way,
    so ``value == value_to - value_from`` holds in the summary as it does on every draw; with a
    mask per column, a draw finite on one side only would enter one mean and not the other.
    """
    val_cols = (value, *extra_values)

    # An empty table has NO columns at all, so `keys` comes back empty and the scalar branch
    # below indexes table[value] -- which does not exist -- raising KeyError('value') four
    # frames deep in pandas. That is a real path, not a defensive check: a change has no item
    # rows when no replicate has clones at BOTH covariate levels, which is exactly what happens
    # if the covariate is constant within replicate (a between-group label like treatment arm
    # rather than a within-patient axis). The metric is right to find nothing; crashing on it
    # hides why.
    if table is None or not len(table):
        # The columns the non-empty shape would have had, not just `value`. An empty frame
        # that is missing its declared extras is indistinguishable from a frame computed
        # without them, which is how an empty item level -- a real and documented outcome --
        # would read as "no reference was computed" rather than "no rows".
        return pd.DataFrame(columns=list(val_cols))

    keys = [c for c in table.columns if c not in ("draw", *val_cols)]

    def _masked(chunk):
        if not one_mask:
            return chunk
        finite = np.isfinite(chunk[list(val_cols)].to_numpy(dtype=float)).all(axis=1)
        return chunk.loc[finite]

    def _extra(chunk):
        # further value columns (a change's endpoints, a denominator): their posterior means. A
        # column with no number in it (a replicate with no joint) has a NaN mean, an expected
        # outcome rather than a numpy warning.
        out = {}
        for c in extra_values:
            v = chunk[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            out[c] = float(v.mean()) if v.size else np.nan
        return out

    if not keys:
        chunk = _masked(table)
        agg = summarize(chunk[value].to_numpy())
        row = {value: agg.pop("mean"), **agg, **_extra(chunk)}
        return pd.DataFrame([row])
    rows = []
    for label, chunk in table.groupby(keys, observed=True, dropna=False):
        label = label if isinstance(label, tuple) else (label,)
        row = dict(zip(keys, label))
        chunk = _masked(chunk)
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
    ``mean(adjusted)`` stops equalling ``mean(value) - mean(null_value)``. A single-column call
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


def build_stats(result, *, groupby, splitby, value="value", change=None):
    """The tests of one quantity: the covariate rows of a change, then the split rows; ``None``
    when nothing is compared.

    ``value`` names the column contrasted -- ``"value"``, or ``"adjusted"`` once a reference has
    been computed -- and the frame carries it back in a ``quantity`` column, so a reader of a
    stored ``stats`` knows which column the p-value is about without the payload around it.

    The replicate unit is the GROUP. When the metric has an item axis, ``result`` holds one row
    per (group, item) -- so the item rows are averaged to one value per group FIRST, and the
    contrast is over groups. That is what makes pseudoreplication structurally impossible here:
    15 clones from 2 patients contribute n=2, not n=15, because there are only 2 group rows to
    compare.

    The contrast math itself is NOT here -- it is ``_compare.compare_groups``, so the package
    has one implementation of "Mann-Whitney two levels and star the p". This function owns the
    part that is specific to a metric result: collapsing items to groups first, and attaching
    the between-replicate spread of each arm.

    Every pair of split levels that has a replicate row gets its row, whatever the values: an
    arm with no finite value has ``n`` 0 and the row has NaN statistics and blank stars, so the
    rows are fixed by the replicates and their splits. ``comparison`` is ``"split"`` on these
    rows.

    ``change`` is ``{"cov_from", "cov_to", "distance"}`` for a change result, which adds its
    covariate rows (:func:`_covariate_rows`).

    The labels of a row (the ``splitby`` column, ``level_a``, ``level_b``) are stored as text:
    one column holds a split level on a covariate row and ``"a vs b"`` on a split row, and
    ``level_a`` a covariate level or a split level, so a boolean or numeric label beside a string
    would make a column ``.h5ad`` cannot write.
    """
    if result is None or not len(result) or value not in result.columns:
        return None
    rows = []
    if change is not None:
        rows += _covariate_rows(result, groupby=groupby, splitby=splitby, value=value, **change)
    if splitby is not None and groupby is not None:
        rows += _split_rows(result, groupby=groupby, splitby=splitby, value=value)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    for column in ("level_a", "level_b", splitby):
        if column is not None and column in frame.columns:
            frame[column] = frame[column].map(lambda v: v if pd.isna(v) else str(v))
    return frame


def _split_rows(result, *, groupby, splitby, value):
    from .._stats import compare_groups

    # THE pseudoreplication step. Everything after this sees one number per group. Shared with
    # the plotting layer so the marks cannot describe a different unit from the p-value.
    per_group = collapse_to_replicates(result, groupby=groupby, splitby=splitby, value=value)

    contrasts = compare_groups(per_group, value=value, splitby=splitby, replicate=groupby)
    tested = {(c["group_a"], c["group_b"]): c for _, c in contrasts.iterrows()}
    levels = sorted(per_group[splitby].dropna().unique().tolist(), key=str)
    rows = []
    for a, b in itertools.combinations(levels, 2):
        row = {"comparison": "split", splitby: f"{a} vs {b}", "level_a": a, "level_b": b,
               "replicate_unit": groupby, "quantity": value}
        for suffix, level in (("a", a), ("b", b)):
            arm = across_groups(per_group.loc[per_group[splitby] == level, value])
            row[f"mean_{suffix}"] = arm["value"]
            row[f"sd_{suffix}"] = arm["sd"]
            row[f"ci_low_{suffix}"] = arm["ci_low"]
            row[f"ci_high_{suffix}"] = arm["ci_high"]
            row[f"n_{suffix}"] = arm["n_groups"]
        c = tested.get((a, b))
        if c is None:
            # an arm with no finite value: compare_groups has no row for it
            row.update(delta=row["mean_b"] - row["mean_a"], stat=np.nan, p=np.nan, stars="")
        else:
            row.update(delta=float(c["delta"]), stat=float(c["stat"]), p=float(c["p"]),
                       stars=c["stars"])
        rows.append(row)
    return rows


def _covariate_rows(result, *, groupby, splitby, value, cov_from, cov_to, distance=False):
    """The paired test of a change, one row per split level, or one over every replicate
    without ``splitby``: a Wilcoxon signed-rank test on each replicate's change.

    The arms are the change's endpoints of the tested quantity (``value_from``/``value_to``, or
    ``adjusted_from``/``adjusted_to``), and ``level_a``/``level_b`` are ``cov_from``/``cov_to``,
    so ``delta = mean_b - mean_a`` as on a split row. A replicate enters when both endpoints are
    finite and its change is not exactly 0; the arms, ``delta`` and ``n`` are all over the
    replicates the test used. Fewer than two leave the row with NaN statistics and blank stars.

    ``distance=True`` is a change that is a distance between the two levels (flux), with no
    endpoints and never negative: the row tests the adjusted value against 0, its arms are NaN and
    ``delta`` is the mean over the replicates used. Without a usable reference (``value`` is
    ``"value"``) a test against 0 rejects for any data, so p is NaN and stars are blank.

    With no replicate the whole object is one replicate: the row is present, with ``n`` of 1 or 0
    and NaN statistics.
    """
    from .._stats import compare_groups

    frame = result if groupby is not None else result.assign(_replicate=0)
    unit = groupby if groupby is not None else "_replicate"
    if splitby is not None and splitby in frame.columns:
        levels = sorted(frame[splitby].dropna().unique().tolist(), key=str)
    else:
        levels = [None]
    ends = (None if distance else
            ("adjusted_from", "adjusted_to") if value == "adjusted" else
            ("value_from", "value_to"))
    rows = []
    for level in levels:
        sub = frame if level is None else frame.loc[frame[splitby] == level]
        sub = collapse_to_replicates(sub, groupby=unit, splitby=None,
                                     value=[value] if distance else list(ends))
        if distance:
            a = np.zeros(len(sub))
            b = sub[value].to_numpy(dtype=float)
        else:
            a = sub[ends[0]].to_numpy(dtype=float)
            b = sub[ends[1]].to_numpy(dtype=float)
        used = np.isfinite(a) & np.isfinite(b) & (b - a != 0)
        row = {"comparison": "covariate"}
        if level is not None:
            row[splitby] = level
        row.update(level_a=cov_from, level_b=cov_to,
                   replicate_unit=groupby if groupby is not None else np.nan, quantity=value)
        for suffix, arm in (("a", a[used]), ("b", b[used])):
            summary = across_groups(arm)
            row[f"mean_{suffix}"] = np.nan if distance else summary["value"]
            row[f"sd_{suffix}"] = np.nan if distance else summary["sd"]
            row[f"ci_low_{suffix}"] = np.nan if distance else summary["ci_low"]
            row[f"ci_high_{suffix}"] = np.nan if distance else summary["ci_high"]
            row[f"n_{suffix}"] = int(used.sum())
        row["delta"] = (float(np.mean(b[used] - a[used])) if used.any() else np.nan)
        row.update(stat=np.nan, p=np.nan, stars="")
        if not (distance and value != "adjusted"):
            ids = sub[unit].to_numpy()[used]
            paired = pd.DataFrame({unit: np.concatenate([ids, ids]),
                                   "_arm": [0] * len(ids) + [1] * len(ids),
                                   "_v": np.concatenate([a[used], b[used]])})
            tested = compare_groups(paired, value="_v", splitby="_arm", replicate=unit,
                                    paired=True)
            if len(tested):
                c = tested.iloc[0]
                row.update(stat=float(c["stat"]), p=float(c["p"]), stars=c["stars"])
        rows.append(row)
    return rows


class ZeroNormalizer(ValueError):
    """A replicate kernel's normalizer is 0, so its normalized value is undefined.

    Raised inside ``compute``, where the replicate is not known; :func:`metric_frames` re-raises
    it as a ``ValueError`` naming the replicate and the covariate level. ``level`` is the level
    whose joint raised, set on a change, where the call has two.
    """

    def __init__(self, message, level=None):
        super().__init__(message)
        self.level = level


def at_level(level, kernel, *args, **kwargs):
    """``kernel(*args, **kwargs)``, with a :class:`ZeroNormalizer` it raises naming ``level``."""
    try:
        return kernel(*args, **kwargs)
    except ZeroNormalizer as err:
        raise ZeroNormalizer(str(err), level=level) from None


def check_covariates(adata, *, covariate, cov_from, cov_to, metric):
    """Whether a call is at one covariate level (``"covariate"``) or a change (``"change"``).

    Raises ``ValueError`` unless exactly one of ``covariate`` or the pair ``cov_from`` +
    ``cov_to`` is given, naming both call forms, and on a change when either level is not a
    registered covariate level. ``cov_from == cov_to`` is allowed: its change is 0.
    """
    pair = (cov_from is not None, cov_to is not None)
    if covariate is not None and not any(pair):
        return "covariate"
    if covariate is None and all(pair):
        levels = list(adata.uns[K.COVARIATE_CATEGORIES])
        for name, level in (("cov_from", cov_from), ("cov_to", cov_to)):
            if level not in levels:
                raise ValueError(
                    f"{metric}: {name}={level!r} is not a registered covariate level; the "
                    f"levels are {levels}."
                )
        return "change"
    raise ValueError(
        f"{metric}: pass covariate= for the value at one covariate level, or cov_from= with "
        f"cov_to= for the change from one level to another (got covariate={covariate!r}, "
        f"cov_from={cov_from!r}, cov_to={cov_to!r})."
    )


def resolve_seed(random_state) -> int:
    """One integer seed for a call: every ``tl`` call turns a stateful ``random_state`` into one
    (:func:`~tcri._state.storage.tl_result`), and a change turns ``None`` into one too.

    Both levels of a change are drawn from one shared sample: the engine samples every clone x
    covariate row and then selects a level's block, so two engine calls carrying one integer seed
    realize the same draw, and a level against itself changes by exactly 0. A stateful source
    would hand each engine call a fresh seed and decouple the two levels, so it is turned into an
    integer once per call: ``None`` generates one, an integer is itself, a ``torch.Generator``
    gives its initial seed, and anything else (a numpy ``Generator``, ``RandomState``,
    ``BitGenerator`` or ``SeedSequence``) gives one draw.

    This is common random numbers: the estimand is a difference between two levels of the same
    clones, so the draws are coupled and the interval is the coupled quantity's. Independent draws
    would report the sum of the two posteriors' spread plus a positive bias that does not vanish.
    """
    if random_state is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    import torch

    if isinstance(random_state, torch.Generator):
        return int(random_state.initial_seed())
    return int(np.random.default_rng(random_state).integers(0, 2**31 - 1))


def change_draws(adata, cov_from, cov_to, *, clones, seed, n_samples, weighted, temperature,
                 device=None, fit=None):
    """Both levels' joints of a set of clones, from one shared draw: ``(draws_from, draws_to,
    cols)``, each a list of ``(clone_ids, J)`` per draw as :func:`joint_draws` returns.

    ``seed`` is an integer (:func:`resolve_seed`). A level against itself reuses one draw, so its
    change is exactly 0.
    """
    kw = dict(n_samples=n_samples, weighted=weighted, temperature=temperature, clones=clones,
              random_state=seed, device=device, fit=fit)
    draws_from, cols = joint_draws(adata, cov_from, **kw)
    draws_to = draws_from if cov_to == cov_from else joint_draws(adata, cov_to, **kw)[0]
    return draws_from, draws_to, cols


def rows_of(clone_ids, J, wanted):
    """The rows of ``J`` for the clones in ``wanted``, in the order of ``wanted``."""
    at = {c: i for i, c in enumerate(clone_ids)}
    J = np.asarray(J, dtype=np.float64)
    return J[[at[c] for c in wanted]].reshape(len(wanted), J.shape[-1])


def _groups(adata, *, groupby, splitby, clones, exclude_pools, fit, base, skip_empty,
            validate=True):
    """Yield ``(label_row, clones)`` for every group: its labels and the clones it counts.

    The group loop lives here so every metric means the same thing by ``groupby``. The clones
    counted come from :func:`_resolve_clones`. Clone restriction is intersected with the group's
    clones, never shadowed: ``groupby=...`` together with ``clones=[...]`` keeps only the clones
    in both, in the order of ``clones``. Without ``clones``, leaving out the pools keeps each
    group's clones in the group's own order. With no ``groupby`` the whole object is one group,
    with no group label. ``skip_empty`` leaves out a group that counts no clone; otherwise it is
    yielded with an empty list.
    """
    obs = adata.obs
    clone_labels = fit_clone_labels(adata, fit)
    counted = _resolve_clones(adata, clones, exclude_pools)
    if groupby is None:
        yield dict(base), counted
        return

    if validate:
        _validate_group_clones(clone_labels, obs[groupby], groupby,
                               hint=_refit_hint(adata, fit, groupby))
    n_missing = int(obs[groupby].isna().sum())
    if n_missing:
        warnings.warn(
            f"groupby={groupby!r}: {n_missing} of {len(obs)} cells "
            f"({100 * n_missing / max(len(obs), 1):.1f}%) have no group label and are excluded "
            f"from every row of this result.",
            UserWarning, stacklevel=4,
        )

    for g in obs[groupby].dropna().unique().tolist():
        gmask = obs[groupby] == g
        group_clones = clone_labels[gmask.to_numpy()].dropna().unique().tolist()
        if counted is not None:
            if clones is None:
                kept = set(counted)
                group_clones = [c for c in group_clones if c in kept]
            else:
                allowed = set(group_clones)
                group_clones = [c for c in counted if c in allowed]
            if not group_clones and skip_empty:
                continue
        label_row = {**base, groupby: g}
        if splitby is not None:
            label_row[splitby] = obs.loc[gmask, splitby].iloc[0]
        yield label_row, group_clones


def movable_clones(adata, levels=None):
    """The clonotype labels whose cells the covariate null can move, at ``levels`` or at any.

    The covariate null shuffles each cell's covariate level within (clonotype, batch), so it can
    move a clonotype's cells at a level only where one of its (clonotype, batch) groups holds a
    cell at that level and a cell at another level. Decided from ``obs`` alone, not from a drawn
    shuffle, so the answer is the same on every run and for any number of nulls. ``levels=None``
    asks about any level.
    """
    frame = _cell_levels(adata)
    keys = [frame["clone"], frame["batch"]]
    mixed = frame["level"].groupby(keys, sort=False).nunique() > 1
    if levels is not None:
        wanted = _level_names(adata, levels)
        mixed &= frame["level"].isin(wanted).groupby(keys, sort=False).any()
    return set(mixed[mixed].index.get_level_values("clone"))


def _cell_levels(adata):
    """Each cell's clonotype, batch and covariate level, as text."""
    meta = adata.uns[K.METADATA]
    obs = adata.obs
    return pd.DataFrame({
        "clone": obs[meta[K.Config.CLONE_COL]].astype(str).to_numpy(),
        "batch": obs[meta[K.Config.BATCH_COL]].astype(str).to_numpy(),
        "level": obs[meta[K.Config.COVARIATE_COL]].astype(str).to_numpy(),
    })


def _level_names(adata, levels):
    """The text of the registered categories ``levels`` name, by the engine's rule (equality)."""
    cats = list(adata.uns[K.COVARIATE_CATEGORIES])
    return {str(cats[cats.index(level)]) for level in levels if level in cats}


def unmovable_groups(adata, *, levels, groupby, splitby, clones, exclude_pools, fit=None):
    """The label rows of the groups the covariate null cannot move at ``levels``: groups with a
    counted clone at one of ``levels``, none of which is in :func:`movable_clones`. A group with
    no counted clone at those levels has no values to adjust and is not one of them. The groups
    are those of :func:`_groups`, without its clone-disjointness check: the question is which
    clones a group's cells belong to, which a tool grouping cells (gene importance) asks too."""
    movable = movable_clones(adata, levels)
    frame = _cell_levels(adata)
    present = set(frame.loc[frame["level"].isin(_level_names(adata, levels)), "clone"])
    out = []
    with warnings.catch_warnings():
        # the metric's own pass over the groups has already warned about cells with no group
        warnings.simplefilter("ignore", UserWarning)
        groups = list(_groups(adata, groupby=groupby, splitby=splitby, clones=clones,
                              exclude_pools=exclude_pools, fit=fit, base={}, skip_empty=False,
                              validate=False))
    for label_row, group_clones in groups:
        counted = (group_clones if group_clones is not None
                   else adata.uns[K.CLONOTYPE_CATEGORIES])
        at = [str(c) for c in counted if str(c) in present]
        if at and not any(c in movable for c in at):
            out.append(label_row)
    return out


def metric_frames(adata, *, covariate=None, cov_from=None, cov_to=None, groupby, splitby, clones,
                  exclude_pools, item_col, compute, metric, fit=None):
    """Build a metric's per-draw frames: ``(replicate_draws, item_draws)``.

    At one covariate level, ``compute(clone_subset)`` returns one ``(replicate, items)`` pair per
    draw: ``replicate`` maps value columns to the replicate's number for its whole joint, and
    ``items`` maps each item to a mapping of value columns, or is ``None`` for a metric with no
    items. ``replicate_draws`` has one row per (covariate, group, draw); ``item_draws`` one row per
    (covariate, group, item, draw), or is ``None`` when ``item_col`` is ``None``.

    On a change (``cov_from`` and ``cov_to``), ``compute(clone_subset, persistent)`` also receives
    the group's persistent clones, those with cells at both levels, in ascending clone-code order;
    the rows are labeled ``cov_from`` and ``cov_to`` instead of ``covariate``. Two warnings name
    what the inputs leave out: the groups with no counted clone at a level, and a call where no
    group has a persistent clone, whose usual cause is a covariate that does not vary within a
    replicate.

    Every group has its replicate rows: a group with no joint at a level, or none of its clones
    left by ``clones=`` or ``exclude_pools``, gets them as NaN rather than no rows, so the row set
    is fixed by the inputs. Its item rows are whatever ``compute`` returns for an empty joint.

    Raises
    ------
    ValueError
        When a replicate's normalizer is 0 (:class:`ZeroNormalizer`), naming the replicate and
        the covariate level.
    """
    change = cov_from is not None
    if change:
        base = {"cov_from": cov_from, "cov_to": cov_to}
    else:
        base = {"covariate": covariate} if covariate is not None else {}
    rep_rows, item_rows = [], []
    n_groups, missing, persistent = 0, 0, False
    for label_row, group_clones in _groups(adata, groupby=groupby, splitby=splitby,
                                           clones=clones, exclude_pools=exclude_pools, fit=fit,
                                           base=base, skip_empty=False):
        n_groups += 1
        args = (group_clones,)
        if change:
            at_from = clones_at(adata, cov_from, fit=fit, group_clones=group_clones)
            at_to = set(clones_at(adata, cov_to, fit=fit, group_clones=group_clones))
            shared = [c for c in at_from if c in at_to]
            persistent = persistent or bool(shared)
            missing += not (at_from and at_to)
            args = (group_clones, shared)
        try:
            per_draw = compute(*args)
        except ZeroNormalizer as err:
            where = (f"{groupby}={label_row[groupby]!r}" if groupby is not None
                     else "the whole object")
            level = err.level if err.level is not None else covariate
            raise ValueError(f"{metric}: {where} at covariate {level!r}: {err}") from None
        for draw, (replicate, items) in enumerate(per_draw):
            rep_rows.append({**label_row, "draw": draw, **replicate})
            for item, vals in (items or {}).items():
                item_rows.append({**label_row, item_col: item, "draw": draw, **vals})
    if change:
        _warn_change(metric, groupby=groupby, cov_from=cov_from, cov_to=cov_to,
                     n_groups=n_groups, missing=missing, persistent=persistent,
                     has_items=item_col is not None)
    item_draws = pd.DataFrame(item_rows) if item_col is not None else None
    return pd.DataFrame(rep_rows), item_draws


#: Warnings are attributed to the caller's line, never to a frame inside the package. The
#: trailing separator keeps a sibling directory whose name starts with the package's out of it.
_PACKAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "")


def _warn_change(metric, *, groupby, cov_from, cov_to, n_groups, missing, persistent, has_items):
    unit = groupby or "replicate"
    if missing:
        warnings.warn(
            f"{metric}: {missing} of {n_groups} {unit} groups have no counted clone at "
            f"{cov_from!r} or at {cov_to!r}; their values are NaN and they leave the tests.",
            UserWarning, skip_file_prefixes=(_PACKAGE_DIR,),
        )
    if not persistent:
        items = ", so there are no item rows" if has_items else ""
        warnings.warn(
            f"{metric}: no {unit} has a clone at both {cov_from!r} and {cov_to!r}{items}. A "
            f"change is taken within each {unit}: it needs a covariate that varies inside a "
            f"single {unit} (a timepoint, say). A covariate that is constant within each {unit} "
            f"only distinguishes one {unit} from another.",
            UserWarning, skip_file_prefixes=(_PACKAGE_DIR,),
        )

