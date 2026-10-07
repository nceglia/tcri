"""``pl.gene_importance`` -- cache renderer for the in-silico perturbation.

Two views of one cached result, selected by ``kind``:

``"rank"`` (default)
    The top genes by importance, one x position per gene, the mark following the package's
    one rule: replicates when they vary (one dot per patient, boxed; the split as hue, with
    that gene's contrast starred above it), draws when only they vary (a violin per gene),
    a point with its HDI otherwise.

``"shift"``
    Which way each top gene moves the call: a gene × phenotype heatmap of ``shift``
    (``baseline − silenced``, averaged over groups), diverging and centred on zero, so a red
    cell is mass the phenotype LOSES when the gene is silenced and a blue cell is mass it gains.

Like every twin it takes no metric arguments: the genes, groups, split and draws are the ones
``perturb.gene_importance`` used, read from the cache.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._state import _reference
from ._base import (_axes, _boxstrip, _colors_for, _empty, _finish, _points,
                    _reference_legend, _violins, _zero_rule, check_quantity, filter_quantity,
                    order_from, resolve_quantity)

__all__ = ["gene_importance"]

_YLABEL = "gene importance (L1 shift of the phenotype call)"


def _top_genes(result, n_top, order, quantity="value"):
    """Genes ranked by their mean ``quantity`` over groups, or the caller's ``order``."""
    mean = result.groupby("gene", observed=True, sort=False)[quantity].mean()
    ranked = mean.sort_values(ascending=False).index.tolist()
    if order is not None:
        known = set(ranked)
        ranked = [g for g in order if g in known]
    return ranked[: int(n_top)] if n_top is not None else ranked


def _star_labels(ax, stats, genes, *, groupby, quantity):
    """That gene's own contrast (over ``groupby`` units), starred above its x position.

    Filtered by quantity FIRST, so the star is about the column on the axis: a value panel of a
    result whose contrast is on the adjusted value gets no rows and no star. The stars are then
    collected into a mapping keyed by gene, and one row per gene is what the panel has room to
    draw; a gene left with more than one row has a ``splitby`` of three or more levels, and the
    panel warns and draws no stars rather than resolving the duplicate by position.
    """
    stats = filter_quantity(stats, quantity)
    if stats is None or not len(stats) or "gene" not in stats.columns:
        return
    if stats["gene"].duplicated().any():
        # More than one CONTRAST per gene is what `splitby` with three or more levels produces,
        # and it is a limitation of a panel that puts ONE star above each gene: there is no
        # position for the other pairs. Draw nothing and say why, rather than starring one pair
        # and letting it read as the whole comparison.
        warnings.warn(
            f"not starring the ranking: `splitby` has more than two levels, so each gene "
            f"carries {stats.groupby('gene', observed=True).size().max()} contrasts and this "
            f"panel has one position per gene. Read them from "
            f"`tcri.get.gene_importance(adata, which='stats')`, or plot two levels at a time.",
            UserWarning, stacklevel=3,
        )
        return
    lut = {row["gene"]: (row.get("stars") or "ns") for _, row in stats.iterrows()}
    ymin, ymax = ax.get_ylim()
    y = ymax + 0.03 * ((ymax - ymin) or 1.0)
    drawn = 0
    for i, g in enumerate(genes):
        if g in lut:
            ax.text(i, y, lut[g], ha="center", va="bottom", fontsize=8, c="0.2", clip_on=False)
            drawn += 1
    if drawn:
        ax.set_ylim(ymin, ymax + 0.12 * ((ymax - ymin) or 1.0))


def _rank(adata, payload, params, *, genes, quantity, hue_order, palette, ax, figsize, save,
          show):
    result, table, stats = payload["result"], payload.get("table"), payload.get("stats")
    groupby, splitby = params.get("groupby"), params.get("splitby")
    fig, ax = _axes(ax, figsize)
    if result is None or not len(result) or "value" not in result.columns:
        return _finish(fig, _empty(ax, "no data for gene_importance",
                                   _reference.label_for(result, quantity, _YLABEL)),
                       save=save, show=show)
    d = result[result["gene"].isin(genes)].dropna(subset=[quantity])
    # The reference is drawn behind the value and nowhere else, and only when something in it
    # is finite: the rule `render_metric` applies, so an all-NaN null is "(no reference)" here
    # too rather than an all-NaN y handed to seaborn.
    ref = None
    if quantity == "value" and "null_value" in d.columns:
        if _reference.usable_reference(d):
            ref = "null_value"
        else:
            result = result.drop(columns=["null_value"])
    ylabel = _reference.label_for(result, quantity, _YLABEL)

    has_groups = groupby is not None and groupby in d.columns and d[groupby].nunique() > 1
    if has_groups:
        hue = splitby if (splitby and splitby in d.columns) else None
        if ref is not None:
            # the SAME hue and hue_order as the value pass. With hue=None the grey box pools
            # both arms into one distribution that belongs to neither, and the panel shows a
            # single reference where the value has two -- so a gene whose null differs between
            # arms reads as if it did not.
            _boxstrip(adata, d, x="gene", y=ref, hue=hue, order=genes, hue_order=hue_order,
                      palette=palette, ax=ax, ylabel=ylabel, rotation=90, reference=True)
        _boxstrip(adata, d, x="gene", y=quantity, hue=hue, order=genes, hue_order=hue_order,
                  palette=palette, ax=ax, ylabel=ylabel, rotation=90)
        if hue is not None:
            _star_labels(ax, stats, genes, groupby=groupby, quantity=quantity)
    elif (table is not None and "draw" in table.columns and table["draw"].nunique() > 1
          and quantity == "value"):
        # `quantity == "value"` only. The draws are a distribution of the VALUE; the adjusted
        # value in `table` is a per-group constant, so a violin of it is a spike at one number.
        # `render_metric` makes the same refusal for the same reason.
        t = table[table["gene"].isin(genes)].dropna(subset=[quantity])
        _violins(adata, t, x="gene", y=quantity, palette=palette, ax=ax, ylabel=ylabel,
                 rotation=90, order=genes)
    else:
        _points(adata, d, x="gene", y=quantity, palette=palette, ax=ax, ylabel=ylabel,
                rotation=90, order=genes, ref=ref)
    if quantity != "value":
        _zero_rule(ax)
    _reference_legend(ax, ref is not None)
    ax.set_xlabel("gene")
    return _finish(fig, ax, save=save, show=show)


def _shift(adata, payload, params, *, genes, ax, figsize, save, show):
    import matplotlib.pyplot as plt

    from .._state import keys as K

    shift = payload.get("shift")
    fig, ax = _axes(ax, figsize)
    if shift is None or not len(shift):
        return _finish(fig, _empty(ax, "no shift table for gene_importance", ""),
                       save=save, show=show)
    phenotypes = list(adata.uns.get(K.PHENOTYPE_CATEGORIES, shift["phenotype"].unique()))
    # mean over groups (and the covariate label, when present): one cell per gene x phenotype
    mat = (shift[shift["gene"].isin(genes)]
           .groupby(["gene", "phenotype"], observed=True)["shift"].mean()
           .unstack("phenotype").reindex(index=genes, columns=phenotypes))
    values = mat.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = limit or 1.0
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(phenotypes)))
    ax.set_xticklabels([str(p) for p in phenotypes], rotation=90)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels([str(g) for g in genes], fontsize=8)
    ax.set_xlabel("phenotype")
    ax.set_ylabel("gene")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("shift (baseline − silenced)")
    return _finish(fig, ax, save=save, show=show)


def gene_importance(adata, *, kind="rank", quantity="auto", n_top=25, key=None, order=None,
                    hue_order=None, palette=None, ax=None, figsize=(8, 4), save=None,
                    show=None, return_df=False):
    """The cached ``perturb.gene_importance``: a ranking (``kind="rank"``) or the signed
    gene × phenotype shift (``kind="shift"``) of the ``n_top`` most important genes.

    ``order`` restricts and orders the genes shown; ``hue_order`` orders the split levels;
    ``return_df`` hands back the cached ``result`` frame instead of drawing.

    ``quantity`` defaults to ``"auto"``: the adjusted importance whenever the cached result
    carries a reference, and the bare value otherwise, as on every twin. The correction matters
    most here. Silencing a gene is an intervention whose size scales with the gene's counts, and
    the encoder responds to that whatever the gene says about phenotype; the adjusted importance
    subtracts the same gene's importance under the permutation null, leaving the part of the
    ranking that a gene's abundance does not account for.

    ``quantity="value"`` draws the bare ranking, and is worth looking at once: the gap between
    the two panels IS the abundance confound, and it is a property of this estimand rather than
    of a fit. Which genes are shown and which quantity is drawn are separate decisions: the gene
    set is ranked by the adjusted importance whenever the result carries one, unless
    ``quantity="value"`` is asked for, so ``kind="rank"`` and ``kind="shift"`` describe the
    same genes.

    ``"adjusted"`` is refused for ``kind="shift"``: the heatmap is the per-phenotype
    decomposition of the importance, and the reference has no such decomposition stored, so an
    adjusted shift would have to be invented from a sum that does not decompose the same way.
    """
    from .. import get as _get

    if kind not in ("rank", "shift"):
        raise ValueError(f"kind must be 'rank' or 'shift', got {kind!r}")
    check_quantity(quantity, value_only=(kind == "shift"), message=(
        f"kind='shift' has no {quantity!r}: the heatmap decomposes the importance across "
        f"phenotypes, and the reference is stored as one number per gene and group, not as "
        f"a shift. Use kind='rank', quantity={quantity!r} for the adjusted importance, or "
        f"kind='shift', quantity='value' for the decomposition."))
    payload = _get.result(adata, "gene_importance", key=key)
    params = _get.params(adata, "gene_importance", key=key)
    result = payload["result"]
    # Which genes are SHOWN and which quantity is PLOTTED are separate decisions, and keeping
    # them separate is what lets the two panels of one figure stay about the same genes: the
    # heatmap has only one quantity it can draw, so tying its gene set to that quantity would
    # make `kind="rank"` and `kind="shift"` disagree by default.
    #
    # The rule, in one sentence: the gene set is ranked by the ADJUSTED importance whenever the
    # result carries a usable one, unless the caller explicitly asked for `quantity="value"`.
    # Decided from the caller's argument BEFORE it is resolved: on the heatmap "auto" and
    # "value" both resolve to the value, and only the unresolved argument tells them apart.
    rank_by = "value" if quantity == "value" else (
        "adjusted" if _reference.usable_reference(result, "adjusted") else "value")
    if return_df:
        return result
    # Resolved ONCE and handed to both views, so a half-threaded quantity cannot make the two
    # panels of one figure rank a different set of genes. Past `return_df`, and only for a
    # frame with rows: an empty result draws its empty panel whatever the quantity.
    if result is not None and len(result):
        quantity = resolve_quantity(result, quantity, name="gene_importance",
                                    value_only=(kind == "shift"))
    genes = _top_genes(result, n_top, order, rank_by) if (
        result is not None and len(result)) else []
    if kind == "rank":
        return _rank(adata, payload, params, genes=genes, quantity=quantity,
                     hue_order=hue_order, palette=palette, ax=ax, figsize=figsize,
                     save=save, show=show)
    return _shift(adata, payload, params, genes=genes, ax=ax, figsize=figsize,
                  save=save, show=show)
