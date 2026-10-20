"""``pl.gene_importance`` -- cache renderer for the in-silico perturbation.

Two views of one cached result, selected by ``view``:

``"replicate"`` (default)
    The top genes by adjusted importance, one x position per gene, the mark following the
    package's one rule: replicates when they vary (one dot per patient, boxed; the split as hue,
    with that gene's contrast starred above it), draws when only they vary (a violin per gene),
    a point with its HDI otherwise.

``"item"``
    Which way each top gene moves the call: a gene × phenotype heatmap of the item summary's
    adjusted shift (``baseline − silenced``, less the null's, averaged over groups), diverging and
    centered on zero, so a red cell is mass the phenotype LOSES when the gene is silenced, beyond
    what the null loses, and a blue cell is mass it gains.

Only the adjusted value is stored, so it is the one quantity drawn.

Like every twin it takes no metric arguments: the genes, groups, split and draws are the ones
``perturb.gene_importance`` used, read from the cache.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._base import (_axes, _boxstrip, _empty, _finish, _points, _violins, _zero_rule,
                    filter_quantity)

__all__ = ["gene_importance"]

_YLABEL = "adjusted gene importance (L1 shift of the phenotype call)"

#: The one quantity a gene_importance result stores.
_QUANTITY = "adjusted"


def _top_genes(result, n_top, order):
    """Genes ranked by their mean adjusted importance over groups, or the caller's ``order``."""
    mean = result.groupby("gene", observed=True, sort=False)[_QUANTITY].mean()
    ranked = mean.sort_values(ascending=False).index.tolist()
    if order is not None:
        known = set(ranked)
        ranked = [g for g in order if g in known]
    return ranked[: int(n_top)] if n_top is not None else ranked


def _star_labels(ax, stats, genes):
    """That gene's own contrast (over ``groupby`` units), starred above its x position.

    Filtered to the adjusted value FIRST, so the star is about the column on the axis. The stars
    are then collected into a mapping keyed by gene, and one row per gene is what the panel has
    room to draw; a gene left with more than one row has a ``splitby`` of three or more levels,
    and the panel warns and draws no stars rather than resolving the duplicate by position.
    """
    stats = filter_quantity(stats, _QUANTITY)
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
            f"`tcri.get.stats(adata, 'gene_importance')`, or plot two levels at a time.",
            UserWarning, stacklevel=3,
        )
        return
    # blank stars are a test that was not run (an undefined p): nothing is drawn for that gene
    lut = {row["gene"]: row.get("stars") for _, row in stats.iterrows() if row.get("stars")}
    ymin, ymax = ax.get_ylim()
    y = ymax + 0.03 * ((ymax - ymin) or 1.0)
    drawn = 0
    for i, g in enumerate(genes):
        if g in lut:
            ax.text(i, y, lut[g], ha="center", va="bottom", fontsize=8, c="0.2", clip_on=False)
            drawn += 1
    if drawn:
        ax.set_ylim(ymin, ymax + 0.12 * ((ymax - ymin) or 1.0))


def _rank(adata, payload, params, *, genes, hue_order, palette, legend, ax, figsize, save,
          show):
    level, stats = payload["replicate"], payload.get("stats")
    result, table = level["summary"], level["draws"]
    groupby, splitby = params.get("groupby"), params.get("splitby")
    fig, ax = _axes(ax, figsize)
    if result is None or not len(result) or _QUANTITY not in result.columns:
        return _finish(fig, _empty(ax, "no data for gene_importance", _YLABEL),
                       save=save, show=show, legend=legend)
    d = result[result["gene"].isin(genes)].dropna(subset=[_QUANTITY])
    if not len(d):
        return _finish(fig, _empty(ax, "no finite gene_importance", _YLABEL),
                       save=save, show=show, legend=legend)

    has_groups = groupby is not None and groupby in d.columns and d[groupby].nunique() > 1
    if has_groups:
        hue = splitby if (splitby and splitby in d.columns) else None
        _boxstrip(adata, d, x="gene", y=_QUANTITY, hue=hue, order=genes, hue_order=hue_order,
                  palette=palette, ax=ax, ylabel=_YLABEL, rotation=90)
        if hue is not None:
            _star_labels(ax, stats, genes)
    elif table is not None and "draw" in table.columns and table["draw"].nunique() > 1:
        # draws are the sample: each paired draw's adjusted importance
        t = table[table["gene"].isin(genes)].dropna(subset=[_QUANTITY])
        _violins(adata, t, x="gene", y=_QUANTITY, palette=palette, ax=ax, ylabel=_YLABEL,
                 rotation=90, order=genes)
    else:
        _points(adata, d, x="gene", y=_QUANTITY, palette=palette, ax=ax, ylabel=_YLABEL,
                rotation=90, order=genes)
    _zero_rule(ax)
    ax.set_xlabel("gene")
    return _finish(fig, ax, save=save, show=show, legend=legend)


def _heatmap(adata, result, *, genes, ax, figsize, save, show):
    from .._state import keys as K

    fig, ax = _axes(ax, figsize)
    if result is None or not len(result):
        return _finish(fig, _empty(ax, "no shift for gene_importance", ""),
                       save=save, show=show)
    phenotypes = list(adata.uns.get(K.PHENOTYPE_CATEGORIES, result["phenotype"].unique()))
    # mean over groups (and the covariate label, when present): one cell per gene x phenotype
    mat = (result[result["gene"].isin(genes)]
           .groupby(["gene", "phenotype"], observed=True)[_QUANTITY].mean()
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
    cbar.set_label("adjusted shift (baseline − silenced)")
    return _finish(fig, ax, save=save, show=show)


def gene_importance(adata, *, view="replicate", n_top=25, key=None, order=None,
                    hue_order=None, palette=None, legend=True, ax=None, figsize=(8, 4),
                    save=None, show=None, return_df=False):
    """The cached ``perturb.gene_importance``: a ranking of the ``n_top`` genes with the largest
    adjusted importance (``view="replicate"``) or their gene × phenotype adjusted shift
    (``view="item"``).

    ``order`` restricts and orders the genes shown; ``hue_order`` orders the split levels;
    ``legend=False`` draws no legend; ``return_df`` hands back the summary the view draws, the
    replicate summary or the item summary, instead of drawing.

    The result stores only the adjusted value, so both views draw it, against zero, and both
    show the same genes, ranked by the adjusted importance: the gene's importance less the same
    gene's importance under the permutation null. The adjusted shift is the shift minus the
    null's, per phenotype; its absolute values summed over phenotypes are not the adjusted
    importance, so the heatmap is not a decomposition of the ranking.
    """
    from .. import get as _get

    if view not in ("replicate", "item"):
        raise ValueError(f"view must be 'replicate' or 'item', got {view!r}")
    payload = _get.result(adata, "gene_importance", key=key)
    params = _get.params(adata, "gene_importance", key=key)
    ranked = payload["replicate"]["summary"]
    drawn = ranked if view == "replicate" else payload["item"]["summary"]
    if return_df:
        return drawn
    genes = _top_genes(ranked, n_top, order) if (
        ranked is not None and len(ranked) and _QUANTITY in ranked.columns) else []
    if view == "replicate":
        return _rank(adata, payload, params, genes=genes, hue_order=hue_order,
                     palette=palette, legend=legend, ax=ax, figsize=figsize, save=save,
                     show=show)
    return _heatmap(adata, drawn, genes=genes, ax=ax, figsize=figsize, save=save, show=show)
