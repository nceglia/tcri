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

import numpy as np

from ._base import _axes, _boxstrip, _colors_for, _empty, _finish, _points, _violins

__all__ = ["gene_importance"]

_YLABEL = "gene importance (L1 shift of the phenotype call)"


def _top_genes(result, n_top, order):
    """Genes ranked by their mean importance over groups, or the caller's ``order``."""
    mean = result.groupby("gene", observed=True, sort=False)["value"].mean()
    ranked = mean.sort_values(ascending=False).index.tolist()
    if order is not None:
        known = set(ranked)
        ranked = [g for g in order if g in known]
    return ranked[: int(n_top)] if n_top is not None else ranked


def _star_labels(ax, stats, genes, *, groupby):
    """That gene's own contrast (over ``groupby`` units), starred above its x position."""
    if stats is None or not len(stats) or "gene" not in stats.columns:
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


def _rank(adata, payload, params, *, n_top, order, hue_order, palette, ax, figsize, save,
          show):
    result, table, stats = payload["result"], payload.get("table"), payload.get("stats")
    groupby, splitby = params.get("groupby"), params.get("splitby")
    fig, ax = _axes(ax, figsize)
    if result is None or not len(result) or "value" not in result.columns:
        return _finish(fig, _empty(ax, "no data for gene_importance", _YLABEL),
                       save=save, show=show)
    genes = _top_genes(result, n_top, order)
    d = result[result["gene"].isin(genes)].dropna(subset=["value"])

    has_groups = groupby is not None and groupby in d.columns and d[groupby].nunique() > 1
    if has_groups:
        hue = splitby if (splitby and splitby in d.columns) else None
        _boxstrip(adata, d, x="gene", y="value", hue=hue, order=genes, hue_order=hue_order,
                  palette=palette, ax=ax, ylabel=_YLABEL, rotation=90)
        if hue is not None:
            _star_labels(ax, stats, genes, groupby=groupby)
    elif table is not None and "draw" in table.columns and table["draw"].nunique() > 1:
        t = table[table["gene"].isin(genes)].dropna(subset=["value"])
        _violins(adata, t, x="gene", y="value", palette=palette, ax=ax, ylabel=_YLABEL,
                 rotation=90, order=genes)
    else:
        _points(adata, d, x="gene", y="value", palette=palette, ax=ax, ylabel=_YLABEL,
                rotation=90)
    ax.set_xlabel("gene")
    return _finish(fig, ax, save=save, show=show)


def _shift(adata, payload, params, *, n_top, order, ax, figsize, save, show):
    import matplotlib.pyplot as plt

    from .._state import keys as K

    result, shift = payload["result"], payload.get("shift")
    fig, ax = _axes(ax, figsize)
    if shift is None or not len(shift):
        return _finish(fig, _empty(ax, "no shift table for gene_importance", ""),
                       save=save, show=show)
    genes = _top_genes(result, n_top, order)
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


def gene_importance(adata, *, kind="rank", n_top=25, key=None, order=None, hue_order=None,
                    palette=None, ax=None, figsize=(8, 4), save=None, show=None,
                    return_df=False):
    """The cached ``perturb.gene_importance``: a ranking (``kind="rank"``) or the signed
    gene × phenotype shift (``kind="shift"``) of the ``n_top`` most important genes.

    ``order`` restricts and orders the genes shown; ``hue_order`` orders the split levels;
    ``return_df`` hands back the cached ``result`` frame instead of drawing.
    """
    from .. import get as _get

    if kind not in ("rank", "shift"):
        raise ValueError(f"kind must be 'rank' or 'shift', got {kind!r}")
    payload = _get.result(adata, "gene_importance", key=key)
    params = _get.params(adata, "gene_importance", key=key)
    if return_df:
        return payload["result"]
    if kind == "rank":
        return _rank(adata, payload, params, n_top=n_top, order=order, hue_order=hue_order,
                     palette=palette, ax=ax, figsize=figsize, save=save, show=show)
    return _shift(adata, payload, params, n_top=n_top, order=order, ax=ax, figsize=figsize,
                  save=save, show=show)
