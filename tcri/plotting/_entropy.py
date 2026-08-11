"""``pl.clonotypic_entropy`` / ``pl.phenotypic_entropy`` (§8.1) — cache renderers that call
the ``tl`` twin and box/strip the tidy result. No metric math here."""
from __future__ import annotations

from ._base import _barplot, _finish, _metric_boxplot

__all__ = ["clonotypic_entropy", "phenotypic_entropy"]


def clonotypic_entropy(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, weighted=False, normalized=True,
                       order=None, hue_order=None, palette=None, ax=None, figsize=(8, 4),
                       save=None, show=None, return_df=False):
    """Per-phenotype clonotypic entropy (bits), boxed over ``groupby`` units and split by
    ``splitby`` (x = phenotype)."""
    from .. import tools as tl

    # inplace=False: this still recomputes (pass 2 rewires `pl` to read `tcri.get`), but with
    # store-once a recomputing plot would OVERWRITE the user's cached result with the plot's own
    # arguments -- a manufactured groupby, a different n_samples -- which is worse than the
    # duplicate work it does today.
    res = tl.clonotypic_entropy(
        adata, covariate=covariate, groupby=groupby, splitby=splitby, n_samples=n_samples,
        temperature=temperature, clones=clones, weighted=weighted, normalized=normalized,
        inplace=False,
    )["result"]
    if return_df:
        return res
    if groupby is None:
        fig, ax = _barplot(res.set_index("phenotype")["value"],
                           ylabel="clonotypic entropy (bits)", palette=palette, ax=ax,
                           figsize=figsize)
    else:
        fig, ax = _metric_boxplot(res, x="phenotype", y="value", hue=splitby,
                                  order=order, hue_order=hue_order, palette=palette, ax=ax,
                                  figsize=figsize, ylabel="clonotypic entropy (bits)")
    return _finish(fig, ax, save=save, show=show)


def phenotypic_entropy(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                       temperature=1.0, clones=None, weighted=False, normalized=True,
                       order=None, hue_order=None, palette=None, ax=None, figsize=(8, 4),
                       save=None, show=None, return_df=False):
    """Distribution of per-clone phenotypic entropy (plasticity), boxed by ``splitby`` (or
    ``groupby``) with each clone a dot."""
    from .. import tools as tl

    # inplace=False: this still recomputes (pass 2 rewires `pl` to read `tcri.get`), but with
    # store-once a recomputing plot would OVERWRITE the user's cached result with the plot's own
    # arguments -- a manufactured groupby, a different n_samples -- which is worse than the
    # duplicate work it does today.
    res = tl.phenotypic_entropy(
        adata, covariate=covariate, groupby=groupby, splitby=splitby, n_samples=n_samples,
        temperature=temperature, clones=clones, weighted=weighted, normalized=normalized,
        inplace=False,
    )["result"]
    if return_df:
        return res
    if groupby is None:
        fig, ax = _barplot(res.set_index("clonotype")["value"],
                           ylabel="phenotypic entropy (bits)", palette=palette, ax=ax,
                           figsize=figsize)
    else:
        x = splitby if (splitby is not None and splitby in res.columns) else groupby
        fig, ax = _metric_boxplot(res, x=x, y="value", hue=None, order=order,
                                  palette=palette, ax=ax, figsize=figsize,
                                  ylabel="phenotypic entropy (bits)")
    return _finish(fig, ax, save=save, show=show)
