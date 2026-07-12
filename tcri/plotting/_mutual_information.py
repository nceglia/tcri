"""``pl.mutual_information`` (§8.2) — cache renderer: per-unit MI boxed by ``splitby``."""
from __future__ import annotations

from ._base import _finish, _metric_boxplot

__all__ = ["mutual_information"]


def mutual_information(adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
                      temperature=1.0, clones=None, weighted=False, normalized=True,
                      normalize_mode="min", order=None, hue_order=None, palette=None, ax=None,
                      figsize=(8, 4), save=None, show=None, return_df=False):
    """Clone↔phenotype MI (bits, normalized). With ``groupby`` (units, e.g. patient) and
    ``splitby`` (cohort, e.g. response): one MI per unit, boxed by cohort."""
    from .. import tools as tl
    from .. import _keys as K

    # a boxplot needs per-unit MI; default the aggregation unit to the batch (patient) column
    gb = groupby if groupby is not None else adata.uns[K.METADATA]["batch_col"]

    res = tl.mutual_information(
        adata, covariate=covariate, groupby=gb, splitby=splitby, n_samples=n_samples,
        temperature=temperature, clones=clones, weighted=weighted, normalized=normalized,
        normalize_mode=normalize_mode,
    )
    if return_df:
        return res
    x = splitby if (splitby is not None and splitby in res.columns) else gb
    fig, ax = _metric_boxplot(res, x=x, y="MI", hue=None, order=order, palette=palette,
                              ax=ax, figsize=figsize, ylabel="mutual information (bits)")
    return _finish(fig, ax, save=save, show=show)
