"""``pl.phenotypic_flux`` (§8.3) — cache renderer for per-clonotype phenotype-distribution
flux across a covariate ``order``. Renders a box of the per-clone flux magnitudes split by
``splitby``. (A phenotype-flow Sankey over ``order`` is a deferred enhancement, tracked in
``REFACTOR_NOTES``; the tidy flux values are available via ``return_axes=True``.)
"""
from __future__ import annotations

from ._base import _finish, _metric_boxplot

__all__ = ["phenotypic_flux"]


def phenotypic_flux(adata, *, order, groupby=None, splitby=None, n_samples=0, temperature=1.0,
                    clones=None, weighted=False, distance_metric="l1", palette=None, ax=None,
                    figsize=(8, 4), save=None, show=None, return_axes=False):
    """Per-clone phenotype-distribution flux from ``order[0]`` to ``order[-1]``, boxed by
    cohort. ``order`` is the covariate sequence (>= 2 values)."""
    from .. import tools as tl
    from .. import _keys as K

    order = list(order)
    if len(order) < 2:
        raise ValueError("phenotypic_flux needs `order` with >= 2 covariate values.")
    cov_from, cov_to = order[0], order[-1]
    gb = groupby if groupby is not None else adata.uns[K.METADATA]["batch_col"]

    res = tl.phenotypic_flux(
        adata, cov_from=cov_from, cov_to=cov_to, groupby=gb, splitby=splitby,
        n_samples=n_samples, temperature=temperature, clones=clones, weighted=weighted,
        distance_metric=distance_metric,
    )
    if return_axes:
        return res
    x = splitby if (splitby is not None and splitby in res.columns) else gb
    fig, ax = _metric_boxplot(res, x=x, y="phenotypic_flux", hue=None, palette=palette, ax=ax,
                              figsize=figsize, ylabel=f"phenotypic flux ({distance_metric})")
    return _finish(fig, ax, save=save, show=show)
