"""``pl.phenotypic_flux`` (§8.3) — cache renderer for per-clone phenotype-distribution flux.

(A phenotype-flow Sankey over a covariate ``order`` is a deferred enhancement, tracked as its
own issue; the tidy flux values are available via ``return_df=True``.)
"""
from __future__ import annotations

from ._base import render_metric

__all__ = ["phenotypic_flux"]


def phenotypic_flux(adata, *, key=None, order=None, hue_order=None, palette=None, ax=None,
                    figsize=(8, 4), save=None, show=None, return_df=False):
    """Per-clone flux from ``cov_from`` to ``cov_to``, from the cached ``tl.phenotypic_flux``.

    The endpoints and the distance metric come from the cached ``params``. ``pl`` used to
    take its own ``distance_metric``, defaulting to ``"l1"`` while ``tl`` defaulted to
    ``"kl"`` -- so the axis label and the numbers beneath it could describe different
    quantities. There is now one place the distance is chosen.
    """
    from .. import get as _get

    metric = _get.params(adata, "phenotypic_flux", key=key).get("distance_metric", "kl")
    return render_metric(adata, "phenotypic_flux", ylabel=f"phenotypic flux ({metric})",
                         item_col="clonotype", key=key, order=order, hue_order=hue_order,
                         palette=palette, ax=ax, figsize=figsize, save=save, show=show,
                         return_df=return_df)
