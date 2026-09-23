"""``pl.phenotypic_flux`` — cache renderer for per-clone phenotype-distribution flux.

The tidy flux values are available via ``return_df=True``.
"""
from __future__ import annotations

from ._base import render_metric

__all__ = ["phenotypic_flux"]


def phenotypic_flux(adata, *, quantity="value", key=None, order=None, hue_order=None, palette=None, ax=None,
                    figsize=(8, 4), save=None, show=None, return_df=False):
    """Per-clone flux from ``cov_from`` to ``cov_to``, from the cached ``tl.phenotypic_flux``.

    The endpoints and the distance metric come from the cached ``params``, so the axis label
    and the numbers beneath it always describe the same quantity. The distance is chosen in
    one place, ``tl.phenotypic_flux``.
    """
    from .. import get as _get

    metric = _get.params(adata, "phenotypic_flux", key=key).get("distance_metric", "kl")
    return render_metric(adata, "phenotypic_flux", ylabel=f"phenotypic flux ({metric})",
                         item_col="clonotype", quantity=quantity, key=key, order=order, hue_order=hue_order,
                         palette=palette, ax=ax, figsize=figsize, save=save, show=show,
                         return_df=return_df)
