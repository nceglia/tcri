"""``pl.phenotypic_flux`` — cache renderer for per-clone phenotype-distribution flux.

The tidy flux values are available via ``return_df=True``.
"""
from __future__ import annotations

from ._base import render_metric

__all__ = ["phenotypic_flux"]


def phenotypic_flux(adata, *, quantity="auto", key=None, order=None, hue_order=None,
                    palette=None, legend=True, ax=None, figsize=(8, 4), save=None, show=None,
                    return_df=False):
    """Per-clone flux from ``cov_from`` to ``cov_to``, from the cached ``tl.phenotypic_flux``.

    The endpoints and the distance come from the cached ``params``, so the axis label and the
    numbers beneath it describe the same quantity; the distance is chosen in one place,
    ``tl.phenotypic_flux``.

    Parameters
    ----------
    adata
        The object ``tl.phenotypic_flux`` stored its result in.
    quantity
        ``"auto"`` draws the adjusted value when the result carries a reference and the value
        otherwise; ``"adjusted"`` or ``"value"`` asks for one by name.
    key
        The ``uns`` key of a result stored with ``key_added``.
    order
        The x order; by default the category order of the axis.
    hue_order
        The order of the split levels.
    palette
        Colors for the levels: a dict, a list, or a colormap name.
    legend
        ``False`` draws no legend, for a panel in a row that carries one.
    ax
        The axes to draw on; ``None`` makes a new figure.
    figsize
        The size of a new figure.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the cached ``result`` frame instead of drawing.

    Returns
    -------
    matplotlib.axes.Axes or pandas.DataFrame
        The axes drawn on, or the cached ``result`` with ``return_df=True``.
    """
    from .. import get as _get

    metric = _get.params(adata, "phenotypic_flux", key=key).get("distance_metric", "kl")
    return render_metric(adata, "phenotypic_flux", ylabel=f"phenotypic flux ({metric})",
                         item_col="clonotype", quantity=quantity, key=key, order=order,
                         hue_order=hue_order, palette=palette, legend=legend, ax=ax,
                         figsize=figsize, save=save, show=show, return_df=return_df)
