"""``pl.mutual_information`` — cache renderer for I(c;phi)."""
from __future__ import annotations

from ._base import render_metric

__all__ = ["mutual_information"]


def mutual_information(adata, *, quantity="auto", key=None, order=None, hue_order=None,
                       palette=None, legend=True, ax=None, figsize=(8, 4), save=None,
                       show=None, return_df=False):
    """Clone-phenotype mutual information (bits, normalized), from the cached
    ``tl.mutual_information``.

    The axes are whatever that call used: with ``groupby``, one value per group; with
    ``splitby``, the groups boxed by split with the contrast from ``stats`` bracketed above.

    Parameters
    ----------
    adata
        The object ``tl.mutual_information`` stored its result in.
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
    return render_metric(adata, "mutual_information", ylabel="mutual information (bits)",
                         quantity=quantity, key=key, order=order, hue_order=hue_order,
                         palette=palette, legend=legend, ax=ax, figsize=figsize, save=save,
                         show=show, return_df=return_df)
