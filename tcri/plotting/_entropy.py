"""``pl.clonotypic_entropy`` / ``pl.phenotypic_entropy`` — cache renderers.

The two differ in one thing, and it is a property of the metric rather than a style choice:
H(c|phi) has one value per PHENOTYPE, a handful of named categories that belong on the x
axis; H(phi|c) has one value per CLONE, which is the whole repertoire and belongs in the
distribution.
"""
from __future__ import annotations

from ._base import render_metric

__all__ = ["clonotypic_entropy", "phenotypic_entropy"]


def clonotypic_entropy(adata, *, quantity="auto", key=None, order=None, hue_order=None,
                       palette=None, legend=True, ax=None, figsize=(8, 4), save=None,
                       show=None, return_df=False):
    """Per-phenotype clonotypic entropy, from the cached ``tl.clonotypic_entropy``.

    Phenotypes are on x, with the split as hue when the result has one.

    Parameters
    ----------
    adata
        The object ``tl.clonotypic_entropy`` stored its result in.
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
    return render_metric(adata, "clonotypic_entropy", ylabel="clonotypic entropy (bits)",
                         item_col="phenotype", item_as_x=True, quantity=quantity, key=key,
                         order=order, hue_order=hue_order, palette=palette, legend=legend, ax=ax,
                         figsize=figsize, save=save, show=show, return_df=return_df)


def phenotypic_entropy(adata, *, quantity="auto", key=None, order=None, hue_order=None,
                       palette=None, legend=True, ax=None, figsize=(8, 4), save=None,
                       show=None, return_df=False):
    """Per-clone phenotypic entropy (plasticity), from the cached ``tl.phenotypic_entropy``.

    With replicates, one dot per replicate: its clones are averaged first.

    Parameters
    ----------
    adata
        The object ``tl.phenotypic_entropy`` stored its result in.
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
    return render_metric(adata, "phenotypic_entropy", ylabel="phenotypic entropy (bits)",
                         item_col="clonotype", quantity=quantity, key=key, order=order,
                         hue_order=hue_order, palette=palette, legend=legend, ax=ax,
                         figsize=figsize, save=save, show=show, return_df=return_df)
