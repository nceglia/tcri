"""``pl.clonotypic_entropy`` / ``pl.phenotypic_entropy`` — cache renderers.

Each draws its metric's replicate values by default, one value per replicate, and its items with
``view="item"``. The two item views differ in one thing, and it is a property of the metric
rather than a style choice: H(c|phi) has one value per PHENOTYPE, a handful of named categories
that belong on the x axis; H(phi|c) has one value per CLONE, which is the whole repertoire and is
drawn within each replicate.
"""
from __future__ import annotations

from ._base import render_metric

__all__ = ["clonotypic_entropy", "phenotypic_entropy"]


def clonotypic_entropy(adata, *, view="replicate", quantity="auto", key=None, order=None,
                       hue_order=None, palette=None, legend=True, ax=None, figsize=(8, 4),
                       save=None, show=None, return_df=False):
    """Clonotypic entropy, from the cached ``tl.clonotypic_entropy``.

    The replicate view draws each replicate's H(C|P), by split when the result has one, with the
    split test starred. The item view puts phenotypes on x, with the split as hue. A change
    result (``cov_from``, ``cov_to``) draws each replicate's change, or each phenotype's with
    ``view="item"``, against a zero rule.

    Parameters
    ----------
    adata
        The object ``tl.clonotypic_entropy`` stored its result in.
    view
        ``"replicate"`` (default) draws one value per replicate; ``"item"`` draws one value per
        replicate and phenotype.
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
        Return the summary the view draws instead of drawing.

    Returns
    -------
    matplotlib.axes.Axes or pandas.DataFrame
        The axes drawn on, or the summary with ``return_df=True``.
    """
    return render_metric(adata, "clonotypic_entropy", ylabel="clonotypic entropy (bits)",
                         item_col="phenotype", item_as_x=True, view=view, quantity=quantity,
                         key=key, order=order, hue_order=hue_order, palette=palette,
                         legend=legend, ax=ax, figsize=figsize, save=save, show=show,
                         return_df=return_df)


def phenotypic_entropy(adata, *, view="replicate", quantity="auto", key=None, order=None,
                       hue_order=None, palette=None, legend=True, ax=None, figsize=(8, 4),
                       save=None, show=None, return_df=False):
    """Phenotypic entropy (plasticity), from the cached ``tl.phenotypic_entropy``.

    The replicate view draws each replicate's H(P|C), by split when the result has one, with the
    split test starred. The item view draws one point per clone within each replicate. A change
    result (``cov_from``, ``cov_to``) draws each replicate's change, or each persistent clone's
    with ``view="item"``, against a zero rule.

    Parameters
    ----------
    adata
        The object ``tl.phenotypic_entropy`` stored its result in.
    view
        ``"replicate"`` (default) draws one value per replicate; ``"item"`` draws one value per
        replicate and clone.
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
        Return the summary the view draws instead of drawing.

    Returns
    -------
    matplotlib.axes.Axes or pandas.DataFrame
        The axes drawn on, or the summary with ``return_df=True``.
    """
    return render_metric(adata, "phenotypic_entropy", ylabel="phenotypic entropy (bits)",
                         item_col="clonotype", view=view, quantity=quantity, key=key,
                         order=order, hue_order=hue_order, palette=palette, legend=legend,
                         ax=ax, figsize=figsize, save=save, show=show, return_df=return_df)
