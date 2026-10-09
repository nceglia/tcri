"""``pl.delta_*`` — cache renderers for the paired entropies.

Two views of one cached result, selected by ``kind``:

``"delta"`` (default)
    The change itself, with a zero rule. Mass either side of zero is the direction.

``"endpoints"``
    ``cov_from`` and ``cov_to`` side by side, connected per replicate. This view exists
    *here* rather than on ``pl.phenotypic_entropy`` for a reason worth stating: the endpoints
    live in the delta's payload, where they were computed over the intersected clone set. The
    same figure drawn from two separate single-covariate results would use a DIFFERENT clone
    set on each side, and the values differ substantially. Rendering it only from the delta
    result makes the unmatched version unreachable rather than merely discouraged.
"""
from __future__ import annotations

from ._base import render_delta

__all__ = ["delta_clonotypic_entropy", "delta_phenotypic_entropy"]


def delta_clonotypic_entropy(adata, *, kind="delta", quantity="auto", key=None, order=None,
                             hue_order=None, palette=None, legend=True, ax=None,
                             figsize=(8, 4), save=None, show=None, return_df=False):
    """Per-phenotype change in clonotypic entropy, from the cached
    ``tl.delta_clonotypic_entropy``.

    No connecting lines in the ``"endpoints"`` view, and no matched-count sizing: the item is a
    phenotype, a category measured twice rather than an entity that persisted. A line would
    assert persistence, and the matched CLONE count is not in this result at all — those clones
    were summed over inside H(c|phi), so sizing by item rows would size by phenotype.

    Parameters
    ----------
    adata
        The object ``tl.delta_clonotypic_entropy`` stored its result in.
    kind
        ``"delta"`` draws the change against a zero rule; ``"endpoints"`` draws ``cov_from``
        and ``cov_to`` side by side, one pair per replicate.
    quantity
        ``"auto"`` draws the adjusted value when the result carries a reference and the value
        otherwise; ``"adjusted"`` or ``"value"`` asks for one by name. ``"endpoints"`` draws
        the value only.
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
    return render_delta(adata, "delta_clonotypic_entropy",
                        ylabel="Δ clonotypic entropy (bits)", item_col="phenotype",
                        item_as_x=True, entity_matched=False, kind=kind, quantity=quantity,
                        key=key, order=order, hue_order=hue_order, palette=palette,
                        legend=legend, ax=ax, figsize=figsize, save=save, show=show,
                        return_df=return_df)


def delta_phenotypic_entropy(adata, *, kind="delta", quantity="auto", key=None, order=None,
                             hue_order=None, palette=None, legend=True, ax=None,
                             figsize=(8, 4), save=None, show=None, return_df=False):
    """Per-clone change in phenotypic entropy, from the cached
    ``tl.delta_phenotypic_entropy``.

    The ``"endpoints"`` view connects each replicate across the two levels — the same clone
    set on both sides, so the line is a matched-identity claim the data supports. Dot area is
    the number of clones matched for that replicate, which varies per replicate and is the n
    the value rests on.

    Parameters
    ----------
    adata
        The object ``tl.delta_phenotypic_entropy`` stored its result in.
    kind
        ``"delta"`` draws the change against a zero rule; ``"endpoints"`` draws ``cov_from``
        and ``cov_to`` side by side, one connected pair per replicate.
    quantity
        ``"auto"`` draws the adjusted value when the result carries a reference and the value
        otherwise; ``"adjusted"`` or ``"value"`` asks for one by name. ``"endpoints"`` draws
        the value only.
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
    return render_delta(adata, "delta_phenotypic_entropy",
                        ylabel="Δ phenotypic entropy (bits)", item_col="clonotype",
                        entity_matched=True, kind=kind, quantity=quantity, key=key,
                        order=order, hue_order=hue_order, palette=palette, legend=legend,
                        ax=ax, figsize=figsize, save=save, show=show, return_df=return_df)
