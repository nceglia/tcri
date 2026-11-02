"""``pl.joint_distribution`` — the twin of ``tl.joint_distribution``.

``view="sankey"`` draws the phenotype-flow Sankey. At each covariate value there is a bar of the
phenotypes the clones' cells carry. Between adjacent values, bands carry each clone's phenotype
mass from one value to the next.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch

from .._compute._repertoire import _size_counts
from .._state import keys as K
from ._base import _axes, _finish
from ._colors import resolve_colors
from ._sankey import _render_sankey, _sankey_layout

__all__ = ["joint_distribution"]

_VIEWS = ("sankey",)

#: The text size of the axis labels and the legend.
_FONTSIZE = 12


def joint_distribution(adata, *, view="sankey", normalize=True, key=None, order=None,
                       palette=None, legend=True, ax=None, figsize=(8, 4), save=None, show=None,
                       return_df=False):
    """The phenotype-flow Sankey across covariate values, from the stored
    ``tl.joint_distribution``.

    Every cell carries its clone's phenotype mix at its covariate value. A clone's mass at a value
    is its row of the stored summary, scaled to sum to 1, times its cells at that value in
    ``adata.obs``. At each value, the bar stacks the phenotype totals over clones in category
    order. Between adjacent values, a band runs from each origin phenotype to each destination
    phenotype, carrying the mass of the clones present at both values. Its two ends are each
    clone's own mass at its value, spread over the other value's mix. The band's color runs from
    the origin phenotype's to the destination phenotype's.

    The clones drawn are the rows of the stored joint. A joint stored with ``clones=`` draws that
    subset, and with ``normalize=True`` its columns fill to 1, so it shows the mix of the subset's
    cells rather than their share of all cells. The default joint leaves out the pools of a pooled
    clonotype column, and their cells are in no column; a joint stored with
    ``exclude_pools=False`` draws each pool as one clone.

    Parameters
    ----------
    adata
        The object ``tl.joint_distribution`` stored its result in.
    view
        ``"sankey"``, the phenotype-flow Sankey.
    normalize
        ``True`` draws each covariate value's bar as fractions of its cells, summing to 1;
        ``False`` draws cell counts.
    key
        The ``uns`` key of a joint stored with ``key_added``.
    order
        The covariate values to draw, left to right. By default, every value in the stored
        joint, in category order.
    palette
        Colors for the phenotypes: a dict, a list, or a colormap name. By default, the colors
        stored for the phenotype column, or the package palette.
    legend
        ``False`` draws no phenotype legend.
    ax
        The axes to draw on; ``None`` makes a new figure.
    figsize
        The size of a new figure.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the stored summary instead of drawing.

    Returns
    -------
    matplotlib.axes.Axes or pandas.DataFrame
        The axes drawn on, or the stored summary with ``return_df=True``.

    Raises
    ------
    ValueError
        When ``view`` is not ``"sankey"``, when ``order`` names a covariate value the stored
        joint does not have, or when the joint was computed on a permutation-null fit, whose
        cells are reassigned by permutation and so are not the cells counted in ``adata.obs``.
    KeyError
        When no joint is stored under the key; the message names the call to run.
    """
    from .. import get as _get

    if view not in _VIEWS:
        raise ValueError(f"view must be 'sankey', got {view!r}")
    summary = _get.result(adata, "joint_distribution", key=key)["summary"]
    if return_df:
        return summary
    params = _get.params(adata, "joint_distribution", key=key)
    _refuse_null_fit(adata, params)

    values = _values(adata, summary, params, order)
    phenotypes = [str(p) for p in summary.columns]
    mass = _mass(adata, summary, values, normalize)
    layout = _sankey_layout(mass, len(phenotypes))

    meta = adata.uns[K.METADATA]
    colors = resolve_colors(adata, meta[K.PHENOTYPE_COL], phenotypes, palette=palette)
    n = len(values)
    created = ax is None
    fig, ax = _axes(ax, figsize)
    _render_sankey(ax, layout, phenotypes=phenotypes, colors=colors,
                   dx=(n - 1) / 500 if n > 1 else 0.2)

    if n > 1:
        ax.set_xlim(0, n - 1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(values)
    top = max(float((nodes[:, 0] + nodes[:, 1]).max()) for nodes in layout["nodes"])
    if top > 0:
        ax.set_ylim(0, min(top * 1.02, 1) if normalize else top * 1.02)
    ax.set_xlabel(meta[K.COVARIATE_COL], fontsize=_FONTSIZE)
    ax.set_ylabel("Fraction" if normalize else "Cell Counts", fontsize=_FONTSIZE)
    if legend:
        ax.legend(handles=[Patch(facecolor=colors[p], label=p) for p in phenotypes],
                  frameon=True, fontsize=_FONTSIZE, bbox_to_anchor=(1.05, 1), loc="upper left")
    if created:
        fig.tight_layout()
    return _finish(fig, ax, save=save, show=show, legend=legend)


def _refuse_null_fit(adata, params):
    """Raise when the stored joint was computed on a permutation-null fit.

    The Sankey weights each clone by its cells in ``adata.obs``. A null fit reassigns cells to
    clones or covariate values by a stored permutation, so those counts are not its clones'.
    """
    fit = params.get("fit")
    if fit is None:
        return
    fit = K.resolve_fit(adata, fit)
    if K.fit_key(K.PERMUTATION, fit) in adata.uns:
        raise ValueError(
            f"the stored joint was computed on the permutation-null fit {fit!r}. The Sankey "
            f"weights each clone by its cells in adata.obs, and a null reassigns cells by "
            f"permutation, so those counts are not its clones'. Draw a joint of the main fit, "
            f"stored by tcri.tl.joint_distribution(adata)."
        )


def _values(adata, summary, params, order):
    """The covariate values to draw, as strings, from ``order`` or from the stored joint."""
    if "covariate" in summary.index.names:
        stored = [str(v) for v in dict.fromkeys(summary.index.get_level_values("covariate"))]
    else:
        stored = [str(params["covariate"])]
    if order is None:
        categories = [str(c) for c in adata.uns[K.COVARIATE_CATEGORIES]]
        return ([v for v in categories if v in stored]
                + [v for v in stored if v not in categories])
    values = [str(v) for v in order]
    missing = [v for v in values if v not in stored]
    if missing:
        raise ValueError(f"order names covariate values the stored joint does not have: "
                         f"{missing}; it has {stored}")
    return values


def _mass(adata, summary, values, normalize):
    """``{clone: phenotype mass}`` at each value: the clone's mix times its cells there.

    With ``normalize``, each value's masses are divided by their total, so its bar sums to 1.
    """
    meta = adata.uns[K.METADATA]
    units = _size_counts(adata.obs, meta[K.CLONE_COL], meta[K.COVARIATE_COL])
    cells = dict(zip(zip(units["covariate"].astype(str), units["clonotype"].astype(str)),
                     units["n_cells"]))
    has_level = "covariate" in summary.index.names
    level = summary.index.get_level_values("covariate").astype(str) if has_level else None

    mass = []
    for value in values:
        rows = summary[level == value].droplevel("covariate") if has_level else summary
        rows = rows.groupby(level=0, sort=False).sum()
        mix = rows.to_numpy(dtype=float)
        sums = mix.sum(axis=1, keepdims=True)
        mix = np.divide(mix, sums, out=np.zeros_like(mix), where=sums > 0)
        clones = [str(c) for c in rows.index]
        weights = np.array([cells.get((value, c), 0) for c in clones], dtype=float)
        vectors = mix * weights[:, None]
        if normalize and vectors.sum() > 0:
            vectors = vectors / vectors.sum()
        mass.append({c: v for c, v in zip(clones, vectors) if v.sum() > 0})
    return mass
