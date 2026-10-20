"""``pl`` twins of the repertoire summaries in ``tcri.pp``.

Each draws the result its ``pp`` function stored, read through :mod:`tcri.get`, and computes no
number of its own. Groups are the levels of the stored call's ``groupby``; with ``groupby=None``
the whole dataset is one group, labeled ``"all"``. With ``splitby``, each group's label takes the
color of its split level from ``adata.uns[f"{splitby}_colors"]``.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ._colors import resolve_colors, tcri_colors

__all__ = ["repertoire_summary", "clonotype_sharing", "clone_persistence"]

#: Where two groups, or two levels, share nothing: apart from a small share, which is pale.
NONE_COLOR = "#e8726e"

#: The label of the one group ``groupby=None`` makes.
_ALL = "all"

#: The panels of :func:`repertoire_summary`: a column of the table and its title.
_SUMMARY_PANELS = [("n_cells", "Cells"), ("n_clones", "Clones"),
                   ("singleton_fraction", "Singleton fraction"), ("clonality", "Clonality")]

#: Groups up to which a sharing heatmap prints its values in the cells.
_ANNOTATE_UP_TO = 15


def _stored(adata, name, key, per_flag=None):
    """The stored result of ``tcri.pp.<name>`` and its arguments.

    ``per_flag`` names the argument that stores a frame per clonotype or per clone instead of
    the tables per group; a result stored that way raises, since the plot draws the tables.
    """
    from .. import get as _get

    payload = _get.result(adata, name, key=key)
    params = _get.params(adata, name, key=key)
    if per_flag is not None and not isinstance(payload, dict):
        raise ValueError(
            f"the stored {name} result was computed with {per_flag}=True, one row per "
            f"{per_flag.removeprefix('per_')}; tcri.pl.{name} draws the tables per group. Run "
            f"tcri.pp.{name}(adata, ..., {per_flag}=False), or pass key= for a result stored "
            f"with key_added."
        )
    return payload, params


def _groups(table, groupby, order):
    """The groups of a stored table, as strings, in its row order, or in ``order``."""
    present = [_ALL] if groupby is None else list(table[groupby].astype(str))
    if order is None:
        return present
    order = [str(group) for group in order]
    missing = [group for group in order if group not in present]
    if missing:
        raise ValueError(f"order names groups the stored result does not have: {missing[:5]}")
    return order


def _by_label(adata, column, labels, palette=None):
    """``{label: color}`` for string ``labels`` of an ``obs`` column, through
    :func:`~tcri.pl.resolve_colors`, so a level keeps its color across figures."""
    if column in adata.obs:
        colors = resolve_colors(adata, column, palette=palette)
        colors = {str(level): color for level, color in colors.items()}
        missing = [label for label in labels if label not in colors]
        if missing:
            colors.update(resolve_colors(adata, column, missing, palette=palette, persist=False))
        return colors
    return resolve_colors(adata, column, labels, palette=palette, persist=False)


def _split_colors(adata, table, groupby, splitby, groups):
    """``({group: color}, {split level: color})`` for a table with a split column, or two
    empty dicts without one."""
    if splitby is None or groupby is None:
        return {}, {}
    split_of = dict(zip(table[groupby].astype(str), table[splitby].astype(str)))
    levels = list(dict.fromkeys(split_of[group] for group in groups if split_of.get(group)))
    colors = _by_label(adata, splitby, levels)
    by_group = {group: colors[split_of[group]] for group in groups if split_of.get(group)}
    return by_group, {level: colors[level] for level in levels}


def _color_labels(labels, groups, group_colors):
    for label, group in zip(labels, groups):
        if group in group_colors:
            label.set_color(group_colors[group])


def _legend(fig, sections, **where):
    """One figure legend in titled sections, ``[(title, [(label, color or patch)])]``: two
    sections can hold the same color, the levels of the covariate and of the split, so each
    says which column it names. A section without entries is left out."""
    handles = []
    for title, entries in sections:
        if not entries:
            continue
        if title:
            handles.append(Patch(visible=False, label=title))
        handles += [entry if isinstance(entry, Patch) else Patch(color=entry, label=label)
                    for label, entry in entries]
    if handles:
        fig.legend(handles=handles, frameon=False, fontsize=8, **where)


def _finish(fig, *, save=None, show=None):
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def _heatmap(ax, values, none, *, vmax, annotate, digits):
    """A square heatmap of ``values`` with the diagonal blank and ``none`` drawn red."""
    shown = values.astype(float).copy()
    np.fill_diagonal(shown, np.nan)
    image = ax.imshow(shown, cmap="Purples", vmin=0, vmax=vmax)
    red = np.where(none, 1.0, np.nan)
    np.fill_diagonal(red, np.nan)
    ax.imshow(red, cmap=ListedColormap([NONE_COLOR]), vmin=0, vmax=1)
    if annotate:
        for i in range(len(shown)):
            for j in range(len(shown)):
                if i != j and np.isfinite(shown[i, j]):
                    dark = vmax > 0 and shown[i, j] > 0.6 * vmax
                    ax.text(j, i, f"{shown[i, j]:.{digits}f}", ha="center", va="center",
                            fontsize=8, color="white" if dark else "black")
    return image


def repertoire_summary(adata, *, key=None, order=None, palette=None, legend=True, figsize=None,
                       save=None, show=None, return_df=False):
    """Cells, clones, singleton fraction and clonality per group, from the stored
    ``tcri.pp.repertoire_summary``.

    One panel per quantity, the groups on x. With ``covariate_key``, the levels of each group
    stand side by side, in the level colors of ``adata.uns[f"{covariate_key}_colors"]``. On a
    pooled column the cells in pools are shaded inside each cell bar.

    Parameters
    ----------
    adata
        The object ``tcri.pp.repertoire_summary`` stored its table in.
    key
        The ``uns`` key of a table stored with ``key_added``.
    order
        The groups to draw, in this order; by default every group, in the table's order.
    palette
        Colors for the covariate levels: a dict, a list, or a colormap name.
    legend
        ``False`` draws no legend.
    figsize
        The size of the figure; by default it grows with the number of groups.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the stored table instead of drawing.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        The four axes, or the stored table with ``return_df=True``.
    """
    table, params = _stored(adata, "repertoire_summary", key)
    if return_df:
        return table
    groupby, splitby = params.get("groupby"), params.get("splitby")
    covariate = params.get("covariate_key")
    groups = _groups(table if groupby is None else table.drop_duplicates(groupby), groupby, order)
    rows = table.assign(_group=_ALL if groupby is None else table[groupby].astype(str))

    if covariate is None:
        levels = [None]
        level_colors = {None: tcri_colors[0]}
    else:
        observed = set(rows["covariate"].astype(str))
        categories = [str(c) for c in adata.obs[covariate].astype("category").cat.categories] \
            if covariate in adata.obs else sorted(observed)
        levels = [level for level in categories if level in observed]
        level_colors = _by_label(adata, covariate, levels, palette)
    group_colors, split_colors = _split_colors(adata, table, groupby, splitby, groups)

    width = 0.8 / len(levels)
    x = np.arange(len(groups))
    fig, axes = plt.subplots(
        1, len(_SUMMARY_PANELS),
        figsize=figsize or (max(12.0, 4 * (1.2 + 0.28 * len(groups))), 3.8))
    pooled = "cells_in_pools" in rows.columns
    for ax, (column, title) in zip(axes, _SUMMARY_PANELS):
        for k, level in enumerate(levels):
            sub = rows if level is None else rows[rows["covariate"].astype(str) == level]
            values = sub.set_index("_group")[column].reindex(groups).to_numpy(float)
            offset = (k - (len(levels) - 1) / 2) * width
            ax.bar(x + offset, values, width, color=level_colors[level])
            if column == "n_cells" and pooled:
                in_pools = sub.set_index("_group")["cells_in_pools"].reindex(groups)
                ax.bar(x + offset, in_pools.to_numpy(float), width, color="white", alpha=0.45,
                       hatch="///", edgecolor=level_colors[level], linewidth=0)
        ax.set_title(title)
        ax.set_xticks(x, groups, rotation=90)
        _color_labels(ax.get_xticklabels(), groups, group_colors)
        if groupby is not None:
            ax.set_xlabel(groupby)
        ax.spines[["top", "right"]].set_visible(False)

    if legend:
        pools = []
        if pooled and (rows["cells_in_pools"] > 0).any():
            pools = [(None, Patch(facecolor="white", edgecolor="gray", hatch="///",
                                  label="cells in pools"))]
        _legend(fig, [(covariate, [(level, level_colors[level]) for level in levels
                                   if level is not None]),
                      (None, pools),
                      (splitby, list(split_colors.items()))],
                loc="upper left", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    _finish(fig, save=save, show=show)
    return axes


def clonotype_sharing(adata, *, key=None, order=None, legend=True, figsize=None, save=None,
                      show=None, return_df=False):
    """Clonotypes shared between groups, from the stored ``tcri.pp.clonotype_sharing``.

    Left, a group × group heatmap: cell (a, b) is the percentage of group a's clonotypes also
    found in group b, red where the two groups share none, with the values printed when there
    are at most 15 groups. Right, each group's ``shared_fraction`` as a percentage: its
    clonotypes found in any other group. Each row label carries the group's clonotype count.

    Parameters
    ----------
    adata
        The object ``tcri.pp.clonotype_sharing`` stored its result in.
    key
        The ``uns`` key of a result stored with ``key_added``.
    order
        The groups to draw, in this order; by default every group, in the table's order.
    legend
        ``False`` draws no legend.
    figsize
        The size of the figure; by default it grows with the number of groups.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the stored ``{table, pairs}`` instead of drawing.

    Returns
    -------
    numpy.ndarray or dict
        The two axes, or the stored result with ``return_df=True``.

    Raises
    ------
    ValueError
        When the stored result was computed with ``per_clonotype=True``.
    """
    result, params = _stored(adata, "clonotype_sharing", key, per_flag="per_clonotype")
    if return_df:
        return result
    groupby, splitby = params["groupby"], params.get("splitby")
    table = result["table"][result["table"][groupby].astype(str) != "All"]
    pairs = result["pairs"]
    groups = _groups(table, groupby, order)
    first, second = f"{groupby}_a", f"{groupby}_b"
    percent = (pairs.pivot(index=first, columns=second, values="shared_fraction")
               .reindex(index=groups, columns=groups).to_numpy(float) * 100)
    none = (pairs.pivot(index=first, columns=second, values="shared_clonotypes")
            .reindex(index=groups, columns=groups).to_numpy(float) == 0)
    rows = table.set_index(table[groupby].astype(str))
    group_colors, split_colors = _split_colors(adata, table, groupby, splitby, groups)

    size = max(4.5, 0.45 * len(groups) + 2)
    fig, (left, right) = plt.subplots(1, 2, figsize=figsize or (2 * size + 2, size),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    finite = percent[np.isfinite(percent)]
    image = _heatmap(left, percent, none, vmax=finite.max() if finite.size else 1.0,
                     annotate=len(groups) <= _ANNOTATE_UP_TO, digits=2)
    left.set_xticks(range(len(groups)), groups, rotation=90)
    left.set_yticks(range(len(groups)),
                    [f"{group} ({int(rows.loc[group, 'n_clonotypes'])})" for group in groups])
    _color_labels(left.get_xticklabels(), groups, group_colors)
    _color_labels(left.get_yticklabels(), groups, group_colors)
    left.set_xlabel(groupby)
    left.set_title("Shared clonotypes")
    fig.colorbar(image, ax=left, fraction=0.046,
                 label=f"% of the row {groupby}'s clonotypes found in the column {groupby}")

    shared = rows["shared_fraction"].reindex(groups).to_numpy(float) * 100
    colors = [group_colors.get(group, tcri_colors[0]) for group in groups]
    right.bar(range(len(groups)), shared, color=colors)
    for k, value in enumerate(shared):
        if np.isfinite(value):
            right.text(k, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    right.set_xticks(range(len(groups)), groups, rotation=90)
    _color_labels(right.get_xticklabels(), groups, group_colors)
    right.set_xlabel(groupby)
    right.set_ylabel(f"% of the {groupby}'s clonotypes found in another {groupby}")
    right.set_title("Shared fraction")
    right.spines[["top", "right"]].set_visible(False)

    if legend:
        _legend(fig, [(splitby, list(split_colors.items())),
                      (None, [("no shared clonotypes", NONE_COLOR)] if none.any() else [])],
                loc="upper left", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    _finish(fig, save=save, show=show)
    return np.array([left, right])


def clone_persistence(adata, *, key=None, order=None, legend=True, figsize=None, save=None,
                      show=None, return_df=False):
    """The clones each group carries between covariate levels, from the stored
    ``tcri.pp.clone_persistence``.

    One heatmap per group, in rows of three: cell (a, b) is the share of the group's clones at
    level a also seen at level b, counted in clones, red where the two levels share none. Each
    row label carries the level's clone count. Levels are in the covariate's category order, and
    a group seen at one level only has no pair to draw.

    Parameters
    ----------
    adata
        The object ``tcri.pp.clone_persistence`` stored its result in.
    key
        The ``uns`` key of a result stored with ``key_added``.
    order
        The groups to draw, in this order; by default every group, in the table's order.
    legend
        ``False`` draws no legend.
    figsize
        The size of the figure; by default it grows with the number of groups.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the stored ``{table, pairs}`` instead of drawing.

    Returns
    -------
    numpy.ndarray or dict
        The axes, one per group, or the stored result with ``return_df=True``.

    Raises
    ------
    ValueError
        When the stored result was computed with ``per_clone=True``.
    """
    result, params = _stored(adata, "clone_persistence", key, per_flag="per_clone")
    if return_df:
        return result
    groupby, splitby = params.get("groupby"), params.get("splitby")
    covariate = params["covariate_key"]
    table, pairs = result["table"], result["pairs"]
    groups = _groups(table, groupby, order)
    group_of = (pd.Series(_ALL, index=pairs.index) if groupby is None
                else pairs[groupby].astype(str))
    categories = ([str(c) for c in adata.obs[covariate].astype("category").cat.categories]
                  if covariate in adata.obs else sorted(set(pairs["covariate_a"].astype(str))))
    group_colors, split_colors = _split_colors(adata, table, groupby, splitby, groups)

    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                             figsize=figsize or (5.2 * ncols, 4.8 * nrows))
    image, drew_none = None, False
    for ax, group in zip(axes.ravel(), groups):
        sub = pairs[group_of == group]
        present = set(sub["covariate_a"].astype(str))
        levels = [level for level in categories if level in present]
        ax.set_title(f"Clone overlap - {group}", color=group_colors.get(group, "black"))
        if len(levels) < 2:
            ax.text(0.5, 0.5, "one level", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        a, b = sub["covariate_a"].astype(str), sub["covariate_b"].astype(str)
        share = (sub.assign(a=a, b=b).pivot(index="a", columns="b", values="shared_fraction")
                 .reindex(index=levels, columns=levels).to_numpy(float))
        none = (sub.assign(a=a, b=b).pivot(index="a", columns="b", values="shared_clones")
                .reindex(index=levels, columns=levels).to_numpy(float) == 0)
        drew_none |= bool(none.any())
        n_clones = sub.assign(a=a).drop_duplicates("a").set_index("a")["n_clones"]
        image = _heatmap(ax, share, none, vmax=1.0, annotate=False, digits=2)
        ax.set_xticks(range(len(levels)), levels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(levels)),
                      [f"{level} ({int(n_clones[level])})" for level in levels], fontsize=8)
    for ax in axes.ravel()[len(groups):]:
        ax.set_axis_off()
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.02, aspect=40,
                     label="share of the row level's clones also seen at the column level")

    if legend:
        _legend(fig, [(splitby, list(split_colors.items())),
                      (None, [("no shared clones", NONE_COLOR)] if drew_none else [])],
                loc="outside upper right")
    _finish(fig, save=save, show=show)
    return axes.ravel()[:len(groups)]
