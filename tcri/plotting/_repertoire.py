"""``pl`` twins of the repertoire summaries in ``tcri.pp``.

Each draws the result its ``pp`` function stored, read through :mod:`tcri.get`, and computes no
number of its own. Groups are the levels of the stored call's ``groupby``; with ``groupby=None``
the whole dataset is one group, labeled ``"all"``. With ``splitby``, each group's label takes the
color of its split level from ``adata.uns[f"{splitby}_colors"]``.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex, to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ._base import _fit_ylabel
from ._colors import resolve_colors

__all__ = ["repertoire_summary", "clonotype_sharing", "clone_persistence"]

#: The band colors of the sharing panels, and red also where two levels share no clone.
GREEN, YELLOW, RED = "#7e9437", "#f0bd00", "#cd442a"

#: The bands of the sharing panels, ``(low, high)``: green below ``low``, yellow from ``low``
#: to ``high``, red above ``high``. A pair's share and a group's shared fraction are in
#: percent; the last is the number of groups carrying a shared clonotype.
_PAIR_BANDS = (1.0, 2.0)
_GROUP_BANDS = (3.0, 8.0)
_SPREAD_BANDS = (4, 5)

#: The scale every ``pp`` plot draws in: light green to dark green, through GREEN. The
#: summary's covariate levels are shades of it, and the persistence heatmap runs from white
#: through it.
_GREENS = LinearSegmentedColormap.from_list("tcri_greens", ["#d6e2b3", GREEN, "#3b4a14"])
_WHITE_GREENS = LinearSegmentedColormap.from_list("tcri_white_greens",
                                                  ["#ffffff", GREEN, "#3b4a14"])


def _shades(n):
    """``n`` shades of green, light to dark; GREEN itself for one."""
    if n == 1:
        return [GREEN]
    return [to_hex(_GREENS(x)) for x in np.linspace(0.0, 1.0, n)]

#: The label of the one group ``groupby=None`` makes.
_ALL = "all"

#: The panels of :func:`repertoire_summary`: a column of the table and its title.
_SUMMARY_PANELS = [("n_cells", "Cells"), ("n_clones", "Clones"),
                   ("singleton_fraction", "Singleton fraction"), ("clonality", "Clonality")]

#: Groups up to which a sharing heatmap prints its values in the cells.
_ANNOTATE_UP_TO = 15


def _stored(adata, name, key):
    """The stored result of ``tcri.pp.<name>`` and its arguments."""
    from .. import get as _get

    return _get.result(adata, name, key=key), _get.params(adata, name, key=key)


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


def _band(value, low, high):
    """GREEN below ``low``, YELLOW from ``low`` to ``high``, RED above ``high``."""
    if value < low:
        return GREEN
    return YELLOW if value <= high else RED


def _finish(fig, *, save=None, show=None):
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def repertoire_summary(adata, *, key=None, order=None, palette=None, legend=True, figsize=None,
                       save=None, show=None, return_df=False):
    """Cells, clones, singleton fraction and clonality per group, from the stored
    ``tcri.pp.repertoire_summary``.

    One panel per quantity, the groups on x. With ``covariate_key``, the levels of each group
    stand side by side, in shades of green from light to dark, or in ``palette``. On a pooled
    column the cells in pools are shaded inside each cell bar.

    Parameters
    ----------
    adata
        The object ``tcri.pp.repertoire_summary`` stored its table in.
    key
        The ``uns`` key of a table stored with ``key_added``.
    order
        The groups to draw, in this order; by default every group, in the table's order.
    palette
        Colors for the covariate levels in place of the shades of green: a dict, a list, or a
        colormap name.
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
        level_colors = {None: GREEN}
    else:
        observed = set(rows["covariate"].astype(str))
        categories = [str(c) for c in adata.obs[covariate].astype("category").cat.categories] \
            if covariate in adata.obs else sorted(observed)
        levels = [level for level in categories if level in observed]
        level_colors = (_by_label(adata, covariate, levels, palette) if palette is not None
                        else dict(zip(levels, _shades(len(levels)))))
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

    Three panels:

    - a group × group heatmap: cell (a, b) is the percentage of group a's clonotypes also found
      in group b, green below 1%, yellow from 1% to 2%, red above 2%, with the values printed
      when there are at most 15 groups; each row label carries the group's clonotype count;
    - each group's shared fraction, the percentage of its clonotypes found in any other group,
      green below 3%, yellow from 3% to 8%, red above 8%;
    - the shared clonotypes by the number of groups that carry them, green up to 3 groups,
      yellow at 4 and 5, red at 6 or more.

    With ``splitby``, the group labels take the colors of their split levels.

    Parameters
    ----------
    adata
        The object ``tcri.pp.clonotype_sharing`` stored its result in.
    key
        The ``uns`` key of a result stored with ``key_added``.
    order
        The groups of the first two panels, in this order; by default every group, in the
        stored order.
    legend
        ``False`` draws no legend.
    figsize
        The size of the figure; by default it grows with the number of groups.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the stored ``{pairs, groups, clonotypes}`` instead of drawing.

    Returns
    -------
    numpy.ndarray or dict
        The three axes, or the stored result with ``return_df=True``.
    """
    result, params = _stored(adata, "clonotype_sharing", key)
    if return_df:
        return result
    groupby, splitby = params["groupby"], params.get("splitby")
    table, pairs = result["groups"], result["pairs"]
    groups = _groups(table, groupby, order)
    percent = np.full((len(groups), len(groups)), np.nan)
    at = {group: i for i, group in enumerate(groups)}
    first, second = pairs[f"{groupby}_a"].astype(str), pairs[f"{groupby}_b"].astype(str)
    for a, b, share_a, share_b in zip(first, second, pairs["shared_fraction_a"],
                                      pairs["shared_fraction_b"]):
        if a in at and b in at:
            percent[at[a], at[b]] = 100 * share_a
            percent[at[b], at[a]] = 100 * share_b
    rows = table.set_index(table[groupby].astype(str))
    group_colors, split_colors = _split_colors(adata, table, groupby, splitby, groups)
    spread = result["clonotypes"]["n_groups"].value_counts()
    carriers = np.arange(2, max(len(table), int(spread.index.max()) if len(spread) else 2) + 1)
    spread = spread.reindex(carriers, fill_value=0)

    size = max(4.5, 0.45 * len(groups) + 2)
    fig, (left, middle, right) = plt.subplots(
        1, 3, figsize=figsize or (2 * size + 0.3 * len(carriers) + 5, size),
        gridspec_kw={"width_ratios": [1.15, 1, 0.8]})
    image = np.ones((len(groups), len(groups), 4))
    for i in range(len(groups)):
        for j in range(len(groups)):
            if i != j and np.isfinite(percent[i, j]):
                image[i, j] = to_rgba(_band(percent[i, j], *_PAIR_BANDS))
    left.imshow(image)
    if len(groups) <= _ANNOTATE_UP_TO:
        for i in range(len(groups)):
            for j in range(len(groups)):
                if i != j and np.isfinite(percent[i, j]):
                    yellow = _band(percent[i, j], *_PAIR_BANDS) == YELLOW
                    left.text(j, i, f"{percent[i, j]:.2f}", ha="center", va="center",
                              fontsize=8, color="black" if yellow else "white")
    left.set_xticks(range(len(groups)), groups, rotation=90)
    left.set_yticks(range(len(groups)),
                    [f"{group} ({int(rows.loc[group, 'n_clonotypes'])})" for group in groups])
    _color_labels(left.get_xticklabels(), groups, group_colors)
    _color_labels(left.get_yticklabels(), groups, group_colors)
    left.set_xlabel(groupby)
    left.set_title("Shared clonotypes")
    scale = fig.colorbar(
        ScalarMappable(norm=BoundaryNorm([0, *_PAIR_BANDS, 2 * _PAIR_BANDS[1]], 3),
                       cmap=ListedColormap([GREEN, YELLOW, RED])),
        ax=left, fraction=0.046, ticks=[0, *_PAIR_BANDS],
        label=f"% of the row {groupby}'s clonotypes found in the column {groupby}")

    shared = rows["shared_fraction"].reindex(groups).to_numpy(float) * 100
    middle.bar(range(len(groups)), shared,
               color=[_band(value, *_GROUP_BANDS) for value in shared])
    for k, value in enumerate(shared):
        if np.isfinite(value):
            middle.text(k, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    middle.set_xticks(range(len(groups)), groups, rotation=90)
    _color_labels(middle.get_xticklabels(), groups, group_colors)
    middle.set_xlabel(groupby)
    middle.set_ylabel(f"% of the {groupby}'s clonotypes found in another {groupby}")
    middle.set_title("Shared fraction")

    right.bar(carriers, spread.to_numpy(),
              color=[_band(n, *_SPREAD_BANDS) for n in carriers])
    for n, count in spread.items():
        right.text(n, count, str(count), ha="center", va="bottom", fontsize=8)
    right.set_xticks(carriers)
    right.set_xlabel(f"{groupby}s carrying the clonotype")
    right.set_ylabel("clonotypes")
    right.set_title(f"Clonotypes by number of {groupby}s")
    for ax in (middle, right):
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

    if legend:
        _legend(fig, [(splitby, list(split_colors.items()))],
                loc="upper left", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    for ax in (middle, right, scale.ax):
        _fit_ylabel(ax)
    fig.tight_layout()
    _finish(fig, save=save, show=show)
    return np.array([left, middle, right])


def clone_persistence(adata, *, key=None, order=None, legend=True, figsize=None, save=None,
                      show=None, return_df=False):
    """The clones each group carries between covariate levels, from the stored
    ``tcri.pp.clone_persistence``.

    One heatmap per group, in rows of three: cell (a, b) is the share of the group's clones at
    level a also seen at level b, counted in clones, red where the two levels share none. Each
    row label carries the level's clone count, and the x axis is labeled with the covariate.
    Levels are in the covariate's category order. A group with clones at one level only has no
    pair, so no row in the frame and no panel.

    Parameters
    ----------
    adata
        The object ``tcri.pp.clone_persistence`` stored its frame in.
    key
        The ``uns`` key of a frame stored with ``key_added``.
    order
        The groups to draw, in this order; by default every group, in the stored order.
    legend
        ``False`` draws no legend.
    figsize
        The size of the figure; by default it grows with the number of groups.
    save
        A path to save the figure to.
    show
        Show the figure.
    return_df
        Return the stored frame instead of drawing.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        The axes, one per group, or the stored frame with ``return_df=True``.
    """
    pairs, params = _stored(adata, "clone_persistence", key)
    if return_df:
        return pairs
    groupby, splitby = params.get("groupby"), params.get("splitby")
    covariate = params["covariate_key"]
    group_of = (pd.Series(_ALL, index=pairs.index) if groupby is None
                else pairs[groupby].astype(str))
    present = list(dict.fromkeys(group_of)) or [_ALL]
    groups = present if order is None else _groups(
        pd.DataFrame({groupby: present}), groupby, order)
    categories = ([str(c) for c in adata.obs[covariate].astype("category").cat.categories]
                  if covariate in adata.obs else
                  sorted(set(pairs["covariate_a"]) | set(pairs["covariate_b"])))
    group_colors, split_colors = _split_colors(
        adata, pairs.drop_duplicates(groupby) if groupby is not None else pairs, groupby,
        splitby, groups)

    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                             figsize=figsize or (5.2 * ncols, 4.8 * nrows))
    image, drew_none = None, False
    for ax, group in zip(axes.ravel(), groups):
        sub = pairs[(group_of == group).to_numpy()]
        named = set(sub["covariate_a"].astype(str)) | set(sub["covariate_b"].astype(str))
        levels = [level for level in categories if level in named]
        ax.set_title(f"Clone overlap - {group}", color=group_colors.get(group, "black"))
        if len(levels) < 2:
            ax.text(0.5, 0.5, "one level", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        at = {level: i for i, level in enumerate(levels)}
        share = np.full((len(levels), len(levels)), np.nan)
        none = np.zeros_like(share, dtype=bool)
        n_clones = {}
        for row in sub.itertuples(index=False):
            a, b = at[str(row.covariate_a)], at[str(row.covariate_b)]
            share[a, b], share[b, a] = row.shared_fraction_a, row.shared_fraction_b
            none[a, b] = none[b, a] = row.shared_clones == 0
            n_clones[a], n_clones[b] = row.n_clones_a, row.n_clones_b
        drew_none |= bool(none.any())
        image = ax.imshow(share, cmap=_WHITE_GREENS, vmin=0, vmax=1)
        ax.imshow(np.where(none, 1.0, np.nan), cmap=ListedColormap([RED]), vmin=0, vmax=1)
        ax.set_xticks(range(len(levels)), levels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(levels)),
                      [f"{level} ({int(n_clones[i])})" for i, level in enumerate(levels)],
                      fontsize=8)
        ax.set_xlabel(covariate)
    for ax in axes.ravel()[len(groups):]:
        ax.set_axis_off()
    if image is not None:
        scale = fig.colorbar(image, ax=axes, fraction=0.02, aspect=40,
                             label="share of the row level's clones also seen at the column "
                                   "level")
        _fit_ylabel(scale.ax)

    if legend:
        _legend(fig, [(splitby, list(split_colors.items())),
                      (None, [("no shared clones", RED)] if drew_none else [])],
                loc="outside upper right")
    _finish(fig, save=save, show=show)
    return axes.ravel()[:len(groups)]
