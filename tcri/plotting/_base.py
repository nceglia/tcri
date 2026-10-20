"""``pl`` private plotting engine — the shared cache renderer and the mark rule.

The twins are **cache renderers** in the strict sense: they read ``uns`` through
:mod:`tcri.get` and draw. There is no metric math here and no call into ``tl``.

That matters more than it sounds. The plot and the frame in the caller's hand cannot disagree
about ``n_samples``, the draw or the distance metric, because both read the one cached result.
The replicate axis comes from that result's ``params`` as well: a twin never manufactures a
``groupby``, so the axis a figure is grouped by is whichever one the metric resolved and
recorded — the caller's, or the registered replicate column when the caller named none.

**The mark rule.** A mark shows ONE variance component. Within each x position the sample is
the *coarsest* unit that varies there, ranked replicate > item > draw:

    replicates vary  -> box + strip, one dot per replicate (items collapsed first)
    only items vary  -> box + strip, one dot per item
    only draws vary  -> violin over draws, read from `table`
    nothing varies   -> a point

Two things this prevents. Pooling draws across replicates would render 6 patients x 100 draws
as 600 samples and produce a tight violin for a reason unrelated to evidence -- the same
pseudoreplication ``build_stats`` collapses away, drawn as a picture. And a bar with an HDI
whisker is a lossy summary of a distribution the package already stored, so wherever draws are
the sample the violin shows what was measured instead of two numbers off it.

**Small multiples.** A twin is often one panel of a row, so its defaults are the ones a row
needs. The x axis is in the category order of the column on it (the obs categories, or the
registered phenotype and clonotype categories), never sorted by this panel's values, so a level
sits at the same position in every panel; ``order=`` overrides it. ``legend=False`` draws no
legend, so a row can carry one. Dots that are replicates are colored by replicate unless a split
is the hue, and an item axis is never colored per item: its marks take one color, because a
color per item leaves the replicates within each item indistinguishable. A y label longer than the axes is wrapped onto
as many lines as it needs.

**Connecting lines.** A line between x positions implies the two points are the same entity
observed twice, which is a claim only matched data supports, so it is drawn from an identity
key and never from adjacency; no view draws one. The gray reference points are never
joined, and a significance bracket
-- which spans x positions by design -- carries its own label so the guard can tell it from a
matched-identity line.
"""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from .._state import _reference
from .._state import keys as K
from ._colors import resolve_colors

__all__: list[str] = []  # private module

#: The one color of a mark whose x levels carry no color of their own (an item axis).
tcri_bar_color = "#66D9EF"

#: Coarsest first. The sample at an x position is the first of these that varies there.
_UNIT_ORDER = ("replicate", "item", "draw")

#: Significance brackets span x positions by design. Leading underscore keeps them out of
#: legends; the name lets the connector guard tell them apart from a matched-identity line.
BRACKET_LABEL = "_tcri_bracket"

#: The permutation reference, drawn gray and hollow behind the value. Labeled so a test can
#: find the collection and check it sits at the value's own x positions, and so the "nothing
#: spans two x positions" guard can tell it from a matched-identity line.
REFERENCE_LABEL = "_tcri_reference"

#: What the gray marks are called in the legend. REFERENCE_LABEL is underscore-prefixed, which
#: is exactly what keeps matplotlib from listing every reference artist individually -- so
#: without one explicit handle a panel shows gray boxes beside the colored ones and nothing
#: says what they are. A reader can reasonably take them for a third arm.
REFERENCE_LEGEND = "permutation null"


def _reference_legend(ax, drawn):
    """Name the gray marks, keeping whatever legend the panel already has.

    Appends rather than replaces: the box-and-strip legend carries the split levels, which are
    still wanted. Re-creating the
    legend is the only way matplotlib lets a handle be added, so the existing entries and the
    title are read back off it first.
    """
    if not drawn:
        return
    from matplotlib.lines import Line2D

    existing = ax.get_legend()
    handles = list(getattr(existing, "legend_handles", [])) if existing is not None else []
    labels = [t.get_text() for t in existing.get_texts()] if existing is not None else []
    if REFERENCE_LEGEND in labels:
        return
    title = existing.get_title().get_text() if existing is not None else None
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                          markeredgecolor="0.55", markeredgewidth=1.1, markersize=7))
    labels.append(REFERENCE_LEGEND)
    ax.legend(handles=handles, labels=labels, title=title or None,
              bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False,
              fontsize=8, title_fontsize=8)


def _fit_ylabel(ax, *, min_width=12):
    """Wrap the y label onto as many lines as it takes to fit the height of the axes.

    A label longer than its axes runs past the figure edge and is cut off, which on a short panel
    in a row of them is most labels. The text is wrapped at word boundaries and never shortened.
    """
    label = ax.yaxis.label
    words = label.get_text().split()
    if not words:
        return
    limit = ax.get_window_extent().height
    flat = " ".join(words)
    label.set_text(flat)
    width = len(flat)
    while label.get_window_extent().height > limit and width > min_width:
        width = max(min_width, int(width * 0.8))
        label.set_text(textwrap.fill(flat, width))


def _finish(fig, ax, *, save=None, show=None, legend=True):
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    _fit_ylabel(ax)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return ax


def _axes(ax, figsize):
    if ax is None:
        return plt.subplots(1, 1, figsize=figsize)
    return ax.figure, ax


def _empty(ax, message, ylabel):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    return ax


def _zero_rule(ax):
    """Zero is a real position on an adjusted or a change axis, so mark it."""
    ax.axhline(0.0, lw=0.8, ls="--", c="0.45", zorder=0, label=BRACKET_LABEL)


def _colors_for(adata, key, levels, palette):
    """A ``{level: hex}`` map, persisted under ``uns[f"{key}_colors"]`` when ``key`` names an
    obs column so the same level keeps its color in every later figure (and in scanpy's)."""
    persist = key is not None and key in adata.obs
    return resolve_colors(adata, key or "tcri_level", levels, palette=palette, persist=persist)


def filter_quantity(stats, quantity, *, comparison="split"):
    """The rows of ``stats`` for one comparison and quantity; a frame without a ``comparison``
    or ``quantity`` column is not filtered on it.

    A ``stats`` frame holds one row per contrast, says in ``comparison`` what it compares, and
    names the column it contrasted in ``quantity``. The star a panel draws has to be about the
    comparison and the quantity on its axis, so the rows are selected by both here: a value
    panel of a result whose contrast is on the adjusted value gets no rows, and no star, rather
    than a star about a different column; and comparison and split levels that share a label
    cannot match each other's rows.
    """
    if stats is None or not len(stats):
        return stats
    if "comparison" in stats.columns:
        stats = stats.loc[stats["comparison"] == comparison]
    if "quantity" not in stats.columns:
        return stats
    return stats.loc[stats["quantity"] == quantity]


def check_quantity(quantity, *, value_only=False, message=None):
    """The ``quantity`` argument alone, before anything is loaded.

    Rejects a name that is not ``"auto"``, ``"value"`` or ``"adjusted"``, and an explicit
    ``"adjusted"`` on a view that can draw nothing but the value (the shift heatmap), which raises
    with that view's own ``message``. An argument error never depends on
    the data: it fires whether or not a result exists and whether or not ``return_df`` was
    passed.
    """
    if quantity not in ("auto", "value", "adjusted"):
        raise ValueError(f"quantity must be 'auto', 'value' or 'adjusted', got {quantity!r}")
    if quantity == "adjusted" and value_only:
        raise ValueError(message)


def resolve_quantity(result, quantity, *, name, value_only=False):
    """The column that goes on y, for a non-empty ``result``.

    ``"auto"`` is ``adjusted`` when the result carries one with something finite in it, else
    ``value``; on a value-only view it is always ``value``. An explicit ``"adjusted"`` with no
    such column raises: the result was computed with ``null_model=None``. Called after
    ``return_df`` and after the empty-frame branch, so a frame the caller asked for is handed
    back untouched and an empty result draws its empty panel whatever the quantity.
    """
    if quantity == "auto":
        if value_only or not _reference.usable_reference(result, "adjusted"):
            return "value"
        return "adjusted"
    if quantity == "adjusted" and "adjusted" not in result.columns:
        raise ValueError(
            f"this {name} result has no 'adjusted' column: it was computed with "
            f"null_model=None, so there is nothing to compare against. Re-run the metric with "
            f"a reference (tcri.null.all(model, adata) first), or plot quantity='value'."
        )
    return quantity


def _stat_label(stats, a, b, *, quantity="value"):
    """The stars for the (a, b) contrast, in either order, or ``None``. Levels are matched as
    text, the form ``stats`` stores them in."""
    stats = filter_quantity(stats, quantity)
    if stats is None or not len(stats):
        return None
    hits = [row for _, row in stats.iterrows()
            if {str(row.get("level_a")), str(row.get("level_b"))} == {str(a), str(b)}]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} stats rows match the contrast ({a!r}, {b!r}) at "
            f"quantity={quantity!r}. Every reader of this frame picks one row per contrast, so "
            f"a duplicate makes a silent choice. Quantities present: "
            f"{sorted(set(stats.get('quantity', ['value'])))}."
        )
    row = hits[0]
    stars = row.get("stars") or ""
    p = row.get("p")
    if stars in ("", "ns") and (p is None or not np.isfinite(p)):
        return None
    return f"{stars or 'ns'}", float(p) if p is not None else np.nan


def _annotate_contrasts(ax, stats, levels, *, quantity="value"):
    """Bracket + stars over each significant pair, drawn above the data.

    A bracket is drawn only where ``stats`` has a row for that exact pair of x levels. That
    level match — not the ``x == splitby`` check at the call site — is what stops a response
    contrast being bracketed over the phenotype axis: phenotype names never match split
    names, so no row is found and nothing is drawn. The call-site check is a second layer,
    and it matters if this is ever handed the HUE levels rather than the x order, which for
    ``item_as_x`` metrics ARE the split levels.
    """
    if stats is None or not len(stats) or len(levels) < 2:
        return
    pos = {lv: i for i, lv in enumerate(levels)}
    ymin, ymax = ax.get_ylim()
    span = (ymax - ymin) or 1.0
    step = 0.08 * span
    drawn = 0
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            label = _stat_label(stats, a, b, quantity=quantity)
            if label is None:
                continue
            text, _p = label
            y = ymax + step * (drawn + 0.4)
            xa, xb = pos[a], pos[b]
            # labeled so the "nothing connects two x positions" guard can tell a
            # significance bracket from a claim that two points are the same entity
            ax.plot([xa, xa, xb, xb], [y, y + step * 0.2, y + step * 0.2, y],
                    lw=0.9, c="0.3", clip_on=False, label=BRACKET_LABEL)
            ax.text((xa + xb) / 2, y + step * 0.25, text, ha="center", va="bottom",
                    fontsize=9, c="0.2", clip_on=False)
            drawn += 1
    if drawn:
        ax.set_ylim(ymin, ymax + step * (drawn + 0.9))


def _sample_unit(frame, table, *, x, groupby, item_col):
    """Which unit varies within an x position — the coarsest one, per the mark rule.

    Returns one of :data:`_UNIT_ORDER`, or ``None`` when a single value sits at each x.
    """
    def _varies(col, source):
        if col is None or source is None or col not in source.columns:
            return False
        counts = source.groupby(x, observed=True)[col].nunique() if x in source.columns \
            else source[col].nunique()
        return int(np.max(np.atleast_1d(counts))) > 1

    if groupby != x and _varies(groupby, frame):
        return "replicate"
    if item_col != x and _varies(item_col, frame):
        return "item"
    if table is not None and "draw" in table.columns and table["draw"].nunique() > 1:
        return "draw"
    return None


def category_order(adata, x, levels):
    """The default x order: the levels present, in the category order of the column on x.

    The phenotype and clonotype item axes take the registered categories; any other column takes
    its ``obs`` categories. Levels the categories do not list follow, sorted. Panels given the
    same categories therefore share an x order whatever their values are.

    Computed ONCE by the caller and handed to every mark, because a reference is a second pass
    over the same axes and must land at the same positions.
    """
    present = list(dict.fromkeys(levels))
    if x == "phenotype" and K.PHENOTYPE_CATEGORIES in adata.uns:
        canon = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    elif x == "clonotype" and K.CLONOTYPE_CATEGORIES in adata.uns:
        canon = list(adata.uns[K.CLONOTYPE_CATEGORIES])
    elif x in adata.obs.columns:
        canon = list(adata.obs[x].astype("category").cat.categories)
    else:
        canon = []
    by_name = {str(lv): lv for lv in present}
    ordered = [by_name[str(c)] for c in canon if str(c) in by_name]
    seen = set(map(str, ordered))
    return ordered + sorted((lv for lv in present if str(lv) not in seen), key=str)


def _violins(adata, d, *, x, y, palette, ax, ylabel, rotation, order=None, color_x=True):
    """One violin per x position over the DRAW distribution.

    Reached only when draws are the coarsest varying unit, so a violin never spans replicates.
    ``color_x`` colors each violin by its x level; ``False`` draws them all in one color.
    """
    import seaborn as sns

    if order is None:
        order = category_order(adata, x, d[x].dropna().unique().tolist())
    colors = (_colors_for(adata, x, order, palette) if color_x
               else {lv: tcri_bar_color for lv in order})
    sns.violinplot(data=d, x=x, y=y, order=order, ax=ax, palette=colors,
                   inner="quartile", cut=0, density_norm="width", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    return order


def _reference_means(ax, d, *, x, ref, order):
    """The reference as its mean: one hollow gray marker per x position, behind the violins.

    Its spread over draws is the posterior spread of one null fit, and that spread is already
    inside the adjusted value's interval. Drawn as a violin it would read as a permutation
    distribution, which one null fit does not have.
    """
    per = d.groupby(d[x].astype(str), observed=True)[ref].mean()
    ax.scatter(np.arange(len(order)), [per.get(str(o), np.nan) for o in order], s=45,
               facecolors="none", edgecolors="0.55", linewidths=1.1, zorder=2,
               label=REFERENCE_LABEL)


def _mark_reference(ax, first_collection, first_patch, first_line):
    """Label everything drawn since the given offsets as the reference, and push it behind.

    Seaborn draws a box-and-strip as several unlabeled artists, so the reference pass cannot
    be found afterwards by color. Labeling it is what lets a test assert that the gray marks
    sit at the value's own x positions, and what keeps the "one dot per replicate" counts in
    this package's own tests counting one thing.
    """
    for c in ax.collections[first_collection:]:
        c.set_label(REFERENCE_LABEL)
        c.set_zorder(1.5)
    for pch in ax.patches[first_patch:]:
        pch.set_label(REFERENCE_LABEL)
        pch.set_zorder(1.4)
    # ...and the LINES. Seaborn draws a box's whiskers, caps and median as Line2D at zorder 2
    # and 2.1, above the value boxes, so without this a reference drawn "behind" still puts
    # gray lines across the marks in front of it.
    for ln in ax.lines[first_line:]:
        ln.set_label(REFERENCE_LABEL)
        ln.set_zorder(1.45)


def _replicate_legend(ax, colors, *, title):
    """Name each replicate's color, so a replicate can be followed from panel to panel."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=c,
                      markeredgecolor="black", markeredgewidth=0.3, markersize=6)
               for c in colors.values()]
    ax.legend(handles=handles, labels=[str(r) for r in colors], title=title,
              bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8,
              title_fontsize=8)


def _boxstrip(adata, d, *, x, y, hue, order, hue_order, palette, ax, ylabel, rotation, s=20,
              reference=False, color_x=True, replicate=None):
    """Box + strip of a tidy frame, colored through the shared palette.

    ``reference`` draws the same marks gray, hollow and behind: the permutation reference is a
    second pass over the same axes, never a second axis and never a second figure.

    With a ``hue`` the marks take the hue's colors. Without one, ``color_x`` colors each box by
    its x level and ``False`` draws every box in one color; ``replicate`` names the column the
    dots are replicates of, and colors each dot by its replicate, with a legend naming them.
    """
    import seaborn as sns

    if order is None:
        order = category_order(adata, x, d[x].dropna().unique().tolist())
    color_key = hue if hue is not None else x
    levels = hue_order if (hue is not None and hue_order is not None) else (
        sorted(d[hue].dropna().unique().tolist(), key=str) if hue is not None else order
    )
    by_replicate = (replicate is not None and hue is None and not reference
                    and replicate in d.columns and replicate != x)
    if hue is not None or color_x:
        colors = _colors_for(adata, color_key, levels, palette)
    else:
        # one color; a white box under dots colored by replicate, so no replicate's color
        # disappears into the box behind it
        colors = {lv: "white" if by_replicate else tcri_bar_color for lv in levels}

    common = dict(data=d, x=x, y=y, order=order, ax=ax)
    if hue is not None:
        common.update(hue=hue, hue_order=levels)
    first_collection, first_patch, first_line = (len(ax.collections), len(ax.patches),
                                                 len(ax.lines))
    if reference:
        colors = {lv: "0.85" for lv in levels}
    sns.boxplot(**common, palette=colors, showfliers=False, boxprops=dict(alpha=0.5))
    if by_replicate:
        reps = category_order(adata, replicate, d[replicate].dropna().unique().tolist())
        rep_colors = _colors_for(adata, replicate, reps, None)
        sns.stripplot(data=d, x=x, y=y, order=order, ax=ax, hue=replicate, hue_order=reps,
                      palette=rep_colors, dodge=False, size=s / 3, edgecolor="black",
                      linewidth=0.3, legend=False)
        _replicate_legend(ax, rep_colors, title=replicate)
    else:
        sns.stripplot(**common, palette=colors, dodge=hue is not None, size=s / 3,
                      edgecolor="black", linewidth=0.3, legend=False)
    if reference:
        _mark_reference(ax, first_collection, first_patch, first_line)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    if hue is not None and ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)
    return order, levels


def _points(adata, d, *, x, y, palette, ax, ylabel, rotation, order=None, ref=None,
            color_x=True):
    """One point per x position, with the posterior HDI as an error bar.

    The floor of the mark rule: nothing varies within an x position, so there is no
    distribution to draw. This is the only place a summary interval stands in for a sample,
    and it is drawn as a point-with-interval rather than a bar because a bar's area encodes a
    magnitude from zero that these metrics do not have.

    ``ref`` names a reference column, drawn gray and hollow at the SAME x positions in the same
    pass rather than by calling this twice -- two passes would re-sort the axis under the first.

    The interval is the HDI of ``y`` over its draws: ``hdi_*`` for the value, and
    ``adjusted_hdi_*`` for the adjusted value, whose draws are the paired differences. It is NaN,
    and nothing is drawn, at ``n_samples=0``. The reference is drawn as its mean.

    ``color_x`` colors each point by its x level; ``False`` draws them all in one color.
    """
    if order is None:
        order = category_order(adata, x, d[x].dropna().unique().tolist())
    d = d.set_index(d[x].astype(str)).reindex([str(o) for o in order]).dropna(subset=[y])
    labels = d[x].astype(str).tolist()
    values = d[y].to_numpy(dtype=float)
    colors = (_colors_for(adata, x, labels, palette) if color_x
               else {lv: tcri_bar_color for lv in labels})

    yerr = None
    low, high = (_reference.summary_column(y, "hdi_low"),
                 _reference.summary_column(y, "hdi_high"))
    if {low, high} <= set(d.columns):
        lo = d[low].to_numpy(dtype=float)
        hi = d[high].to_numpy(dtype=float)
        if np.isfinite(lo).any():
            # asymmetric about the mean, so a single +/- would misstate it
            yerr = np.vstack([np.nan_to_num(values - lo), np.nan_to_num(hi - values)])

    pos = np.arange(len(d))
    if yerr is not None:
        ax.errorbar(pos, values, yerr=yerr, fmt="none", ecolor="0.3", elinewidth=0.9,
                    capsize=3)
    if ref is not None and ref in d.columns:
        ax.scatter(pos, d[ref].to_numpy(dtype=float), s=45, facecolors="none",
                   edgecolors="0.55", linewidths=1.1, zorder=2, label=REFERENCE_LABEL)
    ax.scatter(pos, values, s=45, c=[colors[l] for l in labels], zorder=3)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    return ax


def render_metric(adata, name, *, quantity, ylabel, item_col=None, item_as_x=False, key=None,
                  order=None, hue_order=None, palette=None, legend=True, ax=None,
                  figsize=(8, 4), save=None, show=None, return_df=False, annotate=True,
                  rotation=90, decorate=None, view="replicate", distance=False):
    """Draw a cached ``tl`` result. The axes come from its ``params``, not from arguments.

    ``view`` picks the level of the stored result: ``"replicate"`` draws the replicate summary,
    one value per replicate, and stars the split tests; ``"item"`` draws the item summary, one
    value per replicate and item. A metric with no items raises for ``"item"``.

    ``item_as_x`` puts the metric's own item axis on x — right for clonotypic entropy, whose
    items are a handful of phenotypes, wrong for phenotypic entropy, whose items are every
    clone in the repertoire.

    ``quantity`` selects what goes on y. ``"auto"``, the twins' default, is the adjusted value
    when the result carries one and the raw value otherwise. ``"adjusted"`` draws
    ``value - null_value`` against a zero rule, under the same mark rule as the value: its
    per-draw differences make a violin where draws are the sample and an interval on a point;
    ``"value"`` draws the metric with its permutation reference behind it, gray, when the
    result carries one. The star drawn is always the one for the quantity on the axis, so a
    value panel of a result whose ``stats`` is on the adjusted value carries none.

    A change result (``cov_from`` and ``cov_to`` in ``params``) draws each replicate's change, or
    each item's with ``view="item"``, against a zero rule, since a change can be negative, and
    prefixes its y label with ``Δ``.
    ``distance=True`` marks a change that is a distance between the levels (flux), never
    negative, whose zero rule is on its adjusted panel only.

    ``order`` overrides the category order of the x axis; ``legend=False`` draws no legend.
    """
    from .. import get as _get
    from .._compute._tables import collapse_to_replicates

    check_quantity(quantity)
    if view not in ("replicate", "item"):
        raise ValueError(f"view must be 'replicate' or 'item', got {view!r}")
    payload = _get.result(adata, name, key=key)
    params = _get.params(adata, name, key=key)
    change = params.get("cov_from") is not None
    if change and not distance:
        # a change's axis says so; flux is between two levels by definition and keeps its label
        ylabel = f"Δ {ylabel}"
    if view == "item":
        level = payload.get("item")
        if level is None:
            raise ValueError(
                f"{name} has no item view: it has one value per replicate and no items.")
    else:
        level = payload["replicate"]
        item_col, item_as_x = None, False
    result, table = level["summary"], level["draws"]
    stats = payload.get("stats")
    if return_df:
        return result

    groupby = params.get("groupby")
    splitby = params.get("splitby")
    fig, ax = _axes(ax, figsize)

    if result is None or not len(result) or "value" not in result.columns:
        ylabel = _reference.label_for(result, quantity, ylabel)
        return _finish(fig, _empty(ax, f"no data for {name}", ylabel), save=save, show=show,
                       legend=legend)
    quantity = resolve_quantity(result, quantity, name=name)
    d = result.dropna(subset=[quantity])
    if not len(d):
        return _finish(fig, _empty(ax, f"no finite {name}", ylabel), save=save, show=show,
                       legend=legend)
    # The reference is drawn behind the value and nowhere else: on the adjusted panel zero is
    # the reference, and the zero rule says so. A column with nothing finite in it is not a
    # reference either: seaborn raises on an all-NaN y, so the panel is the value with
    # "(no reference)" on the label rather than a blank axes.
    ref = None
    if quantity == "value" and "null_value" in d.columns:
        if _reference.usable_reference(d):
            ref = "null_value"
        else:
            result = result.drop(columns=["null_value"])
    ylabel = _reference.label_for(result, quantity, ylabel)
    # a change can be negative, so 0 is a position on its axis; a distance (flux) cannot
    if quantity == "adjusted" or (change and not distance):
        decorate = decorate or _zero_rule

    has_groups = groupby is not None and groupby in d.columns
    single = False
    if item_as_x and item_col in d.columns:
        x, hue = item_col, (splitby if (splitby and splitby in d.columns) else None)
    elif view == "item" and has_groups:
        # one point per item within each replicate: the replicates are on x
        x, hue = groupby, (splitby if (splitby and splitby in d.columns) else None)
    elif splitby is not None and splitby in d.columns:
        x, hue = splitby, None
    elif has_groups:
        x, hue = groupby, None
    elif item_col is not None and item_col in d.columns:
        # no groups: the items ARE the observations. Keep them on x when they are few and
        # named; otherwise they become the distribution at a single position.
        if d[item_col].nunique() <= 30:
            x, hue = item_col, None
        else:
            d = d.assign(_all="all")
            x, hue = "_all", None
    else:
        position = (f"{params['cov_from']} → {params['cov_to']}" if change
                    else params.get("covariate") or name)
        d = d.assign(_one=str(position))
        x, hue = "_one", None
        single = True

    unit = _sample_unit(d, table, x=x, groupby=groupby if has_groups else None,
                        item_col=item_col)

    if unit == "replicate":
        # collapse items -> replicates with the SAME function build_stats uses, so the dots
        # and the p-value beneath them cannot describe different units. Everything about to be
        # drawn is preserved -- collapsing away the hue would silently drop the split.
        #
        # ONE call over the quantity AND its reference: the single-column collapse drops
        # `null_value` outright on this path, so the gray mark would die here whatever the rest
        # of the threading did, and averaging the two under different masks would leave the
        # gray mark and the value mark resting on different replicate sets.
        d = collapse_to_replicates(d, groupby=groupby,
                                   value=[c for c in (quantity, ref) if c],
                                   keep=[c for c in (x, hue) if c and c in d.columns])
        hue = hue if (hue and hue in d.columns) else None
        ref = ref if (ref and ref in d.columns) else None
        if d is None or not len(d):
            # the shared mask can empty the frame even though `value` was finite everywhere,
            # when the reference is non-finite on every row. Say so rather than drawing an
            # axes with no marks on it.
            return _finish(fig, _empty(ax, f"no finite {name}", ylabel), save=save, show=show,
                           legend=legend)

    # ONE order, computed here and handed to every mark, so the reference pass lands at the
    # value's own positions.
    _order = order if order is not None else category_order(adata, x, d[x].dropna().unique())
    # A split or replicate axis has colors of its own; an item axis does not, and coloring each
    # item would leave the replicates within it indistinguishable.
    color_x = x in (splitby, groupby)
    replicate = groupby if unit == "replicate" else None

    if unit in ("replicate", "item"):
        if ref is not None:
            # the SAME hue and hue_order as the value pass. Drawn with hue=None the gray marks
            # land at the category center while the value marks dodge either side of it, so the
            # reference sits between its own two arms instead of behind them -- and the gray box
            # pools both arms into one distribution that belongs to neither.
            _boxstrip(adata, d, x=x, y=ref, hue=hue, order=_order, hue_order=hue_order,
                      palette=palette, ax=ax, ylabel=ylabel, reference=True,
                      rotation=0 if single else rotation, color_x=color_x)
        _order, _levels = _boxstrip(adata, d, x=x, y=quantity, hue=hue, order=_order,
                                    hue_order=hue_order, palette=palette, ax=ax,
                                    ylabel=ylabel, rotation=0 if single else rotation,
                                    color_x=color_x, replicate=replicate)
    elif unit == "draw":
        t = table.dropna(subset=[quantity]) if quantity in table.columns else table
        if quantity not in t.columns:
            return _finish(fig, _empty(ax, f"no {quantity} in the draws", ylabel),
                           save=save, show=show, legend=legend)
        if x in ("_all", "_one"):
            t = t.assign(**{x: d[x].iloc[0]})
        elif x not in t.columns:
            return _finish(fig, _empty(ax, f"no {x} axis in the draws", ylabel),
                           save=save, show=show, legend=legend)
        _order = _violins(adata, t, x=x, y=quantity, palette=palette, ax=ax, ylabel=ylabel,
                          rotation=0 if single else rotation, order=_order, color_x=color_x)
        if ref is not None:
            _reference_means(ax, d, x=x, ref=ref, order=_order)
    else:
        _points(adata, d, x=x, y=quantity, palette=palette, ax=ax, ylabel=ylabel,
                rotation=0 if single else rotation, order=_order, ref=ref, color_x=color_x)
        _order = [str(o) for o in _order]

    if decorate is not None:
        decorate(ax)
    _reference_legend(ax, ref is not None)
    if single or x == "_all":
        ax.set_xlabel("")
    if annotate and x == splitby:
        _annotate_contrasts(ax, stats, _order, quantity=quantity)
    return _finish(fig, ax, save=save, show=show, legend=legend)

