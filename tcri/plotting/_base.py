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

**Connecting lines.** A line between x positions implies the two points are the same entity
observed twice, which is a claim only matched data supports, so it is drawn from an identity
key and never from adjacency: the endpoints view joins one replicate's two endpoint values,
computed over one intersected clone set, and only where the item is an entity that persists
across the two levels. The grey reference points are never joined, and a significance bracket
-- which spans x positions by design -- carries its own label so the guard can tell it from a
matched-identity line.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._state import _reference
from ._colors import resolve_colors

__all__: list[str] = []  # private module

tcri_bar_color = "#66D9EF"

#: Coarsest first. The sample at an x position is the first of these that varies there.
_UNIT_ORDER = ("replicate", "item", "draw")

#: Significance brackets span x positions by design. Leading underscore keeps them out of
#: legends; the name lets the connector guard tell them apart from a matched-identity line.
BRACKET_LABEL = "_tcri_bracket"

#: The permutation reference, drawn grey and hollow behind the value. Labelled so a test can
#: find the collection and check it sits at the value's own x positions, and so the "nothing
#: spans two x positions" guard can tell it from a matched-identity line.
REFERENCE_LABEL = "_tcri_reference"

#: What the grey marks are called in the legend. REFERENCE_LABEL is underscore-prefixed, which
#: is exactly what keeps matplotlib from listing every reference artist individually -- so
#: without one explicit handle a panel shows grey boxes beside the coloured ones and nothing
#: says what they are. A reader can reasonably take them for a third arm.
REFERENCE_LEGEND = "permutation null"


def _reference_legend(ax, drawn):
    """Name the grey marks, keeping whatever legend the panel already has.

    Appends rather than replaces: the box-and-strip legend carries the split levels and the
    endpoints view carries the matched-clone sizes, and both are still wanted. Re-creating the
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


def _finish(fig, ax, *, save=None, show=None):
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


def _colors_for(adata, key, levels, palette):
    """A ``{level: hex}`` map, persisted under ``uns[f"{key}_colors"]`` when ``key`` names an
    obs column so the same level keeps its colour in every later figure (and in scanpy's)."""
    persist = key is not None and key in adata.obs
    return resolve_colors(adata, key or "tcri_level", levels, palette=palette, persist=persist)


def filter_quantity(stats, quantity):
    """The rows of ``stats`` for one quantity, or ``stats`` unchanged when it carries none.

    A ``stats`` frame holds one row per (contrast, quantity) once a reference has been
    computed. Every reader of it needs a single row per contrast, so the quantity is selected
    here rather than left to row order, which would make a SILENT choice between the value's
    contrast and the excess's.
    """
    if stats is None or not len(stats) or "quantity" not in stats.columns:
        return stats
    return stats.loc[stats["quantity"] == quantity]


def _stat_label(stats, a, b, *, quantity="value"):
    """The stars for the (a, b) contrast, in either order, or ``None``."""
    stats = filter_quantity(stats, quantity)
    if stats is None or not len(stats):
        return None
    hits = [row for _, row in stats.iterrows()
            if {row.get("level_a"), row.get("level_b")} == {a, b}]
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
            # labelled so the "nothing connects two x positions" guard can tell a
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


def order_from(d, x, y):
    """The x order a mark would have chosen for itself: descending median of ``y``.

    Computed ONCE by the caller and handed to every mark, because a reference is a SECOND pass
    over the same axes: a mark that re-derives the order from its own y re-sorts the axis and
    rewrites the tick labels under marks already drawn.
    """
    return d.groupby(x, observed=True)[y].median().sort_values(ascending=False).index.tolist()


def _violins(adata, d, *, x, y, palette, ax, ylabel, rotation, order=None):
    """One violin per x position over the DRAW distribution.

    Reached only when draws are the coarsest varying unit, so a violin never spans replicates.
    """
    import seaborn as sns

    if order is None:
        order = order_from(d, x, y)
    colours = _colors_for(adata, x, order, palette)
    sns.violinplot(data=d, x=x, y=y, order=order, ax=ax, palette=colours,
                   inner="quartile", cut=0, density_norm="width", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    return order


def _mark_reference(ax, first_collection, first_patch, first_line):
    """Label everything drawn since the given offsets as the reference, and push it behind.

    Seaborn draws a box-and-strip as several unlabelled artists, so the reference pass cannot
    be found afterwards by colour. Labelling it is what lets a test assert that the grey marks
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
    # grey lines across the marks in front of it.
    for ln in ax.lines[first_line:]:
        ln.set_label(REFERENCE_LABEL)
        ln.set_zorder(1.45)


def _boxstrip(adata, d, *, x, y, hue, order, hue_order, palette, ax, ylabel, rotation, s=20,
              reference=False):
    """Box + strip of a tidy frame, coloured through the shared palette.

    ``reference`` draws the same marks grey, hollow and behind: the permutation reference is a
    second pass over the same axes, never a second axis and never a second figure.
    """
    import seaborn as sns

    if order is None:
        order = order_from(d, x, y)
    colour_key = hue if hue is not None else x
    levels = hue_order if (hue is not None and hue_order is not None) else (
        sorted(d[hue].dropna().unique().tolist(), key=str) if hue is not None else order
    )
    colours = _colors_for(adata, colour_key, levels, palette)

    common = dict(data=d, x=x, y=y, order=order, ax=ax)
    if hue is not None:
        common.update(hue=hue, hue_order=levels)
    first_collection, first_patch, first_line = (len(ax.collections), len(ax.patches),
                                                 len(ax.lines))
    if reference:
        colours = {lv: "0.85" for lv in levels}
    sns.boxplot(**common, palette=colours, showfliers=False, boxprops=dict(alpha=0.5))
    sns.stripplot(**common, palette=colours, dodge=hue is not None, size=s / 3,
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


def _points(adata, d, *, x, y, palette, ax, ylabel, rotation, order=None, ref=None):
    """One point per x position, with the posterior HDI as an error bar.

    The floor of the mark rule: nothing varies within an x position, so there is no
    distribution to draw. This is the only place a summary interval stands in for a sample,
    and it is drawn as a point-with-interval rather than a bar because a bar's area encodes a
    magnitude from zero that these metrics do not have.

    ``ref`` names a reference column, drawn grey and hollow at the SAME x positions in the same
    pass rather than by calling this twice -- two passes would re-sort the axis under the first.

    The interval is drawn only for ``y == "value"``. An HDI is the posterior spread of the
    value: the reference has none to show here, and an excess cannot have one at all, because
    an interval on a difference needs paired draws and draws are never paired across two fits.
    """
    d = d.sort_values(y, ascending=False) if order is None else \
        d.set_index(d[x].astype(str)).reindex([str(o) for o in order]).dropna(subset=[y])
    labels = d[x].astype(str).tolist()
    values = d[y].to_numpy(dtype=float)
    colours = _colors_for(adata, x, labels, palette)

    yerr = None
    if y == "value" and {"hdi_low", "hdi_high"} <= set(d.columns):
        lo = d["hdi_low"].to_numpy(dtype=float)
        hi = d["hdi_high"].to_numpy(dtype=float)
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
    ax.scatter(pos, values, s=45, c=[colours[l] for l in labels], zorder=3)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    return ax


def render_metric(adata, name, *, ylabel, item_col=None, item_as_x=False, key=None,
                  order=None, hue_order=None, palette=None, ax=None, figsize=(8, 4),
                  save=None, show=None, return_df=False, annotate=True, rotation=90,
                  decorate=None, quantity="value"):
    """Draw a cached ``tl`` result. The axes come from its ``params``, not from arguments.

    ``item_as_x`` puts the metric's own item axis on x — right for clonotypic entropy, whose
    items are a handful of phenotypes, wrong for phenotypic entropy, whose items are every
    clone in the repertoire.

    ``quantity`` selects what goes on y. ``"value"`` draws the metric with its permutation
    reference behind it, grey and hollow, when the result carries one. ``"excess"`` draws
    ``value - null_value`` against a zero rule and no interval. Nothing switches on its own:
    the ``stats`` frame carries both, and the star drawn is always the one for the quantity on
    the axis.
    """
    from .. import get as _get
    from .._compute._tables import collapse_to_replicates

    payload = _get.result(adata, name, key=key)
    params = _get.params(adata, name, key=key)
    result = payload["result"]
    table = payload.get("table")
    stats = payload.get("stats")
    if return_df:
        return result

    groupby = params.get("groupby")
    splitby = params.get("splitby")
    fig, ax = _axes(ax, figsize)

    if result is None or not len(result) or "value" not in result.columns:
        ylabel = _reference.label_for(result, quantity, ylabel)
        return _finish(fig, _empty(ax, f"no data for {name}", ylabel), save=save, show=show)
    if quantity not in result.columns:
        raise ValueError(
            f"this {name} result has no {quantity!r} column: it was computed with "
            f"null_model=None, so there is nothing to compare against. Re-run the metric with "
            f"a reference (tcri.null.all(model, adata) first), or plot quantity='value'."
        )
    d = result.dropna(subset=[quantity])
    if not len(d):
        return _finish(fig, _empty(ax, f"no finite {name}", ylabel), save=save, show=show)
    #: The reference column for this quantity: drawn behind the value, and meaningless on an
    #: excess panel, where zero is the reference and the zero rule already says so. A column
    #: with nothing finite in it is NOT a reference: seaborn raises on an all-NaN y, and the
    #: honest panel is the value with "no reference" on the label rather than a blank axes.
    ref = "null_value" if (quantity == "value" and "null_value" in d.columns) else None
    if ref is not None and not np.isfinite(
            pd.to_numeric(d[ref], errors="coerce").to_numpy(dtype=float)).any():
        ref = None
        result = result.drop(columns=["null_value"])
    ylabel = _reference.label_for(result, quantity, ylabel)
    if quantity != "value":
        decorate = decorate or _zero_rule

    has_groups = groupby is not None and groupby in d.columns
    single = False
    if item_as_x and item_col in d.columns:
        x, hue = item_col, (splitby if (splitby and splitby in d.columns) else None)
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
        d = d.assign(_one=str(params.get("covariate") or name))
        x, hue = "_one", None
        single = True

    unit = _sample_unit(d, table, x=x, groupby=groupby if has_groups else None,
                        item_col=item_col)
    if unit == "draw" and quantity != "value":
        # `excess` is a difference of two SUMMARIES, so `attach` broadcasts one number per row
        # group onto every draw: a violin of it is a spike at that number, and five of six
        # groups collapse to a degenerate KDE seaborn declines to draw. The quantity exists per
        # group, so the group-level mark is the honest one.
        unit = "item" if (item_col is not None and item_col in d.columns) else None

    if unit == "replicate":
        # collapse items -> replicates with the SAME function build_stats uses, so the dots
        # and the p-value beneath them cannot describe different units. Everything about to be
        # drawn is preserved -- collapsing away the hue would silently drop the split.
        #
        # ONE call over the quantity AND its reference: the single-column collapse drops
        # `null_value` outright on this path, so the grey mark would die here whatever the rest
        # of the threading did, and averaging the two under different masks would leave the
        # grey mark and the value mark resting on different replicate sets.
        d = collapse_to_replicates(d, groupby=groupby,
                                   value=[c for c in (quantity, ref) if c],
                                   keep=[c for c in (x, hue) if c and c in d.columns])
        hue = hue if (hue and hue in d.columns) else None
        ref = ref if (ref and ref in d.columns) else None
        if d is None or not len(d):
            # the shared mask can empty the frame even though `value` was finite everywhere,
            # when the reference is non-finite on every row. Say so rather than drawing an
            # axes with no marks on it.
            return _finish(fig, _empty(ax, f"no finite {name}", ylabel), save=save, show=show)

    # ONE order, computed here from the quantity on the axis and handed to every mark. A mark
    # that derives it from its own y re-sorts the axis on the reference pass.
    _order = order if order is not None else order_from(d, x, quantity)

    if unit in ("replicate", "item"):
        if ref is not None:
            # the SAME hue and hue_order as the value pass. Drawn with hue=None the grey marks
            # land at the category centre while the value marks dodge either side of it, so the
            # reference sits between its own two arms instead of behind them -- and the grey box
            # pools both arms into one distribution that belongs to neither.
            _boxstrip(adata, d, x=x, y=ref, hue=hue, order=_order, hue_order=hue_order,
                      palette=palette, ax=ax, ylabel=ylabel, reference=True,
                      rotation=0 if single else rotation)
        _order, _levels = _boxstrip(adata, d, x=x, y=quantity, hue=hue, order=_order,
                                    hue_order=hue_order, palette=palette, ax=ax,
                                    ylabel=ylabel, rotation=0 if single else rotation)
    elif unit == "draw":
        t = table.dropna(subset=[quantity]) if quantity in table.columns else table
        if quantity not in t.columns:
            return _finish(fig, _empty(ax, f"no {quantity} in the draws", ylabel),
                           save=save, show=show)
        if x in ("_all", "_one"):
            t = t.assign(**{x: d[x].iloc[0]})
        elif x not in t.columns:
            return _finish(fig, _empty(ax, f"no {x} axis in the draws", ylabel),
                           save=save, show=show)
        _order = _violins(adata, t, x=x, y=quantity, palette=palette, ax=ax, ylabel=ylabel,
                          rotation=0 if single else rotation, order=_order)
        if ref is not None:
            # the reference has no draw distribution of its own to show here -- its own draws
            # belong to a different fit and are not paired with these -- so it is drawn as the
            # one number it is, a grey marker per x position behind the violin
            per = d.drop_duplicates(x).set_index(d.drop_duplicates(x)[x].astype(str))
            ys = [per[ref].get(str(o), np.nan) for o in _order]
            ax.scatter(np.arange(len(_order)), ys, s=45, facecolors="none", edgecolors="0.55",
                       linewidths=1.1, zorder=2, label=REFERENCE_LABEL)
    else:
        _points(adata, d, x=x, y=quantity, palette=palette, ax=ax, ylabel=ylabel,
                rotation=0 if single else rotation, order=_order, ref=ref)
        _order = [str(o) for o in _order]

    if decorate is not None:
        decorate(ax)
    _reference_legend(ax, ref is not None)
    if single or x == "_all":
        ax.set_xlabel("")
    if annotate and x == splitby:
        _annotate_contrasts(ax, stats, _order, quantity=quantity)
    return _finish(fig, ax, save=save, show=show)


# ── the delta family ─────────────────────────────────────────────────────────

#: Connectors carry this so the "nothing spans two x positions" guard can tell a
#: matched-identity line from a claim made out of adjacency.
CONNECTOR_LABEL = "_tcri_matched"


def _zero_rule(ax):
    """Zero is a real position on a delta axis, so mark it."""
    ax.axhline(0.0, lw=0.8, ls="--", c="0.45", zorder=0, label=BRACKET_LABEL)


def _matched_counts(result, *, groupby, item_col):
    """Clones matched per replicate — the n each replicate's value rests on.

    Derived, not stored: the intersection already decided which rows exist, so counting them
    is the count. It varies per replicate (one patient may match 240 of 300, another 30 of
    400), which is why it is encoded per point rather than stated once in a title.
    """
    if groupby is None or groupby not in result.columns or item_col not in result.columns:
        return None
    return result.groupby(groupby, observed=True)[item_col].nunique()


def _sizes_from(counts, labels, *, lo=25.0, hi=190.0):
    """Marker AREA proportional to the matched count, clamped to a legible band.

    `s` is already area in points^2, so proportionality is perceptually right; the clamp stops
    a 400-clone replicate from swallowing a 12-clone one at 33x the area.
    """
    n = np.array([counts.get(l, np.nan) for l in labels], dtype=float)
    if not np.isfinite(n).any():
        return None, None
    lo_n, hi_n = np.nanmin(n), np.nanmax(n)
    if not np.isfinite(lo_n) or hi_n <= lo_n:
        return np.full(len(n), (lo + hi) / 2), n
    return lo + (hi - lo) * (n - lo_n) / (hi_n - lo_n), n


def _size_legend(ax, counts):
    """Three reference dots spanning the observed range, labelled with real counts."""
    vals = np.array(sorted(set(int(v) for v in counts.values if np.isfinite(v))), dtype=float)
    if vals.size == 0:
        return
    picks = np.unique(np.percentile(vals, [0, 50, 100]).round().astype(int))
    sizes, _ = _sizes_from(counts, list(counts.index))
    lut = dict(zip(counts.values.astype(int), sizes))
    handles = [plt.scatter([], [], s=lut.get(int(v), 60), c="0.6", edgecolor="black",
                           linewidth=.3, label=str(int(v))) for v in picks]
    ax.legend(handles=handles, title="clones matched", bbox_to_anchor=(1.02, 1.0),
              loc="upper left", frameon=False, fontsize=8, title_fontsize=8, labelspacing=1.1)


def render_delta(adata, name, *, ylabel, item_col, kind="delta", quantity="value",
                 item_as_x=False, entity_matched=False, key=None, order=None, hue_order=None,
                 palette=None, ax=None, figsize=(8, 4), save=None, show=None,
                 return_df=False, rotation=90):
    """Render a cached delta result — the change, or its two endpoints.

    ``entity_matched`` says the ITEM is an entity that persists across the two levels (a
    clonotype), rather than a category measured twice (a phenotype). It gates two things at
    once because both are claims about the same fact:

    * connectors — a line asserts "this is the same thing, later";
    * dot area = matched count — the number of matched CLONES the value rests on.

    The second is not merely cosmetic to gate. For a phenotype-item metric the matched clone
    count is not in ``result`` at all: those clones were summed over inside ``H(c|phi)``, so
    counting item rows would count PHENOTYPES and label them "clones matched".
    """
    from .. import get as _get
    from .._compute._tables import collapse_to_replicates

    if kind not in ("delta", "endpoints"):
        raise ValueError(f"kind must be 'delta' or 'endpoints', got {kind!r}")
    if kind == "endpoints" and quantity != "value":
        raise ValueError(
            f"kind='endpoints' has no {quantity!r}: the view draws the two levels a delta is "
            f"taken between, and it draws the reference's own two endpoints beside them. A "
            f"single excess axis would have to pick one of the two. Use kind='delta', "
            f"quantity={quantity!r}, or kind='endpoints', quantity='value'."
        )

    payload = _get.result(adata, name, key=key)
    params = _get.params(adata, name, key=key)
    result = payload["result"]
    if return_df:
        return result

    if kind == "delta":
        return render_metric(adata, name, ylabel=ylabel, item_col=item_col,
                             item_as_x=item_as_x, key=key, order=order, hue_order=hue_order,
                             palette=palette, ax=ax, figsize=figsize, save=save, show=show,
                             annotate=True, rotation=rotation, quantity=quantity,
                             decorate=_zero_rule)

    # ── the endpoints view ──────────────────────────────────────────────────
    groupby = params.get("groupby")
    fig, ax = _axes(ax, figsize)
    if groupby is None or groupby not in result.columns:
        return _finish(fig, _empty(
            ax, f"{name} endpoints need a replicate axis\n(re-run with groupby=)", ylabel),
            save=save, show=show)

    counts = _matched_counts(result, groupby=groupby,
                             item_col=item_col) if entity_matched else None

    # ONE collapse over every endpoint column present, selected by presence rather than
    # hard-coded so the branch works at null_model=None. Collapsing the endpoints separately
    # and merging would give each its own non-finite mask, so a replicate could contribute
    # one endpoint and not the other and the pair drawn would not be a pair.
    endpoints = [c for c in ("value_from", "value_to") if c in result.columns]
    reference = [c for c in ("null_value_from", "null_value_to") if c in result.columns]
    per = collapse_to_replicates(
        result, groupby=groupby, value=endpoints + reference,
        keep=[params.get("splitby")] if params.get("splitby") else [])

    levels = [str(params["cov_from"]), str(params["cov_to"])]
    reps = per[groupby].astype(str).tolist()
    colours = _colors_for(adata, groupby, reps, palette)
    sizes, _n = _sizes_from(counts, reps) if counts is not None else (None, None)

    for i, rep in enumerate(reps):
        if len(reference) == 2:
            # NO connector between the grey points: a line asserts matched identity across a
            # PERMUTED fit, which is the claim CONNECTOR_LABEL exists to prevent. And a FIXED
            # size, not `_sizes_from`: the null's matched clone count is provably the parent's,
            # so sizing these would repeat one number and make the legend ambiguous about which
            # collection it describes.
            ax.scatter([0, 1], [per[reference[0]].iloc[i], per[reference[1]].iloc[i]],
                       s=[55, 55], facecolors="none", edgecolors="0.55", linewidths=1.1,
                       zorder=2, label=REFERENCE_LABEL)
        y = [per["value_from"].iloc[i], per["value_to"].iloc[i]]
        if entity_matched:
            ax.plot([0, 1], y, lw=0.9, c=colours[rep], alpha=0.7, zorder=1,
                    label=CONNECTOR_LABEL)
        s = 60 if sizes is None else sizes[i]
        # one replicate, one size: the matched clone set is the SAME on both sides, so a
        # difference between a replicate's two dots would mean the intersection did not hold
        ax.scatter([0, 1], y, s=[s, s], color=colours[rep], edgecolor="black",
                   linewidth=0.3, zorder=3)

    ax.set_xticks([0, 1]); ax.set_xticklabels(levels, rotation=0)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel.replace("Δ ", "").replace(" (bits)", " (bits)"))
    ax.set_xlabel(params.get("groupby") and "")
    if counts is not None:
        _size_legend(ax, counts)
    _reference_legend(ax, len(reference) == 2)
    return _finish(fig, ax, save=save, show=show)
