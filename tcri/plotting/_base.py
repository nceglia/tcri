"""``pl`` private plotting engine (§8.5) — the shared cache renderer and the mark rule.

The twins are **cache renderers** in the strict sense: they read ``uns`` through
:mod:`tcri.get` and draw. There is no metric math here and no call into ``tl``.

That matters more than it sounds. When each ``pl`` recomputed its metric, the plot and the
frame in the caller's hand could disagree — different ``n_samples``, a different draw, a
``distance_metric`` default that was ``"kl"`` in ``tl`` and ``"l1"`` in ``pl``. And because
a box plot needs per-unit values, ``pl.mutual_information`` and ``pl.phenotypic_flux``
*manufactured* a ``groupby`` from ``batch_col`` when the caller gave none, so the figure was
grouped by a column the caller never named.

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

**Connecting lines.** Nothing here draws a line between x positions. A line implies the two
points are the same entity observed twice, which is a claim only matched data supports -- and
matched data does not exist on this side of the API yet. When it does, the line must be drawn
from an identity key, never from adjacency.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._colors import resolve_colors

__all__: list[str] = []  # private module

tcri_bar_color = "#66D9EF"

#: Coarsest first. The sample at an x position is the first of these that varies there.
_UNIT_ORDER = ("replicate", "item", "draw")

#: Significance brackets span x positions by design. Leading underscore keeps them out of
#: legends; the name lets the connector guard tell them apart from a matched-identity line.
BRACKET_LABEL = "_tcri_bracket"


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


def _stat_label(stats, a, b):
    """The stars for the (a, b) contrast, in either order, or ``None``."""
    if stats is None or not len(stats):
        return None
    for _, row in stats.iterrows():
        if {row.get("level_a"), row.get("level_b")} == {a, b}:
            stars = row.get("stars") or ""
            p = row.get("p")
            if stars in ("", "ns") and (p is None or not np.isfinite(p)):
                return None
            return f"{stars or 'ns'}", float(p) if p is not None else np.nan
    return None


def _annotate_contrasts(ax, stats, levels):
    """Bracket + stars over each significant pair, drawn above the data.

    A bracket is drawn only where ``stats`` has a row for that exact pair of x levels. That
    level match — not the ``x == splitby`` check at the call site — is what stops a response
    contrast being bracketed over the phenotype axis: phenotype names never match split
    names, so no row is found and nothing is drawn. The call-site check is a second layer,
    and it matters if this is ever handed the HUE levels rather than the x order, which for
    ``item_as_x`` metrics ARE the split levels. Mutating either alone leaves the figure
    correct; mutating both draws a bracket the numbers never supported.
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
            label = _stat_label(stats, a, b)
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


def _violins(adata, d, *, x, y, palette, ax, ylabel, rotation, order=None):
    """One violin per x position over the DRAW distribution.

    Reached only when draws are the coarsest varying unit, so a violin never spans replicates.
    """
    import seaborn as sns

    if order is None:
        order = d.groupby(x, observed=True)[y].median().sort_values(ascending=False).index.tolist()
    colours = _colors_for(adata, x, order, palette)
    sns.violinplot(data=d, x=x, y=y, order=order, ax=ax, palette=colours,
                   inner="quartile", cut=0, density_norm="width", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    return order


def _boxstrip(adata, d, *, x, y, hue, order, hue_order, palette, ax, ylabel, rotation, s=20):
    """Box + strip of a tidy frame, coloured through the shared palette."""
    import seaborn as sns

    if order is None:
        order = d.groupby(x, observed=True)[y].median().sort_values(ascending=False).index.tolist()
    colour_key = hue if hue is not None else x
    levels = hue_order if (hue is not None and hue_order is not None) else (
        sorted(d[hue].dropna().unique().tolist(), key=str) if hue is not None else order
    )
    colours = _colors_for(adata, colour_key, levels, palette)

    common = dict(data=d, x=x, y=y, order=order, ax=ax)
    if hue is not None:
        common.update(hue=hue, hue_order=levels)
    sns.boxplot(**common, palette=colours, showfliers=False, boxprops=dict(alpha=0.5))
    sns.stripplot(**common, palette=colours, dodge=hue is not None, size=s / 3,
                  edgecolor="black", linewidth=0.3, legend=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    if hue is not None and ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)
    return order, levels


def _points(adata, d, *, x, y, palette, ax, ylabel, rotation):
    """One point per x position, with the posterior HDI as an error bar.

    The floor of the mark rule: nothing varies within an x position, so there is no
    distribution to draw. This is the only place a summary interval stands in for a sample,
    and it is drawn as a point-with-interval rather than a bar because a bar's area encodes a
    magnitude from zero that these metrics do not have.
    """
    d = d.sort_values(y, ascending=False)
    labels = d[x].astype(str).tolist()
    values = d[y].to_numpy(dtype=float)
    colours = _colors_for(adata, x, labels, palette)

    yerr = None
    if {"hdi_low", "hdi_high"} <= set(d.columns):
        lo = d["hdi_low"].to_numpy(dtype=float)
        hi = d["hdi_high"].to_numpy(dtype=float)
        if np.isfinite(lo).any():
            # asymmetric about the mean, so a single +/- would misstate it
            yerr = np.vstack([np.nan_to_num(values - lo), np.nan_to_num(hi - values)])

    pos = np.arange(len(d))
    if yerr is not None:
        ax.errorbar(pos, values, yerr=yerr, fmt="none", ecolor="0.3", elinewidth=0.9,
                    capsize=3)
    ax.scatter(pos, values, s=45, c=[colours[l] for l in labels], zorder=3)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    return ax


def render_metric(adata, name, *, ylabel, item_col=None, item_as_x=False, key=None,
                  order=None, hue_order=None, palette=None, ax=None, figsize=(8, 4),
                  save=None, show=None, return_df=False, annotate=True, rotation=90):
    """Draw a cached ``tl`` result. The axes come from its ``params``, not from arguments.

    ``item_as_x`` puts the metric's own item axis on x — right for clonotypic entropy, whose
    items are a handful of phenotypes, wrong for phenotypic entropy, whose items are every
    clone in the repertoire.
    """
    from .. import get as _get
    from ..tools._common import collapse_to_replicates

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
        return _finish(fig, _empty(ax, f"no data for {name}", ylabel), save=save, show=show)
    d = result.dropna(subset=["value"])
    if not len(d):
        return _finish(fig, _empty(ax, f"no finite {name}", ylabel), save=save, show=show)

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

    if unit == "replicate":
        # collapse items -> replicates with the SAME function build_stats uses, so the dots
        # and the p-value beneath them cannot describe different units. Everything about to be
        # drawn is preserved -- collapsing away the hue would silently drop the split.
        d = collapse_to_replicates(d, groupby=groupby,
                                   keep=[c for c in (x, hue) if c and c in d.columns])
        hue = hue if (hue and hue in d.columns) else None

    if unit in ("replicate", "item"):
        _order, _levels = _boxstrip(adata, d, x=x, y="value", hue=hue, order=order,
                                    hue_order=hue_order, palette=palette, ax=ax,
                                    ylabel=ylabel, rotation=0 if single else rotation)
    elif unit == "draw":
        keys = [c for c in table.columns if c not in ("draw", "value")]
        t = table.dropna(subset=["value"])
        if x in ("_all", "_one"):
            t = t.assign(**{x: d[x].iloc[0]})
        elif x not in t.columns:
            return _finish(fig, _empty(ax, f"no {x} axis in the draws", ylabel),
                           save=save, show=show)
        _order = _violins(adata, t, x=x, y="value", palette=palette, ax=ax, ylabel=ylabel,
                          rotation=0 if single else rotation, order=order)
    else:
        _points(adata, d, x=x, y="value", palette=palette, ax=ax, ylabel=ylabel,
                rotation=0 if single else rotation)
        _order = d[x].astype(str).tolist()

    if single or x == "_all":
        ax.set_xlabel("")
    if annotate and x == splitby:
        _annotate_contrasts(ax, stats, _order)
    return _finish(fig, ax, save=save, show=show)
