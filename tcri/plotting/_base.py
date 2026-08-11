"""``pl`` private plotting engine (§8.5) — the shared cache renderer and the save/show finisher.

The twins are **cache renderers** in the strict sense: they read ``uns`` through
:mod:`tcri.get` and draw. There is no metric math here and no call into ``tl``.

That matters more than it sounds. When each ``pl`` recomputed its metric, the plot and the
frame in the caller's hand could disagree — different ``n_samples``, a different draw, a
``distance_metric`` default that was ``"kl"`` in ``tl`` and ``"l1"`` in ``pl``. And because
a box plot needs per-unit values, ``pl.mutual_information`` and ``pl.phenotypic_flux``
*manufactured* a ``groupby`` from ``batch_col`` when the caller gave none, so the figure
was grouped by a column the caller never named. Reading the cache removes the whole class:
the axes are whatever the ``tl`` call actually used, recovered from its ``params``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._colors import resolve_colors

__all__: list[str] = []  # private module

tcri_bar_color = "#66D9EF"


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

    Only drawn when the x axis IS the split — annotating a contrast over an axis it was not
    computed on is how a figure comes to claim something the numbers never said.
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
            ax.plot([xa, xa, xb, xb], [y, y + step * 0.2, y + step * 0.2, y],
                    lw=0.9, c="0.3", clip_on=False)
            ax.text((xa + xb) / 2, y + step * 0.25, text, ha="center", va="bottom",
                    fontsize=9, c="0.2", clip_on=False)
            drawn += 1
    if drawn:
        ax.set_ylim(ymin, ymax + step * (drawn + 0.9))


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


def _bars(adata, d, *, x, y, palette, ax, ylabel, rotation):
    """One bar per row, with the posterior HDI as an error bar when the metric has draws."""
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

    ax.bar(range(len(d)), values, yerr=yerr, capsize=2 if yerr is not None else 0,
           color=[colours[l] for l in labels],
           error_kw=dict(lw=0.8, ecolor="0.3"))
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

    payload = _get.result(adata, name, key=key)
    params = _get.params(adata, name, key=key)
    result = payload["result"]
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
    if item_as_x and item_col in d.columns and has_groups:
        x, hue = item_col, (splitby if splitby in d.columns else None)
    elif item_as_x and item_col in d.columns:
        # one row per item and nothing to box over: bar them, with the posterior HDI as the
        # error bar. Boxing a single point per category draws a flat line and hides the
        # interval that IS the uncertainty here.
        _bars(adata, d, x=item_col, y="value", palette=palette, ax=ax, ylabel=ylabel,
              rotation=rotation)
        return _finish(fig, ax, save=save, show=show)
    elif splitby is not None and splitby in d.columns:
        x, hue = splitby, None
    elif groupby is not None and groupby in d.columns:
        x, hue = groupby, None
    elif item_col is not None and item_col in d.columns:
        # no groups: the items ARE the observations. Bar them when they are few and named
        # (phenotypes); box the distribution when they are the whole repertoire.
        if item_as_x or d[item_col].nunique() <= 30:
            _bars(adata, d, x=item_col, y="value", palette=palette, ax=ax, ylabel=ylabel,
                  rotation=rotation)
            return _finish(fig, ax, save=save, show=show)
        d = d.assign(_all="all")
        x, hue = "_all", None
    else:
        _bars(adata, d.assign(_one=str(params.get("covariate") or name)), x="_one", y="value",
              palette=palette, ax=ax, ylabel=ylabel, rotation=0)
        return _finish(fig, ax, save=save, show=show)

    if hue is not None and hue not in d.columns:
        hue = None
    _order, _levels = _boxstrip(adata, d, x=x, y="value", hue=hue, order=order,
                                hue_order=hue_order, palette=palette, ax=ax, ylabel=ylabel,
                                rotation=rotation)
    if annotate and x == splitby:
        _annotate_contrasts(ax, stats, _order)
    return _finish(fig, ax, save=save, show=show)
