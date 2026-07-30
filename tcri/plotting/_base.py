"""``pl`` private plotting engine (§8.5) — the generic tidy-DataFrame box/strip renderer
shared by the metric twins, and the save/show finisher.

The twins are **cache renderers**: they call the ``tl`` metric (which already does its own
``groupby`` via full-space ``clones=`` restriction), get a tidy DataFrame, and hand it here.
Nothing here slices the AnnData or re-computes metric math (fixes the old ``tcri_boxplot``
slice-and-call pattern that tripped the engine alignment guard).
"""
from __future__ import annotations

import matplotlib.pyplot as plt

__all__: list[str] = []  # private module


def _finish(fig, ax, *, save=None, show=None):
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return ax


def _metric_boxplot(df, *, x, y, hue=None, order=None, hue_order=None, palette=None,
                    ax=None, figsize=(8, 4), s=20, ylabel=None, rotation=90):
    """Box + strip of a tidy metric DataFrame: box over the ``x`` categories, optional
    ``hue`` split, individual units as strip dots. Returns ``(fig, ax)``."""
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if df is None or len(df) == 0 or y not in getattr(df, "columns", []):
        ax.text(0.5, 0.5, f"no data for {y}", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylabel if ylabel is not None else y)
        return fig, ax
    d = df.dropna(subset=[y])
    if len(d) == 0:
        ax.text(0.5, 0.5, f"no finite {y}", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylabel if ylabel is not None else y)
        return fig, ax
    if order is None and x in d:
        order = (d.groupby(x)[y].median().sort_values(ascending=False).index.tolist())

    common = dict(data=d, x=x, y=y, order=order, ax=ax)
    if hue is not None and hue in d.columns:
        common.update(hue=hue, hue_order=hue_order)
    sns.boxplot(**common, palette=palette, showfliers=False,
                boxprops=dict(alpha=0.5))
    sns.stripplot(**common, palette=palette, dodge=hue is not None, size=s / 3,
                  edgecolor="black", linewidth=0.3)
    ax.set_ylabel(ylabel if ylabel is not None else y)
    ax.set_xlabel(x)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    if hue is not None and ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)
    return fig, ax


def _barplot(series, *, ylabel, palette=None, ax=None, figsize=(8, 4), rotation=90):
    """Fallback for a no-groupby metric (a Series over phenotypes/clones): a simple bar."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    s = series.dropna()
    ax.bar(range(len(s)), s.values,
           color=(palette if isinstance(palette, str) else None) or tcri_bar_color)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(list(s.index), rotation=rotation)
    ax.set_ylabel(ylabel)
    return fig, ax


tcri_bar_color = "#66D9EF"
