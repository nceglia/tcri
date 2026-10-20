"""Figures for the computing ``tcri.diag`` functions, drawn from the values each one just
computed. Nothing here computes a diagnostic; each function takes what its caller holds.

Shared conventions: ``ax`` is one Axes or a sequence with one Axes per panel, ``None`` makes the
figure; ``save`` is a path written at 150 dpi. The figure is never returned, so a caller's return
value does not depend on whether it plotted.
"""
from __future__ import annotations

import numpy as np

from ..plotting._base import _fit_ylabel


def _sizes_from(counts, labels, *, lo=25.0, hi=190.0):
    """Marker AREA proportional to each label's count, clamped to a legible band.

    `s` is already area in points^2, so proportionality is perceptually right; the clamp stops
    the largest count from swallowing the smallest at many times the area. Returns
    ``(sizes, counts)``, or ``(None, None)`` when no count is finite.
    """
    n = np.array([counts.get(label, np.nan) for label in labels], dtype=float)
    if not np.isfinite(n).any():
        return None, None
    lo_n, hi_n = np.nanmin(n), np.nanmax(n)
    if not np.isfinite(lo_n) or hi_n <= lo_n:
        return np.full(len(n), (lo + hi) / 2), n
    return lo + (hi - lo) * (n - lo_n) / (hi_n - lo_n), n

_LEGEND = dict(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8,
               title_fontsize=8)

#: A PPC panel holds at least this many clones. Below it the 95% band is wider than +/-0.56,
#: so the panel says almost nothing and one clone's ``p`` reads as a flat line; a sparse size bin
#: is merged into its smaller neighbor instead, so its clones stay in the figure. The
#: ``calibration`` frame keeps every bin.
_MIN_PANEL_CLONES = 5


def _panels(n, ax, *, ncols=4, panel_size=(4.0, 3.6), sharex=False, sharey=False):
    """``(figure, axes)`` for ``n`` panels: a new grid of at most ``ncols`` columns, or the
    caller's ``ax``, which must hold exactly ``n`` Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        ncols = max(1, min(ncols, n))
        nrows = int(np.ceil(n / ncols))
        fig, grid = plt.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey,
                                 layout="constrained",
                                 figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
        axes = list(grid.ravel())
        for unused in axes[n:]:
            unused.remove()
        return fig, axes[:n]
    axes = list(np.ravel(ax)) if np.iterable(ax) else [ax]
    if len(axes) != n:
        raise ValueError(
            f"this figure has {n} panel(s) and ax= holds {len(axes)} Axes; pass one Axes per "
            f"panel, or ax=None."
        )
    return axes[0].figure, axes


def _done(fig, axes, save):
    for a in axes:
        if a.get_visible():
            _fit_ylabel(a)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)


def _bin_label(lo, hi):
    if lo == hi:
        return "1 cell" if lo == 1 else f"{lo} cells"
    return f"{lo}-{hi} cells"


def _panel_bins(n_cells):
    """The PPC figure's size panels for one covariate: ``[(label, mask), ...]``, smallest first.

    Doubling bins as in ``calibration``, then every bin of fewer than ``_MIN_PANEL_CLONES`` clones
    merged into its smaller neighbor (the smallest bin into its larger one), largest first,
    until every panel holds enough clones or one panel is left.
    """
    n_cells = np.asarray(n_cells, dtype=np.int64)
    exp = np.floor(np.log2(np.maximum(n_cells, 1))).astype(np.int64)
    bins = [[2 ** int(e), 2 ** (int(e) + 1) - 1, exp == e] for e in np.unique(exp)]
    while len(bins) > 1:
        sparse = [j for j, b in enumerate(bins) if b[2].sum() < _MIN_PANEL_CLONES]
        if not sparse:
            break
        j = sparse[-1]
        k = j - 1 if j > 0 else 1
        lo, hi = sorted((j, k))
        bins[lo:hi + 1] = [[bins[lo][0], bins[hi][1], bins[lo][2] | bins[hi][2]]]
    return [(_bin_label(lo, hi), mask) for lo, hi, mask in bins]


def _quantile_steps(p):
    """The empirical quantile function of ``p`` as a step polyline ``(u, y)`` from u = 0 to 1.

    Its largest vertical distance from the diagonal is exactly the one-sample KS distance of ``p``
    from uniform: each step's two ends are the two terms of the KS maximum at that order
    statistic. So the polyline leaves the band ``|y - u| <= d`` exactly when KS exceeds ``d``.
    """
    p = np.sort(np.asarray(p, dtype=float))
    n = len(p)
    return np.repeat(np.arange(n + 1) / n, 2)[1:-1], np.repeat(p, 2)


def ppc_figure(result, *, save=None):
    """QQ of the PPC's ``p`` against uniform as small multiples: a row per covariate, a column
    per clone-size panel and one for all clones.

    Each panel draws every posterior draw's empirical quantile function of ``p`` as a faint step
    curve over a gray band, the exact 95% Kolmogorov-Smirnov range for that panel's clone count.
    A draw's curve leaves the band exactly when its KS distance from uniform exceeds the 95%
    critical value; for a panel that is one size bin, or all clones, that distance is the draw's
    ``ks`` in ``calibration``. Size bins of fewer than ``_MIN_PANEL_CLONES`` clones are merged
    into a neighbor (see :func:`_panel_bins`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import kstwo

    draws = result["draws"]
    covs = list(dict.fromkeys(draws["covariate"]))
    rows = []
    for m in covs:
        d = draws[draws["covariate"] == m]
        per_clone = d.groupby("clonotype", sort=False)["n_cells"].first()
        pmat = d.pivot(index="draw", columns="clonotype", values="p")[per_clone.index]
        panels = _panel_bins(per_clone.to_numpy())
        panels.append(("all clones", np.ones(len(per_clone), dtype=bool)))
        rows.append((m, pmat.to_numpy(), panels))

    n_cols = max([len(panels) for _, _, panels in rows], default=1)
    fig, axes = _panels(max(1, len(rows)) * n_cols, None, ncols=n_cols, sharex=True,
                        sharey=True, panel_size=(2.6, 2.6))
    u = np.linspace(0.0, 1.0, 201)
    for i, (m, pmat, panels) in enumerate(rows):
        for j in range(n_cols):
            a = axes[i * n_cols + j]
            if j >= len(panels):
                a.set_visible(False)
                continue
            label, mask = panels[j]
            n = int(mask.sum())
            band = kstwo.ppf(0.95, n)
            a.fill_between(u, np.clip(u - band, 0, 1), np.clip(u + band, 0, 1), color="0.88",
                           linewidth=0)
            steps = [np.column_stack(_quantile_steps(row[mask])) for row in pmat]
            alpha = float(np.clip(3.0 / max(1, len(steps)), 0.03, 0.6))
            a.add_collection(LineCollection(steps, colors="C0", alpha=alpha, linewidths=0.8))
            a.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=0.8)
            a.set(xlim=(0, 1), ylim=(0, 1), title=f"{m} - {label} ({n})")
            a.set_aspect("equal")
            a.title.set_fontsize(9)
            if i == len(rows) - 1:
                a.set_xlabel("uniform quantile")
            if j == 0:
                a.set_ylabel("quantile of p")
    if rows:
        axes[len(rows[0][2]) - 1].legend(handles=[Patch(color="0.88", label="95% range, correct fit"),
                                         Line2D([], [], color="C0", label="one posterior draw")],
                                **_LEGEND)
    _done(fig, axes, save)


def calibration_figure(result, *, ax=None, save=None):
    """Reliability diagram: mean predicted probability against observed accuracy per bin, the
    marker area scaled by the bin's cell count, the diagonal as reference."""
    fig, axes = _panels(1, ax, panel_size=(4.2, 4.0))
    a = axes[0]
    df, ece = result["bins"], float(result["summary"]["ece"].iloc[0])
    d = df[df["count"] > 0]
    counts = d.set_index("bin")["count"]
    sizes, _ = _sizes_from(counts, list(counts.index))
    a.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    a.scatter(d["mean_pred"], d["emp_freq"], s=sizes, color="C0", edgecolor="black",
              linewidth=0.3, zorder=3)
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability",
          ylabel="observed accuracy", title=f"Calibration - ECE {ece:.3f}")
    a.set_aspect("equal")
    if len(counts):
        lut = dict(zip(counts.to_numpy(), sizes))
        picks = np.unique(np.percentile(counts.to_numpy(), [0, 50, 100]).round().astype(int))
        nearest = [counts.to_numpy()[np.abs(counts.to_numpy() - v).argmin()] for v in picks]
        handles = [a.scatter([], [], s=lut[v], c="C0", edgecolor="black", linewidth=0.3,
                             label=str(int(v))) for v in dict.fromkeys(nearest)]
        a.legend(handles=handles, title="cells", labelspacing=1.1, **_LEGEND)
    _done(fig, axes, save)


#: Ratios a reconstruction axis may label, on both sides of 1.
_RATIO_TICKS = (1.1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)


def _ratio_ticks(half):
    """Labeled ticks for a log ratio axis spanning ``10**-half`` to ``10**half``: 1 and the
    ratios of ``_RATIO_TICKS`` and their reciprocals inside it, at least a quarter of the
    half-range apart in log, so a ratio near 1 can be read on a narrow axis and labels never
    crowd on a wide one."""
    gap = half / 4.0
    up, last = [], 0.0
    for r in _RATIO_TICKS:
        x = np.log10(r)
        if x > half:
            break
        if x - last >= gap:
            up.append(r)
            last = x
    return sorted([1.0, *up, *(1.0 / r for r in up)])


def reconstruction_figure(df, per_sim, *, ax=None, save=None):
    """Simulated / observed for each statistic on a log axis centered on 1, so a ratio and its
    reciprocal sit equally far from the line: one dot per simulation and the mean as a bar. A
    statistic whose observed value is 0 has no ratio; its column says so. A simulation whose
    statistic is 0 is left out, as a log axis cannot place it."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = _panels(1, ax, panel_size=(5.5, 3.8))
    a = axes[0]
    jitter = np.random.default_rng(0)
    spread = [0.0]
    empty = []
    for i, row in enumerate(df.itertuples(index=False)):
        ratio = (np.asarray(per_sim[row.statistic], dtype=float) / row.observed
                 if np.isfinite(row.observed) and row.observed != 0 else np.array([]))
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if not ratio.size:
            empty.append(i)
            continue
        x = i + jitter.uniform(-0.15, 0.15, len(ratio))
        a.scatter(x, ratio, s=12, color="C0", alpha=0.6, linewidth=0)
        a.hlines(ratio.mean(), i - 0.3, i + 0.3, color="C0", linewidth=2)
        spread.append(float(np.abs(np.log10(ratio)).max()))
    a.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    a.set_yscale("log")
    half = max(max(spread) * 1.15, np.log10(1.3))
    a.set_ylim(10 ** -half, 10 ** half)
    a.yaxis.set_major_locator(FixedLocator(_ratio_ticks(half)))
    a.yaxis.set_minor_locator(NullLocator())
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
    for i in empty:
        a.text(i, 1.0, "observed 0:\nno ratio", ha="center", va="center", fontsize=8,
               color="0.35", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    a.set_xticks(range(len(df)))
    a.set_xticklabels([str(s).replace("_", " ") for s in df["statistic"]], rotation=30,
                      ha="right")
    a.set(xlim=(-0.6, len(df) - 0.4), ylabel="simulated / observed", title="Reconstruction")
    _done(fig, axes, save)


def permutation_null_figure(df, nulls, *, groupby=None, ax=None, save=None):
    """Per returned row, a histogram of the permuted values with the observed value marked."""
    fig, axes = _panels(max(1, len(df)), ax, sharex=True)
    ncols = axes[0].get_gridspec().ncols if ax is None and len(df) else 1
    for k, (a, row, null) in enumerate(zip(axes, df.to_dict("records"), nulls)):
        a.hist(null, bins=30, color="0.75", label="permuted labels")
        a.axvline(row["observed"], color="C3", linewidth=2, label="observed")
        title = f"Mutual information - {row['covariate']}"
        if groupby is not None:
            title += f" - {row[groupby]}"
        a.set(xlabel="mutual information", title=title)
        if ax is not None or k % ncols == 0:
            a.set_ylabel("permutations")
    if len(df):
        if ax is None:
            axes[min(ncols, len(axes)) - 1].legend(**_LEGEND)
        else:
            axes[0].legend(fontsize=8, frameon=False)
    _done(fig, axes, save)
