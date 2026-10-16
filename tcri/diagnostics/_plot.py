"""Figures for the computing ``tcri.diag`` functions, drawn from the values each one just
computed. Nothing here computes a diagnostic; each function takes what its caller holds.

Shared conventions: ``ax`` is one Axes or a sequence with one Axes per panel, ``None`` makes the
figure; ``save`` is a path written at 150 dpi. The figure is never returned, so a caller's return
value does not depend on whether it plotted.
"""
from __future__ import annotations

import numpy as np

from ..plotting._base import _fit_ylabel, _sizes_from

_LEGEND = dict(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8,
               title_fontsize=8)


def _panels(n, ax, *, ncols=4, panel_size=(4.0, 3.6), sharex=False, sharey=False):
    """``(figure, axes)`` for ``n`` panels: a new grid of at most ``ncols`` columns, or the
    caller's ``ax``, which must hold exactly ``n`` Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        ncols = max(1, min(ncols, n))
        nrows = int(np.ceil(n / ncols))
        fig, grid = plt.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey,
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
        _fit_ylabel(a)
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)


def ppc_figure(result, *, ax=None, save=None):
    """QQ of the PPC's ``p`` against uniform as small multiples: a row per covariate, a column
    per clone-size bin and one for all clones.

    Each panel draws the median quantile curve over draws and its 2.5-97.5% band over draws,
    over a grey band: the exact 95% Kolmogorov-Smirnov range for that panel's clone count, where
    a correct fit's curve stays. A curve leaving it is the same event as that bin's ``ks`` in
    the calibration frame exceeding its 95% critical value. The band is wide for a bin of a few
    clones and narrow for a bin of many, so no bin is hidden and none is over-read. A
    (covariate, bin) with no clones is left blank.
    """
    from scipy.stats import kstwo

    from ._ppc import _size_bins

    draws = result["draws"]
    covs = list(dict.fromkeys(draws["covariate"]))
    per_cov = {}
    bins = []
    for m in covs:
        d = draws[draws["covariate"] == m]
        per_clone = d.groupby("clonotype", sort=False)["n_cells"].first()
        labels, present = _size_bins(per_clone.to_numpy())
        per_cov[m] = (d, per_clone, labels)
        bins.extend(b for b in present if b not in bins)
    bins = sorted(bins, key=lambda b: int(b.split("-")[0])) + ["all"]

    n_rows, n_cols = max(1, len(covs)), len(bins)
    fig, axes = _panels(n_rows * n_cols, ax, ncols=n_cols, sharex=True, sharey=True,
                        panel_size=(2.6, 2.6))
    grid = np.linspace(0.0, 1.0, 101)
    for i, m in enumerate(covs):
        d, per_clone, labels = per_cov[m]
        pmat = d.pivot(index="draw", columns="clonotype", values="p")[per_clone.index]
        for j, b in enumerate(bins):
            a = axes[i * n_cols + j]
            keep = per_clone.index[(labels == b) | (b == "all")]
            if not len(keep):
                a.set_visible(False)
                continue
            band = kstwo.ppf(0.95, len(keep))
            a.fill_between(grid, np.clip(grid - band, 0, 1), np.clip(grid + band, 0, 1),
                           color="0.88", linewidth=0, label="95% range, correct fit")
            curves = np.quantile(pmat[keep].to_numpy(), grid, axis=1).T      # [draws, grid]
            lo, mid, hi = np.percentile(curves, [2.5, 50.0, 97.5], axis=0)
            a.fill_between(grid, lo, hi, color="C0", alpha=0.3, linewidth=0,
                           label="2.5-97.5% over draws")
            a.plot(grid, mid, color="C0", linewidth=1.3, label="median over draws")
            a.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=0.8)
            cells = {"all": "all clones", "1": "1 cell"}.get(b, f"{b} cells")
            a.set(xlim=(0, 1), ylim=(0, 1), title=f"{m} - {cells} ({len(keep)})")
            a.set_aspect("equal")
            a.title.set_fontsize(9)
            if i == n_rows - 1:
                a.set_xlabel("uniform quantile")
            if j == 0:
                a.set_ylabel("quantile of p")
    if covs:
        axes[n_cols - 1].legend(**_LEGEND)
    _done(fig, axes, save)


def calibration_figure(df, *, ax=None, save=None):
    """Reliability diagram: mean predicted probability against observed accuracy per bin, the
    marker area scaled by the bin's cell count, the diagonal as reference."""
    import matplotlib.pyplot as plt

    fig, axes = _panels(1, ax, panel_size=(4.2, 4.0))
    a = axes[0]
    d = df[df["count"] > 0]
    counts = d.set_index("bin")["count"]
    sizes, _ = _sizes_from(counts, list(counts.index))
    a.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    a.scatter(d["mean_pred"], d["emp_freq"], s=sizes, color="C0", edgecolor="black",
              linewidth=0.3, zorder=3)
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability",
          ylabel="observed accuracy", title=f"Calibration - ECE {df.attrs['ECE']:.3f}")
    a.set_aspect("equal")
    if len(counts):
        lut = dict(zip(counts.to_numpy(), sizes))
        picks = np.unique(np.percentile(counts.to_numpy(), [0, 50, 100]).round().astype(int))
        nearest = [counts.to_numpy()[np.abs(counts.to_numpy() - v).argmin()] for v in picks]
        handles = [a.scatter([], [], s=lut[v], c="C0", edgecolor="black", linewidth=0.3,
                               label=str(int(v))) for v in dict.fromkeys(nearest)]
        a.legend(handles=handles, title="cells", labelspacing=1.1, **_LEGEND)
    _done(fig, axes, save)


def reconstruction_figure(df, per_sim, *, ax=None, save=None):
    """Simulated / observed for each statistic on a log axis centered on 1, so a ratio and its
    reciprocal sit equally far from the line: one dot per simulation and the mean as a bar. A
    statistic whose observed value is 0 has no ratio and is left out, as is a simulation whose
    statistic is 0, which a log axis cannot place."""
    from matplotlib.ticker import FuncFormatter

    fig, axes = _panels(1, ax, panel_size=(5.5, 3.8))
    a = axes[0]
    jitter = np.random.default_rng(0)
    spread = [0.0]
    for i, row in enumerate(df.itertuples(index=False)):
        if not np.isfinite(row.observed) or row.observed == 0:
            continue
        ratio = np.asarray(per_sim[row.statistic], dtype=float) / row.observed
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if not ratio.size:
            continue
        x = i + jitter.uniform(-0.15, 0.15, len(ratio))
        a.scatter(x, ratio, s=12, color="C0", alpha=0.6, linewidth=0)
        a.hlines(ratio.mean(), i - 0.3, i + 0.3, color="C0", linewidth=2)
        spread.append(float(np.abs(np.log10(ratio)).max()))
    a.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    a.set_yscale("log")
    half = max(max(spread) * 1.15, np.log10(1.5))
    a.set_ylim(10 ** -half, 10 ** half)
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xticks(range(len(df)))
    a.set_xticklabels([str(s).replace("_", " ") for s in df["statistic"]], rotation=30,
                      ha="right")
    a.set(xlim=(-0.6, len(df) - 0.4), ylabel="simulated / observed", title="Reconstruction")
    _done(fig, axes, save)


def permutation_null_figure(df, nulls, *, groupby=None, ax=None, save=None):
    """Per returned row, a histogram of the permuted values with the observed value marked."""
    fig, axes = _panels(max(1, len(df)), ax, sharex=True)
    for a, row, null in zip(axes, df.to_dict("records"), nulls):
        a.hist(null, bins=30, color="0.75")
        a.axvline(row["observed"], color="C3", linewidth=2)
        title = f"Mutual information - {row['covariate']}"
        if groupby is not None:
            title += f" - {row[groupby]}"
        a.set(xlabel="mutual information", ylabel="permutations", title=title)
    _done(fig, axes, save)
