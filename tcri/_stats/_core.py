"""Shared statistics helpers.

The significance helpers plus the posterior-summary primitives the metric layer needs
(true HDI, signed-direction probability).

``hdi`` is the interval primitive the metric layer uses.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def stars(p: float) -> str:
    """Significance stars for a p-value."""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mann_whitney(a, b, *, alternative: str = "two-sided"):
    """Mann–Whitney U + two-sided p (thin wrapper for a single import site)."""
    return mannwhitneyu(np.asarray(a, float), np.asarray(b, float), alternative=alternative)


# ── posterior-summary primitives (metric layer) ──────────────────────────────
def hdi(samples, *, prob: float = 0.94):
    """True highest-density interval: the *narrowest* window holding ``prob`` mass.

    Sounder than an equal-tailed interval for the bounded, skewed entropy/flux
    posteriors, but noisier from few draws near a boundary (use ``n_samples ≳ 500``
    when tight).
    """
    s = np.sort(np.asarray(samples, float))
    n = s.size
    if n == 0:
        return (np.nan, np.nan)
    inc = max(1, int(np.floor(prob * n)))   # points spanned by the interval
    if inc >= n:
        return (float(s[0]), float(s[-1]))
    widths = s[inc:] - s[:n - inc]          # width of every inc-spanning window
    i = int(np.argmin(widths))
    return (float(s[i]), float(s[i + inc]))


def prob_direction(delta):
    """Signed-contrast probabilities for a difference-draw vector."""
    d = np.asarray(delta, float)
    p_gt = float((d > 0).mean())
    return p_gt, 1.0 - p_gt
