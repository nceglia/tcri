"""Shared statistics helpers.

The significance helpers plus the posterior-summary primitive the metric layer needs (a true
HDI).

``hdi`` is the interval primitive the metric layer uses.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def stars(p: float) -> str:
    """Significance stars for a p-value; blank for an undefined one.

    A NaN p is a test that was not run, so it gets no marker at all rather than ``"ns"``, which
    reads as a test that was run and found nothing.
    """
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mann_whitney(a, b):
    """Mann–Whitney U + two-sided p (thin wrapper for a single import site)."""
    return mannwhitneyu(np.asarray(a, float), np.asarray(b, float), alternative="two-sided")


def signed_rank(differences):
    """Wilcoxon signed-rank W + two-sided p on paired differences (thin wrapper for a single
    import site). Zero differences are left out, scipy's default."""
    return wilcoxon(np.asarray(differences, float), alternative="two-sided")


# ── posterior-summary primitive (metric layer) ───────────────────────────────
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
