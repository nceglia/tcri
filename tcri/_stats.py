"""Shared statistics helpers.

Consolidates the significance / AUROC helpers that lived in ``utils`` (used by
plotting) and adds the posterior-summary primitives the metric layer needs
(true HDI, equal-tailed interval, signed-direction probability). The metric and
plotting layers import from here; ``utils`` will drop its copies during adoption.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


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


def auc_and_label_permutation(scores, labels, pos_label=None,
                              n_perm=200_000, seed=42, max_exact=200_000):
    """Observed AUROC + a label-permutation p-value (exact when feasible)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if pos_label is None:
        pos_label = sorted(set(labels))[-1]
    y = (labels == pos_label).astype(int)
    obs_auc = roc_auc_score(y, scores)
    n_pos = int(y.sum())
    n_exact = math.comb(len(y), n_pos)
    if n_exact <= max_exact:
        perm_stats = np.array([
            roc_auc_score(np.isin(np.arange(len(y)), idx).astype(int), scores)
            for idx in itertools.combinations(range(len(y)), n_pos)
        ])
        perm_mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        perm_stats = np.array([
            roc_auc_score(rng.permutation(y), scores) for _ in range(n_perm)
        ])
        perm_mode = "mc"
    p_perm = np.mean(np.abs(perm_stats - 0.5) >= np.abs(obs_auc - 0.5))
    return obs_auc, p_perm, perm_stats, perm_mode


def bootstrap_auc(scores, labels, pos_label=None, n_boot=5000, seed=42):
    """Bootstrap 95% CI for AUROC."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if pos_label is None:
        pos_label = sorted(set(labels))[-1]
    y = (labels == pos_label).astype(int)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    while len(aucs) < n_boot:
        samp = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[samp])) < 2:
            continue
        aucs.append(roc_auc_score(y[samp], scores[samp]))
    return np.quantile(aucs, [0.025, 0.975])


# ── posterior-summary primitives (metric layer) ──────────────────────────────
def eti(samples, *, prob: float = 0.94):
    """Equal-tailed credible interval (percentile). Stable, transform-invariant."""
    s = np.asarray(samples, float)
    lo = (1.0 - prob) / 2.0
    return tuple(np.percentile(s, [100 * lo, 100 * (1 - lo)]))


def hdi(samples, *, prob: float = 0.94):
    """True highest-density interval: the *narrowest* window holding ``prob`` mass.

    Sounder than ``eti`` for the bounded, skewed entropy/flux posteriors, but
    noisier from few draws near a boundary (use ``n_samples ≳ 500`` when tight).
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
