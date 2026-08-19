"""Shared statistics helpers.

The significance / AUROC helpers plus the posterior-summary primitives the metric layer
needs (true HDI, signed-direction probability).

``auc_and_label_permutation`` and ``bootstrap_auc`` are re-exported as ``tcri.ut.*``. They
are complementary -- a permutation p-value against shuffled labels, and a bootstrap CI on
the AUC itself -- and were reachable only through this private module, which is why nothing
called them.

``eti`` (equal-tailed interval) is gone: it had no caller outside its own unit test, and the
package reports HDIs. ``hdi`` is the primitive everything actually uses.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import mannwhitneyu, rankdata
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
    """Observed AUROC + a label-permutation p-value (exact when feasible).

    Under label permutation the *scores* never change, so the ranks are computed
    ONCE and each permuted AUROC is a rank-sum over the permuted positive set via
    the Mann–Whitney identity

        AUC = (Σ ranks[pos] − n_pos(n_pos+1)/2) / (n_pos·n_neg)

    which is exact (midranks reproduce ``roc_auc_score``'s tie handling) and turns
    each draw from an O(n log n) re-sort into an O(n_pos) sum. Measured 137× on the
    Monte-Carlo path (191 s → 1.4 s at the default ``n_perm``).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if pos_label is None:
        pos_label = sorted(set(labels))[-1]
    y = (labels == pos_label).astype(int)
    obs_auc = roc_auc_score(y, scores)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:  # AUROC undefined; keep the old failure mode
        perm_stats = np.array([])
        return obs_auc, float("nan"), perm_stats, "degenerate"

    # midranks: ties get the average rank, matching roc_auc_score exactly
    ranks = rankdata(scores)
    denom = float(n_pos) * float(n_neg)
    offset = n_pos * (n_pos + 1) / 2.0

    n_exact = math.comb(n, n_pos)
    if n_exact <= max_exact:
        perm_stats = np.fromiter(
            ((ranks[list(idx)].sum() - offset) / denom
             for idx in itertools.combinations(range(n), n_pos)),
            dtype=float, count=n_exact,
        )
        perm_mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        perm_stats = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            # a random size-n_pos subset of ranks == a random label permutation
            perm_stats[i] = (rng.permutation(ranks)[:n_pos].sum() - offset) / denom
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
