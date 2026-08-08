"""``tl.compare_groups`` (§7.6) — the public group-comparison orchestrator that replaces
the deleted ``mi_compare`` / ``delta_entropy_table`` / ``flux_table``.

Turns a tidy ``groupby`` result (per-unit point estimates, e.g. per patient) into group
contrasts. Unpaired: Mann–Whitney U + two-sided p + Δ of means. Paired (``paired=True``):
per-unit posterior-draw vectors aligned by ``pair_on`` → signed Δ draws, HDI, and the
direction probability ``p_gt`` via ``prob_direction`` (the only place a direction prob is
emitted).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .._stats import hdi, mann_whitney, prob_direction, stars

__all__ = ["compare_groups"]


def compare_groups(df, *, value, splitby, reference=None, paired=False, pair_on=None,
                   hdi_prob=0.94, alternative="two-sided"):
    """Contrast a metric across cohorts, with direction probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        A grouped metric table (e.g. from ``mutual_information(..., groupby=...)``) with a
        ``value`` column and a ``splitby`` cohort column.
    value : str
        The metric column to contrast (e.g. ``"MI"`` for mutual information).
    splitby : str
        The cohort column whose levels are contrasted (e.g. response).
    reference : str | None
        Contrast this level against every other; ``None`` contrasts all pairs.
    paired : bool
        Align draws per unit (``pair_on``) and contrast the paired differences.
    pair_on : str | None
        The per-unit id used to align draws when ``paired=True``.
    hdi_prob : float
        Mass of the reported highest-density interval on the contrast.
    alternative : {"two-sided", "greater", "less"}
        Direction the reported probability is evaluated against.

    Returns
    -------
    pandas.DataFrame
        One row per contrast, with the mean delta, direction probability, and HDI.
    """
    levels = [g for g in df[splitby].dropna().unique().tolist()]
    if reference is not None:
        if reference not in levels:
            raise ValueError(f"reference {reference!r} not in {splitby} levels {levels}")
        pairs = [(reference, g) for g in levels if g != reference]
    else:
        pairs = list(itertools.combinations(sorted(levels, key=str), 2))

    rows = []
    for a, b in pairs:
        da = df[df[splitby] == a]
        db = df[df[splitby] == b]
        if paired:
            if pair_on is None:
                raise ValueError("paired=True requires pair_on (the per-unit id to align draws on).")
            sa = da.set_index(pair_on)[value]
            sb = db.set_index(pair_on)[value]
            diffs = []
            for u in sa.index.intersection(sb.index):
                va = np.asarray(sa[u], dtype=float).ravel()
                vb = np.asarray(sb[u], dtype=float).ravel()
                n = min(va.size, vb.size)
                if n:
                    diffs.append(vb[:n] - va[:n])
            delta = np.concatenate(diffs) if diffs else np.array([])
            if delta.size == 0:
                continue
            p_gt, p_lt = prob_direction(delta)
            lo, hi = hdi(delta, prob=hdi_prob) if delta.size > 1 else (float(delta[0]), float(delta[0]))
            rows.append(_row(a, b, delta=float(np.mean(delta)), p_gt=float(p_gt),
                             p_lt=float(p_lt), hdi_low=lo, hdi_high=hi))
        else:
            va = pd.to_numeric(da[value], errors="coerce").dropna().to_numpy()
            vb = pd.to_numeric(db[value], errors="coerce").dropna().to_numpy()
            if va.size == 0 or vb.size == 0:
                continue
            try:
                U, p = mann_whitney(va, vb, alternative=alternative)
            except ValueError:  # e.g. all-identical values
                U, p = np.nan, np.nan
            rows.append(_row(a, b, mean_a=float(np.mean(va)), mean_b=float(np.mean(vb)),
                             delta=float(np.mean(vb) - np.mean(va)), U=float(U), p=float(p),
                             stars=stars(p)))
    return pd.DataFrame(rows, columns=_COLUMNS)


_COLUMNS = ["group_a", "group_b", "mean_a", "mean_b", "delta", "U", "p", "stars",
            "p_gt", "p_lt", "hdi_low", "hdi_high"]


def _row(a, b, **vals):
    """A contrast row with the unified §7.6 schema; inapplicable stats are NaN."""
    row = {c: np.nan for c in _COLUMNS}
    row["group_a"], row["group_b"] = a, b
    row.update(vals)
    return row
