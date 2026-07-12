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
    """Contrast ``value`` across ``df[splitby]`` groups. Returns a tidy DataFrame, one row
    per contrast (``reference`` vs each other level, or all pairs when ``reference`` is None)."""
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
            p_gt, _p_lt = prob_direction(delta)
            lo, hi = hdi(delta, prob=hdi_prob) if delta.size > 1 else (float(delta[0]), float(delta[0]))
            rows.append({"group_a": a, "group_b": b, "delta": float(np.mean(delta)),
                         "p_gt": float(p_gt), "hdi_low": lo, "hdi_high": hi})
        else:
            va = pd.to_numeric(da[value], errors="coerce").dropna().to_numpy()
            vb = pd.to_numeric(db[value], errors="coerce").dropna().to_numpy()
            if va.size == 0 or vb.size == 0:
                continue
            try:
                U, p = mann_whitney(va, vb, alternative=alternative)
            except ValueError:  # e.g. all-identical values
                U, p = np.nan, np.nan
            delta = float(np.mean(vb) - np.mean(va))
            rows.append({"group_a": a, "group_b": b, "mean_a": float(np.mean(va)),
                         "mean_b": float(np.mean(vb)), "delta": delta, "U": float(U),
                         "p": float(p), "stars": stars(p)})
    return pd.DataFrame(rows)
