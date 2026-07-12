"""Shared reduction helpers for the ``tl`` metrics.

Each metric pulls the clone×phenotype joint from the engine (:func:`joint_distribution`,
``use_logits=True``), reduces per draw, and — for ``n_samples>0`` — summarizes the draw
distribution (mean / sd / HDI). ``groupby`` is implemented here by **restricting clones
per group** (§7.1: full-space clone masks + ``clones=``, never slicing the AnnData), which
relies on clones being disjoint across groups (a TCR clone never spans two patients).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._stats import hdi
from ._joint import joint_distribution


def is_precomputed_joint(x) -> bool:
    """A precomputed joint (fast path, §7.9) is a plain DataFrame, not an AnnData."""
    return isinstance(x, pd.DataFrame)


def joint_draws(adata, covariate, *, n_samples, weighted, temperature, clones, random_state,
                use_logits=True):
    """Return ``(draws, phenotype_cols)`` where ``draws`` is a list of ``(clone_ids, [C, P])``
    per posterior draw (length 1 for ``n_samples=0``)."""
    jd = joint_distribution(
        adata, covariate=covariate, use_logits=use_logits, n_samples=n_samples,
        weighted=weighted, temperature=temperature, clones=clones, random_state=random_state,
    )
    cols = list(jd.columns)
    if n_samples and int(n_samples) > 0:
        draws = []
        for _sid, sub in jd.groupby(level="sample_id", sort=True):
            sub = sub.droplevel("sample_id")
            draws.append((list(sub.index), sub.values.astype(float)))
        return draws, cols
    return [(list(jd.index), jd.values.astype(float))], cols


def summarize(values, *, hdi_prob=0.94) -> dict:
    """Summarize a 1-D array of per-draw metric values → mean / sd / hdi_low / hdi_high."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": np.nan, "sd": np.nan, "hdi_low": np.nan, "hdi_high": np.nan}
    if v.size == 1:
        return {"mean": float(v[0]), "sd": 0.0, "hdi_low": float(v[0]), "hdi_high": float(v[0])}
    lo, hi = hdi(v, prob=hdi_prob)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "hdi_low": lo, "hdi_high": hi}


def clone_col(adata):
    from .. import _keys as K
    return adata.uns[K.METADATA]["clone_col"]


def grouped_scalar(adata, *, groupby, splitby, value, compute, hdi_prob=0.94):
    """Loop over ``adata.obs[groupby]`` values, restrict to each group's clones, compute a
    scalar-valued metric per group, and tidy into a DataFrame with the ``splitby`` label.

    ``compute(clones) -> (point, draws_or_None)``: ``point`` is the ``n_samples=0`` scalar (or
    the draw mean), ``draws`` is the per-draw vector (or ``None`` at ``n_samples=0``).
    """
    cc = clone_col(adata)
    obs = adata.obs
    rows = []
    for g in obs[groupby].dropna().unique().tolist():
        gmask = obs[groupby] == g
        clones_g = obs.loc[gmask, cc].dropna().unique().tolist()
        point, draws = compute(clones_g)
        row = {groupby: g, value: point}
        if splitby is not None and splitby in obs.columns:
            row[splitby] = obs.loc[gmask, splitby].iloc[0]
        if draws is not None:
            row.update(summarize(draws, hdi_prob=hdi_prob))
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_series(adata, *, groupby, splitby, item_name, value, compute, hdi_prob=0.94):
    """Like :func:`grouped_scalar` but the metric is a per-item (phenotype/clone) map.

    ``compute(clones) -> (point, draws_or_None)``: ``point`` is ``{item: value}``;
    ``draws`` is ``{item: [per-draw values]}`` (or ``None`` at ``n_samples=0``). Tidies to
    one row per (group, item).
    """
    cc = clone_col(adata)
    obs = adata.obs
    rows = []
    for g in obs[groupby].dropna().unique().tolist():
        gmask = obs[groupby] == g
        clones_g = obs.loc[gmask, cc].dropna().unique().tolist()
        point, draws = compute(clones_g)
        split_val = obs.loc[gmask, splitby].iloc[0] if (splitby and splitby in obs.columns) else None
        for item, val in point.items():
            row = {groupby: g, item_name: item, value: val}
            if split_val is not None:
                row[splitby] = split_val
            if draws is not None and item in draws:
                row.update(summarize(draws[item], hdi_prob=hdi_prob))
            rows.append(row)
    return pd.DataFrame(rows)
