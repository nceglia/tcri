"""I(c; phi) — mutual information between clonotype and phenotype.

METRICS eq 5 (MI) and eq 6 (NMI). See ``tcri/tools/_metrics_contract.py`` for the frozen
definitions; this module computes them and stores the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from ._common import (
    build_result,
    build_stats,
    is_precomputed_joint,
    joint_draws,
    reject_stacked_covariate_joint,
    resolve_groupby,
    validate_splitby,
)

__all__ = ["mutual_information"]

_EPS = 1e-12


def _mi_from_joint(J: np.ndarray, *, normalized: bool = True, mode: str = "min") -> float:
    """MI (bits) of a clone×phenotype table ``J`` (any scale — renormalized here)."""
    J = np.asarray(J, dtype=np.float64)
    total = J.sum()
    if total <= 0:
        return np.nan
    pxy = J / total
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = pxy * np.log2(pxy / (px * py + _EPS) + _EPS)
    mi = float(np.nansum(np.where(pxy > 0, terms, 0.0)))
    if not normalized:
        return mi
    hx = float(-np.nansum(np.where(px > 0, px * np.log2(px + _EPS), 0.0)))
    hy = float(-np.nansum(np.where(py > 0, py * np.log2(py + _EPS), 0.0)))
    denom = min(hx, hy) if mode == "min" else 0.5 * (hx + hy)
    return 0.0 if denom <= 0 else mi / denom


@tl_result(key=K.MUTUAL_INFORMATION, version=1, schema=schemas.MutualInformation)
def mutual_information(
    adata, *, covariate, groupby=None, splitby=None, n_samples=0, temperature=1.0,
    clones=None, weighted=False, normalized=True, normalize_mode="min",
    random_state=None, device=None, key_added=None, inplace=True,
):
    """I(c;phi | covariate) in bits, computed once and cached.

    Returns ``{"table", "result", "stats"}`` and stores the same object under
    ``uns[key_added or 'tcri_mutual_information']``. ``pl.mutual_information`` renders from that
    cache rather than recomputing, so the plot cannot disagree with the frame in your hand.

    ``groupby`` is the replicate. Left implicit it resolves to the column registered as
    ``replicate`` at ``setup_anndata``, and the effective value is recorded in provenance.
    ``splitby`` requires ``groupby`` and must be constant within group.
    """
    if is_precomputed_joint(adata):
        reject_stacked_covariate_joint(adata)
        if n_samples and int(n_samples) > 0:
            raise ValueError("precomputed-joint fast path is valid only at n_samples=0 (§7.9).")
        value = _mi_from_joint(adata.values, normalized=normalized, mode=normalize_mode)
        frame = pd.DataFrame([{"covariate": None, "item": None, "draw": 0, "value": value}])
        return {"table": frame, "result": frame.drop(columns=["draw"]), "stats": None}

    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)

    obs = adata.obs
    cc = obs[adata.uns[K.METADATA][K.Config.CLONE_COL]]

    def _one(clone_subset):
        draws, _cols = joint_draws(
            adata, covariate, n_samples=n_samples, weighted=weighted, device=device,
            temperature=temperature, clones=clone_subset, random_state=random_state,
        )
        return [_mi_from_joint(J, normalized=normalized, mode=normalize_mode)
                for _ids, J in draws]

    rows = []
    if gkey is None:
        for d, v in enumerate(_one(clones)):
            rows.append({"covariate": covariate, "item": None, "draw": d, "value": v})
    else:
        from ._common import _validate_group_clones
        _validate_group_clones(obs, gkey, cc.name)
        for g in obs[gkey].dropna().unique().tolist():
            gmask = obs[gkey] == g
            group_clones = cc[gmask].dropna().unique().tolist()
            if clones is not None:
                allowed = set(group_clones)
                group_clones = [c for c in clones if c in allowed]
                if not group_clones:
                    continue
            row_base = {"covariate": covariate, gkey: g, "item": None}
            if splitby is not None:
                row_base[splitby] = obs.loc[gmask, splitby].iloc[0]
            for d, v in enumerate(_one(group_clones)):
                rows.append({**row_base, "draw": d, "value": v})

    table = pd.DataFrame(rows)
    result = build_result(table, groupby=gkey, splitby=splitby, item_col=None)
    stats = build_stats(result, groupby=gkey, splitby=splitby)

    payload = {"table": table, "result": result, "stats": stats}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload
