"""Preprocessing helpers: clonotype grouping and clone sizes.

Deliberately light on imports — this module is loaded by ``import tcri``, so an
eager ``import umap`` here cost ~2.9 s of every import (umap → pynndescent →
numba/llvmlite) for a dependency this file never used.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .._state import keys as K
from .._state._resolution import resolve_clonotype_source

__all__ = ["group_singletons", "clone_size", "from_mudata"]


_SCIRPY_KEYS = {
    "gex_mod": "gex",
    "airr_mod": "airr",
    "sample_key": "sample",
    "tissue_key": "tissue",
    "site_key": "site",
    "batch_key": "patient",
    "clonotype_key": "clone_id",
}


def group_singletons(adata, *, clonotype_key="trb", groupby="patient",
                     target_col="trb_unique", min_clone_size=10):
    adata.obs["trb_candidate"] = adata.obs[clonotype_key].astype(str) + "_" + adata.obs[groupby].astype(str)
    clone_counts = adata.obs["trb_candidate"].value_counts()
    def collapse_singleton(row):
        candidate = row["trb_candidate"]
        if clone_counts[candidate] < min_clone_size:
            return f"Singleton_{row[groupby]}"
        else:
            return candidate
    adata.obs[target_col] = adata.obs.apply(collapse_singleton, axis=1)


def clone_size(adata, *, key_added=K.CLONE_SIZE, return_counts=False):
    # Canonical source is uns[METADATA]['clone_col'] (written by to_anndata). This
    # used to read the legacy uns['tcri_clone_key'] shadow key — the last reader of
    # it, which is why the shim outlived Phase 4.
    meta = adata.uns.get(K.METADATA)
    if not meta or K.CLONE_COL not in meta:
        raise KeyError(
            f"adata.uns[{K.METADATA!r}][{K.CLONE_COL!r}] is missing — run "
            "model.to_anndata(adata) first (or load a session) so the clonotype "
            "column is registered."
        )
    tcr_key = meta[K.CLONE_COL]
    res = np.unique(adata.obs[tcr_key].tolist(), return_counts=True)
    clone_sizes = dict(zip(res[0],res[1]))
    sizes = []
    for clone in adata.obs[tcr_key]:
        sizes.append(clone_sizes[clone])
    adata.obs[key_added] = sizes
    if return_counts:
        return clone_sizes


def _coerce_keys(key_profile: str, keys: Mapping[str, str] | None) -> dict[str, str]:
    if key_profile != "scirpy":
        raise ValueError(
            f"unknown key_profile={key_profile!r}; currently supported: 'scirpy'"
        )
    resolved = dict(_SCIRPY_KEYS)
    if keys is None:
        return resolved
    if not isinstance(keys, Mapping):
        raise TypeError("keys must be a mapping of adapter key names to column/modality names")
    for k, v in keys.items():
        if k not in resolved:
            raise KeyError(
                f"unknown keys override {k!r}; allowed: {sorted(resolved)}"
            )
        if not isinstance(v, str) or not v:
            raise TypeError(f"keys[{k!r}] must be a non-empty string")
        resolved[k] = v
    return resolved


def from_mudata(
    mdata,
    *,
    key_profile: str = "scirpy",
    keys: Mapping[str, str] | None = None,
    clonotype_key: str = "auto",
    covariate_cols: str | None = None,
    drop_missing_clonotype: bool = True,
    counts_layer: str = "counts",
    fill_counts_from_X: bool = True,
    strict: bool = True,
    copy: bool = True,
):
    """Return a tcri-ready AnnData from a MuData object (Scirpy defaults by default).

    The adapter aligns the GEX and AIRR modalities on ``obs_names``, selects a clonotype
    column (explicitly or by ``clonotype_key='auto'``), optionally wires one covariate
    column from ``obs``, and ensures a counts layer exists.

    It records only resolved adapter provenance under ``adata.uns['tcri_adapter']``; key-profile
    defaults are not persisted globally.
    """
    resolved = _coerce_keys(key_profile, keys)
    gex_mod = resolved["gex_mod"]
    airr_mod = resolved["airr_mod"]
    if gex_mod not in mdata.mod:
        raise KeyError(f"mdata.mod has no {gex_mod!r} modality")
    if airr_mod not in mdata.mod:
        raise KeyError(f"mdata.mod has no {airr_mod!r} modality")

    gex = mdata.mod[gex_mod]
    airr = mdata.mod[airr_mod]
    common = gex.obs_names.intersection(airr.obs_names)
    if len(common) == 0:
        raise ValueError(
            f"no shared obs_names between mdata.mod[{gex_mod!r}] and mdata.mod[{airr_mod!r}]"
        )

    adata = gex[common].copy() if copy else gex[common]
    airr_view = airr[adata.obs_names]

    source, key, family, candidates = resolve_clonotype_source(
        {"gex": adata.obs, "airr": airr_view.obs, "mdata": mdata.obs},
        clonotype_key=clonotype_key, source_prefix=airr_mod,
    )
    if source == "gex":
        clone = adata.obs[key]
    elif source == "airr":
        clone = airr_view.obs[key]
    else:
        clone = mdata.obs[key].reindex(adata.obs_names)
    clone = pd.Series(clone, index=adata.obs_names, dtype="object")
    if strict and clone.isna().any():
        n = int(clone.isna().sum())
        raise ValueError(
            f"resolved clonotype column {key!r} has {n} missing values; pass "
            "drop_missing_clonotype=True to drop them or strict=False to continue."
        )

    resolved_clone_col = str(key)
    adata.obs[resolved_clone_col] = clone.astype("string")
    if drop_missing_clonotype:
        adata = adata[adata.obs[resolved_clone_col].notna()].copy()
    adata.obs[resolved_clone_col] = adata.obs[resolved_clone_col].astype("category")

    cov_col: str | None = None
    if covariate_cols is not None:
        if not isinstance(covariate_cols, str) or not covariate_cols:
            raise TypeError(
                "covariate_cols must be a single non-empty column name (str). "
                "For composite covariates, create the combined column upstream and pass it here."
            )
        cov_col = covariate_cols
        if cov_col not in adata.obs:
            raise KeyError(
                f"covariate_cols={cov_col!r} is not in adata.obs"
            )
        cov = adata.obs[cov_col].astype("string")
        if strict and cov.isna().any():
            n_missing = int(cov.isna().sum())
            raise ValueError(
                f"covariate source column {cov_col!r} has {n_missing} missing values"
            )
        cov = cov.fillna("NA")
        adata.obs[cov_col] = pd.Categorical(cov.astype(str))

    if counts_layer not in adata.layers:
        if fill_counts_from_X:
            adata.layers[counts_layer] = adata.X.copy()
        else:
            raise KeyError(
                f"layer {counts_layer!r} is missing and fill_counts_from_X=False"
            )

    adata.uns["tcri_adapter"] = {
        "source": "tcri.pp.from_mudata",
        "key_profile": key_profile,
        "keys": {k: str(v) for k, v in resolved.items()},
        "resolved": {
            "gex_mod": gex_mod,
            "airr_mod": airr_mod,
            "clonotype_source": source,
            "clonotype_key": str(key),
            "clonotype_family": family,
            "clonotype_col": resolved_clone_col,
            "covariate_col": cov_col,
            "counts_layer": counts_layer,
            "n_obs_shared": int(len(common)),
            "n_obs_output": int(adata.n_obs),
        },
        "auto_candidates": [
            {"family": fam, "canonical": canon, "source": src, "key": k}
            for fam, canon, src, k in candidates
        ],
    }
    return adata
