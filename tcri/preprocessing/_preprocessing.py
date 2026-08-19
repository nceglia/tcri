"""Preprocessing helpers: clonotype grouping and clone sizes.

Deliberately light on imports — this module is loaded by ``import tcri``, so an
eager ``import umap`` here cost ~2.9 s of every import (umap → pynndescent →
numba/llvmlite) for a dependency this file never used.
"""
import numpy as np

from .._state import keys as K

__all__ = ["group_singletons", "clone_size"]


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



# ------------ helper to extract logits -------- #

# ------------ main routine -------------------- #









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


