"""Building a permutation, and deciding which cells it may move between.

A null is the parent's model on ONE permuted label vector. Everything that makes it a
*reference* rather than noise is in the strata: the permutation shuffles only within cells that
share the columns the metric is not asking about, so the quantity being scored is destroyed and
nothing else is. This module holds the default strata per kind, validates a caller's
refinement, and turns strata into an integer permutation of ``range(n_obs)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K

#: The permutation's own random stream, per kind. A null's TRAINING seed is the parent's exactly
#: -- same initialisation, same split, same minibatch order -- so the permuted labels are the
#: only difference between a null and its parent (plan §8 decision 3). Name-keyed, never
#: positional: a reordering of this dict must not silently change every recorded permutation.
OFFSET = {"phenotype": 10_000, "clonotype": 20_000, "condition": 30_000}

KINDS = ("phenotype", "clonotype", "condition")

#: Which label vector each kind permutes. The model's ``permutation=(axis, perm)`` names the
#: same three axes, so the kind IS the axis; they are spelled separately because a future null
#: could permute an axis under a different name.
AXIS = {"phenotype": "phenotype", "clonotype": "clonotype", "condition": "condition"}


def _cols(adata):
    meta = adata.uns[K.METADATA]
    return (meta[K.Config.COVARIATE_COL], meta[K.Config.CLONE_COL],
            meta[K.Config.PHENOTYPE_COL], meta[K.Config.BATCH_COL],
            meta.get(K.Config.REPLICATE))


def required_strata(adata, kind: str):
    """The columns a ``within`` for ``kind`` must contain.

    ``within`` exists to *refine* the default, so a refinement has to keep what the default is
    protecting. Built as a computed list rather than a literal because the metric-time check in
    ``_compute/_tables.py`` compares the effective ``groupby`` against the same list, and two
    spellings of one rule drift.
    """
    covariate_col, _, _, _, replicate = _cols(adata)
    if kind == "condition":
        return []
    return [covariate_col] + ([replicate] if replicate else [])


def default_strata(adata, kind: str):
    """The default strata for ``kind``, as columns of ``obs``.

    ``(batch, covariate, replicate)`` for the phenotype and clonotype nulls -- the replicate
    column only when one is registered -- and ``(clonotype, batch)`` for the condition null.
    """
    covariate_col, clone_col, _, batch_col, replicate = _cols(adata)
    if kind == "condition":
        # Each clone keeps its cells and its size; which condition a cell sits at is random.
        return [clone_col, batch_col]
    strata = [batch_col, covariate_col]
    if replicate and replicate not in strata:
        strata.append(replicate)
    return strata


def resolve_strata(adata, kind: str, within=None):
    """``within`` if given and legal, else the default. Never silently extended.

    A ``within`` that drops a required column is rejected rather than unioned: strata are fixed
    at fit time and recorded, so quietly adding a column would make the recorded strata differ
    from the strata the caller believes they asked for. Measured on a covariate-sparse fixture,
    a ``within`` that drops the covariate takes ``n_ct`` from 42 to 72 and the shared-clone set
    from 6 to 36, so the ``null_value`` join would be onto a row set 71% larger.
    """
    if within is None:
        return default_strata(adata, kind)
    if kind == "condition":
        raise ValueError(
            "the condition null takes no `within`: its strata are fixed at (clonotype, batch) "
            "so that each clone keeps its cells and its size while the condition a cell sits "
            "at is randomised. Drop `within`, or permute a different axis."
        )
    within = [within] if isinstance(within, str) else list(within)
    missing = [c for c in within if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"within={within!r}: {missing!r} not in adata.obs")
    absent = [c for c in required_strata(adata, kind) if c not in within]
    if absent:
        raise ValueError(
            f"within={within!r} omits {absent!r}, which the {kind} null must stratify on. "
            f"`within` refines the default strata, it does not replace them: a permutation "
            f"that crosses {absent!r} changes the clone x covariate index, so the null's rows "
            f"no longer correspond to the main fit's and the reference cannot be joined to it. "
            f"Pass within={required_strata(adata, kind) + [c for c in within if c not in required_strata(adata, kind)]!r}."
        )
    return within


def build_permutation(adata, strata, rng):
    """An integer permutation of ``range(n_obs)`` that shuffles only within each stratum.

    Returns ``(perm, sizes)``. A cell whose stratum has one member is its own image, which is
    why ``sizes`` is recorded: singleton strata make the permutation the identity for those
    cells and pull the null toward the observed value for a combinatorial reason.
    """
    obs = adata.obs
    perm = np.arange(adata.n_obs, dtype=np.int64)
    if not strata:
        return rng.permutation(perm), [adata.n_obs]
    keys = pd.MultiIndex.from_arrays(
        [obs[c].astype(str).fillna("<NA>").to_numpy() for c in strata]
    )
    sizes = []
    for _, idx in pd.Series(np.arange(adata.n_obs), index=keys).groupby(level=list(range(len(strata))),
                                                                       observed=True, sort=False):
        idx = idx.to_numpy()
        sizes.append(int(len(idx)))
        perm[idx] = rng.permutation(idx)
    return perm, sizes
