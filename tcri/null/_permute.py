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
#: -- same initialization, same split, same minibatch order -- so the permuted labels are the
#: only difference between a null and its parent. Name-keyed, never positional: a reordering
#: of this dict must not silently change every recorded permutation.
OFFSET = {"phenotype": 10_000, "clonotype": 20_000, "covariate": 30_000}


def spawn_key(kind: str, key_added, index: int) -> tuple:
    """The spawn key of one null's permutation stream: ``(OFFSET[kind], name, index)``.

    ``name`` is 0 for the unnamed reference and ``1 + crc32(key_added)`` for a named one, so the
    stream is a function of the null's own kind, name and index and of nothing else on the
    object: nulls fitted on separate copies, or in any order, draw the same permutations. A
    stream keyed by the fits already present would give a null fitted on a fresh copy the same
    stream as its first sibling.
    """
    import zlib

    name = 0 if key_added is None else 1 + zlib.crc32(str(key_added).encode("utf-8"))
    return (OFFSET[kind], int(name), int(index))


def permutation_rng(seed: int, kind: str, key_added, index: int):
    """The generator of one null's permutation: ``seed`` spawned by :func:`spawn_key`."""
    sequence = np.random.SeedSequence(int(seed) % 2**64,
                                      spawn_key=spawn_key(kind, key_added, index))
    return np.random.default_rng(sequence)

KINDS = ("phenotype", "clonotype", "covariate")

#: Which label vector each kind permutes. The model's ``permutation=(axis, perm)`` names the
#: same three axes, so the kind IS the axis; they are spelled separately because a future null
#: could permute an axis under a different name.
AXIS = {"phenotype": "phenotype", "clonotype": "clonotype", "covariate": "covariate"}


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
    if kind == "covariate":
        return []
    return [covariate_col] + ([replicate] if replicate else [])


def default_strata(adata, kind: str):
    """The default strata for ``kind``, as columns of ``obs``.

    ``(batch, covariate, replicate)`` for the phenotype and clonotype nulls -- the replicate
    column only when one is registered -- and ``(clonotype, batch)`` for the covariate null.
    """
    covariate_col, clone_col, _, batch_col, replicate = _cols(adata)
    if kind == "covariate":
        # Each clone keeps its cells and its size; which covariate level a cell sits at is random.
        return [clone_col, batch_col]
    strata = [batch_col, covariate_col]
    if replicate and replicate not in strata:
        strata.append(replicate)
    return strata


def resolve_strata(adata, kind: str, within=None):
    """``within`` if given and legal, else the default. Never silently extended.

    A ``within`` that drops a required column is rejected rather than unioned: strata are fixed
    at fit time and recorded, so quietly adding a column would make the recorded strata differ
    from the strata the caller believes they asked for. Dropping the covariate also changes the row
    set: it permutes across covariate levels, so the null carries clone-by-covariate rows the
    parent fit does not, and the ``null_value`` join is onto a different set of rows.
    """
    if within is None:
        return default_strata(adata, kind)
    if kind == "covariate":
        raise ValueError(
            "the covariate null takes no `within`: its strata are fixed at (clonotype, batch) "
            "so that each clone keeps its cells and its size while the covariate level a cell "
            "sits at is randomized. Drop `within`, or permute a different axis."
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
