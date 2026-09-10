"""Frame builders for ``perturb.gene_importance``.

``metric_table`` is clone-oriented (its callback receives a clone subset), and the
perturbation reduces over cells, so the long frames are assembled here from the accumulated
arrays instead. ``build_result``/``build_stats`` from ``_compute`` then apply unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._compute._tables import build_stats

__all__: list[str] = []  # private module


def cell_groups(adata, *, cov_col, covariate, groupby, splitby):
    """Group id per ``obs`` row (``-1`` = excluded), the group labels, and the split label per
    group.

    ``covariate`` restricts to the cells at that level; ``groupby`` partitions them. Cells with
    no group label are excluded with the same warning ``metric_table`` gives. No clone
    disjointness is required: the reduction is over cells.
    """
    obs = adata.obs
    keep = np.ones(adata.n_obs, dtype=bool)
    if covariate is not None:
        levels = obs[cov_col].astype(str)
        mask = (levels == str(covariate)).to_numpy()
        if not mask.any():
            raise ValueError(
                f"covariate {covariate!r} not found among {sorted(levels.unique().tolist())}"
            )
        keep &= mask

    if groupby is None:
        gid = np.where(keep, 0, -1).astype(np.int64)
        return gid, [None], None

    if groupby not in obs.columns:
        raise ValueError(f"groupby={groupby!r} is not a column of adata.obs")
    g = obs[groupby]
    n_missing = int(g.isna().sum())
    if n_missing:
        warnings.warn(
            f"groupby={groupby!r}: {n_missing} of {len(obs)} cells "
            f"({100 * n_missing / max(len(obs), 1):.1f}%) have no group label and are excluded "
            f"from every row of this result.",
            UserWarning, stacklevel=3,
        )
    labels = g.dropna().unique().tolist()
    code = {lab: i for i, lab in enumerate(labels)}
    gid = np.array([-1 if pd.isna(v) else code[v] for v in g.tolist()], dtype=np.int64)
    gid[~keep] = -1
    split = None
    if splitby is not None:
        split = {lab: obs.loc[(g == lab).to_numpy(), splitby].iloc[0] for lab in labels}
    return gid, labels, split


def _label_columns(*, covariate, groupby, labels, split, splitby, group_index):
    """The label columns shared by ``table`` and ``shift`` for a flat group index."""
    cols = {}
    if covariate is not None:
        cols["covariate"] = np.full(len(group_index), covariate, dtype=object)
    if groupby is not None:
        lab = np.asarray(labels, dtype=object)[group_index]
        cols[groupby] = lab
        if splitby is not None:
            cols[splitby] = np.asarray([split[l] for l in lab], dtype=object)
    return cols


def importance_table(importance, *, names, labels, present, covariate, groupby, splitby,
                     split):
    """``importance[G, S, n_groups]`` -> one row per (gene, group, draw) with ``value``.

    Groups with no cell (``present`` false) are dropped, as ``metric_table`` drops a group
    with no clones.
    """
    G, S, n_groups = importance.shape
    groups = np.flatnonzero(present)
    n = G * S * len(groups)
    gene_i = np.repeat(np.arange(G), S * len(groups))
    draw_i = np.tile(np.repeat(np.arange(S), len(groups)), G)
    group_i = np.tile(groups, G * S)
    frame = {"gene": np.asarray(names, dtype=object)[gene_i]}
    frame.update(_label_columns(covariate=covariate, groupby=groupby, labels=labels,
                                split=split, splitby=splitby, group_index=group_i))
    frame["draw"] = draw_i
    frame["value"] = importance[gene_i, draw_i, group_i]
    assert len(frame["value"]) == n
    return pd.DataFrame(frame)


def shift_table(baseline, perturbed, *, names, phenotypes, labels, present, covariate,
                groupby, splitby, split):
    """The signed decomposition, averaged over draws.

    ``baseline[S, n_groups, P]``, ``perturbed[G, S, n_groups, P]`` -> one row per
    (gene, phenotype, group) with ``baseline``, ``perturbed`` and ``shift = baseline −
    perturbed``, so a positive shift means silencing the gene REMOVED mass from that phenotype.
    """
    G, S, n_groups, P = perturbed.shape
    groups = np.flatnonzero(present)
    b = baseline.mean(axis=0)            # [n_groups, P]
    p = perturbed.mean(axis=1)           # [G, n_groups, P]
    gene_i = np.repeat(np.arange(G), len(groups) * P)
    group_i = np.tile(np.repeat(groups, P), G)
    pheno_i = np.tile(np.arange(P), G * len(groups))
    frame = {"gene": np.asarray(names, dtype=object)[gene_i],
             "phenotype": np.asarray(phenotypes, dtype=object)[pheno_i]}
    frame.update(_label_columns(covariate=covariate, groupby=groupby, labels=labels,
                                split=split, splitby=splitby, group_index=group_i))
    frame["baseline"] = b[group_i, pheno_i]
    frame["perturbed"] = p[gene_i, group_i, pheno_i]
    frame["shift"] = frame["baseline"] - frame["perturbed"]
    return pd.DataFrame(frame)


def stats_per_gene(result, *, groupby, splitby):
    """The between-split contrast, once per gene.

    ``build_stats`` averages items to one value per group before contrasting, which for a
    gene axis would answer "is the average gene more important in arm A". The question a
    ranking asks is per gene, so the contrast runs per gene and the rows carry a ``gene``
    column. Uncorrected across genes, as every contrast in the package is uncorrected across
    pairs; multiplicity is the caller's.
    """
    if splitby is None or groupby is None or result is None or not len(result):
        return None
    frames = []
    for gene, sub in result.groupby("gene", sort=False, observed=True):
        st = build_stats(sub, groupby=groupby, splitby=splitby)
        if st is not None and len(st):
            frames.append(st.assign(gene=gene))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out[["gene"] + [c for c in out.columns if c != "gene"]]
