"""Frame builders for ``perturb.gene_importance``.

``metric_frames`` is clone-oriented (its callback receives a clone subset), and the
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
    no group label are excluded with the same warning the metrics give. No clone
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
    """The label columns shared by both levels for a flat group index."""
    cols = {}
    if covariate is not None:
        cols["covariate"] = np.full(len(group_index), covariate, dtype=object)
    if groupby is not None:
        lab = np.asarray(labels, dtype=object)[group_index]
        cols[groupby] = lab
        if splitby is not None:
            cols[splitby] = np.asarray([split[l] for l in lab], dtype=object)
    return cols


def importance_frames(shift, *, names, phenotypes, labels, covariate, groupby, splitby, split):
    """The per-draw frames of both levels, from ``shift[G, S, n_groups, P]``.

    ``shift`` is ``baseline − perturbed`` per draw, NaN for a group with no cell. Returns
    ``(replicate_draws, item_draws)``: one row per (gene, group, draw) whose ``value`` is the
    importance ``Σ_p |shift|``, and one row per (gene, group, phenotype, draw) whose ``value`` is
    the shift. Every group has its rows, NaN where it has no cell, so the row set is fixed by the
    genes, the groups and the phenotypes.
    """
    G, S, n_groups, P = shift.shape
    importance = np.abs(shift).sum(axis=-1)                     # [G, S, n_groups]

    gene_i = np.repeat(np.arange(G), S * n_groups)
    draw_i = np.tile(np.repeat(np.arange(S), n_groups), G)
    group_i = np.tile(np.arange(n_groups), G * S)
    replicate = {"gene": np.asarray(names, dtype=object)[gene_i]}
    replicate.update(_label_columns(covariate=covariate, groupby=groupby, labels=labels,
                                    split=split, splitby=splitby, group_index=group_i))
    replicate["draw"] = draw_i
    replicate["value"] = importance[gene_i, draw_i, group_i]

    gene_i = np.repeat(gene_i, P)
    draw_i = np.repeat(draw_i, P)
    group_i = np.repeat(group_i, P)
    pheno_i = np.tile(np.arange(P), G * S * n_groups)
    item = {"gene": np.asarray(names, dtype=object)[gene_i],
            "phenotype": np.asarray(phenotypes, dtype=object)[pheno_i]}
    item.update(_label_columns(covariate=covariate, groupby=groupby, labels=labels,
                               split=split, splitby=splitby, group_index=group_i))
    item["draw"] = draw_i
    item["value"] = shift[gene_i, draw_i, group_i, pheno_i]
    return pd.DataFrame(replicate), pd.DataFrame(item)


def in_gene_order(frame, names, phenotypes=None):
    """``frame`` with its rows in the caller's gene order and, on the item level, the phenotype
    categories' order.

    ``build_result`` sorts its keys; the caller's gene order is the contract ("one importance
    each, in this order"), so it is restored with a stable sort that keeps the group order.
    """
    if frame is None or not len(frame):
        return frame
    rank = frame["gene"].map({g: i for i, g in enumerate(names)}).to_numpy()
    if phenotypes is not None:
        rank = rank * len(phenotypes) + frame["phenotype"].map(
            {p: i for i, p in enumerate(phenotypes)}).to_numpy()
    order = np.argsort(rank, kind="stable")
    return frame.iloc[order].reset_index(drop=True)


def stats_per_gene(summary, *, groupby, splitby, value="value"):
    """The between-split contrast, once per gene.

    ``build_stats`` averages items to one value per group before contrasting, which for a
    gene axis would answer "is the average gene more important in arm A". The question a
    ranking asks is per gene, so the contrast runs per gene and the rows carry a ``gene``
    column. Uncorrected across genes, as every contrast in the package is uncorrected across
    pairs; multiplicity is the caller's.

    ``summary`` is the replicate summary, one row per gene and group. ``value`` names the column
    contrasted, as in ``build_stats``; the frame carries one row per (gene, contrast) and a
    ``quantity`` column naming that column.
    """
    if splitby is None or groupby is None or summary is None or not len(summary):
        return None
    frames = []
    for gene, sub in summary.groupby("gene", sort=False, observed=True):
        st = build_stats(sub, groupby=groupby, splitby=splitby, value=value)
        if st is not None and len(st):
            frames.append(st.assign(gene=gene))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out[["gene"] + [c for c in out.columns if c != "gene"]]
