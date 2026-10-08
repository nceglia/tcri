"""Summaries of a clonotype column, read before fitting.

Each function reads ``adata.obs`` and the clonotype derivation record in ``adata.uns`` and returns
a table. None of them filters cells, writes to ``adata`` or needs scirpy, and none returns a
verdict or a threshold. Cells without a clonotype are left out and not counted, and group labels
and clonotypes appear in the tables as strings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._compute._repertoire import (
    _clonotype_sharing,
    _clonotype_source,
    _missing_clonotypes,
    _pool_labels,
    _size_counts,
)

__all__ = ["repertoire_summary", "clonotype_sharing"]

#: How each count of :func:`repertoire_summary` is reduced over the units of a row.
_REDUCTIONS = {
    "n_cells": "sum",
    "n_clones": "sum",
    "n_singleton_clones": "sum",
    "largest_clone_cells": "max",
    "n_pools": "sum",
    "cells_in_pools": "sum",
}

#: The value columns of :func:`repertoire_summary`, in order. ``n_phenotypes_observed`` is kept
#: only with a phenotype column, and the two pool columns only on a pooled column.
_SUMMARY_COLUMNS = ["n_cells", "n_clones", "n_singleton_clones", "singleton_fraction",
                    "largest_clone_cells", "largest_clone_share", "n_phenotypes_observed",
                    "n_pools", "cells_in_pools"]

#: The label of the row of :func:`clonotype_sharing` that counts all groups together.
_ALL = "All"


def repertoire_summary(adata, *, clonotype_key, groupby=None, covariate_key=None,
                       phenotype_key=None) -> pd.DataFrame:
    """Cells, clones and clone sizes per group, read before fitting.

    One row per group of ``groupby``, or per (group, covariate level) pair when
    ``covariate_key`` is given; ``groupby=None`` makes the whole dataset one group. A row exists
    for every group, or pair, that holds a cell with a clonotype. Within a row each distinct
    clonotype is one clone, so a clonotype that two groups carry is a clone in each, and a clone
    seen at two covariate levels is counted at both.

    Cells without a clonotype (NaN or None, an empty or whitespace-only string, or the string
    ``"nan"``) are left out and not counted; setup refuses them. A cell with no group, or no
    covariate level, is in no row. The function reads ``adata.obs`` and the clonotype derivation
    record in ``adata.uns``; it filters nothing and writes nothing to ``adata``.

    On a pooled column, one whose derivation record lists the pools it was written with, pools
    are counted in their own columns and not as clones. Their cells are part of ``n_cells``.

    Parameters
    ----------
    adata
        The object to summarize, fitted or not.
    clonotype_key
        The clonotype column of ``adata.obs``: a source column, a pooled column, or the column
        registered at setup.
    groupby
        The column naming each cell's group, usually the individual. ``None`` makes the whole
        dataset one group.
    covariate_key
        A column whose levels split each group into one row per level.
    phenotype_key
        A phenotype column; adds ``n_phenotypes_observed``.

    Returns
    -------
    pd.DataFrame
        One row per group, or per (group, covariate level) pair, in the category order of each
        column as ``astype("category")`` orders it. Its columns:

        - the group, in a column named after ``groupby`` (with ``groupby`` only);
        - ``covariate``: the level of ``covariate_key`` (with ``covariate_key`` only);
        - ``n_cells``: the cells with a clonotype;
        - ``n_clones``: the distinct clonotypes among them, pools not counted;
        - ``n_singleton_clones``: the clones with one cell;
        - ``singleton_fraction``: ``n_singleton_clones / n_clones``, NaN in a row with no clone;
        - ``largest_clone_cells``: the cells of the largest clone, 0 in a row with no clone;
        - ``largest_clone_share``: ``largest_clone_cells / n_cells``;
        - ``n_phenotypes_observed``: the distinct phenotypes among the cells (with
          ``phenotype_key`` only);
        - ``n_pools`` and ``cells_in_pools``: the pools and their cells (on a pooled column
          only).

        Groups and covariate levels are strings.

    Raises
    ------
    ValueError
        When a column is not in ``adata.obs``, or when ``groupby`` has the name of one of the
        summary's other columns.

    Notes
    -----
    How the columns bear on the metrics:

    - On the registered clonotype column, with ``groupby`` set to the metric's effective
      ``groupby`` (the one it is given, or the registered replicate when it is given none),
      ``n_clones`` of a (group, level) row is the clone count in the denominator
      :func:`tcri.tl.clonotypic_entropy` uses at that level when ``n_clones_ref`` is unset: the
      normalized value is the entropy divided by the base-2 log of that count. A row from
      ``groupby=None`` is an upper bound on every group's count, because :mod:`tcri.tl` resolves
      ``groupby=None`` to the registered replicate, while here it means the whole dataset.
    - Setting ``n_clones_ref`` to at least the largest ``n_clones`` across rows keeps every
      normalized value at or below 1.
    - A high ``singleton_fraction`` is where mutual information saturates at default settings.
    - A large ``largest_clone_share`` is also the symptom of sequence-cluster chaining in
      scirpy, where clonotypes defined by sequence similarity are joined through chains of close
      matches into one cluster.
    """
    obs = adata.obs
    if clonotype_key not in obs.columns:
        raise ValueError(f"clonotype_key={clonotype_key!r} is not a column of adata.obs")
    for argument, column in (("groupby", groupby), ("covariate_key", covariate_key),
                             ("phenotype_key", phenotype_key)):
        if column is not None and column not in obs.columns:
            raise ValueError(f"{argument}={column!r} is not a column of adata.obs")

    taken = _SUMMARY_COLUMNS + ([] if covariate_key is None else ["covariate"])
    if groupby in taken:
        raise ValueError(
            f"groupby={groupby!r} would name the group column of the summary, which already has "
            f"a column of that name. Group by a copy of the column under another name."
        )
    # each row label: its name in the summary -> (its column in the units, its column in obs)
    labels = {}
    if groupby is not None:
        labels[groupby] = ("group", groupby)
    if covariate_key is not None:
        labels["covariate"] = ("covariate", covariate_key)

    # Without a covariate the clonotype column takes its place, so each unit is one clonotype
    # of a group with all of its cells.
    units = _size_counts(obs, clonotype_key,
                         clonotype_key if covariate_key is None else covariate_key, groupby)
    pools = _pool_labels(adata, clonotype_key)
    pooled = units["clonotype"].isin(pools).to_numpy()
    cells = units["n_cells"].to_numpy()
    counts = pd.DataFrame({
        "n_cells": cells,
        "n_clones": ~pooled,
        "n_singleton_clones": ~pooled & (cells == 1),
        "largest_clone_cells": np.where(pooled, 0, cells),
        "n_pools": pooled,
        "cells_in_pools": np.where(pooled, cells, 0),
    })
    if labels:
        # categorical keys, so that rows follow each column's category order
        keys = []
        for label, (unit_column, column) in labels.items():
            categories = obs[column].astype("category").cat.categories
            keys.append(pd.Series(pd.Categorical(units[unit_column], categories=categories),
                                  name=label))
        summary = counts.groupby(keys, observed=True, sort=True).agg(_REDUCTIONS).reset_index()
        for label in labels:
            summary[label] = summary[label].astype(str)
    else:
        # one row for the whole dataset, even when no cell has a clonotype
        whole = np.zeros(len(counts), dtype=np.int64)
        summary = (counts.groupby(whole).agg(_REDUCTIONS)
                   .reindex([0], fill_value=0).reset_index(drop=True))
    summary = summary.astype({column: "int64" for column in _REDUCTIONS})

    n_clones, n_cells = summary["n_clones"], summary["n_cells"]
    summary["singleton_fraction"] = summary["n_singleton_clones"] / n_clones.where(n_clones > 0)
    summary["largest_clone_share"] = summary["largest_clone_cells"] / n_cells.where(n_cells > 0)

    if phenotype_key is not None:
        # the cells the units count: a clonotype, and a group and a level wherever they apply
        kept = ~_missing_clonotypes(obs[clonotype_key]).to_numpy()
        for _, column in labels.values():
            kept &= obs[column].notna().to_numpy()
        phenotypes = obs.loc[kept, phenotype_key]
        if labels:
            cell_labels = [obs.loc[kept, column].astype(str).rename(label)
                           for label, (_, column) in labels.items()]
            observed = (phenotypes.groupby(cell_labels).nunique()
                        .rename("n_phenotypes_observed").reset_index())
            summary = summary.merge(observed, on=list(labels), how="left")
        else:
            summary["n_phenotypes_observed"] = phenotypes.nunique()
        summary = summary.astype({"n_phenotypes_observed": "int64"})

    absent = set() if pools else {"n_pools", "cells_in_pools"}
    columns = [column for column in _SUMMARY_COLUMNS
               if column in summary.columns and column not in absent]
    return summary[[*labels, *columns]]


def _shared_per_group(labels: np.ndarray, groups: np.ndarray, n_groups: int):
    """Distinct labels and labels present in more than one group, per group and over all groups.

    ``labels`` and ``groups`` are category codes, one per cell, with no missing value. Returns
    ``(shared, n_labels, n_shared)``: the dict :func:`_clonotype_sharing` gives for the codes, and
    two arrays of length ``n_groups + 1``, one entry per group code and a last entry over all
    groups, where each distinct label counts once however many groups carry it.
    """
    shared = _clonotype_sharing(pd.Series(labels), pd.Series(groups))
    pairs = np.unique(labels * n_groups + groups)
    pair_labels, pair_groups = np.divmod(pairs, n_groups)
    pair_shared = np.isin(pair_labels, list(shared))
    n_labels = np.append(np.bincount(pair_groups, minlength=n_groups), len(np.unique(labels)))
    n_shared = np.append(np.bincount(pair_groups[pair_shared], minlength=n_groups), len(shared))
    return shared, n_labels, n_shared


def clonotype_sharing(adata, *, clonotype_key, groupby, per_clonotype=False):
    """How the clonotypes of ``clonotype_key`` are shared across the groups of ``groupby``.

    A clone is a set of cells within one individual, and a clonotype is its name, usually its
    receptor sequence. Two individuals can carry the same clonotype but never the same clone: a
    clonotype carried by two individuals is two clones, one per individual. A clonotype is shared
    when it is present in more than one group. For each group, the table counts its clonotypes,
    the ones it shares with another group, and the cells that carry them.

    Most clonotypes are found in one individual only, so low sharing between individuals is
    expected biology, not a problem with the data. Some clonotypes recur in several individuals,
    which is expected biology as well.

    When tcri derived ``clonotype_key`` from another column, the derivation record in
    ``adata.uns["tcri_clonotype_derivations"]`` leads back to the column it was made from, and
    sharing is measured there: ids made specific to each replicate never span two replicates, so
    sharing between individuals shows only on the clonotypes those ids were made from.
    ``source_column`` names the column used. ``registered_shared_clonotypes`` counts the ids of
    ``clonotype_key`` itself that are present in more than one group. A metric with a
    ``groupby`` needs every registered id inside one group, so on the column registered at setup,
    a registered shared clonotype makes a metric grouped by the same column raise.

    A cell is counted when it has a group and a clonotype, in ``clonotype_key`` and in the source
    column; every other cell is left out.

    Parameters
    ----------
    adata
        The object to summarize. Only ``adata.obs`` and the derivation record are read.
    clonotype_key
        The clonotype column: a source column, or one tcri derived from it.
    groupby
        The column naming each cell's group, usually the individual. Required: sharing is
        defined between groups.
    per_clonotype
        ``False`` gives one row per group; ``True`` gives one row per shared clonotype.

    Returns
    -------
    pd.DataFrame
        By default, one row per group with a counted cell, in the order of
        ``obs[groupby].astype("category")``, then a row labeled ``All`` for all groups together.
        The index holds the group labels and is named after ``groupby``. Columns:

        - ``n_clonotypes``: the distinct clonotypes in the group.
        - ``shared_clonotypes``: those of them also present in another group.
        - ``shared_fraction``: ``shared_clonotypes / n_clonotypes``.
        - ``n_cells``: the cells counted in the group.
        - ``cells_in_shared_clonotypes``: the cells of the group that carry a shared clonotype.
        - ``source_column``: the column sharing was measured on.
        - ``registered_shared_clonotypes``: the ids of ``clonotype_key`` present in the group and
          in another group.

        The ``All`` row counts each distinct clonotype once, however many groups carry it; its
        cell counts are the sums of the group rows.

        With ``per_clonotype=True``, one row per shared clonotype of the source column, in the
        order of ``astype("category")`` on that column. Columns: ``clonotype``, ``n_groups``,
        ``groups`` (the group labels, in the order of the group rows, joined with ``"|"``),
        ``n_cells``, and ``cells_per_group`` (the cells in each of those groups, in the same
        order, joined with ``"|"``).

    Raises
    ------
    ValueError
        When ``clonotype_key``, ``groupby`` or the column the derivation record leads back to
        is not a column of ``adata.obs``, or when a group is named ``All``.
    """
    obs = adata.obs
    for name, column in (("clonotype_key", clonotype_key), ("groupby", groupby)):
        if column not in obs.columns:
            raise ValueError(f"{name}={column!r} is not a column of adata.obs")
    source = _clonotype_source(adata, clonotype_key)
    if source is None:
        source = clonotype_key
    elif source not in obs.columns:
        raise ValueError(
            f"clonotype_key={clonotype_key!r} was derived from {source!r}, which is not a column "
            f"of adata.obs. Sharing is measured on the column a derived clonotype column was made "
            f"from."
        )

    groups = obs[groupby].astype("category")
    group_codes = groups.cat.codes.to_numpy().astype(np.int64)
    group_names = np.asarray(groups.cat.categories.astype(str), dtype=object)
    if _ALL in set(group_names[np.unique(group_codes[group_codes >= 0])]):
        raise ValueError(
            f"groupby={groupby!r} has a group named {_ALL!r}, the label of the row for all "
            f"groups. Rename that group to summarize this column."
        )
    keep = ((group_codes >= 0)
            & ~_missing_clonotypes(obs[source]).to_numpy()
            & ~_missing_clonotypes(obs[clonotype_key]).to_numpy())
    clonotypes = obs[source].astype("category")
    labels = clonotypes.cat.codes.to_numpy().astype(np.int64)[keep]
    cell_groups = group_codes[keep]
    n_groups = len(group_names)

    shared, n_clonotypes, n_shared = _shared_per_group(labels, cell_groups, n_groups)
    in_shared = np.isin(labels, list(shared))

    if per_clonotype:
        names = np.asarray(clonotypes.cat.categories.astype(str), dtype=object)
        pairs, pair_cells = np.unique(labels[in_shared] * n_groups + cell_groups[in_shared],
                                      return_counts=True)
        cells_in_pair = dict(zip(pairs.tolist(), pair_cells.tolist()))
        rows = []
        for code in sorted(shared):
            members = sorted(shared[code])
            cells = [cells_in_pair[code * n_groups + member] for member in members]
            rows.append((names[code], len(members), "|".join(group_names[members]), sum(cells),
                         "|".join(map(str, cells))))
        table = pd.DataFrame(rows, columns=["clonotype", "n_groups", "groups", "n_cells",
                                            "cells_per_group"])
        return table.astype({"n_groups": "int64", "n_cells": "int64"})

    registered = obs[clonotype_key].astype("category").cat.codes.to_numpy().astype(np.int64)[keep]
    _, _, n_registered_shared = _shared_per_group(registered, cell_groups, n_groups)
    n_cells = np.append(np.bincount(cell_groups, minlength=n_groups), len(cell_groups))
    cells_in_shared = np.append(np.bincount(cell_groups[in_shared], minlength=n_groups),
                                in_shared.sum())
    # The groups with a counted cell, in category order, then the last entry: all groups.
    rows = np.append(np.flatnonzero(n_cells[:-1]), n_groups)
    table = pd.DataFrame({
        "n_clonotypes": n_clonotypes[rows],
        "shared_clonotypes": n_shared[rows],
        "n_cells": n_cells[rows],
        "cells_in_shared_clonotypes": cells_in_shared[rows],
        "source_column": source,
        "registered_shared_clonotypes": n_registered_shared[rows],
    }, index=pd.Index([*group_names[rows[:-1]], _ALL], name=groupby))
    table.insert(2, "shared_fraction", table["shared_clonotypes"] / table["n_clonotypes"])
    return table
