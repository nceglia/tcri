"""Summaries of a clonotype column, read before fitting.

Each function reads ``adata.obs`` and the clonotype derivation record in ``adata.uns`` and returns
a table. None of them filters cells, writes to ``adata`` or needs scirpy, and none returns a
verdict or a threshold. Cells without a clonotype are left out and not counted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._compute._repertoire import _missing_clonotypes, _pool_labels, _size_counts

__all__ = ["repertoire_summary", "clone_persistence"]

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


def _check_columns(obs: pd.DataFrame, **columns) -> None:
    """Raise ``ValueError`` naming the first argument whose column is not in ``obs``."""
    for argument, column in columns.items():
        if column is not None and column not in obs.columns:
            raise ValueError(f"{argument}={column!r} is not a column of adata.obs.")


def _frame(columns: list) -> pd.DataFrame:
    """A DataFrame from ``(name, values)`` pairs, refusing two columns of the same name.

    Group and level names come from the data, so one of them can equal another column's name;
    a dict would silently keep only the last.
    """
    names = [name for name, _ in columns]
    repeated = [name for name in dict.fromkeys(names) if names.count(name) > 1]
    if repeated:
        raise ValueError(
            f"two output columns would both be named {repeated[0]!r}. Rename the column, or the "
            f"covariate level, that carries this name."
        )
    return pd.DataFrame(dict(columns))


def _clone_cells(obs, clonotype_key, covariate_key, groupby, pools):
    """Cells per clone at each covariate level, leaving out the labels in ``pools``.

    Returns ``(clones, levels, cells)``: ``clones`` holds one row per clone with columns
    ``group`` (with ``groupby`` only) and ``clonotype``, ordered by group and then clonotype,
    each in ``astype("category")`` order; ``levels`` the observed covariate levels as strings,
    in category order; and ``cells`` the int64 array ``[clone, level]`` of cell counts.
    """
    units = _size_counts(obs, clonotype_key, covariate_key, groupby)
    units = units[~units["clonotype"].isin(pools).to_numpy()]
    keys = ["clonotype"]
    if groupby is not None:
        keys = ["group", "clonotype"]
        # units come ordered by clonotype, then group; a stable sort on the group alone leaves
        # them ordered by group, then clonotype
        rank = {g: i for i, g in enumerate(obs[groupby].astype("category").cat.categories)}
        units = units.iloc[np.argsort(units["group"].map(rank).to_numpy(), kind="stable")]
    level = units["covariate"].map(str)
    observed = set(level)
    categories = obs[covariate_key].astype("category").cat.categories
    levels = [s for s in dict.fromkeys(map(str, categories)) if s in observed]
    clone = units.groupby(keys, sort=False).ngroup().to_numpy()
    clones = units.drop_duplicates(keys)[keys].reset_index(drop=True)
    cells = np.zeros((len(clones), len(levels)), dtype=np.int64)
    np.add.at(cells, (clone, pd.Index(levels).get_indexer(level)), units["n_cells"].to_numpy())
    return clones, levels, cells


def _endpoints(obs, covariate_key, cov_from, cov_to) -> list[str]:
    """``[cov_from, cov_to]`` as strings, once each, after checking both are levels.

    A level is a category of ``obs[covariate_key].astype("category")``, compared as a string.
    """
    declared = list(dict.fromkeys(map(str, obs[covariate_key].astype("category").cat.categories)))
    for argument, value in (("cov_from", cov_from), ("cov_to", cov_to)):
        if str(value) not in declared:
            shown = declared[:10] + (["..."] if len(declared) > 10 else [])
            raise ValueError(
                f"{argument}={value!r} is not a level of adata.obs[{covariate_key!r}], whose "
                f"levels are {shown}."
            )
    return list(dict.fromkeys([str(cov_from), str(cov_to)]))


def clone_persistence(adata, *, clonotype_key, covariate_key, cov_from=None, cov_to=None,
                      groupby=None, per_clone=False) -> pd.DataFrame:
    """Which clones are seen at more than one level of a covariate, per group.

    A summary to read before fitting: how the clones of each group carry across the levels of
    a covariate such as the timepoint. A clone is one clonotype within one group, so with
    ``groupby`` a clonotype that two groups carry counts as one clone in each.

    With ``cov_from`` and ``cov_to``, it counts the clones of each group seen at both levels,
    the persistent clones, and those seen at one of the two only. On the clonotype column
    registered at setup, and with the ``groupby`` the metrics use, the persistent clones are the
    clones ``tl.phenotypic_flux``, ``tl.delta_phenotypic_entropy`` and
    ``tl.delta_clonotypic_entropy`` are computed on, and a group without any has no row in
    those results. The metrics read ``groupby=None`` as the replicate registered at setup, while
    here it means the whole dataset. Without ``cov_from`` and ``cov_to``, it counts the clones
    seen at one level and at more than one.

    On a pooled column, one whose derivation record lists the pools it was written with, a pool
    is not a clone: its cells are left out of every count, as are cells without a clonotype, so
    a pool is never persistent. The metrics still compute a value for a pool seen at both
    levels, as they do for a clone.

    Clones seen at one level only are expected, since each sample holds only part of a
    repertoire.

    Parameters
    ----------
    adata
        The cells to summarize. ``adata.obs`` and the clonotype derivation record in
        ``adata.uns`` are read, and nothing is written.
    clonotype_key
        The clonotype column. Cells without a clonotype (NaN or None, an empty or
        whitespace-only string, or ``"nan"``) are left out and not counted.
    covariate_key
        The covariate column, such as the timepoint.
    cov_from
        The level to contrast from, matched as a string. Give it with ``cov_to``, or give
        neither.
    cov_to
        The level to contrast with ``cov_from``, matched as a string.
    groupby
        The column naming each cell's group, such as the patient. ``None`` summarizes the
        whole dataset in one row, where a clonotype that two groups carry is one clone.
    per_clone
        Return one row per clone instead of one row per group.

    Returns
    -------
    pd.DataFrame
        One row per group with at least one counted cell, in the column's category order, with
        the group, as a string, in a column named ``groupby``. With ``groupby=None``, one row
        and no group column.

        With ``cov_from`` and ``cov_to``, only cells at those two levels are counted:

        - ``clones_from_only``, ``clones_to_only``: clones seen at one of the two levels only.
        - ``persistent_clones``: clones seen at both.
        - ``cells_from_only``, ``cells_to_only``: the cells of the clones seen at one level only.
        - ``cells_in_persistent_clones``: the cells of the persistent clones at the two levels.
        - ``has_persistent_clones``: whether the group has at least one persistent clone.

        Without them, cells at every level are counted:

        - ``n_levels``: the number of levels at which the group has cells.
        - ``clones_at_one_level``, ``clones_at_multiple_levels``: clones seen at exactly one
          level, and at more than one.
        - ``cells_at_one_level``, ``cells_at_multiple_levels``: the cells of those clones.

        With ``per_clone=True``, one row per clone instead, ordered by group and then
        clonotype: the group, as a string, in a column named ``groupby`` (when given), the
        clonotype in ``clonotype``, and one column per covariate level holding the clone's
        cells at that level. Level columns are named by the level as a string, in category
        order. With ``cov_from`` and ``cov_to`` the level columns are those two and the rows
        are the clones seen at either, so the persistent clones are the rows with cells in both
        columns. Without them, every level with a counted cell has a column and every clone a
        row.

    Raises
    ------
    ValueError
        When a column is not in ``adata.obs``, when only one of ``cov_from`` and ``cov_to`` is
        given, when either is not a level of ``covariate_key``, or when two output columns
        would share a name.
    """
    obs = adata.obs
    _check_columns(obs, clonotype_key=clonotype_key, covariate_key=covariate_key,
                   groupby=groupby)
    if (cov_from is None) != (cov_to is None):
        raise ValueError(
            "cov_from and cov_to are given together or not at all: pass both to contrast two "
            "levels, or neither to summarize every level."
        )
    pools = _pool_labels(adata, clonotype_key)
    clones, levels, cells = _clone_cells(obs, clonotype_key, covariate_key, groupby, pools)
    contrast = cov_from is not None
    if contrast:
        ends = _endpoints(obs, covariate_key, cov_from, cov_to)
        cells = np.column_stack(
            [cells[:, levels.index(end)] if end in levels else np.zeros(len(clones), np.int64)
             for end in ends])
        levels = ends

    if per_clone:
        # every clone has cells at some level; in a contrast, only those seen at either end
        rows = cells.sum(axis=1) > 0
        keys = [("clonotype", clones["clonotype"].to_numpy()[rows])]
        if groupby is not None:
            keys.insert(0, (groupby, clones["group"].map(str).to_numpy()[rows]))
        return _frame(keys + [(level, cells[rows, j]) for j, level in enumerate(levels)])

    seen = cells > 0
    if contrast:
        at_from, at_to = seen[:, 0], seen[:, -1]
        persistent, from_only, to_only = at_from & at_to, at_from & ~at_to, at_to & ~at_from
        counts = {
            "clones_from_only": from_only.astype(np.int64),
            "clones_to_only": to_only.astype(np.int64),
            "persistent_clones": persistent.astype(np.int64),
            "cells_from_only": np.where(from_only, cells[:, 0], 0),
            "cells_to_only": np.where(to_only, cells[:, -1], 0),
            "cells_in_persistent_clones": np.where(persistent, cells.sum(axis=1), 0),
        }
    else:
        n_seen, total = seen.sum(axis=1), cells.sum(axis=1)
        counts = {
            "clones_at_one_level": (n_seen == 1).astype(np.int64),
            "clones_at_multiple_levels": (n_seen > 1).astype(np.int64),
            "cells_at_one_level": np.where(n_seen == 1, total, 0),
            "cells_at_multiple_levels": np.where(n_seen > 1, total, 0),
        }

    # the whole dataset is one group of code 0, kept as one row even when no cell is counted
    group = np.zeros(len(clones), np.int64) if groupby is None else clones["group"].to_numpy()
    totals = pd.DataFrame(counts).groupby(group, sort=False).sum()
    if groupby is None:
        totals = totals.reindex([0], fill_value=0)
    columns = [] if groupby is None else [(groupby, totals.index.map(str).to_numpy())]
    if not contrast:
        n_levels = pd.DataFrame(seen).groupby(group, sort=False).any().sum(axis=1)
        columns.append(("n_levels", n_levels.reindex(totals.index, fill_value=0)
                        .to_numpy(dtype=np.int64)))
    columns += [(name, totals[name].to_numpy(dtype=np.int64)) for name in counts]
    if contrast:
        columns.append(("has_persistent_clones", totals["persistent_clones"].to_numpy() > 0))
    return _frame(columns)
