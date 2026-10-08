"""Summaries of a clonotype column, to read before fitting.

Each function reads ``adata.obs`` and reports. None filters cells, writes to ``adata`` or needs
scirpy, and cells without a clonotype are left out of every count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._compute._repertoire import _size_counts

__all__ = ["clone_persistence"]


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


def _clone_cells(obs, clonotype_key, covariate_key, groupby):
    """Cells per clone at each covariate level.

    Returns ``(clones, levels, cells)``: ``clones`` holds one row per clone with columns
    ``group`` (with ``groupby`` only) and ``clonotype``, ordered by group and then clonotype,
    each in ``astype("category")`` order; ``levels`` the observed covariate levels as strings,
    in category order; and ``cells`` the int64 array ``[clone, level]`` of cell counts.
    """
    units = _size_counts(obs, clonotype_key, covariate_key, groupby)
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

    Clones seen at one level only are expected, since each sample holds only part of a
    repertoire.

    Parameters
    ----------
    adata
        The cells to summarize. Only ``adata.obs`` is read, and nothing is written.
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
        the group in a column named ``groupby``. With ``groupby=None``, one row and no group
        column.

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
        clonotype: the group in a column named ``groupby`` (when given), the clonotype in
        ``clonotype``, and one column per covariate level holding the clone's cells at that
        level. Level columns are named by the level as a string, in category order. With
        ``cov_from`` and ``cov_to`` the level columns are those two and the rows are the clones
        seen at either, so the persistent clones are the rows with cells in both columns.
        Without them, every level with a counted cell has a column and every clone a row.

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
    clones, levels, cells = _clone_cells(obs, clonotype_key, covariate_key, groupby)
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
            keys.insert(0, (groupby, clones["group"].to_numpy()[rows]))
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
    columns = [] if groupby is None else [(groupby, totals.index.to_numpy())]
    if not contrast:
        n_levels = pd.DataFrame(seen).groupby(group, sort=False).any().sum(axis=1)
        columns.append(("n_levels", n_levels.reindex(totals.index, fill_value=0)
                        .to_numpy(dtype=np.int64)))
    columns += [(name, totals[name].to_numpy(dtype=np.int64)) for name in counts]
    if contrast:
        columns.append(("has_persistent_clones", totals["persistent_clones"].to_numpy() > 0))
    return _frame(columns)
