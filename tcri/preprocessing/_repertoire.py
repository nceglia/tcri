"""Summaries of a clonotype column, read before fitting.

Each function reads ``adata.obs`` and the clonotype derivation record in ``adata.uns`` and stores
its result the way the ``tl`` tools do: under ``adata.uns[key_added or "tcri_<function>"]``,
returning ``None``, read back with :func:`tcri.get.result` and drawn by the ``tcri.pl`` function
of the same name. With ``inplace=False`` nothing is stored and the result is returned. None of
them filters cells or needs scirpy, and none returns a verdict or a threshold. Cells without a
clonotype are left out and not counted.
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
from .._compute._tables import validate_splitby
from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result

__all__ = ["repertoire_summary", "clone_persistence", "clonotype_sharing"]

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
                    "largest_clone_cells", "largest_clone_share", "clonality",
                    "n_phenotypes_observed", "n_pools", "cells_in_pools"]

#: The value columns of :func:`clonotype_sharing`'s table per group, in order.
_SHARING_COLUMNS = ["n_clonotypes", "shared_clonotypes", "shared_fraction", "n_cells",
                    "cells_in_shared_clonotypes", "source_column", "registered_shared_clonotypes"]

#: The group of the row of :func:`clonotype_sharing` that counts all groups together.
_ALL = "All"


def _check_columns(obs: pd.DataFrame, **columns) -> None:
    """Raise ``ValueError`` naming the first argument whose column is not in ``obs``.

    Each keyword is an argument name and its column; a column of ``None`` is not checked.
    """
    for argument, column in columns.items():
        if column is not None and column not in obs.columns:
            raise ValueError(f"{argument}={column!r} is not a column of adata.obs.")


def _check_group_name(groupby, columns) -> None:
    """Raise ``ValueError`` when ``groupby`` is one of a summary's other ``columns``.

    A summary names its group column after ``groupby``, so a ``groupby`` that has the name of
    another of its columns would replace that column.
    """
    if groupby in columns:
        raise ValueError(
            f"groupby={groupby!r} would name the group column of the summary, which already has "
            f"a column of that name. Group by a copy of the column under another name."
        )


def _check_split(obs, groupby, splitby, columns) -> None:
    """Raise ``ValueError`` unless ``splitby`` can label the groups of a summary.

    The checks are those of the ``tl`` metrics: ``splitby`` needs ``groupby``, must be a column of
    ``obs``, and must be constant within each group. A summary names its split column after
    ``splitby``, so it also raises when ``splitby`` is ``groupby`` or one of the other ``columns``.
    """
    if splitby is None:
        return
    validate_splitby(obs, groupby, splitby)
    if splitby == groupby or splitby in columns:
        raise ValueError(
            f"splitby={splitby!r} would name the split column of the summary, which already has "
            f"a column of that name. Split by a copy of the column under another name."
        )


def _split_labels(obs, groupby, splitby, groups) -> np.ndarray:
    """The split label of each group in ``groups``, given as strings: a string, or ``""`` where
    a group's cells carry none. Strings, like the group labels, so the stored table writes to
    ``.h5ad``.

    :func:`_check_split` has already checked that each group carries one label at most.
    """
    labeled = obs[obs[groupby].notna() & obs[splitby].notna()]
    first = labeled.groupby(labeled[groupby].astype(str), sort=False)[splitby].first()
    return np.asarray([str(first[group]) if group in first.index else "" for group in groups],
                      dtype=object)


def _clonality(sizes: np.ndarray) -> float:
    """``1 - H / log2(N)`` over clone sizes: H the base-2 Shannon entropy of the size
    distribution, N the number of clones. NaN with fewer than two clones."""
    if len(sizes) < 2:
        return np.nan
    share = sizes / sizes.sum()
    return float(1.0 + (share * np.log2(share)).sum() / np.log2(len(sizes)))


def _source_column(adata, clonotype_key, what) -> str:
    """The column ``clonotype_key`` was derived from, or ``clonotype_key`` itself when no tcri
    step derived it. Raises ``ValueError`` when that column is not in ``adata.obs``; ``what``
    names the quantity measured there, for the message."""
    source = _clonotype_source(adata, clonotype_key)
    if source is None:
        return clonotype_key
    if source not in adata.obs.columns:
        raise ValueError(
            f"clonotype_key={clonotype_key!r} was derived from {source!r}, which is not a column "
            f"of adata.obs. {what} is measured on the column a derived clonotype column was made "
            f"from."
        )
    return source


@tl_result(key=K.REPERTOIRE_SUMMARY, version=1)
def repertoire_summary(adata, *, clonotype_key, groupby=None, splitby=None, covariate_key=None,
                       phenotype_key=None) -> pd.DataFrame:
    """Cells, clones, clone sizes and clonality per group, read before fitting.

    One row per group of ``groupby``, or per (group, covariate level) pair when
    ``covariate_key`` is given; ``groupby=None`` makes the whole dataset one group. A row exists
    for every group, or pair, that holds a cell with a clonotype. Within a row each distinct
    clonotype is one clone, so a clonotype that two groups carry is a clone in each, and a clone
    seen at two covariate levels is counted at both.

    Cells without a clonotype (NaN or None, an empty or whitespace-only string, or the string
    ``"nan"``) are left out and not counted; setup refuses them. A cell with no group, or no
    covariate level, is in no row. The function reads ``adata.obs`` and the clonotype derivation
    record in ``adata.uns``, and filters nothing.

    On a pooled column, one whose derivation record lists the pools it was written with, pools
    are counted in their own columns and not as clones. Their cells are part of ``n_cells``.

    Clonality is always measured on the original clonotype column, with every clone, singletons
    included. On a column tcri derived, the derivation record leads back to that column: a pool
    merges a group's rare clones into one label, which would count as one large clone, and
    leaving the pools out would drop the singletons. Either way the repertoire would look more
    clonal than it is.

    The table is stored in ``adata.uns[key_added or "tcri_repertoire_summary"]`` and read back
    with ``tcri.get.result(adata, "repertoire_summary")``; :func:`tcri.pl.repertoire_summary`
    draws it.

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
    splitby
        A column constant within each group, such as the response, that labels the groups; it
        needs ``groupby``. The table carries it after the group column, as a string, and
        :func:`tcri.pl.repertoire_summary` colors the groups by it.
    covariate_key
        A column whose levels split each group into one row per level.
    phenotype_key
        A phenotype column; adds ``n_phenotypes_observed``.
    key_added
        The ``uns`` key to store the table under; ``None`` stores it under
        ``"tcri_repertoire_summary"``.
    inplace
        ``True`` stores the table and returns ``None``; ``False`` stores nothing and returns it.

    Returns
    -------
    pd.DataFrame or None
        ``None``, or with ``inplace=False`` the table: one row per group, or per (group,
        covariate level) pair, in the category order of each column as ``astype("category")``
        orders it. Its columns:

        - the group, in a column named after ``groupby`` (with ``groupby`` only);
        - the group's split label, as a string, in a column named after ``splitby`` (with
          ``splitby`` only; ``""`` for a group whose cells carry none);
        - ``covariate``: the level of ``covariate_key`` (with ``covariate_key`` only);
        - ``n_cells``: the cells with a clonotype;
        - ``n_clones``: the distinct clonotypes among them, pools not counted;
        - ``n_singleton_clones``: the clones with one cell;
        - ``singleton_fraction``: ``n_singleton_clones / n_clones``, NaN in a row with no clone;
        - ``largest_clone_cells``: the cells of the largest clone, 0 in a row with no clone;
        - ``largest_clone_share``: ``largest_clone_cells / n_cells``;
        - ``clonality``: ``1 - H / log2(N)`` on the original clonotype column, where H is the
          base-2 Shannon entropy of the row's clone sizes and N its number of clones; NaN with
          fewer than two clones. On a pooled column N counts every original clone, so it can
          differ from ``n_clones``;
        - ``n_phenotypes_observed``: the distinct phenotypes among the cells (with
          ``phenotype_key`` only);
        - ``n_pools`` and ``cells_in_pools``: the pools and their cells (on a pooled column
          only).

        Groups and covariate levels are strings.

    Raises
    ------
    ValueError
        When a column is not in ``adata.obs``; when ``groupby`` has the name of one of the
        summary's other columns; when ``splitby`` is given without ``groupby``, is not constant
        within each group, or has the name of another column; or when the column the derivation
        record leads back to is not in ``adata.obs``.

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
    _check_columns(obs, clonotype_key=clonotype_key, groupby=groupby, splitby=splitby,
                   covariate_key=covariate_key, phenotype_key=phenotype_key)
    others = _SUMMARY_COLUMNS + ([] if covariate_key is None else ["covariate"])
    _check_group_name(groupby, others)
    _check_split(obs, groupby, splitby, others)
    source = _source_column(adata, clonotype_key, "Clonality")
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

    # clonality on the original column, over the cells the rows count
    counted = obs[~_missing_clonotypes(obs[clonotype_key]).to_numpy()]
    original = _size_counts(counted, source,
                            source if covariate_key is None else covariate_key, groupby)
    if labels:
        unit_columns = [unit_column for unit_column, _ in labels.values()]
        clonality = (original.groupby(unit_columns, sort=False)["n_cells"]
                     .apply(lambda sizes: _clonality(sizes.to_numpy(float)))
                     .rename("clonality").reset_index()
                     .rename(columns={unit_column: label
                                      for label, (unit_column, _) in labels.items()}))
        for label in labels:
            clonality[label] = clonality[label].astype(str)
        summary = summary.merge(clonality, on=list(labels), how="left")
    else:
        summary["clonality"] = _clonality(original["n_cells"].to_numpy(float))

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
    split = []
    if splitby is not None:
        summary[splitby] = _split_labels(obs, groupby, splitby, summary[groupby])
        split = [splitby]
    rows = list(labels)
    return summary[[*rows[:1], *split, *rows[1:], *columns]]


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


@tl_result(key=K.CLONE_PERSISTENCE, version=1, schema=schemas.ClonePersistence)
def clone_persistence(adata, *, clonotype_key, covariate_key, cov_from=None, cov_to=None,
                      groupby=None, splitby=None, per_clone=False) -> dict:
    """Which clones are seen at more than one level of a covariate, per group.

    A summary to read before fitting: how the clones of each group carry across the levels of
    a covariate such as the timepoint. A clone is one clonotype within one group, so with
    ``groupby`` a clonotype that two groups carry counts as one clone in each.

    With ``cov_from`` and ``cov_to``, it counts the clones of each group seen at both levels,
    the persistent clones, and those seen at one of the two only. On the clonotype column
    registered at setup, and with the ``groupby`` the metrics use, the persistent clones are the
    clones ``tl.phenotypic_flux``, ``tl.delta_phenotypic_entropy`` and
    ``tl.delta_clonotypic_entropy`` are computed on at their default ``exclude_pools=True``,
    and a group without any has no row in those results. The metrics read ``groupby=None`` as
    the replicate registered at setup, while here it means the whole dataset. Without
    ``cov_from`` and ``cov_to``, it counts the clones seen at one level and at more than one.

    On a pooled column, one whose derivation record lists the pools it was written with, a pool
    is not a clone: its cells are left out of every count, as are cells without a clonotype, so
    a pool is never persistent.

    Clones seen at one level only are expected, since each sample holds only part of a
    repertoire.

    Beside the table per group, ``pairs`` counts, for each group and each ordered pair of levels
    at which it has clones, the clones of the first level that are also seen at the second.
    It takes any number of levels: a design such as tissue × timepoint is one combined
    covariate column. The result is stored in
    ``adata.uns[key_added or "tcri_clone_persistence"]`` and read back with
    ``tcri.get.result(adata, "clone_persistence")``; :func:`tcri.pl.clone_persistence` draws
    ``pairs``.

    Parameters
    ----------
    adata
        The cells to summarize. ``adata.obs`` and the clonotype derivation record in
        ``adata.uns`` are read.
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
    splitby
        A column constant within each group, such as the response, that labels the groups; it
        needs ``groupby``. The tables per group and per clone carry it after the group column,
        as a string.
    per_clone
        Store one row per clone instead of the tables per group.
    key_added
        The ``uns`` key to store the result under; ``None`` stores it under
        ``"tcri_clone_persistence"``.
    inplace
        ``True`` stores the result and returns ``None``; ``False`` stores nothing and returns it.

    Returns
    -------
    dict or pd.DataFrame or None
        ``None``, or with ``inplace=False`` the result: ``{table, pairs}``, or with
        ``per_clone=True`` the table per clone.

        ``table`` has one row per group with at least one counted cell, in the column's category
        order, with the group, as a string, in a column named ``groupby`` and its split label in
        a column named ``splitby`` (with ``splitby`` only). With ``groupby=None``, one row and
        no group column.

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
        clonotype, as a string, in ``clonotype``, and one column per covariate level holding
        the clone's cells at that level. Level columns are named by the level as a string, in
        category order. With ``cov_from`` and ``cov_to`` the level columns are those two and
        the rows are the clones seen at either, so the persistent clones are the rows with
        cells in both columns. Without them, every level with a counted cell has a column and
        every clone a row.

        ``pairs`` has one row per group and ordered pair of distinct levels at which the group
        has clones, by group, then the first level, then the second, levels in category order
        (only ``cov_from`` and ``cov_to`` when they are given):

        - the group, in a column named ``groupby`` (with ``groupby`` only);
        - ``covariate_a``, ``covariate_b``: the two levels, as strings;
        - ``n_clones``: the group's clones at ``covariate_a``;
        - ``shared_clones``: those also seen at ``covariate_b``;
        - ``shared_fraction``: ``shared_clones / n_clones``.

    Raises
    ------
    ValueError
        When a column is not in ``adata.obs``, when only one of ``cov_from`` and ``cov_to`` is
        given, when either is not a level of ``covariate_key``, when ``splitby`` is given
        without ``groupby`` or is not constant within each group, or when two output columns
        would share a name.
    """
    obs = adata.obs
    _check_columns(obs, clonotype_key=clonotype_key, covariate_key=covariate_key,
                   groupby=groupby, splitby=splitby)
    _check_split(obs, groupby, splitby, [])
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
        keys = [("clonotype", clones["clonotype"].map(str).to_numpy()[rows])]
        if groupby is not None:
            groups = clones["group"].map(str).to_numpy()[rows]
            keys[:0] = [(groupby, groups), *_split_column(obs, groupby, splitby, groups)]
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
    columns = []
    if groupby is not None:
        groups = totals.index.map(str).to_numpy()
        columns = [(groupby, groups), *_split_column(obs, groupby, splitby, groups)]
    if not contrast:
        n_levels = pd.DataFrame(seen).groupby(group, sort=False).any().sum(axis=1)
        columns.append(("n_levels", n_levels.reindex(totals.index, fill_value=0)
                        .to_numpy(dtype=np.int64)))
    columns += [(name, totals[name].to_numpy(dtype=np.int64)) for name in counts]
    if contrast:
        columns.append(("has_persistent_clones", totals["persistent_clones"].to_numpy() > 0))
    return {"table": _frame(columns), "pairs": _level_pairs(clones, levels, seen, groupby)}


def _split_column(obs, groupby, splitby, groups) -> list:
    """``[(splitby, labels)]``, the split label of each of ``groups``, or ``[]`` without
    ``splitby``."""
    if splitby is None:
        return []
    return [(splitby, _split_labels(obs, groupby, splitby, groups))]


def _level_pairs(clones, levels, seen, groupby) -> pd.DataFrame:
    """The ``pairs`` table of :func:`clone_persistence`.

    ``seen`` is the boolean ``[clone, level]`` array of the clones in ``clones``, which holds
    them ordered by group. For each group and each ordered pair of distinct levels at which the
    group has clones: the clones at the first level, and those of them also seen at the second.
    """
    group = (np.zeros(len(clones), dtype=object) if groupby is None
             else clones["group"].map(str).to_numpy())
    rows = []
    for name in pd.unique(group):
        at = seen[group == name].astype(np.int64)
        n = at.sum(axis=0)
        both = at.T @ at
        present = np.flatnonzero(n > 0)
        rows += [(name, levels[a], levels[b], n[a], both[a, b])
                 for a in present for b in present if a != b]
    pairs = pd.DataFrame(rows, columns=["group", "covariate_a", "covariate_b", "n_clones",
                                        "shared_clones"])
    pairs = pairs.astype({"n_clones": "int64", "shared_clones": "int64"})
    pairs["shared_fraction"] = pairs["shared_clones"] / pairs["n_clones"]
    if groupby is None:
        return pairs.drop(columns="group")
    return _frame([(groupby, pairs["group"].to_numpy()),
                   *[(column, pairs[column].to_numpy()) for column in pairs.columns[1:]]])


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


@tl_result(key=K.CLONOTYPE_SHARING, version=1, schema=schemas.ClonotypeSharing)
def clonotype_sharing(adata, *, clonotype_key, groupby, splitby=None, per_clonotype=False) -> dict:
    """How the clonotypes of ``clonotype_key`` are shared across the groups of ``groupby``.

    A clone is a set of cells within one individual, and a clonotype is its name, usually its
    receptor sequence. Two individuals can carry the same clonotype but never the same clone: a
    clonotype carried by two individuals is two clones, one per individual. A clonotype is shared
    when it is present in more than one group. For each group, the table counts its clonotypes,
    the ones it shares with another group, and the cells that carry them.

    Most clonotypes are found in one individual only, so low sharing between individuals is
    expected biology, not a problem with the data. Some clonotypes recur in several individuals,
    which is expected biology as well.

    When :func:`tcri.pp.pool_rare_clones` wrote ``clonotype_key``, the derivation record in
    ``adata.uns["tcri_clonotype_derivations"]`` leads back to the column pooling read, and
    sharing is measured there: pooling makes each id specific to its group, usually the
    individual, so no id spans two groups, and sharing between individuals shows only on the
    clonotypes those ids were made from. ``TCRIModel.setup_anndata`` registers the column it is
    given under its own name, so a registered pooled column leads back the same way.
    ``source_column`` names the column used. ``registered_shared_clonotypes`` counts the ids of
    ``clonotype_key`` itself that are present in more than one group. A metric with a
    ``groupby`` needs every registered id inside one group, so on the column registered at setup,
    a registered shared clonotype makes a metric grouped by the same column raise.

    A cell is counted when it has a group and a clonotype, in ``clonotype_key`` and in the source
    column; every other cell is left out.

    What sharing means follows from ``groupby``. Grouped by the individual, a shared clonotype is
    one found in more than one person. Grouped by a column that splits an individual, such as
    the sample or the timepoint, a clonotype one person carries at two timepoints is shared too,
    so the counts mix sharing between people with persistence within a person
    (:func:`clone_persistence` counts the latter on its own).

    Beside the table per group, ``pairs`` counts, for each ordered pair of groups, the
    clonotypes of the first group also found in the second. The result is stored in
    ``adata.uns[key_added or "tcri_clonotype_sharing"]`` and read back with
    ``tcri.get.result(adata, "clonotype_sharing")``; :func:`tcri.pl.clonotype_sharing` draws
    ``pairs`` and each group's ``shared_fraction``.

    Parameters
    ----------
    adata
        The object to summarize. Only ``adata.obs`` and the derivation record are read.
    clonotype_key
        The clonotype column: a source column, or a column :func:`tcri.pp.pool_rare_clones`
        wrote from one.
    groupby
        The column naming each cell's group, usually the individual. Required: sharing is
        defined between groups.
    splitby
        A column constant within each group, such as the response, that labels the groups. The
        table per group carries it after the group column, as a string.
    per_clonotype
        ``False`` stores the tables per group and per pair; ``True`` stores one row per shared
        clonotype instead.
    key_added
        The ``uns`` key to store the result under; ``None`` stores it under
        ``"tcri_clonotype_sharing"``.
    inplace
        ``True`` stores the result and returns ``None``; ``False`` stores nothing and returns it.

    Returns
    -------
    dict or pd.DataFrame or None
        ``None``, or with ``inplace=False`` the result: ``{table, pairs}``, or with
        ``per_clonotype=True`` the table per shared clonotype.

        ``table`` has one row per group with a counted cell, in the order of
        ``obs[groupby].astype("category")``, then a row for all groups together. Columns:

        - the group, in a column named after ``groupby``, with ``All`` in the row for all
          groups.
        - the group's split label, in a column named after ``splitby`` (with ``splitby`` only),
          with ``All`` in the row for all groups.
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

        ``pairs`` has one row per ordered pair of distinct groups with a counted cell, by the
        first group and then the second, in the order of the group rows:

        - ``<groupby>_a``, ``<groupby>_b``: the two groups, named after ``groupby``;
        - ``n_clonotypes``: the clonotypes of the first group;
        - ``shared_clonotypes``: those also found in the second;
        - ``shared_fraction``: ``shared_clonotypes / n_clonotypes``.

        With ``per_clonotype=True``, one row per shared clonotype of the source column, in the
        order of ``astype("category")`` on that column. Columns: ``clonotype``, ``n_groups``,
        ``groups`` (the group labels, in the order of the group rows, joined with ``"|"``),
        ``n_cells``, and ``cells_per_group`` (the cells in each of those groups, in the same
        order, joined with ``"|"``).

        Group labels and clonotypes are strings.

    Raises
    ------
    ValueError
        When ``clonotype_key``, ``groupby``, ``splitby`` or the column the derivation record
        leads back to is not a column of ``adata.obs``, when a group is named ``All``, when
        ``splitby`` is not constant within each group, or, for the table per group, when
        ``groupby`` or ``splitby`` has the name of one of its other columns.
    """
    obs = adata.obs
    _check_columns(obs, clonotype_key=clonotype_key, groupby=groupby, splitby=splitby)
    if not per_clonotype:
        _check_group_name(groupby, _SHARING_COLUMNS)
    _check_split(obs, groupby, splitby, [] if per_clonotype else _SHARING_COLUMNS)
    source = _source_column(adata, clonotype_key, "Sharing")

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
        groupby: [*group_names[rows[:-1]], _ALL],
        "n_clonotypes": n_clonotypes[rows],
        "shared_clonotypes": n_shared[rows],
        "n_cells": n_cells[rows],
        "cells_in_shared_clonotypes": cells_in_shared[rows],
        "source_column": source,
        "registered_shared_clonotypes": n_registered_shared[rows],
    })
    table.insert(3, "shared_fraction", table["shared_clonotypes"] / table["n_clonotypes"])
    if splitby is not None:
        table.insert(1, splitby, [*_split_labels(obs, groupby, splitby, group_names[rows[:-1]]),
                                  _ALL])
    return {"table": table,
            "pairs": _group_pairs(shared, rows[:-1], group_names, n_clonotypes, groupby)}


def _group_pairs(shared, present, group_names, n_clonotypes, groupby) -> pd.DataFrame:
    """The ``pairs`` table of :func:`clonotype_sharing`.

    ``shared`` maps each shared clonotype code to the codes of the groups that carry it,
    ``present`` holds the codes of the groups with a counted cell, in row order, and
    ``n_clonotypes`` the clonotype count of every group code.
    """
    both = np.zeros((len(group_names), len(group_names)), dtype=np.int64)
    for members in shared.values():
        members = list(members)
        both[np.ix_(members, members)] += 1
    rows = [(group_names[a], group_names[b], n_clonotypes[a], both[a, b])
            for a in present for b in present if a != b]
    pairs = pd.DataFrame(rows, columns=["a", "b", "n_clonotypes", "shared_clonotypes"])
    pairs = pairs.astype({"n_clonotypes": "int64", "shared_clonotypes": "int64"})
    return _frame([(f"{groupby}_a", pairs["a"].to_numpy(dtype=object)),
                   (f"{groupby}_b", pairs["b"].to_numpy(dtype=object)),
                   ("n_clonotypes", pairs["n_clonotypes"].to_numpy()),
                   ("shared_clonotypes", pairs["shared_clonotypes"].to_numpy()),
                   ("shared_fraction",
                    pairs["shared_clonotypes"].to_numpy() / pairs["n_clonotypes"].to_numpy())])
