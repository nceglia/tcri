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

#: The value columns of :func:`clonotype_sharing`'s ``groups`` frame, in order.
_SHARING_COLUMNS = ["n_clonotypes", "shared_clonotypes", "shared_fraction"]

#: The columns of :func:`clone_persistence`'s frame after the group and split columns.
_PERSISTENCE_COLUMNS = ["covariate_a", "covariate_b", "n_clones_a", "n_clones_b",
                        "shared_clones", "shared_fraction_a", "shared_fraction_b"]


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
    if groupby is None:
        raise ValueError(
            f"splitby={splitby!r} needs groupby: a split labels the groups of the summary, and "
            f"without groupby the whole dataset is one group. Pass groupby=, usually the "
            f"individual."
        )
    validate_splitby(obs, groupby, splitby)
    if splitby == groupby or splitby in columns:
        raise ValueError(
            f"splitby={splitby!r} would name the split column of the summary, which already has "
            f"a column of that name. Split by a copy of the column under another name."
        )


def _split_labels(obs, groupby, splitby, groups) -> list:
    """The split label of each group in ``groups``, given as strings: a string, or ``""`` where
    a group's cells carry none. Strings, like the group labels, so the stored table writes to
    ``.h5ad``.

    :func:`_check_split` has already checked that each group carries one label at most.
    """
    labeled = obs[obs[groupby].notna() & obs[splitby].notna()]
    first = labeled.groupby(labeled[groupby].astype(str), sort=False)[splitby].first()
    return [str(first[group]) if group in first.index else "" for group in groups]


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


@tl_result(key=K.CLONE_PERSISTENCE, version=1)
def clone_persistence(adata, *, clonotype_key, covariate_key, groupby=None,
                      splitby=None) -> pd.DataFrame:
    """How many clones each group carries between the levels of a covariate.

    A summary to read before fitting: for each group, such as the patient, and each pair of
    covariate levels at which it has clones, the clones at each level and the clones seen at
    both. A clone is one clonotype within one group, so with ``groupby`` a clonotype that two
    groups carry counts as one clone in each. It takes any number of levels: a design such as
    tissue × timepoint is one combined covariate column. Clones are counted, never cells.

    On the clonotype column registered at setup, and with the ``groupby`` the metrics use,
    ``shared_clones`` of a pair of levels is the number of persistent clones of a change between
    those levels: the item rows of ``tl.phenotypic_flux`` and of the entropies with ``cov_from``
    and ``cov_to``, at their default ``exclude_pools=True``. Each endpoint of an entropy or mutual
    information change counts every clone at its level. A group without a shared clone has no
    item rows (one NaN item row per phenotype for clonotypic entropy) and a NaN flux. The metrics
    read ``groupby=None`` as the replicate registered at setup, while here it means the whole
    dataset.

    On a pooled column, one whose derivation record lists the pools it was written with, a pool
    is not a clone: its cells are left out of every count, as are cells without a clonotype, so
    a pool is never shared between levels.

    Clones seen at one level only are expected, since each sample holds only part of a
    repertoire.

    The frame is stored in ``adata.uns[key_added or "tcri_clone_persistence"]`` and read back
    with ``tcri.get.result(adata, "clone_persistence")``; :func:`tcri.pl.clone_persistence`
    draws it.

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
    groupby
        The column naming each cell's group, such as the patient. ``None`` summarizes the
        whole dataset as one group, where a clonotype that two groups carry is one clone.
    splitby
        A column constant within each group, such as the response, that labels the groups; it
        needs ``groupby``. The frame carries it after the group column, as a string.
    key_added
        The ``uns`` key to store the frame under; ``None`` stores it under
        ``"tcri_clone_persistence"``.
    inplace
        ``True`` stores the frame and returns ``None``; ``False`` stores nothing and returns it.

    Returns
    -------
    pd.DataFrame or None
        ``None``, or with ``inplace=False`` the frame. One row per group and pair of distinct
        levels at which the group has clones, each pair once: by group, then the first level,
        then the second, each in category order, the first level before the second. Columns:

        - the group, as a string, in a column named ``groupby`` (with ``groupby`` only), and
          its split label in a column named ``splitby`` (with ``splitby`` only; ``""`` for a
          group whose cells carry none);
        - ``covariate_a``, ``covariate_b``: the two levels, as strings;
        - ``n_clones_a``, ``n_clones_b``: the group's clones at each level;
        - ``shared_clones``: the clones seen at both;
        - ``shared_fraction_a``, ``shared_fraction_b``: ``shared_clones`` as a share of each
          level's clones.

    Raises
    ------
    ValueError
        When a column is not in ``adata.obs``, when ``splitby`` is given without ``groupby``
        or is not constant within each group, or when ``groupby`` or ``splitby`` has the name
        of another column of the frame.
    """
    obs = adata.obs
    _check_columns(obs, clonotype_key=clonotype_key, covariate_key=covariate_key,
                   groupby=groupby, splitby=splitby)
    _check_group_name(groupby, _PERSISTENCE_COLUMNS)
    _check_split(obs, groupby, splitby, _PERSISTENCE_COLUMNS)
    pools = _pool_labels(adata, clonotype_key)
    clones, levels, cells = _clone_cells(obs, clonotype_key, covariate_key, groupby, pools)
    seen = cells > 0
    group = (np.zeros(len(clones), dtype=object) if groupby is None
             else clones["group"].map(str).to_numpy())
    rows = []
    for name in pd.unique(group):
        at = seen[group == name].astype(np.int64)
        n = at.sum(axis=0)
        both = at.T @ at
        present = np.flatnonzero(n > 0)
        rows += [(name, levels[a], levels[b], n[a], n[b], both[a, b])
                 for i, a in enumerate(present) for b in present[i + 1:]]
    pairs = pd.DataFrame(rows, columns=["group", "covariate_a", "covariate_b", "n_clones_a",
                                        "n_clones_b", "shared_clones"])
    counts = {column: pairs[column].to_numpy(dtype=np.int64)
              for column in ("n_clones_a", "n_clones_b", "shared_clones")}
    columns = []
    if groupby is not None:
        groups = pairs["group"].tolist()
        columns = [(groupby, groups), *_split_column(obs, groupby, splitby, groups)]
    columns += [("covariate_a", pairs["covariate_a"].tolist()),
                ("covariate_b", pairs["covariate_b"].tolist()),
                *counts.items(),
                ("shared_fraction_a", _share(counts["shared_clones"], counts["n_clones_a"])),
                ("shared_fraction_b", _share(counts["shared_clones"], counts["n_clones_b"]))]
    return _frame(columns)


def _share(part, whole) -> np.ndarray:
    """``part / whole`` as floats, NaN where ``whole`` is 0."""
    whole = np.asarray(whole, dtype=float)
    return np.divide(np.asarray(part, dtype=float), whole, out=np.full(len(whole), np.nan),
                     where=whole > 0)


def _split_column(obs, groupby, splitby, groups) -> list:
    """``[(splitby, labels)]``, the split label of each of ``groups``, or ``[]`` without
    ``splitby``."""
    if splitby is None:
        return []
    return [(splitby, _split_labels(obs, groupby, splitby, groups))]


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
def clonotype_sharing(adata, *, clonotype_key, groupby, splitby=None) -> dict:
    """How the clonotypes of ``clonotype_key`` are shared across the groups of ``groupby``.

    A clone is a set of cells within one individual, and a clonotype is its name, usually its
    receptor sequence. Two individuals can carry the same clonotype but never the same clone: a
    clonotype carried by two individuals is two clones, one per individual. A clonotype is shared
    when it is present in more than one group.

    Most clonotypes are found in one individual only, so low sharing between individuals is
    expected biology, not a problem with the data. Some clonotypes recur in several individuals,
    which is expected biology as well.

    What sharing means follows from ``groupby``. Grouped by the individual, a shared clonotype is
    one found in more than one person. Grouped by a column that splits an individual, such as
    the sample or the timepoint, a clonotype one person carries at two timepoints is shared too,
    so the counts mix sharing between people with persistence within a person
    (:func:`clone_persistence` counts the latter on its own).

    When :func:`tcri.pp.pool_rare_clones` wrote ``clonotype_key``, the derivation record in
    ``adata.uns["tcri_clonotype_derivations"]`` leads back to the column pooling read, and
    sharing is measured there: pooling makes each id specific to its group, usually the
    individual, so no id spans two groups, and sharing between individuals shows only on the
    clonotypes those ids were made from. ``TCRIModel.setup_anndata`` registers the column it is
    given under its own name, so a registered pooled column leads back the same way.

    A cell is counted when it has a group and a clonotype, in ``clonotype_key`` and in the source
    column; every other cell is left out. Clonotypes are counted, never cells.

    The result is stored in ``adata.uns[key_added or "tcri_clonotype_sharing"]`` and read back
    with ``tcri.get.result(adata, "clonotype_sharing")``; :func:`tcri.pl.clonotype_sharing`
    draws it.

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
        frames carry it as a string: after the group column in ``groups``, and once for each
        side in ``pairs``.
    key_added
        The ``uns`` key to store the result under; ``None`` stores it under
        ``"tcri_clonotype_sharing"``.
    inplace
        ``True`` stores the result and returns ``None``; ``False`` stores nothing and returns it.

    Returns
    -------
    dict or None
        ``None``, or with ``inplace=False`` the result, ``{pairs, groups, clonotypes}``. Group
        labels and clonotypes are strings, and groups come in the order of
        ``obs[groupby].astype("category")``, only those with a counted cell.

        ``pairs`` has one row per pair of distinct groups, each pair once, by the first group and
        then the second, the first before the second:

        - ``<groupby>_a``, ``<groupby>_b``: the two groups, named after ``groupby``, and their
          split labels in ``<splitby>_a`` and ``<splitby>_b`` (with ``splitby`` only);
        - ``n_clonotypes_a``, ``n_clonotypes_b``: the distinct clonotypes of each group;
        - ``shared_clonotypes``: those both groups carry;
        - ``shared_fraction_a``, ``shared_fraction_b``: ``shared_clonotypes`` as a share of each
          group's clonotypes.

        ``groups`` has one row per group: the group in a column named after ``groupby``, its
        split label in a column named after ``splitby`` (with ``splitby`` only),
        ``n_clonotypes``, ``shared_clonotypes``, those also present in any other group, and
        ``shared_fraction``, ``shared_clonotypes / n_clonotypes``. A clonotype carried by three
        groups counts once here, so this fraction is not a sum over ``pairs``.

        ``clonotypes`` has one row per shared clonotype of the source column, in the order of
        ``astype("category")`` on that column: ``clonotype``, ``n_groups``, ``groups`` (the group
        labels, in group order, joined with ``"|"``), ``n_cells``, and ``cells_per_group`` (the
        cells in each of those groups, in the same order, joined with ``"|"``).

    Raises
    ------
    ValueError
        When ``clonotype_key``, ``groupby``, ``splitby`` or the column the derivation record
        leads back to is not a column of ``adata.obs``, when ``splitby`` is not constant within
        each group, or when ``groupby`` or ``splitby`` has the name of another column.
    """
    obs = adata.obs
    _check_columns(obs, clonotype_key=clonotype_key, groupby=groupby, splitby=splitby)
    _check_group_name(groupby, _SHARING_COLUMNS)
    _check_split(obs, groupby, splitby, _SHARING_COLUMNS)
    source = _source_column(adata, clonotype_key, "Sharing")

    groups = obs[groupby].astype("category")
    group_codes = groups.cat.codes.to_numpy().astype(np.int64)
    group_names = np.asarray(groups.cat.categories.astype(str), dtype=object)
    keep = ((group_codes >= 0)
            & ~_missing_clonotypes(obs[source]).to_numpy()
            & ~_missing_clonotypes(obs[clonotype_key]).to_numpy())
    clonotypes = obs[source].astype("category")
    labels = clonotypes.cat.codes.to_numpy().astype(np.int64)[keep]
    cell_groups = group_codes[keep]
    n_groups = len(group_names)

    shared, n_clonotypes, n_shared = _shared_per_group(labels, cell_groups, n_groups)
    in_shared = np.isin(labels, list(shared))
    # the groups with a counted cell, in category order
    present = np.flatnonzero(np.bincount(cell_groups, minlength=n_groups))
    names = group_names[present].tolist()
    split = _split_column(obs, groupby, splitby, names)
    per_group = _frame([(groupby, names), *split,
                        ("n_clonotypes", n_clonotypes[present]),
                        ("shared_clonotypes", n_shared[present]),
                        ("shared_fraction", _share(n_shared[present], n_clonotypes[present]))])
    return {"pairs": _group_pairs(shared, present, group_names, n_clonotypes,
                                  groupby=groupby, split=dict(split)),
            "groups": per_group,
            "clonotypes": _shared_clonotypes(shared, clonotypes, labels, cell_groups,
                                             in_shared, group_names)}


def _shared_clonotypes(shared, clonotypes, labels, cell_groups, in_shared, group_names):
    """The ``clonotypes`` frame of :func:`clonotype_sharing`: one row per shared clonotype, with
    the groups that carry it and their cells.

    ``shared`` maps each shared clonotype code to the codes of its groups; ``labels`` and
    ``cell_groups`` are the codes of the counted cells, and ``in_shared`` marks those carrying a
    shared clonotype.
    """
    n_groups = len(group_names)
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


def _group_pairs(shared, present, group_names, n_clonotypes, *, groupby, split):
    """The ``pairs`` frame of :func:`clonotype_sharing`.

    ``shared`` maps each shared clonotype code to the codes of the groups that carry it,
    ``present`` holds the codes of the groups with a counted cell, in order, ``n_clonotypes`` the
    clonotype count of every group code, and ``split`` maps ``splitby`` to the split label of
    each group in ``present`` (empty without ``splitby``).
    """
    both = np.zeros((len(group_names), len(group_names)), dtype=np.int64)
    for members in shared.values():
        members = list(members)
        both[np.ix_(members, members)] += 1
    first = np.asarray([a for i, a in enumerate(present) for _ in present[i + 1:]],
                       dtype=np.int64)
    second = np.asarray([b for i, _ in enumerate(present) for b in present[i + 1:]],
                        dtype=np.int64)
    n_a, n_b, n_both = n_clonotypes[first], n_clonotypes[second], both[first, second]
    columns = [(f"{groupby}_a", group_names[first].tolist()),
               (f"{groupby}_b", group_names[second].tolist())]
    for splitby, labels in split.items():
        label_of = dict(zip(present.tolist(), labels))
        columns += [(f"{splitby}_a", [label_of[a] for a in first]),
                    (f"{splitby}_b", [label_of[b] for b in second])]
    columns += [("n_clonotypes_a", n_a), ("n_clonotypes_b", n_b),
                ("shared_clonotypes", n_both),
                ("shared_fraction_a", _share(n_both, n_a)),
                ("shared_fraction_b", _share(n_both, n_b))]
    return _frame(columns)
