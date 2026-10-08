"""Tables of the clonotypes in ``obs``. They need no fitted model, so they can be read before
fitting.

Each summary reads ``adata.obs`` and the clonotype derivation record in ``adata.uns`` and returns
a table. It reports and never filters, and it writes nothing to ``adata``. Cells without a
clonotype (NaN or None, an empty or whitespace-only string, or the literal string ``"nan"``) are
left out and not counted; ``TCRIModel.setup_anndata`` refuses them. Group labels and clonotypes
appear in the tables as strings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._compute._repertoire import _clonotype_sharing, _clonotype_source, _missing_clonotypes

__all__ = ["clonotype_sharing"]

#: The label of the row of :func:`clonotype_sharing` that counts all groups together.
_ALL = "All"


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
