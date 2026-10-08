"""Private helpers for clonotype labels and the clonotype derivation record.

This module may import numpy, pandas and :mod:`tcri._state.keys`, and nothing else from tcri, so
that ``preprocessing``, ``model`` and the metric tables can all import it without reaching across
layers.

- :class:`TCRIDataWarning`: the warning class of tcri's pre-fit data checks.
- :func:`_missing_clonotypes`: the one definition of a cell without a clonotype.
- :func:`_label_problems`: empty strings, literal ``"nan"`` and unused categories in label
  columns.
- :func:`_clonotype_sharing`: the clonotypes present in more than one group.
- :func:`_make_clonotypes_replicate_specific`: the ``@{replicate}`` rule and its collision check.
- :func:`_size_counts`: cells per (clonotype, covariate level), the unit the model builds.
- The derivation record in ``uns[K.CLONOTYPE_DERIVATIONS]``: one writer,
  :func:`_record_derivation`, and two readers, :func:`_clonotype_source` and :func:`_pool_labels`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._state import keys as K


class TCRIDataWarning(UserWarning):
    """Emitted by tcri's pre-fit data checks.

    A class of its own, so these warnings can be filtered or raised as errors apart from the
    warnings scvi emits during registration.
    """


def _missing_clonotypes(values: pd.Series) -> pd.Series:
    """Boolean mask of the cells that have no clonotype.

    A clonotype is missing when it is NaN or None, an empty or whitespace-only string, or the
    literal string ``"nan"``. ``"nan"`` is what ``astype(str)`` makes of NaN, and an empty string
    is how many tables write a missing value, so both are read as missing rather than as a clone
    of that name. This is the one definition of a missing clonotype in tcri.

    Parameters
    ----------
    values
        A clonotype column, of any dtype. For a categorical column each category is checked once
        and cells are read through their codes; an unused category marks no cell.

    Returns
    -------
    pd.Series
        Boolean, on the same index as ``values``; ``True`` where the clonotype is missing.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        bad = np.flatnonzero(_missing_clonotypes(pd.Series(values.cat.categories)).to_numpy())
        codes = values.cat.codes.to_numpy()
        return pd.Series((codes < 0) | np.isin(codes, bad), index=values.index, name=values.name)
    missing = values.isna().to_numpy(dtype=bool)
    if pd.api.types.is_string_dtype(values.dtype):
        text = values.astype("string")
        blank = (text.str.strip().eq("") | text.eq("nan")).fillna(False)
        missing = missing | blank.to_numpy(dtype=bool)
    return pd.Series(missing, index=values.index, name=values.name)


def _label_problems(obs: pd.DataFrame, columns, *, unused_for=()) -> pd.DataFrame:
    """Empty strings, literal ``"nan"`` and unused categories in label columns.

    Three problems are reported, in this order:

    - ``"empty_string"``: an empty or whitespace-only string, which many tables write for a
      missing value.
    - ``"literal_nan"``: the string ``"nan"``, which ``astype(str)`` makes of NaN.
    - ``"unused_category"``: a category that no cell carries. Reported only for the columns in
      ``unused_for``.

    The first two are the strings :func:`_missing_clonotypes` reads as missing in a clonotype
    column. NaN and None are not reported.

    Parameters
    ----------
    obs
        The cell table, usually ``adata.obs``.
    columns
        The label columns to check.
    unused_for
        The columns among ``columns`` whose unused categories are reported.

    Returns
    -------
    pd.DataFrame
        One row per (column, problem, value), with columns ``column``, ``problem``, ``value`` and
        ``n_cells``, the number of cells carrying the value (0 for an unused category). Rows
        follow ``columns``, then the problem order above, then the column's category order, or
        the order of first appearance for a column that is not categorical. Empty when nothing
        is found.

    Raises
    ------
    KeyError
        When a column is not in ``obs``.
    """
    rows = []
    for column in columns:
        values = obs[column]
        categorical = isinstance(values.dtype, pd.CategoricalDtype)
        if categorical:
            codes = values.cat.codes.to_numpy()
            found = list(values.cat.categories)
            counts = np.bincount(codes[codes >= 0], minlength=len(found)).tolist()
        else:
            sizes = values.groupby(values, sort=False, dropna=True).size()
            found, counts = sizes.index.tolist(), sizes.tolist()
        text = [value if isinstance(value, str) else None for value in found]
        hits = {
            "empty_string": [t is not None and n > 0 and not t.strip()
                             for t, n in zip(text, counts)],
            "literal_nan": [t == "nan" and n > 0 for t, n in zip(text, counts)],
            "unused_category": [categorical and column in unused_for and n == 0
                                for n in counts],
        }
        for problem, hit in hits.items():
            rows.extend((column, problem, found[i], counts[i]) for i in np.flatnonzero(hit))
    problems = pd.DataFrame(rows, columns=["column", "problem", "value", "n_cells"])
    return problems.astype({"n_cells": "int64"})


def _clonotype_sharing(clonotypes: pd.Series, groups: pd.Series) -> dict:
    """The clonotypes present in more than one group, with the groups each is present in.

    Parameters
    ----------
    clonotypes, groups
        One clonotype and one group label per cell, aligned by position. A cell whose clonotype
        or group is NA is ignored.

    Returns
    -------
    dict
        ``{clonotype: [group, ...]}`` for every clonotype present in more than one group. Keys
        are in the order the clonotypes first appear, and each list holds that clonotype's
        groups in the order it first appears in them. Empty when no clonotype is shared.
    """
    pairs = pd.DataFrame({"clonotype": np.asarray(clonotypes, dtype=object),
                          "group": np.asarray(groups, dtype=object)}).dropna().drop_duplicates()
    n_groups = pairs.groupby("clonotype", sort=False)["group"].transform("size")
    shared = pairs[(n_groups > 1).to_numpy()]
    return {c: g.tolist() for c, g in shared.groupby("clonotype", sort=False)["group"]}


def _make_clonotypes_replicate_specific(clonotypes: pd.Series,
                                        replicates: pd.Series) -> pd.Series:
    """Clonotype ids made specific to each cell's replicate.

    Each id becomes ``f"{clonotype}@{replicate}"``, except an id that already ends in
    ``f"@{replicate}"`` for that cell, which is copied unchanged. Ids tcri wrote and columns
    already scoped with ``@`` pass through, and applying the rule twice changes nothing.

    Parameters
    ----------
    clonotypes
        One clonotype per cell.
    replicates
        One replicate label per cell, aligned with ``clonotypes`` by position.

    Returns
    -------
    pd.Series
        Categorical, on the index of ``clonotypes``. Its categories are the ids of the observed
        (clonotype, replicate) pairs, ordered by the clonotype's category order, then the
        replicate's. A column that is not categorical is ordered as ``astype("category")``
        orders it.

    Raises
    ------
    ValueError
        When a cell has no clonotype (see :func:`_missing_clonotypes`) or no replicate, or when
        two different (clonotype, replicate) pairs would produce the same id. The message names
        the pairs.
    """
    no_clonotype = int(_missing_clonotypes(clonotypes).sum())
    no_replicate = int(pd.Series(replicates).isna().sum())
    if no_clonotype or no_replicate:
        raise ValueError(
            f"cannot make clonotype ids replicate-specific: {no_clonotype} of {len(clonotypes)} "
            f"cells have no clonotype and {no_replicate} have no replicate."
        )
    source = clonotypes.astype("category")
    group = pd.Series(replicates).astype("category")
    n_groups = len(group.cat.categories)
    key = (source.cat.codes.to_numpy(dtype=np.int64) * n_groups
           + group.cat.codes.to_numpy(dtype=np.int64))
    pairs, cell_pair = np.unique(key, return_inverse=True)
    clonotype_codes, replicate_codes = np.divmod(pairs, n_groups)
    pair_clonotypes = source.cat.categories[clonotype_codes]
    pair_replicates = group.cat.categories[replicate_codes]
    ids = pd.Index([c if c.endswith(f"@{r}") else f"{c}@{r}"
                    for c, r in zip(map(str, pair_clonotypes), map(str, pair_replicates))],
                   dtype=object)
    if ids.has_duplicates:
        collisions: dict = {}
        for i in np.flatnonzero(ids.duplicated(keep=False)):
            collisions.setdefault(ids[i], []).append((pair_clonotypes[i], pair_replicates[i]))
        shown = "; ".join(f"{' and '.join(map(repr, named))} would each become {label!r}"
                          for label, named in list(collisions.items())[:5])
        more = f" ({len(collisions)} ids in all)" if len(collisions) > 5 else ""
        raise ValueError(
            f"two different (clonotype, replicate) pairs would get the same replicate-specific "
            f"id: {shown}{more}. Rename the clonotypes so that every pair keeps its own id."
        )
    return pd.Series(pd.Categorical.from_codes(cell_pair, categories=ids),
                     index=clonotypes.index, name=clonotypes.name)


def _size_counts(obs: pd.DataFrame, clonotype_key: str, covariate_key: str,
                 groupby: str | None = None) -> pd.DataFrame:
    """Cells per (clonotype, covariate level): the unit ``TCRIModel.__init__`` builds.

    Without ``groupby`` the unit is (clonotype, covariate level), and the units are the rows the
    model builds from the column. With ``groupby`` the unit is (group, clonotype, covariate
    level): a clone lives within one individual, so a clonotype carried by two groups at the same
    level is two units, each with its own cell count. When ``groupby`` is the replicate, these
    are the units the model builds from replicate-specific ids (see
    :func:`_make_clonotypes_replicate_specific`); on a column whose ids never span groups the
    split changes nothing. Without ``groupby``, a clonotype that two individuals share is one
    unit, so on a source column, the column replicate-specific ids are made from, the rows are
    not what the model estimates.

    Cells without a clonotype (see :func:`_missing_clonotypes`) are left out; setup refuses
    them. So are cells with no covariate level, which the model refuses, and, with ``groupby``,
    cells with no group.

    Parameters
    ----------
    obs
        The cell table, usually ``adata.obs``.
    clonotype_key, covariate_key
        The clonotype and covariate columns.
    groupby
        The column naming each cell's group, or ``None``.

    Returns
    -------
    pd.DataFrame
        One row per observed unit, with columns ``group`` (with ``groupby`` only),
        ``clonotype``, ``covariate`` and ``n_cells``. Rows are sorted by clonotype, then group,
        then covariate level, each in ``astype("category")`` order. That is the order in which
        the model numbers its units, and, when ``groupby`` is the replicate, the order of the
        units it builds from the replicate-specific ids.

    Raises
    ------
    KeyError
        When a column is not in ``obs``.
    """
    labels = {"clonotype": obs[clonotype_key].astype("category")}
    if groupby is not None:
        labels["group"] = obs[groupby].astype("category")
    labels["covariate"] = obs[covariate_key].astype("category")
    codes = {name: values.cat.codes.to_numpy() for name, values in labels.items()}
    keep = ~_missing_clonotypes(obs[clonotype_key]).to_numpy()
    for name in labels:
        keep &= codes[name] >= 0
    units = (pd.DataFrame({name: code[keep] for name, code in codes.items()})
             .groupby(list(labels), sort=True).size().reset_index(name="n_cells"))
    for name, values in labels.items():
        units[name] = np.asarray(values.cat.categories, dtype=object)[units[name].to_numpy()]
    order = (["group"] if groupby is not None else []) + ["clonotype", "covariate", "n_cells"]
    return units[order].astype({"n_cells": "int64"})


# ── the clonotype derivation record ──────────────────────────────────────────────────────────

#: The ``function`` names of the steps the readers follow.
_SETUP = "setup_anndata"
_POOL = "pool_rare_clones"

#: The fields of a derivation step, each with the value it holds when it does not apply. The
#: record in ``uns[K.CLONOTYPE_DERIVATIONS]`` keeps one list per field with one entry per step,
#: because h5ad stores a dict of equal-length lists and cannot store a list of dicts. No field
#: shares a name with the provenance keys ``tcri.get`` strips from what it reads.
_DERIVATION_FIELDS = {
    "function": "",
    "source": "",
    "key_added": "",
    "replicate": "",
    "groupby": "",
    "min_cells": -1,
    "samples": "",
    "suffixed": False,
    "pool_labels": "",
    "n_clones_pooled": 0,
    "n_cells_pooled": 0,
}


def _derivation_steps(adata) -> list[dict]:
    """The steps of the derivation record, oldest first, each as a dict of Python values.

    h5ad returns each field as a numpy array, so every field is read back into a list here.
    """
    record = adata.uns.get(K.CLONOTYPE_DERIVATIONS)
    if record is None:
        return []
    fields = {field: np.asarray(record[field]).tolist() for field in _DERIVATION_FIELDS}
    return [{field: fields[field][i] for field in _DERIVATION_FIELDS}
            for i in range(len(fields["function"]))]


def _record_derivation(adata, *, function: str, source: str, key_added: str,
                       replicate: str | None = None, groupby: str | None = None,
                       min_cells: int | None = None, samples: str | None = None,
                       suffixed: bool = False, pool_labels=(), n_clones_pooled: int = 0,
                       n_cells_pooled: int = 0) -> None:
    """Record in ``adata.uns`` that ``function`` wrote ``obs[key_added]`` from ``obs[source]``.

    A step with the same ``function`` and ``key_added`` as an earlier one replaces it, so running
    a function again on the same column does not grow the record. The new step always goes last,
    so the last step that names a column in ``key_added`` is the one that wrote it.

    Nothing is recorded when ``source == key_added``: the column was not derived from another
    one, and the step already in the record, if any, still says where it came from.

    Parameters
    ----------
    adata
        The object whose ``uns`` holds the record.
    function
        The name of the tcri function that wrote the column.
    source, key_added
        The column the function read and the column it wrote.
    replicate, groupby, samples
        The columns the step used, or ``None``; stored as ``""`` when ``None``.
    min_cells
        The cell floor the step used, or ``None``; stored as ``-1`` when ``None``.
    suffixed
        Whether the step rewrote any id.
    pool_labels
        The labels of the pools the step created, stored joined with ``"|"``.
    n_clones_pooled, n_cells_pooled
        How many clones and cells the step pooled.

    Raises
    ------
    ValueError
        When a pool label contains ``"|"``, which could not be split back out of the record.
    """
    if source == key_added:
        return
    labels = [str(label) for label in pool_labels]
    joined = [label for label in labels if "|" in label]
    if joined:
        raise ValueError(
            f"pool labels cannot contain '|', which joins them in the derivation record: "
            f"{joined[:5]}"
        )
    step = {
        "function": str(function),
        "source": str(source),
        "key_added": str(key_added),
        "replicate": "" if replicate is None else str(replicate),
        "groupby": "" if groupby is None else str(groupby),
        "min_cells": -1 if min_cells is None else int(min_cells),
        "samples": "" if samples is None else str(samples),
        "suffixed": bool(suffixed),
        "pool_labels": "|".join(labels),
        "n_clones_pooled": int(n_clones_pooled),
        "n_cells_pooled": int(n_cells_pooled),
    }
    steps = [s for s in _derivation_steps(adata)
             if (s["function"], s["key_added"]) != (step["function"], step["key_added"])]
    steps.append(step)
    adata.uns[K.CLONOTYPE_DERIVATIONS] = {field: [s[field] for s in steps]
                                          for field in _DERIVATION_FIELDS}


def _writer_of(steps: list[dict], column: str) -> dict | None:
    """The last step that wrote ``column``, or ``None`` when no step did."""
    for step in reversed(steps):
        if step["key_added"] == column:
            return step
    return None


def _clonotype_source(adata, clonotype_key: str) -> str | None:
    """The source column ``clonotype_key`` was derived from, followed back to its root.

    Each step of the derivation record names the column it read. The walk goes from
    ``clonotype_key`` through those steps until it reaches a column no step wrote, which is the
    root. It stops at a column it has already visited, so a step that names its own column as its
    source ends the walk there.

    Returns
    -------
    str or None
        The root source column, or ``None`` when no step wrote ``clonotype_key``, which means
        the column was not derived by tcri.
    """
    steps = _derivation_steps(adata)
    step = _writer_of(steps, clonotype_key)
    if step is None:
        return None
    column, visited = clonotype_key, {clonotype_key}
    while step is not None and step["source"] not in visited:
        column = step["source"]
        visited.add(column)
        step = _writer_of(steps, column)
    return column


def _pool_labels(adata, clonotype_key: str) -> list[str]:
    """The pool labels of the pooling step that produced ``clonotype_key``.

    When setup wrote ``clonotype_key``, the walk follows setup's step back to the column setup
    read, so the registered column reports the pools of the pooled column it came from.

    Returns
    -------
    list of str
        The labels, in the order the pooling step recorded them. Empty when no pooling step
        produced the column.
    """
    steps = _derivation_steps(adata)
    column, visited = clonotype_key, set()
    step = _writer_of(steps, column)
    while step is not None and step["function"] == _SETUP and column not in visited:
        visited.add(column)
        column = step["source"]
        step = _writer_of(steps, column)
    if step is None or step["function"] != _POOL or not step["pool_labels"]:
        return []
    return step["pool_labels"].split("|")
