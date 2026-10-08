"""The repertoire summaries in ``tcri.pp``.

Each summary reads ``obs`` and the clonotype derivation record and returns a table. It filters
nothing, writes nothing to ``adata`` and needs no fit. ``repertoire_summary`` counts cells and
clones per group: its ``n_clones`` is the clone count clonotypic entropy normalizes by, and on a
pooled column its pools are counted apart from the clones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from tcri._compute._repertoire import _record_derivation
from tcri._state import keys as K

COUNTS = ["n_cells", "n_clones", "n_singleton_clones", "singleton_fraction",
          "largest_clone_cells", "largest_clone_share"]
POOLS = ["n_pools", "cells_in_pools"]


def _adata(obs):
    obs = obs.copy()
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)


def _obs():
    """Two patients at two timepoints. Clonotype c1 is carried by both patients, and three cells
    have no clonotype."""
    return pd.DataFrame({
        "clone_id": ["c1", "c1", "c1", "c2", "c3", "c3", "c1", "c4", "c5", "", None, "nan"],
        "patient": ["P1", "P1", "P1", "P1", "P1", "P1", "P2", "P2", "P2", "P1", "P2", "P2"],
        "covariate": ["t0", "t0", "t1", "t0", "t1", "t1", "t0", "t0", "t1", "t0", "t1", "t0"],
        "phenotype": ["A", "B", "A", "A", "B", "B", "A", "A", None, "B", "A", "C"],
    })


def _expected(labels, rows, columns):
    return pd.DataFrame([list(row) for row in rows], columns=[*labels, *columns])


def test_repertoire_summary_counts_cells_and_clones_per_row():
    """Every column, counted by hand. c1 is a clone in each patient that carries it, the cells
    without a clonotype are in no row, and a cell without a phenotype is not a phenotype."""
    adata = _adata(_obs())
    by_patient = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="patient")
    pd.testing.assert_frame_equal(by_patient, _expected(["patient"], [
        ("P1", 6, 3, 1, 1 / 3, 3, 1 / 2),
        ("P2", 3, 3, 3, 1.0, 1, 1 / 3),
    ], COUNTS), check_dtype=False)

    by_level = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="patient",
                                          covariate_key="covariate", phenotype_key="phenotype")
    pd.testing.assert_frame_equal(by_level, _expected(["patient", "covariate"], [
        ("P1", "t0", 3, 2, 1, 0.5, 2, 2 / 3, 2),
        ("P1", "t1", 3, 2, 1, 0.5, 2, 2 / 3, 2),
        ("P2", "t0", 2, 2, 2, 1.0, 1, 1 / 2, 1),
        ("P2", "t1", 1, 1, 1, 1.0, 1, 1.0, 0),
    ], [*COUNTS, "n_phenotypes_observed"]), check_dtype=False)

    # rows follow each column's category order, and labels are strings whatever the column holds
    adata.obs["patient"] = pd.Categorical(adata.obs["patient"], categories=["P2", "P1"])
    adata.obs["day"] = adata.obs["covariate"].map({"t0": 0, "t1": 7})
    ordered = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="patient",
                                         covariate_key="day")
    assert list(zip(ordered["patient"], ordered["covariate"])) == [
        ("P2", "0"), ("P2", "7"), ("P1", "0"), ("P1", "7")]


def test_groupby_none_is_one_whole_dataset_row():
    """``groupby=None`` is no grouping: one row for the whole dataset and no group column, the
    same row as a group holding every cell. A clonotype two patients carry is one clonotype of
    the dataset. With a covariate the whole dataset has one row per level."""
    adata = _adata(_obs())
    whole = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", phenotype_key="phenotype")
    pd.testing.assert_frame_equal(whole, _expected([], [(9, 5, 3, 0.6, 4, 4 / 9, 2)],
                                                   [*COUNTS, "n_phenotypes_observed"]))

    adata.obs["everyone"] = "all"
    one_group = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="everyone",
                                           phenotype_key="phenotype")
    pd.testing.assert_frame_equal(whole, one_group.drop(columns="everyone"))

    by_level = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id",
                                          covariate_key="covariate")
    pd.testing.assert_frame_equal(by_level, _expected(["covariate"], [
        ("t0", 5, 3, 2, 2 / 3, 3, 3 / 5),
        ("t1", 4, 3, 2, 2 / 3, 2, 1 / 2),
    ], COUNTS), check_dtype=False)


def _pooled():
    """A column pooled within each patient. P1 keeps c1@P1 and pools four cells, more than its
    one clone has; P2 keeps c2@P2 and pools one cell."""
    return _adata(pd.DataFrame({
        "clone_id_pooled": pd.Categorical(
            ["c1@P1"] * 3 + ["pooled@P1"] * 4 + ["c2@P2"] * 3 + ["pooled@P2"]),
        "patient": ["P1"] * 7 + ["P2"] * 4,
        "covariate": ["t0"] * 5 + ["t1"] * 2 + ["t0"] * 3 + ["t1"],
        "phenotype": pd.Categorical(["A", "B"] * 5 + ["A"], categories=["A", "B", "unused"]),
    }))


def _record_pooling(adata):
    _record_derivation(adata, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled", groupby="patient", min_cells=3,
                       pool_labels=["pooled@P1", "pooled@P2"], n_clones_pooled=3,
                       n_cells_pooled=5)


def test_repertoire_summary_counts_pools_separately():
    """On a pooled column the pools are counted in ``n_pools`` and ``cells_in_pools``, never as
    clones, and their cells stay in ``n_cells``. A row holding only a pool has no clone. Which
    labels are pools comes from the derivation record, followed through setup's step: without
    a pooling step the same labels are clones."""
    adata = _pooled()
    _record_pooling(adata)
    by_patient = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id_pooled",
                                            groupby="patient")
    pd.testing.assert_frame_equal(by_patient, _expected(["patient"], [
        ("P1", 7, 1, 0, 0.0, 3, 3 / 7, 1, 4),
        ("P2", 4, 1, 0, 0.0, 3, 3 / 4, 1, 1),
    ], [*COUNTS, *POOLS]), check_dtype=False)

    by_level = tcri.pp.repertoire_summary(adata, clonotype_key="clone_id_pooled",
                                          groupby="patient", covariate_key="covariate")
    pd.testing.assert_frame_equal(by_level, _expected(["patient", "covariate"], [
        ("P1", "t0", 5, 1, 0, 0.0, 3, 3 / 5, 1, 2),
        ("P1", "t1", 2, 0, 0, np.nan, 0, 0.0, 1, 2),
        ("P2", "t0", 3, 1, 0, 0.0, 3, 1.0, 0, 0),
        ("P2", "t1", 1, 0, 0, np.nan, 0, 0.0, 1, 1),
    ], [*COUNTS, *POOLS]), check_dtype=False)

    adata.obs[K.CLONOTYPE] = adata.obs["clone_id_pooled"]
    _record_derivation(adata, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, replicate="patient")
    pd.testing.assert_frame_equal(
        tcri.pp.repertoire_summary(adata, clonotype_key=K.CLONOTYPE, groupby="patient"),
        by_patient)

    unrecorded = tcri.pp.repertoire_summary(_pooled(), clonotype_key="clone_id_pooled",
                                            groupby="patient")
    pd.testing.assert_frame_equal(unrecorded, _expected(["patient"], [
        ("P1", 7, 2, 0, 0.0, 4, 4 / 7),
        ("P2", 4, 2, 1, 0.5, 3, 3 / 4),
    ], COUNTS), check_dtype=False)


def _entropy_denominators(adata, level, *, groupby, group_column):
    """Per group, the clone count ``tl.clonotypic_entropy`` at ``level`` normalized by.

    The normalized value is the entropy divided by the base-2 log of that count, so the count is
    two to the power of the entropy over its normalized value. A phenotype with zero entropy
    shows no count and is skipped.
    """
    arguments = dict(covariate=level, groupby=groupby, inplace=False)
    raw = tcri.tl.clonotypic_entropy(adata, normalized=False, **arguments)["result"]
    scaled = tcri.tl.clonotypic_entropy(adata, **arguments)["result"]
    both = raw.merge(scaled, on=[group_column, "phenotype"], suffixes=("_raw", "_scaled"))
    both = both[both["value_scaled"] > 0]
    counts = np.rint(2.0 ** (both["value_raw"] / both["value_scaled"])).astype(int)
    per_group = counts.groupby(both[group_column].astype(str).to_numpy()).unique()
    assert all(len(found) == 1 for found in per_group), per_group
    return {group: int(found[0]) for group, found in per_group.items()}


def test_repertoire_summary_n_clones_is_the_entropy_denominator(cohort):
    """On the registered column, grouped by the replicate, ``n_clones`` is the clone count
    clonotypic entropy normalizes by at each level when ``n_clones_ref`` is unset.

    The replicate is passed to both functions explicitly. ``tl`` resolves ``groupby=None`` to
    the registered replicate, while ``pp`` reads it as the whole dataset, so the whole-dataset
    row is an upper bound on every patient's count; on ids that never span patients it is their
    sum. Setting ``n_clones_ref`` to the largest ``n_clones`` keeps every normalized value at or
    below 1.
    """
    _, adata = cohort
    meta = adata.uns[K.METADATA]
    replicate = meta[K.Config.REPLICATE]
    columns = dict(clonotype_key=meta[K.CLONE_COL], covariate_key=meta[K.COVARIATE_COL])
    by_replicate = tcri.pp.repertoire_summary(adata, groupby=replicate, **columns)
    whole = tcri.pp.repertoire_summary(adata, **columns)
    largest = int(by_replicate["n_clones"].max())

    for level in adata.uns[K.COVARIATE_CATEGORIES]:
        rows = by_replicate[by_replicate["covariate"] == str(level)]
        n_clones = dict(zip(rows[replicate], rows["n_clones"].tolist()))
        assert len(n_clones) == adata.obs[replicate].nunique()
        assert _entropy_denominators(adata, level, groupby=replicate,
                                     group_column=replicate) == n_clones

        resolved = _entropy_denominators(adata, level, groupby=None, group_column=replicate)
        assert resolved == n_clones
        bound = whole.loc[whole["covariate"] == str(level), "n_clones"].item()
        assert max(resolved.values()) <= bound
        assert bound == sum(n_clones.values())

        pinned = tcri.tl.clonotypic_entropy(adata, covariate=level, n_clones_ref=largest,
                                            inplace=False)["result"]
        assert (pinned["value"] <= 1).all()


def test_repertoire_summary_raises_on_a_missing_or_clashing_column():
    """A column that is not in ``obs`` raises, naming the argument and the column. The group
    column is named after ``groupby``, so a ``groupby`` that names another column of the summary
    raises rather than replacing it."""
    adata = _adata(_obs())
    for argument in ("clonotype_key", "groupby", "covariate_key", "phenotype_key"):
        arguments = {"clonotype_key": "clone_id", argument: "absent"}
        with pytest.raises(ValueError, match=f"{argument}='absent' is not a column"):
            tcri.pp.repertoire_summary(adata, **arguments)

    adata.obs["n_cells"] = adata.obs["patient"]
    with pytest.raises(ValueError, match="groupby='n_cells' would name"):
        tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="n_cells")
    with pytest.raises(ValueError, match="groupby='covariate' would name"):
        tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="covariate",
                                   covariate_key="patient")


#: Each repertoire function, with arguments that read every column it takes.
CALLS = {
    "repertoire_summary": dict(clonotype_key="clone_id_pooled", groupby="patient",
                               covariate_key="covariate", phenotype_key="phenotype"),
}


@pytest.mark.parametrize("function", list(CALLS))
def test_repertoire_functions_do_not_mutate_adata(function):
    """The summaries only read: ``obs`` with its dtypes and categories, ``uns`` with the
    derivation record, and the matrix are left as they were."""
    adata = _pooled()
    _record_pooling(adata)
    before = adata.copy()
    getattr(tcri.pp, function)(adata, **CALLS[function])
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    assert list(adata.uns) == list(before.uns)
    assert adata.uns[K.CLONOTYPE_DERIVATIONS] == before.uns[K.CLONOTYPE_DERIVATIONS]
    np.testing.assert_array_equal(adata.X, before.X)
    assert list(adata.obsm) == list(before.obsm) and list(adata.layers) == list(before.layers)
