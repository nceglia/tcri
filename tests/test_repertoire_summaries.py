"""The repertoire summaries in ``tcri.pp``.

Each summary reads ``obs`` and the clonotype derivation record and stores its result in ``uns``,
returning it with ``inplace=False``. It filters nothing and needs no fit. Cells without a clonotype are left out and
not counted, and group labels and clonotypes come back as strings. ``repertoire_summary`` counts
cells and clones per group: its ``n_clones`` is the clone count clonotypic entropy normalizes by,
and on a pooled column its pools are counted apart from the clones. ``clonotype_sharing``
measures a column tcri derived on the column the record leads back to. ``clone_persistence`` has
a module of its own; every summary is checked here for storing its result and nothing else.
"""
from __future__ import annotations

import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from .conftest import frames
from tcri._compute._repertoire import _record_derivation
from tcri._state import keys as K

DTYPES = ["object", "category", "string", "str"]


def _adata(columns, *, complete=False):
    """An AnnData whose ``obs`` holds ``columns``, a dict or a DataFrame; ``complete`` keeps only
    the cells that have both a clonotype and a patient."""
    obs = pd.DataFrame(columns).copy()
    if complete:
        obs = obs[obs["clone_id"].notna() & obs["clone_id"].ne("") & obs["patient"].notna()]
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)


# ── repertoire_summary ────────────────────────────────────────────────────────────────────────

COUNTS = ["n_cells", "n_clones", "n_singleton_clones", "singleton_fraction",
          "largest_clone_cells", "largest_clone_share", "clonality"]
POOLS = ["n_pools", "cells_in_pools"]


def _obs():
    """Two patients at two timepoints. Clonotype c1 is carried by both patients, and three cells
    have no clonotype."""
    return pd.DataFrame({
        "clone_id": ["c1", "c1", "c1", "c2", "c3", "c3", "c1", "c4", "c5", "", None, "nan"],
        "patient": ["P1", "P1", "P1", "P1", "P1", "P1", "P2", "P2", "P2", "P1", "P2", "P2"],
        "covariate": ["t0", "t0", "t1", "t0", "t1", "t1", "t0", "t0", "t1", "t0", "t1", "t0"],
        "phenotype": ["A", "B", "A", "A", "B", "B", "A", "A", None, "B", "A", "C"],
    })


def _summary(adata, **arguments):
    """The table ``pp.repertoire_summary`` returns with ``inplace=False``."""
    return tcri.pp.repertoire_summary(adata, inplace=False, **arguments)


def _sharing(adata, part="groups", **arguments):
    """What ``pp.clonotype_sharing`` returns with ``inplace=False``: one of its frames, the
    ``groups`` frame by default, or the whole result with ``part=None``."""
    result = tcri.pp.clonotype_sharing(adata, inplace=False, **arguments)
    return result if part is None else result[part]


def _clonality(sizes):
    """``1 - H / log2(N)`` from its definition, over hand-listed clone sizes."""
    if len(sizes) < 2:
        return np.nan
    p = np.asarray(sizes, float) / sum(sizes)
    return 1 - float(-(p * np.log2(p)).sum()) / np.log2(len(sizes))


def _expected(labels, rows, columns):
    return pd.DataFrame([list(row) for row in rows], columns=[*labels, *columns])


def test_repertoire_summary_counts_cells_and_clones_per_row():
    """Every column, counted by hand. c1 is a clone in each patient that carries it, the cells
    without a clonotype are in no row, and a cell without a phenotype is not a phenotype."""
    adata = _adata(_obs())
    by_patient = _summary(adata, clonotype_key="clone_id", groupby="patient")
    pd.testing.assert_frame_equal(by_patient, _expected(["patient"], [
        ("P1", 6, 3, 1, 1 / 3, 3, 1 / 2, _clonality([3, 1, 2])),
        ("P2", 3, 3, 3, 1.0, 1, 1 / 3, _clonality([1, 1, 1])),
    ], COUNTS), check_dtype=False)

    by_level = _summary(adata, clonotype_key="clone_id", groupby="patient",
                                          covariate_key="covariate", phenotype_key="phenotype")
    pd.testing.assert_frame_equal(by_level, _expected(["patient", "covariate"], [
        ("P1", "t0", 3, 2, 1, 0.5, 2, 2 / 3, _clonality([2, 1]), 2),
        ("P1", "t1", 3, 2, 1, 0.5, 2, 2 / 3, _clonality([1, 2]), 2),
        ("P2", "t0", 2, 2, 2, 1.0, 1, 1 / 2, _clonality([1, 1]), 1),
        ("P2", "t1", 1, 1, 1, 1.0, 1, 1.0, np.nan, 0),
    ], [*COUNTS, "n_phenotypes_observed"]), check_dtype=False)

    # rows follow each column's category order, and labels are strings whatever the column holds
    adata.obs["patient"] = pd.Categorical(adata.obs["patient"], categories=["P2", "P1"])
    adata.obs["day"] = adata.obs["covariate"].map({"t0": 0, "t1": 7})
    ordered = _summary(adata, clonotype_key="clone_id", groupby="patient",
                                         covariate_key="day")
    assert list(zip(ordered["patient"], ordered["covariate"])) == [
        ("P2", "0"), ("P2", "7"), ("P1", "0"), ("P1", "7")]


def test_groupby_none_is_one_whole_dataset_row():
    """``groupby=None`` is no grouping: one row for the whole dataset and no group column, the
    same row as a group holding every cell. A clonotype two patients carry is one clonotype of
    the dataset. With a covariate the whole dataset has one row per level."""
    adata = _adata(_obs())
    whole = _summary(adata, clonotype_key="clone_id", phenotype_key="phenotype")
    pd.testing.assert_frame_equal(whole, _expected([], [(9, 5, 3, 0.6, 4, 4 / 9,
                                                         _clonality([4, 1, 2, 1, 1]), 2)],
                                                   [*COUNTS, "n_phenotypes_observed"]))

    adata.obs["everyone"] = "all"
    one_group = _summary(adata, clonotype_key="clone_id", groupby="everyone",
                                           phenotype_key="phenotype")
    pd.testing.assert_frame_equal(whole, one_group.drop(columns="everyone"))

    by_level = _summary(adata, clonotype_key="clone_id",
                                          covariate_key="covariate")
    pd.testing.assert_frame_equal(by_level, _expected(["covariate"], [
        ("t0", 5, 3, 2, 2 / 3, 3, 3 / 5, _clonality([3, 1, 1])),
        ("t1", 4, 3, 2, 2 / 3, 2, 1 / 2, _clonality([1, 2, 1])),
    ], COUNTS), check_dtype=False)


def _pooled():
    """A column pooled within each patient, beside the column it was pooled from. P1 keeps
    c1@P1 and pools four cells of two clones, more cells than its one kept clone has; P2 keeps
    c2@P2 and pools one cell."""
    return _adata(pd.DataFrame({
        "clone_id": ["c1"] * 3 + ["r1", "r1", "r2", "r2"] + ["c2"] * 3 + ["r3"],
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
    by_patient = _summary(adata, clonotype_key="clone_id_pooled",
                                            groupby="patient")
    pd.testing.assert_frame_equal(by_patient, _expected(["patient"], [
        ("P1", 7, 1, 0, 0.0, 3, 3 / 7, _clonality([3, 2, 2]), 1, 4),
        ("P2", 4, 1, 0, 0.0, 3, 3 / 4, _clonality([3, 1]), 1, 1),
    ], [*COUNTS, *POOLS]), check_dtype=False)

    by_level = _summary(adata, clonotype_key="clone_id_pooled",
                                          groupby="patient", covariate_key="covariate")
    pd.testing.assert_frame_equal(by_level, _expected(["patient", "covariate"], [
        ("P1", "t0", 5, 1, 0, 0.0, 3, 3 / 5, _clonality([3, 2]), 1, 2),
        ("P1", "t1", 2, 0, 0, np.nan, 0, 0.0, np.nan, 1, 2),
        ("P2", "t0", 3, 1, 0, 0.0, 3, 1.0, np.nan, 0, 0),
        ("P2", "t1", 1, 0, 0, np.nan, 0, 0.0, np.nan, 1, 1),
    ], [*COUNTS, *POOLS]), check_dtype=False)

    adata.obs[K.CLONOTYPE] = adata.obs["clone_id_pooled"]
    _record_derivation(adata, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, replicate="patient")
    pd.testing.assert_frame_equal(
        _summary(adata, clonotype_key=K.CLONOTYPE, groupby="patient"),
        by_patient)

    unrecorded = _summary(_pooled(), clonotype_key="clone_id_pooled",
                                            groupby="patient")
    pd.testing.assert_frame_equal(unrecorded, _expected(["patient"], [
        ("P1", 7, 2, 0, 0.0, 4, 4 / 7, _clonality([3, 4])),
        ("P2", 4, 2, 1, 0.5, 3, 3 / 4, _clonality([3, 1])),
    ], COUNTS), check_dtype=False)


def _entropy_denominators(adata, level, *, groupby, group_column):
    """Per group, the clone count ``tl.clonotypic_entropy`` at ``level`` normalized by.

    The normalized value is the entropy divided by the base-2 log of that count, so the count is
    two to the power of the entropy over its normalized value. A phenotype with zero entropy
    shows no count and is skipped.
    """
    arguments = dict(covariate=level, groupby=groupby, inplace=False)
    raw = frames(tcri.tl.clonotypic_entropy(adata, normalized=False, **arguments))[1]
    scaled = frames(tcri.tl.clonotypic_entropy(adata, **arguments))[1]
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
    by_replicate = _summary(adata, groupby=replicate, **columns)
    whole = _summary(adata, **columns)
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

        pinned = frames(tcri.tl.clonotypic_entropy(adata, covariate=level, n_clones_ref=largest,
                                            inplace=False))[1]
        assert (pinned["value"] <= 1).all()


def test_repertoire_summary_raises_on_a_missing_or_clashing_column():
    """A column that is not in ``obs`` raises, naming the argument and the column. The group
    column is named after ``groupby``, so a ``groupby`` that names another column of the summary
    raises rather than replacing it."""
    adata = _adata(_obs())
    for argument in ("clonotype_key", "groupby", "covariate_key", "phenotype_key"):
        arguments = {"clonotype_key": "clone_id", argument: "absent"}
        with pytest.raises(ValueError, match=f"{argument}='absent' is not a column"):
            _summary(adata, **arguments)

    adata.obs["n_cells"] = adata.obs["patient"]
    with pytest.raises(ValueError, match="groupby='n_cells' would name"):
        _summary(adata, clonotype_key="clone_id", groupby="n_cells")
    with pytest.raises(ValueError, match="groupby='covariate' would name"):
        _summary(adata, clonotype_key="clone_id", groupby="covariate",
                                   covariate_key="patient")


# ── clonotype_sharing ─────────────────────────────────────────────────────────────────────────

#: Clonotype a is carried by P1 and P2, c by P2 and P3, and e by all three; b and d are private.
#: Two cells have no clonotype and one has no patient.
COHORT = {
    "clone_id": ["a", "a", "b", "e", None, "a", "c", "c", "e", "", "c", "d", "d", "d", "e", "a"],
    "patient": ["P1", "P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2", "P2",
                "P3", "P3", "P3", "P3", "P3", None],
}

#: ``groups`` on COHORT by patient: e counts once in each group's shared clonotypes, however
#: many groups carry it.
EXPECTED_GROUPS = pd.DataFrame({
    "patient": ["P1", "P2", "P3"],
    "n_clonotypes": [3, 3, 3],
    "shared_clonotypes": [2, 3, 2],
    "shared_fraction": [2 / 3, 1.0, 2 / 3],
})

#: ``pairs``, each pair once: P1 and P2 share a and e, P1 and P3 share e, P2 and P3 share c and e.
EXPECTED_PAIRS = pd.DataFrame({
    "patient_a": ["P1", "P1", "P2"],
    "patient_b": ["P2", "P3", "P3"],
    "n_clonotypes_a": [3, 3, 3],
    "n_clonotypes_b": [3, 3, 3],
    "shared_clonotypes": [2, 1, 2],
    "shared_fraction_a": [2 / 3, 1 / 3, 2 / 3],
    "shared_fraction_b": [2 / 3, 1 / 3, 2 / 3],
})

#: ``clonotypes``: each shared clonotype's share of each group's counted cells. P1 and P2 count
#: four cells each, the cell without a clonotype and the one without a patient left out, and P3
#: five.
EXPECTED_CLONOTYPES = pd.DataFrame({
    "clonotype": ["a", "c", "e"],
    "n_groups": [2, 2, 3],
    "groups": ["P1|P2", "P2|P3", "P1|P2|P3"],
    "repertoire_fraction": ["0.5|0.25", "0.5|0.2", "0.25|0.25|0.2"],
})


def _pool_then_register(adata):
    """clone_id -> clone_id_pooled (pooling) -> tcri_clonotype (setup), with every id ending in
    ``@{patient}``, so no registered id is in two patients."""
    ids = (adata.obs["clone_id"].astype(str) + "@" + adata.obs["patient"].astype(str))
    adata.obs["clone_id_pooled"] = ids.astype("category")
    _record_derivation(adata, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled", groupby="patient", min_cells=1,
                       suffixed=True)
    adata.obs[K.CLONOTYPE] = ids.astype("category")
    _record_derivation(adata, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, replicate="patient")
    return adata


def _assert_sharing(result, groups, pairs, clonotypes):
    assert list(result) == ["pairs", "groups", "clonotypes"]
    pd.testing.assert_frame_equal(result["groups"], groups, check_dtype=False)
    pd.testing.assert_frame_equal(result["pairs"], pairs, check_dtype=False)
    pd.testing.assert_frame_equal(result["clonotypes"], clonotypes, check_dtype=False)


@pytest.mark.parametrize("dtype", DTYPES)
def test_clonotype_sharing_counts_groups_pairs_and_clonotypes(dtype):
    """Per group, its clonotypes and those also present in another group; per pair of groups,
    each once, the clonotypes of each and those both carry; per shared clonotype, its groups and
    its share of each group's counted cells. A clonotype carried by three groups counts once in a
    group's shared clonotypes, so a group's fraction is not a sum over its pairs. Cells without a
    clonotype or a group are not counted, and rows follow the group's category order."""
    adata = _adata({**COHORT, "clone_id": pd.Series(COHORT["clone_id"], dtype=dtype)})
    result = _sharing(adata, clonotype_key="clone_id", groupby="patient", part=None)
    _assert_sharing(result, EXPECTED_GROUPS, EXPECTED_PAIRS, EXPECTED_CLONOTYPES)
    p1 = EXPECTED_PAIRS[(EXPECTED_PAIRS["patient_a"] == "P1")]
    assert p1["shared_clonotypes"].sum() > EXPECTED_GROUPS.loc[0, "shared_clonotypes"]

    ordered = _adata({**COHORT, "patient": pd.Categorical(
        COHORT["patient"], categories=["P3", "P1", "P2", "P0"])})
    result = _sharing(ordered, clonotype_key="clone_id", groupby="patient", part=None)
    assert result["groups"]["patient"].tolist() == ["P3", "P1", "P2"]
    assert list(zip(result["pairs"]["patient_a"], result["pairs"]["patient_b"])) == [
        ("P3", "P1"), ("P3", "P2"), ("P1", "P2")]
    assert result["pairs"]["shared_clonotypes"].tolist() == [1, 2, 2]
    assert result["clonotypes"]["groups"].tolist() == ["P1|P2", "P3|P2", "P3|P1|P2"]
    assert result["clonotypes"]["repertoire_fraction"].tolist() == [
        "0.5|0.25", "0.2|0.5", "0.2|0.25|0.25"]

    numbered = _adata(COHORT, complete=True)
    numbered.obs["clone_number"] = numbered.obs["clone_id"].map(
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 10})
    numbered.obs["patient_number"] = numbered.obs["patient"].map({"P1": 1, "P2": 2, "P3": 3})
    result = _sharing(numbered, clonotype_key="clone_number", groupby="patient_number",
                      part=None)
    assert result["groups"]["patient_number"].tolist() == ["1", "2", "3"]
    assert result["pairs"]["patient_number_a"].tolist() == ["1", "1", "2"]
    assert result["clonotypes"]["clonotype"].tolist() == ["1", "3", "10"]
    assert result["clonotypes"]["groups"].tolist() == ["1|2", "2|3", "1|2|3"]

    for clonotype_key, groupby, named in [("trb", "patient", "clonotype_key='trb'"),
                                          ("clone_id", "donor", "groupby='donor'")]:
        with pytest.raises(ValueError, match=f"{named} is not a column of adata.obs"):
            _sharing(adata, clonotype_key=clonotype_key, groupby=groupby)


def test_a_group_sharing_nothing_has_pairs_of_zero():
    adata = _adata({"clone_id": [*COHORT["clone_id"], "z"],
                    "patient": [*COHORT["patient"], "P4"]})
    pairs = _sharing(adata, clonotype_key="clone_id", groupby="patient", part="pairs")
    with_p4 = pairs[pairs["patient_b"] == "P4"]
    assert with_p4[["patient_a", "n_clonotypes_b", "shared_clonotypes",
                    "shared_fraction_b"]].values.tolist() == [
        ["P1", 1, 0, 0.0], ["P2", 1, 0, 0.0], ["P3", 1, 0, 0.0]]


def test_clonotype_sharing_follows_the_derivation_record():
    """On a column tcri derived, sharing is measured on the source column the record leads back
    to; a missing source column raises."""
    adata = _pool_then_register(_adata(COHORT, complete=True))
    for key in ["clone_id", "clone_id_pooled", K.CLONOTYPE]:
        result = _sharing(adata, clonotype_key=key, groupby="patient", part=None)
        _assert_sharing(result, EXPECTED_GROUPS, EXPECTED_PAIRS, EXPECTED_CLONOTYPES)

    del adata.obs["clone_id"]
    with pytest.raises(ValueError, match="derived from 'clone_id', which is not a column"):
        _sharing(adata, clonotype_key=K.CLONOTYPE, groupby="patient")


def test_clonotype_sharing_raises_on_a_clashing_group_name():
    """The group column is named after ``groupby``, so a ``groupby`` that names another column
    raises rather than replacing it."""
    adata = _adata(COHORT, complete=True)
    for clashing in ("n_clonotypes", "shared_fraction"):
        adata.obs[clashing] = adata.obs["patient"]
        with pytest.raises(ValueError, match=f"groupby={clashing!r} would name"):
            _sharing(adata, clonotype_key="clone_id", groupby=clashing)


# ── every summary ─────────────────────────────────────────────────────────────────────────────

def _summarized():
    """COHORT with a categorical patient, a timepoint and a phenotype per cell, pooled and
    registered as in :func:`_pool_then_register`, and with an ``uns`` entry of its own."""
    adata = _pool_then_register(_adata({
        **COHORT, "patient": pd.Categorical(COHORT["patient"]),
        "timepoint": ["t0", "t1"] * 8, "phenotype": ["A", "B", "C", "A"] * 4}))
    adata.uns["kept"] = {"a": [1, 2]}
    return adata


#: Every summary in each of its modes, with arguments that read every column it takes.
PERSISTENCE = dict(clonotype_key=K.CLONOTYPE, covariate_key="timepoint")
CALLS = [
    ("repertoire_summary", dict(clonotype_key=K.CLONOTYPE, groupby="patient",
                                covariate_key="timepoint", phenotype_key="phenotype")),
    ("repertoire_summary", dict(clonotype_key="clone_id_pooled", phenotype_key="phenotype")),
    *[("clone_persistence", dict(PERSISTENCE, groupby=groupby)) for groupby in (None, "patient")],
    ("clonotype_sharing", dict(clonotype_key=K.CLONOTYPE, groupby="patient")),
    ("clonotype_sharing", dict(clonotype_key="clone_id", groupby="patient")),
]


def _unchanged_but_uns(adata, before):
    """``adata`` holds what ``before`` held: ``obs`` with its values, dtypes and categories, the
    derivation record, the matrix and every other slot. Returns the ``uns`` keys it gained."""
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    assert adata.uns[K.CLONOTYPE_DERIVATIONS] == before.uns[K.CLONOTYPE_DERIVATIONS]
    assert adata.uns["kept"] == before.uns["kept"]
    np.testing.assert_array_equal(adata.X, before.X)
    for slot in ("obsm", "varm", "layers", "obsp"):
        assert list(getattr(adata, slot)) == list(getattr(before, slot)), slot
    return [key for key in adata.uns if key not in before.uns]


@pytest.mark.parametrize(("function", "arguments"), [
    pytest.param(function, arguments, id=f"{function}-{i}")
    for i, (function, arguments) in enumerate(CALLS)])
def test_repertoire_functions_store_their_result_and_change_nothing_else(function, arguments):
    """With ``inplace=False`` a summary only reads: cells without a clonotype included, the
    object is left as it was. By default it adds one ``uns`` entry, ``tcri_<function>`` or the
    ``key_added`` it is given, and returns ``None``; ``tcri.get.result`` reads back what
    ``inplace=False`` returns."""
    adata = _summarized()
    before = adata.copy()
    returned = getattr(tcri.pp, function)(adata, inplace=False, **arguments)
    assert _unchanged_but_uns(adata, before) == []
    assert list(adata.uns) == list(before.uns)

    assert getattr(tcri.pp, function)(adata, **arguments) is None
    assert _unchanged_but_uns(adata, before) == [f"tcri_{function}"]
    stored = tcri.get.result(adata, function)
    if isinstance(returned, dict):
        assert set(stored) == set(returned)
        for part in returned:
            pd.testing.assert_frame_equal(stored[part], returned[part])
    else:
        pd.testing.assert_frame_equal(stored, returned)
    signature = inspect.signature(getattr(tcri.pp, function))
    defaults = {name: parameter.default for name, parameter in signature.parameters.items()
                if name not in ("adata", "key_added", "inplace")}
    assert tcri.get.params(adata, function) == {**defaults, **arguments}

    getattr(tcri.pp, function)(adata, key_added="mine", **arguments)
    assert _unchanged_but_uns(adata, before) == [f"tcri_{function}", "mine"]


# ── clonality, splitby, pairs and storage ─────────────────────────────────────────────────────

def test_clonality_is_measured_on_the_original_column():
    """On a pooled column, and on the column setup registered from it, clonality is that of the
    column pooling read, with every clone, singletons included: the derivation record leads back
    to it, and a missing source column raises. It follows the definition, ``1 - H / log2(N)``,
    and is NaN with one clone."""
    adata = _pooled()
    _record_pooling(adata)
    on_source = _summary(adata, clonotype_key="clone_id", groupby="patient",
                         covariate_key="covariate")
    on_pooled = _summary(adata, clonotype_key="clone_id_pooled", groupby="patient",
                         covariate_key="covariate")
    np.testing.assert_allclose(on_pooled["clonality"], on_source["clonality"])

    adata.obs[K.CLONOTYPE] = adata.obs["clone_id_pooled"]
    _record_derivation(adata, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, replicate="patient")
    registered = _summary(adata, clonotype_key=K.CLONOTYPE, groupby="patient",
                          covariate_key="covariate")
    np.testing.assert_allclose(registered["clonality"], on_source["clonality"])

    del adata.obs["clone_id"]
    for key in ("clone_id_pooled", K.CLONOTYPE):
        with pytest.raises(ValueError, match="derived from 'clone_id', which is not a column"):
            _summary(adata, clonotype_key=key, groupby="patient")

    sizes = np.random.default_rng(0).integers(1, 20, size=40)
    ragged = _adata({"clone_id": np.repeat([f"c{i}" for i in range(40)], sizes)})
    assert _summary(ragged, clonotype_key="clone_id")["clonality"].item() == pytest.approx(
        _clonality(list(sizes)))
    even = _adata({"clone_id": ["a", "a", "b", "b", "c", "c"]})
    assert _summary(even, clonotype_key="clone_id")["clonality"].item() == pytest.approx(0.0)
    one = _adata({"clone_id": ["a", "a", "a"]})
    assert np.isnan(_summary(one, clonotype_key="clone_id")["clonality"].item())


def _with_split():
    """COHORT with each patient's response and a timepoint per cell."""
    adata = _adata(COHORT, complete=True)
    adata.obs["response"] = adata.obs["patient"].map({"P1": "R", "P2": "NR", "P3": "R"})
    adata.obs["timepoint"] = (["t0", "t1"] * 8)[:adata.n_obs]
    return adata


def test_splitby_labels_each_group_and_is_checked_like_the_metrics():
    """Each frame carries the split label as a string: after the group column, or once for each
    side of a pair. ``splitby`` needs ``groupby``, must be constant within each group, and may
    not take the name of another column."""
    adata = _with_split()
    summary = _summary(adata, clonotype_key="clone_id", groupby="patient", splitby="response",
                       covariate_key="timepoint")
    assert list(summary.columns[:3]) == ["patient", "response", "covariate"]
    assert summary.drop_duplicates("patient")["response"].tolist() == ["R", "NR", "R"]

    sharing = _sharing(adata, clonotype_key="clone_id", groupby="patient", splitby="response",
                       part=None)
    assert list(sharing["groups"].columns[:2]) == ["patient", "response"]
    assert sharing["groups"]["response"].tolist() == ["R", "NR", "R"]
    assert list(sharing["pairs"].columns[:4]) == ["patient_a", "patient_b", "response_a",
                                                  "response_b"]
    assert list(zip(sharing["pairs"]["response_a"], sharing["pairs"]["response_b"])) == [
        ("R", "NR"), ("R", "R"), ("NR", "R")]

    persistence = tcri.pp.clone_persistence(adata, clonotype_key="clone_id",
                                            covariate_key="timepoint", groupby="patient",
                                            splitby="response", inplace=False)
    assert list(persistence.columns[:2]) == ["patient", "response"]

    adata.obs["response_number"] = adata.obs["patient"].map({"P1": 1, "P2": 0, "P3": 1})
    numbered = _summary(adata, clonotype_key="clone_id", groupby="patient",
                        splitby="response_number")
    assert numbered["response_number"].tolist() == ["1", "0", "1"]

    with pytest.raises(ValueError, match="splitby='response' needs groupby"):
        _summary(adata, clonotype_key="clone_id", splitby="response")
    adata.obs["mixed"] = (["x", "y"] * 8)[:adata.n_obs]
    with pytest.raises(ValueError, match="is not constant within"):
        _sharing(adata, clonotype_key="clone_id", groupby="patient", splitby="mixed")
    adata.obs["n_clonotypes"] = adata.obs["response"]
    with pytest.raises(ValueError, match="splitby='n_clonotypes' would name"):
        _sharing(adata, clonotype_key="clone_id", groupby="patient", splitby="n_clonotypes")
    with pytest.raises(ValueError, match="splitby='patient' would name"):
        _summary(adata, clonotype_key="clone_id", groupby="patient", splitby="patient")


def _stored_parts(adata, name):
    """``{part: frame}`` of a stored result; a result stored as one frame is the part
    ``frame``."""
    stored = tcri.get.result(adata, name)
    return dict(stored) if isinstance(stored, dict) else {"frame": stored}


def test_every_stored_summary_writes_to_h5ad_and_reads_back(tmp_path):
    """Every stored frame survives a write to ``.h5ad``, with integer covariate levels and a
    group whose cells carry no split label."""
    adata = _with_split()
    adata.obs["timepoint"] = adata.obs["timepoint"].map({"t0": 0, "t1": 7})
    adata.obs["response"] = adata.obs["response"].where(adata.obs["patient"] != "P3")
    for function, arguments in [
        ("repertoire_summary", dict(covariate_key="timepoint")),
        ("clonotype_sharing", {}),
        ("clone_persistence", dict(covariate_key="timepoint")),
    ]:
        getattr(tcri.pp, function)(adata, clonotype_key="clone_id", groupby="patient",
                                   splitby="response", **arguments)
    assert tcri.get.result(adata, "clonotype_sharing")["groups"]["response"].tolist() == [
        "R", "NR", ""]
    adata.write_h5ad(tmp_path / "summaries.h5ad")
    back = ad.read_h5ad(tmp_path / "summaries.h5ad")
    for name in ("repertoire_summary", "clonotype_sharing", "clone_persistence"):
        written, read = _stored_parts(adata, name), _stored_parts(back, name)
        assert set(read) == set(written), name
        for part, frame in written.items():
            pd.testing.assert_frame_equal(read[part], frame, check_dtype=False,
                                          obj=f"{name} {part}")
