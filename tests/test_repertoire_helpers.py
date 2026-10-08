"""The private clonotype-label helpers and the clonotype derivation record.

Each rule about clonotype labels is stated once, in ``tcri/_compute/_repertoire.py``: which
values are label problems, which clonotypes are shared, how ids become replicate-specific, what a
(clonotype, covariate level) unit is, and how a derived clonotype column leads back to its source
and its pools. The metric-time clone check reads sharing through the same helper.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

import tcri
from tcri import get as tcri_get
from tcri._compute._repertoire import (
    _DERIVATION_FIELDS,
    TCRIDataWarning,
    _clonotype_sharing,
    _clonotype_source,
    _derivation_steps,
    _label_problems,
    _make_clonotypes_replicate_specific,
    _pool_labels,
    _record_derivation,
    _size_counts,
)
from tcri._compute._tables import _refit_hint, _validate_group_clones
from tcri._state import keys as K
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel

KEYS = dict(clonotype_key="clone_id", phenotype_key="phenotype", covariate_key="covariate",
            batch_key="batch")
KNOBS = dict(n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1, classifier_hidden=8,
             K=2)
DTYPES = ["object", "category", "string", "str"]


def _adata(obs):
    """An AnnData over ``obs`` with two phenotypes and one batch, ready for setup."""
    obs = obs.copy()
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    obs["phenotype"] = pd.Categorical(["A", "B"] * (len(obs) // 2) + ["A"] * (len(obs) % 2))
    obs["batch"] = pd.Categorical(["b0"] * len(obs))
    X = np.random.default_rng(0).poisson(2.0, size=(len(obs), 5)).astype(np.float32)
    return AnnData(X=X, obs=obs)


def _model_units(adata, clonotype_key):
    """The model's (clonotype, covariate level) units as rows: ids, levels and cells per unit."""
    with warnings.catch_warnings():
        # scvi warns about categories with fewer than three cells, which every frame here has
        warnings.simplefilter("ignore", UserWarning)
        TCRIModel.setup_anndata(adata, **{**KEYS, "clonotype_key": clonotype_key})
        module = TCRIModel(adata, **KNOBS).module
    clones = list(adata.obs[clonotype_key].astype("category").cat.categories)
    levels = list(adata.obs["covariate"].astype("category").cat.categories)
    return pd.DataFrame({
        "clonotype": [clones[c] for c in module.ct_to_c.numpy()],
        "covariate": [levels[t] for t in module.ct_to_cov.numpy()],
        "n_cells": np.bincount(module.ct_array.numpy(), minlength=module.ct_count),
    })


def _shared_cohort():
    """Clonotype c1 is carried by two patients and seen in both at t0."""
    return pd.DataFrame({
        "clone_id": ["c1", "c1", "c2", "c1", "c1", "c3", "c2", "c3"],
        "patient": ["P1", "P1", "P1", "P2", "P2", "P2", "P1", "P2"],
        "covariate": ["t0", "t1", "t0", "t0", "t0", "t1", "t1", "t1"],
    })


def test_tcri_data_warning_is_a_user_warning_of_its_own():
    """The data checks' warnings can be raised as errors without touching other UserWarnings."""
    assert issubclass(TCRIDataWarning, UserWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("error", TCRIDataWarning)
        warnings.warn("not a data check", UserWarning)
        with pytest.raises(TCRIDataWarning):
            warnings.warn("a data check", TCRIDataWarning)
    assert [str(w.message) for w in caught] == ["not a data check"]


# ── _label_problems ───────────────────────────────────────────────────────────────────────────

def test_label_problems_reports_each_kind():
    """One row per (column, problem, value) with its cell count: empty or whitespace-only
    strings, the literal ``"nan"``, and unused categories for the columns that ask for them.
    Rows follow the columns, then the problem order, then category order or first appearance."""
    obs = pd.DataFrame({
        "phenotype": pd.Categorical(["A", "", "nan", "A", " "],
                                    categories=["A", "B", "", " ", "nan", "nan2"]),
        "covariate": pd.Series(["t0", "nan", "t1", "", "nan"], dtype=object),
        "batch": pd.Series(["b0"] * 5, dtype="string"),
    })
    problems = _label_problems(obs, ["phenotype", "covariate", "batch"],
                               unused_for=["phenotype"])
    assert list(problems.columns) == ["column", "problem", "value", "n_cells"]
    assert list(problems.itertuples(index=False, name=None)) == [
        ("phenotype", "empty_string", "", 1),
        ("phenotype", "empty_string", " ", 1),
        ("phenotype", "literal_nan", "nan", 1),
        ("phenotype", "unused_category", "B", 0),
        ("phenotype", "unused_category", "nan2", 0),
        ("covariate", "empty_string", "", 1),
        ("covariate", "literal_nan", "nan", 2),
    ]
    clean = _label_problems(obs, ["batch"])
    assert clean.empty and list(clean.columns) == ["column", "problem", "value", "n_cells"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_label_problems_finds_the_strings_in_every_dtype(dtype):
    """The empty and ``"nan"`` strings are found in every dtype; near misses, NaN and None are
    not label problems."""
    values = ["a", "", " \t", "nan", "nan", "NaN", " nan", np.nan, None]
    problems = _label_problems(pd.DataFrame({"x": pd.Series(values, dtype=dtype)}), ["x"])
    assert list(problems.itertuples(index=False, name=None)) == [
        ("x", "empty_string", "", 1),
        ("x", "empty_string", " \t", 1),
        ("x", "literal_nan", "nan", 2),
    ], dtype


def test_label_problems_reports_unused_categories_only_where_asked():
    """An unused category is reported for a column in ``unused_for`` and nowhere else, and an
    unused ``""`` or ``"nan"`` category is unused rather than an empty or ``"nan"`` label,
    because no cell carries it."""
    column = pd.Categorical(["a", "b"], categories=["a", "b", "", "nan"])
    obs = pd.DataFrame({"x": column, "y": column})
    problems = _label_problems(obs, ["x", "y"], unused_for=["y"])
    assert list(problems.itertuples(index=False, name=None)) == [
        ("y", "unused_category", "", 0),
        ("y", "unused_category", "nan", 0),
    ]


# ── _clonotype_sharing ────────────────────────────────────────────────────────────────────────

def test_clonotype_sharing_order():
    """Shared clonotypes in order of first appearance, each with its groups in the order it
    first appears in them. A clonotype seen in one group is not listed."""
    clonotypes = pd.Series(["c2", "c1", "c2", "c1", "c3", "c2", "c3"])
    groups = pd.Series(["B", "A", "A", "B", "A", "C", "A"])
    assert list(_clonotype_sharing(clonotypes, groups).items()) == [
        ("c2", ["B", "A", "C"]),
        ("c1", ["A", "B"]),
    ]
    assert _clonotype_sharing(pd.Series(["c1", "c2"]), pd.Series(["A", "B"])) == {}


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_clonotype_sharing_ignores_na(dtype):
    """A cell with no clonotype or no group joins nothing: an NA clonotype in two groups is not
    shared, and a clonotype is not shared through a cell whose group is NA."""
    clonotypes = pd.Series(["c1", None, None, "c2", "c2", "c3", "c3"], dtype=dtype)
    groups = pd.Series(["A", "A", "B", "A", None, "A", "B"], dtype=dtype)
    assert _clonotype_sharing(clonotypes, groups) == {"c3": ["A", "B"]}


# ── _make_clonotypes_replicate_specific ───────────────────────────────────────────────────────

def test_replicate_specific_ids_pass_through_or_gain_the_replicate():
    """An id that already ends in its own cell's ``@{replicate}`` is copied unchanged; every
    other id gains the suffix, including one that ends in another replicate. Applying the rule
    again changes nothing, and the result keeps the source index."""
    clonotypes = pd.Series(["x", "y@P1", "x", "z@P0", "clone_3@P0"],
                           index=list("abcde"), name="clone_id")
    replicates = pd.Series(["P0", "P1", "P1", "P1", "P0"], index=list("abcde"))
    ids = _make_clonotypes_replicate_specific(clonotypes, replicates)
    assert ids.tolist() == ["x@P0", "y@P1", "x@P1", "z@P0@P1", "clone_3@P0"]
    assert list(ids.index) == list("abcde") and isinstance(ids.dtype, pd.CategoricalDtype)
    again = _make_clonotypes_replicate_specific(ids, replicates)
    assert again.tolist() == ids.tolist()


def test_replicate_specific_collision_names_the_pairs():
    """Two different (clonotype, replicate) pairs that would produce one id raise, and the
    message names both pairs and the id."""
    with pytest.raises(ValueError, match=r"\('x', 'P0'\) and \('x@P0', 'P0'\) would each "
                                         r"become 'x@P0'"):
        _make_clonotypes_replicate_specific(pd.Series(["x", "x@P0"]), pd.Series(["P0", "P0"]))
    with pytest.raises(ValueError, match=r"\('a', 'b@c'\) and \('a@b', 'c'\) would each "
                                         r"become 'a@b@c'"):
        _make_clonotypes_replicate_specific(pd.Series(["a@b", "a"]), pd.Series(["c", "b@c"]))


def test_replicate_specific_category_order():
    """Categories are the observed pairs, ordered by the source category order, then the
    replicate category order; a column that is not categorical is ordered as
    ``astype("category")`` orders it. Unused source categories leave no id behind."""
    clonotypes = pd.Series(pd.Categorical(["c1", "c3", "c1", "c2", "c3"],
                                          categories=["c3", "c1", "c2", "unused"]))
    replicates = pd.Series(pd.Categorical(["P1", "P2", "P2", "P1", "P1"],
                                          categories=["P2", "P1"]))
    ids = _make_clonotypes_replicate_specific(clonotypes, replicates)
    assert list(ids.cat.categories) == ["c3@P2", "c3@P1", "c1@P2", "c1@P1", "c2@P1"]
    assert ids.tolist() == ["c1@P1", "c3@P2", "c1@P2", "c2@P1", "c3@P1"]

    plain = _make_clonotypes_replicate_specific(clonotypes.astype(object),
                                                replicates.astype(object))
    assert list(plain.cat.categories) == ["c1@P1", "c1@P2", "c2@P1", "c3@P1", "c3@P2"]
    assert plain.tolist() == ids.tolist()


@pytest.mark.parametrize("clonotype, replicate, counts", [
    (np.nan, "P0", (1, 0)), ("", "P0", (1, 0)), ("nan", "P0", (1, 0)), ("x", None, (0, 1)),
])
def test_replicate_specific_refuses_missing_values(clonotype, replicate, counts):
    """A cell without a clonotype or a replicate raises instead of becoming an id such as
    ``nan@P0``."""
    with pytest.raises(ValueError, match=rf"{counts[0]} of 2 cells have no clonotype and "
                                         rf"{counts[1]} have no replicate"):
        _make_clonotypes_replicate_specific(pd.Series(["c1", clonotype]),
                                            pd.Series(["P0", replicate]))


# ── _size_counts ──────────────────────────────────────────────────────────────────────────────

def _declared_order():
    """Clone categories declared out of label order, with one unused."""
    obs = pd.DataFrame({
        "clone_id": pd.Categorical(["c2", "c1", "c3", "c1", "c2", "c3"],
                                   categories=["c3", "c1", "unused", "c2"]),
        "covariate": pd.Categorical(["t0", "t1", "t0", "t1", "t1", "t0"]),
    })
    return _adata(obs)


def _simulated():
    return simulate_tcri(n_clones=8, n_phenotypes=3, n_genes=20, n_cells=200, n_covariates=2,
                         omega_concentration=0.4, seed=0)


@pytest.mark.parametrize("make", [_declared_order, _simulated], ids=["declared-order", "simulated"])
def test_size_counts_agrees_with_the_model_units(make):
    """Without ``groupby``, the rows are the model's (clonotype, covariate level) units, in the
    order the model numbers them, with the cells of each unit."""
    adata = make()
    units = _size_counts(adata.obs, "clone_id", "covariate")
    pd.testing.assert_frame_equal(units, _model_units(adata, "clone_id"), check_dtype=False)


def test_size_counts_by_replicate_is_the_unit_of_replicate_specific_ids():
    """With ``groupby`` the unit is (group, clonotype, covariate level): a clonotype carried by
    two patients at the same level is two units, each with its own cells. Grouped by the
    replicate, they are the units the model builds once the ids are replicate-specific, in the
    model's order. Without ``groupby`` the shared clonotype is one unit."""
    obs = _shared_cohort()
    by_patient = _size_counts(obs, "clone_id", "covariate", groupby="patient")
    assert list(by_patient.columns) == ["group", "clonotype", "covariate", "n_cells"]
    assert list(by_patient.itertuples(index=False, name=None)) == [
        ("P1", "c1", "t0", 1),
        ("P1", "c1", "t1", 1),
        ("P2", "c1", "t0", 2),
        ("P1", "c2", "t0", 1),
        ("P1", "c2", "t1", 1),
        ("P2", "c3", "t1", 2),
    ]

    adata = _adata(obs)
    adata.obs["clone_patient"] = _make_clonotypes_replicate_specific(adata.obs["clone_id"],
                                                                     adata.obs["patient"])
    model = _model_units(adata, "clone_patient")
    assert (by_patient["clonotype"] + "@" + by_patient["group"]).tolist() == \
        model["clonotype"].tolist()
    assert by_patient["covariate"].tolist() == model["covariate"].tolist()
    assert by_patient["n_cells"].tolist() == model["n_cells"].tolist()

    whole = _size_counts(obs, "clone_id", "covariate")
    assert whole.loc[(whole["clonotype"] == "c1") & (whole["covariate"] == "t0"),
                     "n_cells"].tolist() == [3]


def test_size_counts_on_ids_that_never_span_groups_is_unchanged_by_groupby():
    """On a column whose ids never span groups, as on the registered column, grouping only adds
    the group label."""
    obs = _shared_cohort()
    obs["clone_patient"] = _make_clonotypes_replicate_specific(obs["clone_id"], obs["patient"])
    grouped = _size_counts(obs, "clone_patient", "covariate", groupby="patient")
    pd.testing.assert_frame_equal(grouped.drop(columns="group"),
                                  _size_counts(obs, "clone_patient", "covariate"))


@pytest.mark.parametrize("dtype", DTYPES)
def test_size_counts_leaves_out_cells_without_a_clonotype(dtype):
    """Cells without a clonotype are in no unit, nor are cells with no covariate level or, with
    ``groupby``, no group."""
    obs = pd.DataFrame({
        "clone_id": pd.Series(["c1", "", "nan", None, " ", "c2", "c1", "c2", "c1"], dtype=dtype),
        "covariate": ["t0", "t0", "t0", "t0", "t1", "t1", "t0", None, "t1"],
        "patient": ["P1"] * 8 + [None],
    })
    assert list(_size_counts(obs, "clone_id", "covariate").itertuples(index=False, name=None)) \
        == [("c1", "t0", 2), ("c1", "t1", 1), ("c2", "t1", 1)], dtype
    grouped = _size_counts(obs, "clone_id", "covariate", groupby="patient")
    assert list(grouped.itertuples(index=False, name=None)) == [
        ("P1", "c1", "t0", 2), ("P1", "c2", "t1", 1)], dtype


# ── the derivation record ─────────────────────────────────────────────────────────────────────

def _blank():
    return AnnData(X=np.zeros((2, 1), dtype=np.float32))


def _pool_then_setup(adata):
    """clone_id -> clone_id_pooled (pooling) -> tcri_clonotype (setup)."""
    _record_derivation(adata, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled", groupby="patient", min_cells=3,
                       samples="sample", suffixed=True, pool_labels=["pooled@P1", "pooled@P2"],
                       n_clones_pooled=4, n_cells_pooled=6)
    _record_derivation(adata, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, replicate="patient", suffixed=False)


def test_derivation_record_fields():
    """The record holds exactly the documented fields, with "", -1 and 0 where a field does not
    apply, and no field is named like the provenance ``tcri.get`` strips."""
    assert list(_DERIVATION_FIELDS) == [
        "function", "source", "key_added", "replicate", "groupby", "min_cells", "samples",
        "suffixed", "pool_labels", "n_clones_pooled", "n_cells_pooled"]
    assert set(_DERIVATION_FIELDS).isdisjoint(tcri_get._PROVENANCE)

    adata = _blank()
    _record_derivation(adata, function="setup_anndata", source="clone_id", key_added=K.CLONOTYPE)
    assert adata.uns[K.CLONOTYPE_DERIVATIONS] == {
        "function": ["setup_anndata"], "source": ["clone_id"], "key_added": [K.CLONOTYPE],
        "replicate": [""], "groupby": [""], "min_cells": [-1], "samples": [""],
        "suffixed": [False], "pool_labels": [""], "n_clones_pooled": [0], "n_cells_pooled": [0]}

    with pytest.raises(ValueError, match=r"cannot contain '\|'"):
        _record_derivation(adata, function="pool_rare_clones", source="clone_id",
                           key_added="clone_id_pooled", pool_labels=["pooled@A|B"])


def test_derivation_record_survives_h5ad_and_can_be_appended(tmp_path):
    """The record is parallel lists, so it survives ``write_h5ad``/``read_h5ad``. h5ad returns
    the lists as arrays, and the writer still appends to it, so a record read from disk keeps
    growing and round-trips again."""
    adata = _blank()
    _pool_then_setup(adata)
    adata.write_h5ad(tmp_path / "one.h5ad")
    back = read_h5ad(tmp_path / "one.h5ad")
    assert _derivation_steps(back) == _derivation_steps(adata)
    assert _clonotype_source(back, K.CLONOTYPE) == "clone_id"
    assert _pool_labels(back, K.CLONOTYPE) == ["pooled@P1", "pooled@P2"]

    _record_derivation(back, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled_5", groupby="patient", min_cells=5,
                       pool_labels=["pooled@P1"], n_clones_pooled=6, n_cells_pooled=9)
    assert all(isinstance(v, list) for v in back.uns[K.CLONOTYPE_DERIVATIONS].values())
    back.write_h5ad(tmp_path / "two.h5ad")
    again = read_h5ad(tmp_path / "two.h5ad")
    steps = _derivation_steps(again)
    assert [s["key_added"] for s in steps] == ["clone_id_pooled", K.CLONOTYPE,
                                               "clone_id_pooled_5"]
    assert steps[-1] == {
        "function": "pool_rare_clones", "source": "clone_id", "key_added": "clone_id_pooled_5",
        "replicate": "", "groupby": "patient", "min_cells": 5, "samples": "", "suffixed": False,
        "pool_labels": "pooled@P1", "n_clones_pooled": 6, "n_cells_pooled": 9}
    assert _pool_labels(again, "clone_id_pooled_5") == ["pooled@P1"]


def test_derivation_record_replaces_a_step_with_the_same_function_and_key():
    """Running a function again on the same column replaces its step, which then goes last;
    another function writing the same column is a step of its own."""
    adata = _blank()
    _pool_then_setup(adata)
    _record_derivation(adata, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled", groupby="patient", min_cells=5,
                       pool_labels=["pooled@P1"])
    _record_derivation(adata, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, replicate="sample")
    steps = _derivation_steps(adata)
    assert [(s["function"], s["key_added"]) for s in steps] == [
        ("pool_rare_clones", "clone_id_pooled"), ("setup_anndata", K.CLONOTYPE)]
    assert steps[0]["min_cells"] == 5 and steps[1]["replicate"] == "sample"

    _record_derivation(adata, function="pool_rare_clones", source="clone_id_pooled",
                       key_added=K.CLONOTYPE, groupby="patient", pool_labels=["pooled@P2"])
    assert len(_derivation_steps(adata)) == 3
    assert _pool_labels(adata, K.CLONOTYPE) == ["pooled@P2"]


def test_derivation_record_skips_a_column_derived_from_itself():
    """``source == key_added`` records nothing: no record is created, and an existing one is
    left exactly as it was."""
    adata = _blank()
    _record_derivation(adata, function="setup_anndata", source=K.CLONOTYPE,
                       key_added=K.CLONOTYPE)
    assert K.CLONOTYPE_DERIVATIONS not in adata.uns

    _pool_then_setup(adata)
    before = _derivation_steps(adata)
    _record_derivation(adata, function="setup_anndata", source=K.CLONOTYPE,
                       key_added=K.CLONOTYPE, replicate="sample")
    assert _derivation_steps(adata) == before
    assert _clonotype_source(adata, K.CLONOTYPE) == "clone_id"


def test_clonotype_source_follows_to_the_root_and_stops_on_a_self_reference():
    """The source of a derived column is the root of its chain; a column tcri did not derive
    has none. A step that names its own column as its source ends the walk."""
    adata = _blank()
    assert _clonotype_source(adata, K.CLONOTYPE) is None
    _pool_then_setup(adata)
    assert _clonotype_source(adata, K.CLONOTYPE) == "clone_id"
    assert _clonotype_source(adata, "clone_id_pooled") == "clone_id"
    assert _clonotype_source(adata, "clone_id") is None

    record = adata.uns[K.CLONOTYPE_DERIVATIONS]
    for field, default in _DERIVATION_FIELDS.items():
        record[field].append(default)
    record["function"][-1] = "pool_rare_clones"
    record["source"][-1] = record["key_added"][-1] = "clone_id"
    assert _clonotype_source(adata, "clone_id") == "clone_id"
    assert _clonotype_source(adata, K.CLONOTYPE) == "clone_id"


def test_pool_labels_follow_setup_back_to_the_pooling_step():
    """The registered column reports the pools of the pooled column setup read; a column with
    no pooling step behind it reports none."""
    adata = _blank()
    assert _pool_labels(adata, K.CLONOTYPE) == []
    _pool_then_setup(adata)
    assert _pool_labels(adata, K.CLONOTYPE) == ["pooled@P1", "pooled@P2"]
    assert _pool_labels(adata, "clone_id_pooled") == ["pooled@P1", "pooled@P2"]
    assert _pool_labels(adata, "clone_id") == []

    direct = _blank()
    _record_derivation(direct, function="setup_anndata", source="clone_id", key_added=K.CLONOTYPE)
    assert _pool_labels(direct, K.CLONOTYPE) == []

    nothing_pooled = _blank()
    _record_derivation(nothing_pooled, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled", groupby="patient", min_cells=1)
    _record_derivation(nothing_pooled, function="setup_anndata", source="clone_id_pooled",
                       key_added=K.CLONOTYPE)
    assert _pool_labels(nothing_pooled, K.CLONOTYPE) == []


# ── the metric-time clone check ───────────────────────────────────────────────────────────────

def _spans_groups(labels, groups):
    """Whether any clonotype spans groups, by scanning the groups one at a time."""
    seen = {}
    for g in groups.dropna().unique().tolist():
        for c in labels[(groups == g).to_numpy()].dropna().unique():
            if seen.setdefault(c, g) != g:
                return True
    return False


def test_clonotype_sharing_helper_agrees_with_metric_validator():
    """The metric check raises exactly when the sharing helper finds a shared clonotype, which
    is exactly when a group-by-group scan finds a clonotype in a second group. The error names
    the helper's first shared clonotype and its first two groups."""
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(1, 30))
        labels = pd.Series(rng.choice(["c1", "c2", "c3", "c4", None], size=n), dtype=object)
        groups = pd.Series(rng.choice(["P1", "P2", "P3", None], size=n), dtype=object)
        if trial % 2:
            labels, groups = labels.astype("category"), groups.astype("category")
        shared = _clonotype_sharing(labels, groups)
        assert bool(shared) == _spans_groups(labels, groups), (labels.tolist(), groups.tolist())
        if not shared:
            _validate_group_clones(labels, groups, "patient")
            continue
        c, spans = next(iter(shared.items()))
        with pytest.raises(ValueError) as exc:
            _validate_group_clones(labels, groups, "patient")
        assert f"clonotype {c!r} spans groups {spans[0]!r} and {spans[1]!r}." in str(exc.value)


def test_validate_group_clones_message_and_hint_unchanged():
    """The message, the hint appended to it, and the model-free caller in ``diag``."""
    labels = pd.Series(["c1", "c2", "c1"])
    groups = pd.Series(["P1", "P1", "P2"])
    message = (
        "groupby='patient': clonotype 'c1' spans groups 'P1' and 'P2'. The metric groupby "
        "restricts by clone id (clones=), which requires clones to be disjoint across groups "
        "(e.g. patient-specific `trb_unique`). Use a clone-disjoint groupby, or pre-filter with "
        "`clones=`."
    )
    with pytest.raises(ValueError) as exc:
        _validate_group_clones(labels, groups, "patient")
    assert str(exc.value) == message
    with pytest.raises(ValueError) as exc:
        _validate_group_clones(labels, groups, "patient", hint=" This is fit 'x'.")
    assert str(exc.value) == message + " This is fit 'x'."

    adata = _adata(pd.DataFrame({"clone_id": labels, "patient": groups,
                                 "covariate": ["t0", "t0", "t0"]}))
    adata.uns[K.fit_key(K.FIT_SETTINGS, "null.clonotype")] = {"kind": "clonotype",
                                                              "strata": ["batch"]}
    hint = _refit_hint(adata, "null.clonotype", "patient")
    with pytest.raises(ValueError) as exc:
        _validate_group_clones(labels, groups, "patient", hint=hint)
    assert str(exc.value) == message + hint
    assert "null.clonotype" in hint and "within=" in hint

    adata.uns[K.METADATA] = {"clone_col": "clone_id", "phenotype_col": "phenotype",
                             "covariate_col": "covariate"}
    adata.uns[K.PHENOTYPE_CATEGORIES] = ["A", "B"]
    adata.uns[K.COVARIATE_CATEGORIES] = ["t0"]
    with pytest.raises(ValueError) as exc:
        tcri.diag.permutation_null(adata, groupby="patient", n_perm=1)
    assert str(exc.value) == message
