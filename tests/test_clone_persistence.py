"""``pp.clone_persistence``: how many clones each group carries between covariate levels.

For each group and pair of levels at which it has clones, the clones at each level and the
clones seen at both. On the registered column, the shared clones of a pair of levels are the
persistent clones a change between them is computed on, which leaves pools out by default; a pool
is not a clone and is never shared. The summary reads ``obs`` and the clonotype derivation
record and stores one frame.
"""
from __future__ import annotations

import contextlib
import io
import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from tcri._compute._repertoire import _record_derivation
from tcri._state import keys as K
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel

KEYS = dict(clonotype_key="clone_id", covariate_key="timepoint")
COUNTS = ["covariate_a", "covariate_b", "n_clones_a", "n_clones_b", "shared_clones",
          "shared_fraction_a", "shared_fraction_b"]
DTYPES = ["object", "category", "string", "str"]


def _persistence(adata, **arguments):
    """The frame ``pp.clone_persistence`` returns with ``inplace=False``."""
    return tcri.pp.clone_persistence(adata, inplace=False, **arguments)


def _adata(obs):
    obs = obs.copy()
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)


def _cohort():
    """Two patients over three timepoints.

    P1: clone a at t0, t1 and t2; b at t0 only; c at t1 only; d at t2 only; and one cell
    without a clonotype, at t0. P2: clonotype a again, at t1 only, and e at t0 only.
    """
    rows = [
        ("a", "P1", "t0"), ("a", "P1", "t0"), ("a", "P1", "t1"), ("a", "P1", "t2"),
        ("b", "P1", "t0"), ("b", "P1", "t0"),
        ("c", "P1", "t1"),
        ("d", "P1", "t2"), ("d", "P1", "t2"),
        ("", "P1", "t0"),
        ("e", "P2", "t0"), ("e", "P2", "t0"),
        ("a", "P2", "t1"), ("a", "P2", "t1"),
    ]
    return _adata(pd.DataFrame(rows, columns=["clone_id", "patient", "timepoint"]))


# ── the pairs ─────────────────────────────────────────────────────────────────────────────────

def test_each_pair_of_levels_counts_the_clones_at_each_and_at_both():
    """Counted by hand on the cohort. P1 has clones a and b at t0, a and c at t1, a and d at t2:
    each pair shares a. P2 has e at t0 and a at t1, which share nothing, and no clone at t2, so
    no pair with t2. Each pair is one row, first level before the second."""
    got = _persistence(_cohort(), **KEYS, groupby="patient")
    assert list(got.columns) == ["patient"] + COUNTS
    assert got.values.tolist() == [
        ["P1", "t0", "t1", 2, 2, 1, 0.5, 0.5],
        ["P1", "t0", "t2", 2, 2, 1, 0.5, 0.5],
        ["P1", "t1", "t2", 2, 2, 1, 0.5, 0.5],
        ["P2", "t0", "t1", 1, 1, 0, 0.0, 0.0],
    ]
    assert (got[["n_clones_a", "n_clones_b", "shared_clones"]].dtypes == "int64").all()


def test_groupby_none_is_the_whole_dataset():
    """Without ``groupby`` the whole dataset is one group with no group column, where a
    clonotype that two patients carry is one clone: a, b and e at t0, a and c at t1, a and d at
    t2. With no counted cell there is no pair."""
    got = _persistence(_cohort(), **KEYS)
    assert list(got.columns) == COUNTS
    assert got.values.tolist() == [
        ["t0", "t1", 3, 2, 1, 1 / 3, 0.5],
        ["t0", "t2", 3, 2, 1, 1 / 3, 0.5],
        ["t1", "t2", 2, 2, 1, 0.5, 0.5],
    ]
    adata = _cohort()
    adata.obs["clone_id"] = ""
    assert _persistence(adata, **KEYS).empty
    assert _persistence(adata, **KEYS, groupby="patient").empty


def test_rows_follow_category_order_and_labels_are_strings():
    """Groups and levels come in each column's category order, the first level of a pair before
    the second, and as strings whatever the columns hold."""
    adata = _cohort()
    adata.obs["patient"] = pd.Categorical(adata.obs["patient"], categories=["P2", "P1"])
    adata.obs["timepoint"] = pd.Categorical(adata.obs["timepoint"],
                                            categories=["t2", "t1", "t0"])
    got = _persistence(adata, **KEYS, groupby="patient")
    assert list(zip(got["patient"], got["covariate_a"], got["covariate_b"])) == [
        ("P2", "t1", "t0"), ("P1", "t2", "t1"), ("P1", "t2", "t0"), ("P1", "t1", "t0")]

    numbered = _cohort()
    numbered.obs["timepoint"] = numbered.obs["timepoint"].map({"t0": 0, "t1": 1, "t2": 2})
    numbered.obs["patient"] = numbered.obs["patient"].map({"P1": 1, "P2": 2})
    got = _persistence(numbered, **KEYS, groupby="patient")
    expected = _persistence(_cohort(), **KEYS, groupby="patient")
    assert got["patient"].tolist() == ["1", "1", "1", "2"]
    assert got["covariate_a"].tolist() == ["0", "0", "1", "0"]
    pd.testing.assert_frame_equal(got[COUNTS[2:]], expected[COUNTS[2:]])


def test_splitby_labels_each_group():
    adata = _cohort()
    adata.obs["response"] = adata.obs["patient"].map({"P1": "R", "P2": "NR"})
    got = _persistence(adata, **KEYS, groupby="patient", splitby="response")
    assert list(got.columns[:2]) == ["patient", "response"]
    assert got["response"].tolist() == ["R", "R", "R", "NR"]
    with pytest.raises(ValueError, match="splitby='response' needs groupby"):
        _persistence(adata, **KEYS, splitby="response")


@pytest.mark.parametrize("argument", ["clonotype_key", "covariate_key", "groupby", "splitby"])
def test_a_missing_column_raises_naming_it(argument):
    arguments = dict(KEYS, groupby="patient")
    arguments[argument] = "absent"
    with pytest.raises(ValueError, match=f"{argument}='absent' is not a column of adata.obs"):
        _persistence(_cohort(), **arguments)


def test_a_group_column_named_like_another_column_raises():
    adata = _cohort()
    adata.obs["shared_clones"] = adata.obs["patient"]
    with pytest.raises(ValueError, match="groupby='shared_clones' would name"):
        _persistence(adata, **KEYS, groupby="shared_clones")


def test_pools_are_never_shared():
    """A pool holds many rare clones, so it is not a clone: on a column whose derivation record
    lists its pools, their cells are in no count. P1 keeps one clone at both timepoints and P2
    one at t0 only, while each patient's pool has cells at both. Without the record nothing
    marks the pools, and each label is a clone."""
    obs = pd.DataFrame({
        "clone_id_pooled": ["c1@P1"] * 4 + ["pooled@P1"] * 4 + ["c2@P2"] * 3 + ["pooled@P2"] * 2,
        "patient": ["P1"] * 8 + ["P2"] * 5,
        "timepoint": ["t0", "t0", "t0", "t1", "t0", "t0", "t1", "t1", "t0", "t0", "t0", "t0",
                      "t1"],
    })
    adata = _adata(obs)
    _record_derivation(adata, function="pool_rare_clones", source="clone_id",
                       key_added="clone_id_pooled", groupby="patient", min_cells=3,
                       pool_labels=["pooled@P1", "pooled@P2"], n_clones_pooled=4,
                       n_cells_pooled=6)
    kw = dict(clonotype_key="clone_id_pooled", covariate_key="timepoint", groupby="patient")
    assert _persistence(adata, **kw).values.tolist() == [["P1", "t0", "t1", 1, 1, 1, 1.0, 1.0]]

    del adata.uns[K.CLONOTYPE_DERIVATIONS]
    assert _persistence(adata, **kw).values.tolist() == [
        ["P1", "t0", "t1", 2, 2, 2, 1.0, 1.0], ["P2", "t0", "t1", 2, 1, 1, 0.5, 1.0]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_cells_without_a_clonotype_are_not_counted(dtype):
    """NaN, None, empty or whitespace-only strings and ``"nan"`` are no clone."""
    obs = pd.DataFrame({
        "clone_id": pd.Series(["a", "a", "", "nan", None, " ", "b"], dtype=dtype),
        "timepoint": ["t0", "t1", "t0", "t1", "t0", "t1", "t0"],
    })
    got = _persistence(_adata(obs), **KEYS)
    assert got.values.tolist() == [["t0", "t1", 2, 1, 1, 0.5, 1.0]], dtype


# ── the clones the metrics use ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ragged_cohort():
    """A fitted three-patient AnnData whose clones are not all seen at both timepoints.

    In P0 and P1, clone 0 loses its cells at ``cov_1`` and clone 1 its cells at ``cov_0``. In
    P2, clones 0 to 5 each lose their cells at one of the two, so P2 has no persistent clone.
    In every patient clone 6 keeps two cells at ``cov_0`` and clone 7 two at ``cov_1``: both are
    rare, and pooling gathers them into one pool per patient with cells at both timepoints. The
    pooled column, ``clone_id_pooled``, is the one registered. Clone ids are patient-scoped,
    which the metrics' ``groupby`` needs.
    """
    logging.disable(logging.INFO)
    parts = []
    for i, patient in enumerate(["P0", "P1", "P2"]):
        block = simulate_tcri(n_clones=8, n_phenotypes=3, n_genes=20, n_cells=160,
                              n_covariates=2, seed=i)
        clone = block.obs["clone_id"].cat.codes.to_numpy()
        level = block.obs["covariate"].cat.codes.to_numpy()
        if patient == "P2":
            drop = (clone < 6) & (clone % 2 == level)
        else:
            drop = ((clone == 0) & (level == 1)) | ((clone == 1) & (level == 0))
        for rare, kept in ((6, 0), (7, 1)):
            drop |= (clone == rare) & (level != kept)
            drop[np.flatnonzero((clone == rare) & (level == kept))[2:]] = True
        block = block[~drop].copy()
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["patient"] = patient
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)

    adata = ad.concat(parts, join="outer", label=None)
    for column in ("clone_id", "phenotype", "covariate", "patient"):
        adata.obs[column] = adata.obs[column].astype("category")
    adata.layers["counts"] = adata.X.copy()
    tcri.pp.pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    # only the fit's clone x covariate rows are read, so a short fit will do, and its warnings
    # about small categories and the epoch budget say nothing about those rows
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", UserWarning)
        TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id_pooled",
                                phenotype_key="phenotype", covariate_key="covariate",
                                batch_key="patient", replicate="patient")
        model = TCRIModel(adata, n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1,
                          classifier_hidden=8, K=3, seed=0, name="persistence")
        model.train(max_epochs=2, batch_size=128, n_epochs_kl_warmup=1, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    logging.disable(logging.NOTSET)
    return adata


def test_shared_clones_are_the_clones_a_change_is_computed_on(ragged_cohort):
    """Per patient, the shared clones of (``cov_0``, ``cov_1``) on the registered column are the
    clones flux and the phenotypic entropy change report, at the metrics' default
    ``exclude_pools=True``. Each patient's pool has cells at both timepoints and is not one of
    them, and P2, with none, has no rows in those results.

    Run at ``null_model=None``: what is compared is which clones are used, not a value.
    """
    adata = ragged_cohort
    obs = adata.obs
    for patient in ("P0", "P1", "P2"):
        pool = obs.loc[obs["clone_id_pooled"] == f"pooled@{patient}", "covariate"]
        assert set(pool) == {"cov_0", "cov_1"}, f"{patient} has no pool at both timepoints"
    pairs = _persistence(adata, clonotype_key="clone_id_pooled", covariate_key="covariate",
                         groupby="patient")
    pair = pairs[(pairs["covariate_a"] == "cov_0") & (pairs["covariate_b"] == "cov_1")]
    shared = dict(zip(pair["patient"], pair["shared_clones"]))
    assert shared["P2"] == 0 and shared["P0"] > 0 and shared["P1"] > 0, \
        "the fixture cannot tell persistent clones apart"

    kw = dict(cov_from="cov_0", cov_to="cov_1", groupby="patient", null_model=None,
              inplace=False)
    with warnings.catch_warnings():
        # the metrics name the clones seen at one level only, which is the point here
        warnings.simplefilter("ignore", UserWarning)
        for name in ("delta_phenotypic_entropy", "phenotypic_flux"):
            result = getattr(tcri.tl, name)(adata, **kw)["result"]
            reported = result.groupby("patient", observed=True)["clonotype"].nunique()
            assert {p: int(reported.get(p, 0)) for p in shared} == shared, name
