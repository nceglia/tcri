"""``pp.clone_persistence``: which clones are seen at more than one covariate level.

The summary counts the clones the paired metrics are computed on: both deltas and
``phenotypic_flux`` use the clones of each group seen at both ``cov_from`` and ``cov_to``. It
reads ``obs`` only and writes nothing to the object.
"""
from __future__ import annotations

import contextlib
import copy
import io
import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
import tcri.tools._delta as delta_module
from tcri.datasets import simulate_tcri
from tcri.model._model import TCRIModel

KEYS = dict(clonotype_key="clone_id", covariate_key="timepoint")
CONTRAST = ["clones_from_only", "clones_to_only", "persistent_clones", "cells_from_only",
            "cells_to_only", "cells_in_persistent_clones", "has_persistent_clones"]
OVERVIEW = ["n_levels", "clones_at_one_level", "clones_at_multiple_levels",
            "cells_at_one_level", "cells_at_multiple_levels"]
DTYPES = ["object", "category", "string", "str"]


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


# ── the three modes ───────────────────────────────────────────────────────────────────────────

def test_contrast_mode_counts_clones_at_one_endpoint_and_at_both():
    """Per group: the clones seen at one of the two levels only, the clones seen at both, and
    their cells at those two levels. Cells at other levels and cells without a clonotype are
    not counted, and a group with no persistent clone keeps its row."""
    got = tcri.pp.clone_persistence(_cohort(), **KEYS, cov_from="t0", cov_to="t1",
                                    groupby="patient")
    assert list(got.columns) == ["patient"] + CONTRAST
    assert got.to_dict("records") == [
        {"patient": "P1", "clones_from_only": 1, "clones_to_only": 1, "persistent_clones": 1,
         "cells_from_only": 2, "cells_to_only": 1, "cells_in_persistent_clones": 3,
         "has_persistent_clones": True},
        {"patient": "P2", "clones_from_only": 1, "clones_to_only": 1, "persistent_clones": 0,
         "cells_from_only": 2, "cells_to_only": 2, "cells_in_persistent_clones": 0,
         "has_persistent_clones": False},
    ]
    assert (got[CONTRAST[:-1]].dtypes == "int64").all()
    assert got["has_persistent_clones"].dtype == bool


def test_a_level_contrasted_with_itself_counts_each_cell_once():
    """Every clone at the level is persistent, and its cells there are counted once."""
    got = tcri.pp.clone_persistence(_cohort(), **KEYS, cov_from="t0", cov_to="t0",
                                    groupby="patient")
    assert got[["persistent_clones", "cells_in_persistent_clones", "clones_from_only",
                "clones_to_only"]].values.tolist() == [[2, 4, 0, 0], [1, 2, 0, 0]]


def test_overview_mode_counts_clones_at_one_level_and_at_several():
    """Without the two levels, every level is counted: the levels each group has cells at,
    and the clones, with their cells, seen at exactly one level or at more than one."""
    got = tcri.pp.clone_persistence(_cohort(), **KEYS, groupby="patient")
    assert list(got.columns) == ["patient"] + OVERVIEW
    assert got.to_dict("records") == [
        {"patient": "P1", "n_levels": 3, "clones_at_one_level": 3,
         "clones_at_multiple_levels": 1, "cells_at_one_level": 5, "cells_at_multiple_levels": 4},
        {"patient": "P2", "n_levels": 2, "clones_at_one_level": 2,
         "clones_at_multiple_levels": 0, "cells_at_one_level": 4, "cells_at_multiple_levels": 0},
    ]
    assert (got[OVERVIEW].dtypes == "int64").all()


def test_per_clone_lists_each_clone_with_its_cells_per_level():
    """One row per (group, clonotype) with a column of cells per level. In a contrast the
    columns are ``cov_from`` then ``cov_to`` and only clones seen at either are listed."""
    adata = _cohort()
    every = tcri.pp.clone_persistence(adata, **KEYS, groupby="patient", per_clone=True)
    assert list(every.columns) == ["patient", "clonotype", "t0", "t1", "t2"]
    assert list(every.itertuples(index=False, name=None)) == [
        ("P1", "a", 2, 1, 1), ("P1", "b", 2, 0, 0), ("P1", "c", 0, 1, 0), ("P1", "d", 0, 0, 2),
        ("P2", "a", 0, 2, 0), ("P2", "e", 2, 0, 0),
    ]
    ends = tcri.pp.clone_persistence(adata, **KEYS, cov_from="t1", cov_to="t0",
                                     groupby="patient", per_clone=True)
    assert list(ends.columns) == ["patient", "clonotype", "t1", "t0"]
    assert list(ends.itertuples(index=False, name=None)) == [
        ("P1", "a", 1, 2), ("P1", "b", 0, 2), ("P1", "c", 1, 0), ("P2", "a", 2, 0),
        ("P2", "e", 0, 2),
    ]
    whole = tcri.pp.clone_persistence(adata, **KEYS, per_clone=True)
    assert list(whole.columns) == ["clonotype", "t0", "t1", "t2"]
    assert whole.loc[whole["clonotype"] == "a", ["t0", "t1", "t2"]].values.tolist() == [[2, 3, 1]]


def test_rows_and_level_columns_follow_category_order():
    """Groups, clonotypes and level columns come in each column's category order."""
    adata = _cohort()
    adata.obs["patient"] = pd.Categorical(adata.obs["patient"], categories=["P2", "P1"])
    adata.obs["timepoint"] = pd.Categorical(adata.obs["timepoint"],
                                            categories=["t2", "t1", "t0"])
    adata.obs["clone_id"] = pd.Categorical(adata.obs["clone_id"],
                                           categories=["e", "d", "c", "b", "a", ""])
    summary = tcri.pp.clone_persistence(adata, **KEYS, groupby="patient")
    assert summary["patient"].tolist() == ["P2", "P1"]
    every = tcri.pp.clone_persistence(adata, **KEYS, groupby="patient", per_clone=True)
    assert list(every.columns) == ["patient", "clonotype", "t2", "t1", "t0"]
    assert list(zip(every["patient"], every["clonotype"])) == [
        ("P2", "e"), ("P2", "a"), ("P1", "d"), ("P1", "c"), ("P1", "b"), ("P1", "a"),
    ]


def test_level_keys_are_strings():
    """Levels that are not strings name their columns as strings, and ``cov_from`` and
    ``cov_to`` match them either way."""
    adata = _cohort()
    adata.obs["timepoint"] = adata.obs["timepoint"].map({"t0": 0, "t1": 1, "t2": 2})
    every = tcri.pp.clone_persistence(adata, **KEYS, groupby="patient", per_clone=True)
    assert list(every.columns) == ["patient", "clonotype", "0", "1", "2"]
    expected = tcri.pp.clone_persistence(_cohort(), **KEYS, cov_from="t0", cov_to="t1",
                                         groupby="patient")
    for ends in (dict(cov_from=0, cov_to=1), dict(cov_from="0", cov_to="1")):
        got = tcri.pp.clone_persistence(adata, **KEYS, groupby="patient", **ends)
        pd.testing.assert_frame_equal(got, expected)


# ── arguments ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ends", [dict(cov_from="t0"), dict(cov_to="t1")])
def test_cov_from_and_cov_to_are_given_together(ends):
    with pytest.raises(ValueError, match="together or not at all"):
        tcri.pp.clone_persistence(_cohort(), **KEYS, **ends)


def test_a_level_must_belong_to_the_covariate():
    """A level the column does not have raises, naming it; a declared category with no cells
    is a level, and nothing is seen there."""
    adata = _cohort()
    with pytest.raises(ValueError, match=r"cov_to='t9' is not a level of "
                                         r"adata.obs\['timepoint'\]"):
        tcri.pp.clone_persistence(adata, **KEYS, cov_from="t0", cov_to="t9")
    adata.obs["timepoint"] = pd.Categorical(adata.obs["timepoint"],
                                            categories=["t0", "t1", "t2", "t3"])
    got = tcri.pp.clone_persistence(adata, **KEYS, cov_from="t0", cov_to="t3",
                                    groupby="patient")
    assert got[["clones_from_only", "clones_to_only", "persistent_clones"]].values.tolist() \
        == [[2, 0, 0], [1, 0, 0]]


@pytest.mark.parametrize("argument", ["clonotype_key", "covariate_key", "groupby"])
def test_a_missing_column_raises_naming_it(argument):
    arguments = dict(KEYS, groupby="patient")
    arguments[argument] = "absent"
    with pytest.raises(ValueError, match=f"{argument}='absent' is not a column of adata.obs"):
        tcri.pp.clone_persistence(_cohort(), **arguments)


def test_two_output_columns_with_one_name_raise():
    """A level named like another output column cannot be given a column of its own."""
    adata = _cohort()
    adata.obs["timepoint"] = adata.obs["timepoint"].replace({"t2": "clonotype"})
    with pytest.raises(ValueError, match="'clonotype'"):
        tcri.pp.clone_persistence(adata, **KEYS, per_clone=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_cells_without_a_clonotype_are_not_counted(dtype):
    """NaN, None, empty or whitespace-only strings and ``"nan"`` are no clone and no cell."""
    obs = pd.DataFrame({
        "clone_id": pd.Series(["a", "a", "", "nan", None, " ", "b"], dtype=dtype),
        "timepoint": ["t0", "t1", "t0", "t1", "t0", "t1", "t0"],
    })
    contrast = tcri.pp.clone_persistence(_adata(obs), **KEYS, cov_from="t0", cov_to="t1")
    assert contrast[CONTRAST].values.tolist() == [[1, 0, 1, 1, 0, 2, True]], dtype
    overview = tcri.pp.clone_persistence(_adata(obs), **KEYS)
    assert overview[OVERVIEW].values.tolist() == [[2, 1, 1, 1, 2]], dtype


# ── the whole dataset, and the object left as it was ──────────────────────────────────────────

def test_groupby_none_is_one_whole_dataset_row():
    """Without ``groupby`` the whole dataset is one row with no group column. On ids that never
    span groups its counts are the sums of the grouped rows; a clonotype that two groups carry
    is one clone instead of two. With no counted cell the row is still there."""
    adata = _cohort()
    for ends, columns in ((dict(cov_from="t0", cov_to="t1"), CONTRAST), ({}, OVERVIEW)):
        whole = tcri.pp.clone_persistence(adata, **KEYS, **ends)
        assert list(whole.columns) == columns and len(whole) == 1

    clone, patient = adata.obs["clone_id"], adata.obs["patient"]
    adata.obs["clone_patient"] = clone.where(clone == "", clone + "@" + patient)
    scoped = dict(clonotype_key="clone_patient", covariate_key="timepoint", cov_from="t0",
                  cov_to="t1")
    by_patient = tcri.pp.clone_persistence(adata, **scoped, groupby="patient")
    whole = tcri.pp.clone_persistence(adata, **scoped)
    assert whole[CONTRAST[:-1]].iloc[0].tolist() == by_patient[CONTRAST[:-1]].sum().tolist()

    # clonotype a is at t0 and t1 in P1 and at t1 in P2: one persistent clone of five cells
    shared = tcri.pp.clone_persistence(adata, **KEYS, cov_from="t0", cov_to="t1")
    assert shared[CONTRAST].values.tolist() == [[2, 1, 1, 4, 1, 5, True]]

    adata.obs["clone_id"] = ""
    for ends, row in ((dict(cov_from="t0", cov_to="t1"), [0] * 6 + [False]), ({}, [0] * 5)):
        assert tcri.pp.clone_persistence(adata, **KEYS, **ends).values.tolist() == [row]
        assert tcri.pp.clone_persistence(adata, **KEYS, **ends, groupby="patient").empty


def test_repertoire_functions_do_not_mutate_adata():
    """No mode writes to the object, converts a column or adds a key."""
    adata = _cohort()
    adata.obs["patient"] = adata.obs["patient"].astype("category")
    adata.uns["kept"] = {"a": [1, 2]}
    obs, uns = adata.obs.copy(deep=True), copy.deepcopy(dict(adata.uns))
    for ends in ({}, dict(cov_from="t0", cov_to="t1")):
        for groupby in (None, "patient"):
            for per_clone in (False, True):
                tcri.pp.clone_persistence(adata, **KEYS, **ends, groupby=groupby,
                                          per_clone=per_clone)
    pd.testing.assert_frame_equal(adata.obs, obs)
    assert dict(adata.uns) == uns
    assert not adata.obsm and not adata.layers and not adata.varm


# ── the clones the metrics use ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ragged_cohort():
    """A fitted three-patient AnnData whose clones are not all seen at both timepoints.

    In P0 and P1, clone 0 loses its cells at ``cov_1`` and clone 1 its cells at ``cov_0``. In
    P2, every clone loses its cells at one of the two, so P2 has no persistent clone. Clone ids
    are patient-scoped, which the metrics' ``groupby`` needs.
    """
    logging.disable(logging.INFO)
    parts = []
    for i, patient in enumerate(["P0", "P1", "P2"]):
        block = simulate_tcri(n_clones=8, n_phenotypes=3, n_genes=20, n_cells=160,
                              n_covariates=2, seed=i)
        clone = block.obs["clone_id"].cat.codes.to_numpy()
        level = block.obs["covariate"].cat.codes.to_numpy()
        if patient == "P2":
            drop = clone % 2 == level
        else:
            drop = ((clone == 0) & (level == 1)) | ((clone == 1) & (level == 0))
        block = block[~drop].copy()
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["patient"] = patient
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)

    adata = ad.concat(parts, join="outer", label=None)
    for column in ("clone_id", "phenotype", "covariate", "patient"):
        adata.obs[column] = adata.obs[column].astype("category")
    adata.layers["counts"] = adata.X.copy()
    # only the fit's clone x covariate rows are read, so a short fit will do, and its warnings
    # about small categories and the epoch budget say nothing about those rows
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", UserWarning)
        TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                                phenotype_key="phenotype", covariate_key="covariate",
                                batch_key="patient", replicate="patient")
        model = TCRIModel(adata, n_latent=4, n_hidden=8, n_layers=1, classifier_n_layers=1,
                          classifier_hidden=8, K=3, seed=0, name="persistence")
        model.train(max_epochs=2, batch_size=128, n_epochs_kl_warmup=1, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    logging.disable(logging.NOTSET)
    return adata


def test_clone_persistence_matches_delta_support(ragged_cohort, monkeypatch):
    """The persistent clones of each group are the clones both deltas hand the engine, and the
    clones ``delta_phenotypic_entropy`` and ``phenotypic_flux`` report. A group without a
    persistent clone is the group those results leave out.

    Run at ``null_model=None``: what is compared is which clones are used, not a value.
    """
    adata = ragged_cohort
    kw = dict(cov_from="cov_0", cov_to="cov_1", groupby="patient")
    keys = dict(clonotype_key="clone_id", covariate_key="covariate")
    clones = tcri.pp.clone_persistence(adata, **keys, **kw, per_clone=True)
    both = clones[(clones["cov_0"] > 0) & (clones["cov_1"] > 0)]
    persistent = set(zip(both["patient"], both["clonotype"]))
    summary = tcri.pp.clone_persistence(adata, **keys, **kw)
    with_persistent = set(summary.loc[summary["has_persistent_clones"], "patient"])
    assert len(clones) > len(both) and with_persistent == {"P0", "P1"}, \
        "the fixture cannot tell persistent clones apart"
    assert summary["persistent_clones"].sum() == len(persistent)

    handed = []
    joint_draws = delta_module.joint_draws

    def spy(adata, covariate, **kwargs):
        handed.extend(kwargs["clones"])
        return joint_draws(adata, covariate, **kwargs)

    monkeypatch.setattr(delta_module, "joint_draws", spy)
    results = {}
    with warnings.catch_warnings():
        # the deltas name the clones seen at one level only, which is the point here
        warnings.simplefilter("ignore", UserWarning)
        for name in ("delta_clonotypic_entropy", "delta_phenotypic_entropy",
                     "phenotypic_flux"):
            handed.clear()
            results[name] = getattr(tcri.tl, name)(adata, **kw, null_model=None,
                                                   inplace=False)["result"]
            if name != "phenotypic_flux":
                assert set(handed) == {c for _, c in persistent}, name
            assert set(results[name]["patient"]) == with_persistent, name
    for name in ("delta_phenotypic_entropy", "phenotypic_flux"):
        result = results[name]
        assert set(zip(result["patient"], result["clonotype"])) == persistent, name
