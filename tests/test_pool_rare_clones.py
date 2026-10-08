"""Pooling rare clones: which clones are rare, the column written, and the derivation step.

A clone is the cells of one clonotype within one group. It is rare when it has fewer than
``min_cells`` cells and all of them come from one sample. Each group's rare clones become one
``pooled@{group}`` label, and every other clone keeps its id, scoped to its group.
"""
from __future__ import annotations

import inspect
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

from tcri._compute._repertoire import (
    TCRIDataWarning,
    _clonotype_source,
    _derivation_steps,
    _pool_labels,
    _pool_rare_clones,
)
from tcri._state import keys as K

DTYPES = ["object", "category", "string", "str"]
MISSING = [np.nan, None, "", "   ", "nan"]


def _adata(dtype="object", **columns):
    """An AnnData over ``columns``; ``clone_id`` is stored in ``dtype``."""
    n = len(next(iter(columns.values())))
    names = [f"cell_{i}" for i in range(n)]
    obs = pd.DataFrame({name: pd.Series(values, index=names,
                                        dtype=dtype if name == "clone_id" else "object")
                        for name, values in columns.items()}, index=names)
    return AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)


def _cohort(dtype="object"):
    """P1: c1 x3, c2 x2, c3 x1. P2: c1 x2, c4 x4. Clonotype c1 is carried by both patients."""
    return _adata(dtype,
                  clone_id=["c1"] * 3 + ["c2"] * 2 + ["c3"] + ["c1"] * 2 + ["c4"] * 4,
                  patient=["P1"] * 6 + ["P2"] * 6)


#: The labels _pool_rare_clones writes for _cohort() at the default min_cells.
COHORT_LABELS = (["c1@P1"] * 3 + ["pooled@P1"] * 3 + ["pooled@P2"] * 2 + ["c4@P2"] * 4)


def _labels(adata, key="clone_id_pooled"):
    return adata.obs[key].astype(str).tolist()


@pytest.mark.parametrize("dtype", DTYPES)
def test_pool_counts_within_group_and_is_order_independent(dtype):
    """Cells are counted per (group, clonotype): c1 has five cells, but three in P1, which
    keep it, and two in P2, which pool it. Shuffling the cells changes no cell's label."""
    adata = _cohort(dtype)
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert _labels(adata) == COHORT_LABELS

    order = np.random.default_rng(0).permutation(adata.n_obs)
    shuffled = _cohort(dtype)[order].copy()
    _pool_rare_clones(shuffled, clonotype_key="clone_id", groupby="patient")
    assert shuffled.obs["clone_id_pooled"].astype(str).to_dict() == \
        adata.obs["clone_id_pooled"].astype(str).to_dict()
    assert list(shuffled.obs["clone_id_pooled"].cat.categories) == \
        list(adata.obs["clone_id_pooled"].cat.categories)
    assert _derivation_steps(shuffled) == _derivation_steps(adata)


def test_pool_shared_clonotype_is_one_clone_per_group():
    """A clonotype carried by two patients is two clones with two ids, each counted on its own
    cells: three cells in each patient keep both, two in each pool both, although the
    clonotype has four cells in all."""
    adata = _adata(clone_id=["c1"] * 6 + ["c2"] * 6, patient=["P1", "P2"] * 6)
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert adata.obs["clone_id_pooled"].value_counts().to_dict() == {
        "c1@P1": 3, "c1@P2": 3, "c2@P1": 3, "c2@P2": 3}

    adata = _adata(clone_id=["c1"] * 4 + ["c2"] * 6, patient=["P1", "P2"] * 2 + ["P1"] * 3
                   + ["P2"] * 3)
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert _labels(adata) == ["pooled@P1", "pooled@P2"] * 2 + ["c2@P1"] * 3 + ["c2@P2"] * 3
    assert _derivation_steps(adata)[0]["n_clones_pooled"] == 2


@pytest.mark.parametrize("min_cells", [2, 3, 5])
def test_pool_min_cells_boundary(min_cells):
    """A clone of exactly min_cells cells is kept; one cell fewer is pooled."""
    adata = _adata(clone_id=["at"] * min_cells + ["under"] * (min_cells - 1),
                   patient=["P1"] * (2 * min_cells - 1))
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", min_cells=min_cells)
    assert _labels(adata) == ["at@P1"] * min_cells + ["pooled@P1"] * (min_cells - 1)


def test_pool_default_min_cells_is_3():
    """The default is 3, in the signature, in what is pooled, and in the record."""
    assert inspect.signature(_pool_rare_clones).parameters["min_cells"].default == 3
    adata = _adata(clone_id=["three"] * 3 + ["two"] * 2, patient=["P1"] * 5)
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert _labels(adata) == ["three@P1"] * 3 + ["pooled@P1"] * 2
    assert _derivation_steps(adata)[0]["min_cells"] == 3


def test_pool_rare_definition():
    """Rare means fewer than min_cells cells, all from one sample. Two cells seen in two samples
    are kept at any min_cells; two cells from one sample are pooled; without samples, the rule
    is min_cells alone."""
    def run(**kwargs):
        adata = _adata(clone_id=["two_samples"] * 2 + ["one_sample"] * 2,
                       patient=["P1"] * 4, sample=["s1", "s2", "s1", "s1"])
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", **kwargs)
        return _labels(adata)

    for min_cells in (3, 5, 100):
        assert run(min_cells=min_cells, samples="sample") == (
            ["two_samples@P1"] * 2 + ["pooled@P1"] * 2), min_cells
    with pytest.warns(TCRIDataWarning):
        assert run(min_cells=3) == ["pooled@P1"] * 4
    assert run(min_cells=2) == ["two_samples@P1"] * 2 + ["one_sample@P1"] * 2

    # samples are counted within each group's clone: a sample shared with another patient does
    # not make a clone seen in one sample of its own patient any less rare
    adata = _adata(clone_id=["c1", "c1", "c1", "c1"], patient=["P1", "P1", "P2", "P2"],
                   sample=["s1", "s1", "s1", "s2"])
    with pytest.warns(TCRIDataWarning, match="patient 'P1' keeps no clone"):
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", samples="sample")
    assert _labels(adata) == ["pooled@P1", "pooled@P1", "c1@P2", "c1@P2"]


def _unchanged(adata, before):
    obs_columns, uns_keys = before
    assert list(adata.obs.columns) == obs_columns, "pooling wrote to obs before raising"
    assert set(adata.uns) == uns_keys, "pooling wrote to uns before raising"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", MISSING, ids=repr)
def test_pool_raises_on_missing_values_and_columns(value, dtype):
    """Missing clonotypes, groups and samples raise with the column and the number of cells,
    and say how to keep only the cells that have one; missing columns raise naming the
    argument and the column. Nothing is written in any case."""
    clones = ["c1"] * 3 + ["c2"] * 3
    clones[1] = clones[4] = value
    adata = _adata(dtype, clone_id=clones, patient=["P1"] * 6, sample=["s1"] * 6)
    before = (list(adata.obs.columns), set(adata.uns))
    with pytest.raises(ValueError) as exc:
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    message = str(exc.value)
    assert "2 of 6 cells have no clonotype in adata.obs['clone_id']" in message
    assert "adata = adata[~missing].copy()" in message
    assert "from_mudata" not in message
    _unchanged(adata, before)

    for column in ("patient", "sample"):
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            continue  # an empty string or "nan" is a group or sample like any other
        adata = _adata(dtype, clone_id=["c1"] * 6, patient=["P1"] * 6, sample=["s1"] * 6)
        adata.obs[column] = adata.obs[column].astype(object)
        adata.obs.iloc[[0, 3], adata.obs.columns.get_loc(column)] = value
        before = (list(adata.obs.columns), set(adata.uns))
        with pytest.raises(ValueError) as exc:
            _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient",
                              samples="sample")
        assert f"2 of 6 cells have no value in adata.obs[{column!r}]" in str(exc.value)
        assert f"adata = adata[adata.obs[{column!r}].notna()].copy()" in str(exc.value)
        _unchanged(adata, before)

    adata = _cohort(dtype)
    before = (list(adata.obs.columns), set(adata.uns))
    for argument, kwargs in [
        ("clonotype_key", dict(clonotype_key="nope", groupby="patient")),
        ("groupby", dict(clonotype_key="clone_id", groupby="nope")),
        ("samples", dict(clonotype_key="clone_id", groupby="patient", samples="nope")),
    ]:
        with pytest.raises(KeyError, match=rf"{argument}='nope' is not a column of adata\.obs"):
            _pool_rare_clones(adata, **kwargs)
        _unchanged(adata, before)


@pytest.mark.parametrize(("min_cells", "error"), [
    (0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError), (False, TypeError),
    ("3", TypeError), (3.0, TypeError),
])
def test_pool_rejects_bad_min_cells(min_cells, error):
    """min_cells is an integer >= 1, and a bool is not an integer here. 1 is accepted and pools
    nothing, and the step is still recorded; a numpy integer is accepted."""
    adata = _cohort()
    before = (list(adata.obs.columns), set(adata.uns))
    with pytest.raises(error, match="min_cells must be"):
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient",
                          min_cells=min_cells)
    _unchanged(adata, before)

    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", min_cells=1)
    assert "pooled@P1" not in _labels(adata) and "pooled@P2" not in _labels(adata)
    step = _derivation_steps(adata)[0]
    assert (step["min_cells"], step["pool_labels"], step["n_clones_pooled"],
            step["n_cells_pooled"]) == (1, "", 0, 0)

    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient",
                      min_cells=np.int64(3))
    assert _labels(adata) == COHORT_LABELS


def test_pool_writes_only_key_added():
    """Only obs[key_added] and the derivation record are written. The source column keeps its
    values and dtype; the default key_added is f"{clonotype_key}_pooled"; key_added equal to
    clonotype_key raises; a second run replaces the column."""
    adata = _cohort("category")
    source = adata.obs["clone_id"].copy()
    columns, uns_keys = list(adata.obs.columns), set(adata.uns)
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert list(adata.obs.columns) == columns + ["clone_id_pooled"]
    assert set(adata.uns) == uns_keys | {K.CLONOTYPE_DERIVATIONS}
    pd.testing.assert_series_equal(adata.obs["clone_id"], source)
    pooled = adata.obs["clone_id_pooled"]
    assert isinstance(pooled.dtype, pd.CategoricalDtype)
    assert list(pooled.cat.categories) == sorted(set(COHORT_LABELS))

    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", key_added="mine")
    assert list(adata.obs.columns) == columns + ["clone_id_pooled", "mine"]
    assert _labels(adata, "mine") == COHORT_LABELS

    before = (list(adata.obs.columns), set(adata.uns))
    with pytest.raises(ValueError, match="key_added must differ from clonotype_key"):
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient",
                          key_added="clone_id")
    _unchanged(adata, before)
    pd.testing.assert_series_equal(adata.obs["clone_id"], source)

    with pytest.warns(TCRIDataWarning):
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", min_cells=5)
    assert list(adata.obs.columns) == columns + ["clone_id_pooled", "mine"]
    assert _labels(adata) == ["pooled@P1"] * 6 + ["pooled@P2"] * 6


def test_pool_suffixes_once():
    """An id that already ends in @{group} is kept as it is; any other id gains the suffix,
    including one scoped to another group. Pooling the pooled column again changes nothing.
    Two kept clones that would share an id raise, naming them."""
    adata = _adata(clone_id=["a@P1"] * 3 + ["b"] * 3 + ["c@P2"] * 3 + ["d@P2"] * 3,
                   patient=["P1"] * 9 + ["P2"] * 3)
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert _labels(adata) == ["a@P1"] * 3 + ["b@P1"] * 3 + ["c@P2@P1"] * 3 + ["d@P2"] * 3
    assert _derivation_steps(adata)[-1]["suffixed"] is True

    _pool_rare_clones(adata, clonotype_key="clone_id_pooled", groupby="patient")
    assert _labels(adata, "clone_id_pooled_pooled") == _labels(adata)
    assert _derivation_steps(adata)[-1]["suffixed"] is False

    scoped = _adata(clone_id=["a@P1"] * 3 + ["b@P1"] * 2, patient=["P1"] * 5)
    _pool_rare_clones(scoped, clonotype_key="clone_id", groupby="patient")
    assert _labels(scoped) == ["a@P1"] * 3 + ["pooled@P1"] * 2
    assert _derivation_steps(scoped)[0]["suffixed"] is False

    collide = _adata(clone_id=["x@P1"] * 3 + ["x"] * 3, patient=["P1"] * 6)
    with pytest.raises(ValueError, match=r"\('x', 'P1'\) and \('x@P1', 'P1'\) would each "
                                         r"become 'x@P1'"):
        _pool_rare_clones(collide, clonotype_key="clone_id", groupby="patient")
    assert "clone_id_pooled" not in collide.obs and K.CLONOTYPE_DERIVATIONS not in collide.uns

    # the same pair is no collision when one of the two is pooled
    collide = _adata(clone_id=["x@P1"] * 3 + ["x"] * 2, patient=["P1"] * 5)
    _pool_rare_clones(collide, clonotype_key="clone_id", groupby="patient")
    assert _labels(collide) == ["x@P1"] * 3 + ["pooled@P1"] * 2


def test_pool_warns_when_a_group_keeps_nothing():
    """A group whose clones are all rare warns once, naming the group; a group that keeps a
    clone does not."""
    adata = _adata(clone_id=["c1"] * 3 + ["c2"] * 2 + ["c3"] * 2 + ["c4"],
                   patient=["P1"] * 3 + ["P2"] * 5, sample=["s1"] * 8)
    with pytest.warns(TCRIDataWarning) as record:
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", samples="sample")
    messages = [str(w.message) for w in record if issubclass(w.category, TCRIDataWarning)]
    assert len(messages) == 1
    assert "patient 'P2' keeps no clone" in messages[0]
    assert "'pooled@P2'" in messages[0] and "'P1'" not in messages[0]
    assert _labels(adata) == ["c1@P1"] * 3 + ["pooled@P2"] * 5

    with warnings.catch_warnings():
        warnings.simplefilter("error", TCRIDataWarning)
        _pool_rare_clones(_cohort(), clonotype_key="clone_id", groupby="patient")


def test_pool_logs_one_line_per_group(caplog):
    """One INFO line per group on the tcri logger: clones kept, clones pooled, and cells pooled
    with their share of the group. A category no cell carries gets no line."""
    adata = _cohort()
    adata.obs["patient"] = pd.Categorical(adata.obs["patient"], categories=["P1", "P2", "P3"])
    with caplog.at_level(logging.INFO, logger="tcri"):
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("tcri")]
    assert lines == [
        "pool_rare_clones: patient 'P1': clones kept 1, clones pooled 2, cells pooled 3 of 6 "
        "(50.0%).",
        "pool_rare_clones: patient 'P2': clones kept 1, clones pooled 1, cells pooled 2 of 6 "
        "(33.3%).",
    ]


def test_pool_step_in_derivation_record(tmp_path):
    """One step per call, with the columns used, the pool labels and what was pooled. It leads
    back to the source column and to its pools, survives h5ad, and a second call on the same
    key_added replaces it."""
    adata = _cohort()
    adata.obs["sample"] = ["s1", "s2"] * 6
    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", samples="sample")
    # with samples, c2 (s1, s2) and c1 in P2 (s1, s2) are seen in two samples and kept
    assert _labels(adata) == (["c1@P1"] * 3 + ["c2@P1"] * 2 + ["pooled@P1"] + ["c1@P2"] * 2
                              + ["c4@P2"] * 4)
    assert _derivation_steps(adata) == [{
        "function": "pool_rare_clones", "source": "clone_id", "key_added": "clone_id_pooled",
        "replicate": "", "groupby": "patient", "min_cells": 3, "samples": "sample",
        "suffixed": True, "pool_labels": "pooled@P1", "n_clones_pooled": 1,
        "n_cells_pooled": 1}]
    assert _clonotype_source(adata, "clone_id_pooled") == "clone_id"
    assert _pool_labels(adata, "clone_id_pooled") == ["pooled@P1"]

    _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
    assert _derivation_steps(adata) == [{
        "function": "pool_rare_clones", "source": "clone_id", "key_added": "clone_id_pooled",
        "replicate": "", "groupby": "patient", "min_cells": 3, "samples": "",
        "suffixed": True, "pool_labels": "pooled@P1|pooled@P2", "n_clones_pooled": 3,
        "n_cells_pooled": 5}]
    assert _pool_labels(adata, "clone_id_pooled") == ["pooled@P1", "pooled@P2"]

    with pytest.warns(TCRIDataWarning):
        _pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", min_cells=5,
                          key_added="clone_id_pooled_5")
    assert [s["key_added"] for s in _derivation_steps(adata)] == ["clone_id_pooled",
                                                                  "clone_id_pooled_5"]

    adata.write_h5ad(tmp_path / "pooled.h5ad")
    back = read_h5ad(tmp_path / "pooled.h5ad")
    assert _derivation_steps(back) == _derivation_steps(adata)
    assert _pool_labels(back, "clone_id_pooled") == ["pooled@P1", "pooled@P2"]
    assert _labels(back) == _labels(adata)
    assert list(back.obs["clone_id_pooled"].cat.categories) == \
        list(adata.obs["clone_id_pooled"].cat.categories)
