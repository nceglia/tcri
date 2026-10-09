"""The repertoire summaries in ``tcri.pp``.

Each summary reads ``obs`` and the clonotype derivation record and returns a table: it never
filters and writes nothing to the object. Cells without a clonotype are left out and not counted,
group labels and clonotypes come back as strings, and a column tcri derived is summarized through
the column the record leads back to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tcri
from tcri._compute._repertoire import _record_derivation
from tcri._state import keys as K

DTYPES = ["object", "category", "string", "str"]

#: Clonotype a is carried by P1 and P2, c by P2 and P3, and e by all three; b and d are private.
#: Two cells have no clonotype and one has no patient.
COHORT = {
    "clone_id": ["a", "a", "b", "e", None, "a", "c", "c", "e", "", "c", "d", "d", "d", "e", "a"],
    "patient": ["P1", "P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2", "P2",
                "P3", "P3", "P3", "P3", "P3", None],
}

#: ``clonotype_sharing`` on COHORT by patient. The All row counts the five distinct clonotypes
#: once each, while its cells are the sum of the group rows.
EXPECTED = pd.DataFrame(
    {
        "n_clonotypes": [3, 3, 3, 5],
        "shared_clonotypes": [2, 3, 2, 3],
        "shared_fraction": [2 / 3, 1.0, 2 / 3, 3 / 5],
        "n_cells": [4, 4, 5, 13],
        "cells_in_shared_clonotypes": [3, 4, 2, 9],
        "source_column": "clone_id",
        "registered_shared_clonotypes": [2, 3, 2, 3],
    },
    index=pd.Index(["P1", "P2", "P3", "All"], name="patient"),
)

EXPECTED_PER_CLONOTYPE = pd.DataFrame({
    "clonotype": ["a", "c", "e"],
    "n_groups": [2, 2, 3],
    "groups": ["P1|P2", "P2|P3", "P1|P2|P3"],
    "n_cells": [3, 3, 3],
    "cells_per_group": ["2|1", "2|1", "1|1|1"],
})


def _adata(columns, *, complete=False):
    """An AnnData whose ``obs`` holds ``columns``; ``complete`` keeps only the cells that have
    both a clonotype and a patient."""
    obs = pd.DataFrame(columns)
    if complete:
        obs = obs[obs["clone_id"].notna() & obs["clone_id"].ne("") & obs["patient"].notna()]
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    return AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)


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


# ── clonotype_sharing ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DTYPES)
def test_clonotype_sharing_counts_and_fractions(dtype):
    """Per group: its clonotypes, those also present in another group, and their cells. The
    All row counts distinct clonotypes across groups, not a column sum, and cells without a
    clonotype or a group are not counted. Rows follow the group's category order, labels are
    strings, and a group named All, the label of the row for all groups, raises."""
    adata = _adata({**COHORT, "clone_id": pd.Series(COHORT["clone_id"], dtype=dtype)})
    table = tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient")
    pd.testing.assert_frame_equal(table, EXPECTED)
    groups = table.drop(index="All")
    assert table.loc["All", "n_clonotypes"] < groups["n_clonotypes"].sum()
    assert table.loc["All", "n_cells"] == groups["n_cells"].sum()

    per_clonotype = tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient",
                                              per_clonotype=True)
    pd.testing.assert_frame_equal(per_clonotype, EXPECTED_PER_CLONOTYPE)

    ordered = _adata({**COHORT, "patient": pd.Categorical(
        COHORT["patient"], categories=["P3", "P1", "P2", "P0"])})
    table = tcri.pp.clonotype_sharing(ordered, clonotype_key="clone_id", groupby="patient")
    pd.testing.assert_frame_equal(table, EXPECTED.loc[["P3", "P1", "P2", "All"]])
    per_clonotype = tcri.pp.clonotype_sharing(ordered, clonotype_key="clone_id",
                                              groupby="patient", per_clonotype=True)
    assert per_clonotype["groups"].tolist() == ["P1|P2", "P3|P2", "P3|P1|P2"]
    assert per_clonotype["cells_per_group"].tolist() == ["2|1", "1|2", "1|1|1"]

    numbered = _adata(COHORT, complete=True)
    numbered.obs["clone_number"] = numbered.obs["clone_id"].map(
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 10})
    numbered.obs["patient_number"] = numbered.obs["patient"].map({"P1": 1, "P2": 2, "P3": 3})
    table = tcri.pp.clonotype_sharing(numbered, clonotype_key="clone_number",
                                      groupby="patient_number")
    assert table.index.tolist() == ["1", "2", "3", "All"]
    per_clonotype = tcri.pp.clonotype_sharing(numbered, clonotype_key="clone_number",
                                              groupby="patient_number", per_clonotype=True)
    assert per_clonotype["clonotype"].tolist() == ["1", "3", "10"]
    assert per_clonotype["groups"].tolist() == ["1|2", "2|3", "1|2|3"]

    named_all = _adata({"clone_id": ["a", "a", "b"], "patient": ["All", "P1", "P1"]})
    with pytest.raises(ValueError, match="group named 'All'"):
        tcri.pp.clonotype_sharing(named_all, clonotype_key="clone_id", groupby="patient")
    for clonotype_key, groupby, named in [("trb", "patient", "clonotype_key='trb'"),
                                          ("clone_id", "donor", "groupby='donor'")]:
        with pytest.raises(ValueError, match=f"{named} is not a column of adata.obs"):
            tcri.pp.clonotype_sharing(adata, clonotype_key=clonotype_key, groupby=groupby)


def test_clonotype_sharing_follows_the_derivation_record():
    """On a column tcri derived, sharing is measured on the source column the record leads back
    to, which the table names. registered_shared_clonotypes counts the ids of the column itself:
    none when they are specific to each patient, the shared clonotypes when they are not."""
    adata = _pool_then_register(_adata(COHORT, complete=True))
    on_source = tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient")
    pd.testing.assert_frame_equal(on_source, EXPECTED)

    for key in ["clone_id_pooled", K.CLONOTYPE]:
        table = tcri.pp.clonotype_sharing(adata, clonotype_key=key, groupby="patient")
        pd.testing.assert_frame_equal(table, EXPECTED.assign(registered_shared_clonotypes=0))
        per_clonotype = tcri.pp.clonotype_sharing(adata, clonotype_key=key, groupby="patient",
                                                  per_clonotype=True)
        pd.testing.assert_frame_equal(per_clonotype, EXPECTED_PER_CLONOTYPE)

    public = _adata(COHORT, complete=True)
    public.obs[K.CLONOTYPE] = public.obs["clone_id"].astype("category")
    _record_derivation(public, function="setup_anndata", source="clone_id",
                       key_added=K.CLONOTYPE, replicate="patient")
    table = tcri.pp.clonotype_sharing(public, clonotype_key=K.CLONOTYPE, groupby="patient")
    pd.testing.assert_frame_equal(table, EXPECTED)

    del adata.obs["clone_id"]
    with pytest.raises(ValueError, match="derived from 'clone_id', which is not a column"):
        tcri.pp.clonotype_sharing(adata, clonotype_key=K.CLONOTYPE, groupby="patient")


# ── every summary ─────────────────────────────────────────────────────────────────────────────

SUMMARIES = [
    pytest.param(lambda adata: tcri.pp.clonotype_sharing(
        adata, clonotype_key=K.CLONOTYPE, groupby="patient"), id="clonotype_sharing"),
    pytest.param(lambda adata: tcri.pp.clonotype_sharing(
        adata, clonotype_key=K.CLONOTYPE, groupby="patient", per_clonotype=True),
        id="clonotype_sharing-per_clonotype"),
    pytest.param(lambda adata: tcri.pp.clonotype_sharing(
        adata, clonotype_key="clone_id", groupby="patient"), id="clonotype_sharing-source"),
]


@pytest.mark.parametrize("summary", SUMMARIES)
def test_repertoire_functions_do_not_mutate_adata(summary):
    """A summary leaves ``obs`` (values, dtypes and categories, cells without a clonotype
    included), ``uns`` with its derivation record, and ``X`` exactly as they were."""
    adata = _pool_then_register(_adata({**COHORT, "patient": pd.Categorical(COHORT["patient"])}))
    before = adata.copy()
    summary(adata)
    pd.testing.assert_frame_equal(adata.obs, before.obs)
    assert list(adata.uns) == list(before.uns)
    assert adata.uns[K.CLONOTYPE_DERIVATIONS] == before.uns[K.CLONOTYPE_DERIVATIONS]
    np.testing.assert_array_equal(adata.X, before.X)
