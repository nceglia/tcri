from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from mudata import MuData
from scipy import sparse

import tcri
from tcri._compute._repertoire import TCRIDataWarning

AIRR_CELLS = ["cell_1", "cell_2", "cell_4", "cell_5"]
CLONES = ["cl_1", "cl_2", "cl_3", "cl_4"]
TISSUES = ["PBMC", "TP", "CSF", "PBMC"]


def _toy_mudata(
    *,
    gex_only_cells: bool = False,
    include_clone_id: bool = True,
    missing_clone: bool = False,
    missing_value: object = None,
    cc_keys: tuple[str, ...] = (),
    counts: bool = True,
):
    """An AIRR modality of four cells and a GEX modality with the same cells, plus two GEX-only
    cells when ``gex_only_cells``. ``mdata.obs`` has no columns, whatever the mudata version."""
    obs_names = [f"cell_{i}" for i in range(6)]
    gex_obs = pd.DataFrame(
        {
            "sample": [
                "GBM1-DFCI1-S1-TP",
                "GBM1-DFCI1-S1-PBMC",
                "GBM1-DFCI2-S1-TP",
                "BTC-GBM-001-001-TP",
                "GBM1-MSK1-S4-CSF",
                "GBM1-DFCI3-S2-PBMC",
            ],
            "tissue": ["TP", "PBMC", "TP", "TP", "CSF", "PBMC"],
            "patient": ["P1", "P1", "P2", "P3", "P4", "P5"],
            "site": ["DFCI1", "DFCI1", "DFCI2", "BTC", "MSK1", "DFCI3"],
        },
        index=obs_names,
    )
    X = np.arange(18, dtype=np.float32).reshape(6, 3)
    gex = AnnData(X=X, obs=gex_obs)
    if counts:
        gex.layers["counts"] = X.copy()
    if not gex_only_cells:
        gex = gex[AIRR_CELLS].copy()

    airr_obs = pd.DataFrame(index=AIRR_CELLS)
    if include_clone_id:
        clones = list(CLONES)
        if missing_clone:
            clones[1] = missing_value
        airr_obs["clone_id"] = pd.Series(clones, index=AIRR_CELLS, dtype="string")
        airr_obs["clone_id_size"] = [5, 2, 7, 3]
    for i, key in enumerate(cc_keys):
        airr_obs[key] = pd.Series(
            [f"cc_{i}_a", f"cc_{i}_b", f"cc_{i}_c", f"cc_{i}_d"], index=AIRR_CELLS, dtype="string"
        )
        airr_obs[f"{key}_size"] = [2, 2, 1, 1]
    airr = AnnData(X=np.zeros((len(AIRR_CELLS), 0), dtype=np.float32), obs=airr_obs)

    mdata = MuData({"gex": gex, "airr": airr})
    mdata.update()
    mdata.obs.drop(columns=list(mdata.obs.columns), inplace=True)
    return mdata


def _place(mdata, column, where):
    """Keep ``column`` in one place only: the "gex" or "airr" obs, or ``mdata.obs[where]``."""
    values = pd.concat([mdata.mod[mod].obs[column] for mod in ("gex", "airr")
                        if column in mdata.mod[mod].obs])
    values = values[~values.index.duplicated()]
    for mod in ("gex", "airr"):
        if column in mdata.mod[mod].obs:
            del mdata.mod[mod].obs[column]
    if where in ("gex", "airr"):
        mdata.mod[where].obs[column] = values.reindex(mdata.mod[where].obs_names)
    else:
        mdata.obs[where] = values.reindex(mdata.obs_names)


def test_from_mudata_reads_the_clonotype_covariate_and_counts():
    mdata = _toy_mudata(gex_only_cells=True)
    assert mdata.obs.shape[1] == 0
    with pytest.warns(TCRIDataWarning,
                      match=r"2 of 6 cells of mdata\.mod\['gex'\] have no AIRR data"):
        adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id", covariate_key="tissue")

    assert adata.obs_names.tolist() == AIRR_CELLS
    assert adata.obs["clone_id"].tolist() == CLONES
    assert adata.obs["tissue"].tolist() == TISSUES
    assert str(adata.obs["clone_id"].dtype) == "category"
    assert str(adata.obs["tissue"].dtype) == "category"
    np.testing.assert_array_equal(adata.layers["counts"],
                                  mdata.mod["gex"][AIRR_CELLS].layers["counts"])
    assert adata.uns["tcri_adapter"]["resolved"] == {
        "gex_mod": "gex",
        "airr_mod": "airr",
        "clonotype_key": "clone_id",
        "clonotype_source": "mdata.mod['airr'].obs['clone_id']",
        "covariate_key": "tissue",
        "covariate_source": "mdata.mod['gex'].obs['tissue']",
        "counts_layer": "counts",
        "n_obs_gex": 6,
        "n_obs_airr": 4,
        "n_obs_shared": 4,
        "n_obs_output": 4,
    }


def test_from_mudata_does_not_warn_when_every_gex_cell_has_airr_data():
    with warnings.catch_warnings():
        warnings.simplefilter("error", TCRIDataWarning)
        adata = tcri.pp.from_mudata(_toy_mudata(), clonotype_key="clone_id")
    assert adata.uns["tcri_adapter"]["resolved"]["n_obs_gex"] == 4


def test_from_mudata_output_round_trips_through_h5ad(tmp_path):
    adata = tcri.pp.from_mudata(_toy_mudata(), clonotype_key="clone_id")
    path = tmp_path / "adapter.h5ad"
    adata.write_h5ad(path)
    back = read_h5ad(path)

    assert back.uns["tcri_adapter"] == adata.uns["tcri_adapter"]
    assert back.obs["clone_id"].tolist() == adata.obs["clone_id"].tolist()


@pytest.mark.parametrize("column", ["clone_id", "tissue"])
@pytest.mark.parametrize(("where", "source"), [
    ("gex", "mdata.mod['gex'].obs['{c}']"),
    ("airr", "mdata.mod['airr'].obs['{c}']"),
    ("{c}", "mdata.obs['{c}']"),
    ("gex:{c}", "mdata.obs['gex:{c}']"),
    ("airr:{c}", "mdata.obs['airr:{c}']"),
], ids=["gex", "airr", "mdata", "mdata-gex", "mdata-airr"])
def test_from_mudata_reads_a_column_from_any_place_and_writes_the_bare_name(column, where, source):
    mdata = _toy_mudata()
    _place(mdata, column, where.format(c=column))

    adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id", covariate_key="tissue")

    role = "clonotype" if column == "clone_id" else "covariate"
    assert adata.obs[column].tolist() == (CLONES if column == "clone_id" else TISSUES)
    assert adata.uns["tcri_adapter"]["resolved"][f"{role}_source"] == source.format(c=column)
    assert not [c for c in adata.obs.columns if ":" in c]


def test_from_mudata_reads_the_first_place_that_has_the_column():
    mdata = _toy_mudata()
    mdata.mod["gex"].obs["clone_id"] = "in_gex"
    mdata.obs["clone_id"] = "in_mdata"
    mdata.obs["airr:clone_id"] = "in_mdata_prefixed"

    def read():
        adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id")
        return sorted(set(adata.obs["clone_id"]))

    assert read() == ["in_gex"]
    del mdata.mod["gex"].obs["clone_id"]
    assert read() == CLONES
    del mdata.mod["airr"].obs["clone_id"]
    assert read() == ["in_mdata"]
    del mdata.obs["clone_id"]
    assert read() == ["in_mdata_prefixed"]


def test_from_mudata_reads_a_cc_definition_by_name():
    mdata = _toy_mudata(include_clone_id=False, cc_keys=("cc_aa_tcrdist", "cc_aa_tcrdist_same_v"))
    adata = tcri.pp.from_mudata(mdata, clonotype_key="cc_aa_tcrdist")
    assert adata.obs["cc_aa_tcrdist"].tolist() == ["cc_0_a", "cc_0_b", "cc_0_c", "cc_0_d"]


def test_from_mudata_unknown_column_raises_and_lists_the_clone_like_columns():
    mdata = _toy_mudata(include_clone_id=False, cc_keys=("cc_aa_tcrdist",))
    with pytest.raises(KeyError, match="clonotype_key='no_such_column' was not found") as err:
        tcri.pp.from_mudata(mdata, clonotype_key="no_such_column")
    assert "Clone-like columns include: cc_aa_tcrdist." in str(err.value)

    with pytest.raises(KeyError, match="covariate_key='no_such_column' was not found"):
        tcri.pp.from_mudata(mdata, clonotype_key="cc_aa_tcrdist", covariate_key="no_such_column")


@pytest.mark.parametrize(("keys", "message"), [
    (dict(clonotype_key=("clone_id",)), "clonotype_key must be one column name"),
    (dict(clonotype_key=""), "clonotype_key must be one column name"),
    (dict(clonotype_key="clone_id", covariate_key=("tissue", "site")),
     "covariate_key must be one column name.*single column first"),
    (dict(clonotype_key="clone_id", covariate_key=""), "covariate_key must be one column name"),
], ids=["clonotype-tuple", "clonotype-empty", "covariate-tuple", "covariate-empty"])
def test_from_mudata_takes_one_column_name_per_key(keys, message):
    with pytest.raises(TypeError, match=message):
        tcri.pp.from_mudata(_toy_mudata(), **keys)


@pytest.mark.parametrize("value", [None, "", "   ", "nan"], ids=repr)
def test_from_mudata_drops_cells_without_a_clonotype_by_default(value):
    mdata = _toy_mudata(missing_clone=True, missing_value=value)
    adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id")
    assert adata.obs_names.tolist() == ["cell_1", "cell_4", "cell_5"]
    assert adata.uns["tcri_adapter"]["resolved"]["n_obs_output"] == 3


def test_from_mudata_without_the_drop_raises_on_cells_without_a_clonotype():
    mdata = _toy_mudata(missing_clone=True, missing_value="")
    with pytest.raises(
        ValueError,
        match=r"1 of 4 cells have no clonotype in mdata\.mod\['airr'\]\.obs\['clone_id'\].*"
              r"Pass drop_missing_clonotype=True",
    ):
        tcri.pp.from_mudata(mdata, clonotype_key="clone_id", drop_missing_clonotype=False)


def test_from_mudata_raises_on_missing_covariate_values():
    mdata = _toy_mudata()
    mdata.mod["gex"].obs.loc["cell_4", "tissue"] = None
    with pytest.raises(ValueError, match=r"1 of 4 cells have no covariate value in "
                                         r"mdata\.mod\['gex'\]\.obs\['tissue'\]"):
        tcri.pp.from_mudata(mdata, clonotype_key="clone_id", covariate_key="tissue")


def test_from_mudata_requires_the_counts_layer():
    mdata = _toy_mudata(counts=False)
    with pytest.raises(KeyError, match=r"no layer 'counts'.*"
                                       r"mdata\.mod\['gex'\]\.layers\['counts'\] = "
                                       r"mdata\.mod\['gex'\]\.X\.copy\(\)"):
        tcri.pp.from_mudata(mdata, clonotype_key="clone_id")


@pytest.mark.parametrize("to_layer", [np.asarray, sparse.csr_matrix], ids=["dense", "sparse"])
@pytest.mark.parametrize("transform", [np.log1p, lambda x: x - x.max()],
                         ids=["normalized", "negative"])
def test_from_mudata_refuses_a_counts_layer_without_raw_counts(transform, to_layer):
    mdata = _toy_mudata()
    gex = mdata.mod["gex"]
    gex.layers["counts"] = to_layer(transform(np.asarray(gex.X)))
    with pytest.raises(ValueError, match=r"layer 'counts' of mdata\.mod\['gex'\] does not hold "
                                         r"raw counts.*X\.copy\(\)"):
        tcri.pp.from_mudata(mdata, clonotype_key="clone_id")


@pytest.mark.parametrize("to_layer", [np.asarray, sparse.csr_matrix], ids=["dense", "sparse"])
def test_from_mudata_accepts_integer_counts(to_layer):
    mdata = _toy_mudata()
    gex = mdata.mod["gex"]
    gex.layers["counts"] = to_layer(np.asarray(gex.X).astype(np.int64))
    adata = tcri.pp.from_mudata(mdata, clonotype_key="clone_id")
    assert sparse.issparse(adata.layers["counts"]) == (to_layer is sparse.csr_matrix)
