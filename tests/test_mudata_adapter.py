from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from mudata import MuData

import tcri


def _toy_mudata(
    *,
    include_site: bool = True,
    include_clone_id: bool = True,
    missing_clone: bool = False,
    missing_value: object = None,
    cc_keys: tuple[str, ...] = (),
):
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
        },
        index=obs_names,
    )
    if include_site:
        gex_obs["site"] = ["DFCI1", "DFCI1", "DFCI2", "BTC", "MSK1", "DFCI3"]
    gex = AnnData(X=np.arange(18, dtype=np.float32).reshape(6, 3), obs=gex_obs)

    airr_idx = ["cell_1", "cell_2", "cell_4", "cell_5"]
    airr_obs = pd.DataFrame(index=airr_idx)
    if include_clone_id:
        clones = ["cl_1", "cl_2", "cl_3", "cl_4"]
        if missing_clone:
            clones[1] = missing_value
        airr_obs["clone_id"] = pd.Series(clones, index=airr_idx, dtype="string")
        airr_obs["clone_id_size"] = [5, 2, 7, 3]
    for i, key in enumerate(cc_keys):
        airr_obs[key] = pd.Series(
            [f"cc_{i}_a", f"cc_{i}_b", f"cc_{i}_c", f"cc_{i}_d"], index=airr_idx, dtype="string"
        )
        airr_obs[f"{key}_size"] = [2, 2, 1, 1]
    airr = AnnData(X=np.zeros((len(airr_idx), 0), dtype=np.float32), obs=airr_obs)

    mdata = MuData({"gex": gex, "airr": airr})
    mdata.update()
    return mdata


def test_from_mudata_scirpy_defaults_with_single_covariate():
    mdata = _toy_mudata()
    adata = tcri.pp.from_mudata(
        mdata,
        covariate_cols="tissue",
    )

    assert adata.n_obs == 4
    assert "clone_id" in adata.obs
    assert str(adata.obs["clone_id"].dtype) == "category"
    assert "counts" in adata.layers
    np.testing.assert_array_equal(np.asarray(adata.layers["counts"]), np.asarray(adata.X))
    assert "tissue" in adata.obs
    assert str(adata.obs["tissue"].dtype) == "category"
    assert set(adata.obs["tissue"].astype(str)) == {"PBMC", "TP", "CSF"}

    meta = adata.uns["tcri_adapter"]
    assert meta["key_profile"] == "scirpy"
    assert meta["resolved"]["clonotype_key"] == "clone_id"
    assert meta["resolved"]["clonotype_source"] == "airr"
    assert meta["resolved"]["clonotype_col"] == "clone_id"
    assert meta["resolved"]["covariate_col"] == "tissue"


def test_from_mudata_output_round_trips_through_h5ad(tmp_path):
    adata = tcri.pp.from_mudata(_toy_mudata())
    path = tmp_path / "adapter.h5ad"
    adata.write_h5ad(path)
    back = read_h5ad(path)

    meta = adata.uns["tcri_adapter"]
    read = dict(back.uns["tcri_adapter"])
    read["auto_candidates"] = {k: list(v) for k, v in read["auto_candidates"].items()}
    assert read == meta
    assert back.obs["clone_id"].tolist() == adata.obs["clone_id"].tolist()


def test_from_mudata_missing_clones_drop_when_non_strict():
    mdata = _toy_mudata(missing_clone=True)
    adata = tcri.pp.from_mudata(
        mdata,
        strict=False,
        drop_missing_clonotype=True,
    )
    assert adata.n_obs == 3
    assert adata.obs["clone_id"].notna().all()


@pytest.mark.parametrize("value", [None, "", "   ", "nan"], ids=repr)
def test_from_mudata_defaults_drop_cells_without_a_clonotype(value):
    adata = tcri.pp.from_mudata(_toy_mudata(missing_clone=True, missing_value=value))
    assert adata.obs_names.tolist() == ["cell_1", "cell_4", "cell_5"]
    assert adata.uns["tcri_adapter"]["resolved"]["n_obs_output"] == 3


def test_from_mudata_without_drop_raises_or_keeps_cells_without_a_clonotype():
    mdata = _toy_mudata(missing_clone=True, missing_value="")
    with pytest.raises(
        ValueError,
        match=r"1 of 4 cells have no clonotype in 'clone_id'.*drop_missing_clonotype=True",
    ):
        tcri.pp.from_mudata(mdata, drop_missing_clonotype=False)

    adata = tcri.pp.from_mudata(mdata, drop_missing_clonotype=False, strict=False)
    assert adata.n_obs == 4


def test_from_mudata_auto_ambiguous_cc_raises():
    mdata = _toy_mudata(
        include_clone_id=False,
        cc_keys=("cc_aa_identity", "cc_nt_identity"),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        tcri.pp.from_mudata(mdata, clonotype_key="auto")


def test_from_mudata_auto_resolves_one_cc_definition_with_its_size_column():
    mdata = _toy_mudata(include_clone_id=False, cc_keys=("cc_aa_tcrdist",))
    # Scirpy also writes both columns to mdata.obs under the modality prefix.
    for col in ("cc_aa_tcrdist", "cc_aa_tcrdist_size"):
        mdata.obs[f"airr:{col}"] = mdata.mod["airr"].obs[col]

    adata = tcri.pp.from_mudata(mdata)

    meta = adata.uns["tcri_adapter"]
    assert meta["resolved"]["clonotype_key"] == "cc_aa_tcrdist"
    assert meta["resolved"]["clonotype_family"] == "cc"
    assert meta["auto_candidates"]["canonical"] == ["cc_aa_tcrdist", "cc_aa_tcrdist"]
    assert meta["auto_candidates"]["key"] == ["cc_aa_tcrdist", "airr:cc_aa_tcrdist"]
    assert adata.n_obs == 4


def test_from_mudata_unknown_clonotype_key_lists_the_cc_definition_not_its_size_column():
    mdata = _toy_mudata(include_clone_id=False, cc_keys=("cc_aa_tcrdist",))
    with pytest.raises(KeyError, match="Clone-like columns include") as err:
        tcri.pp.from_mudata(mdata, clonotype_key="no_such_column")
    assert "cc_aa_tcrdist" in str(err.value)
    assert "cc_aa_tcrdist_size" not in str(err.value)


def test_from_mudata_unknown_profile_raises():
    mdata = _toy_mudata()
    with pytest.raises(ValueError, match="unknown key_profile"):
        tcri.pp.from_mudata(mdata, key_profile="custom")


def test_from_mudata_rejects_tuple_covariate_cols():
    mdata = _toy_mudata()
    with pytest.raises(TypeError, match="single non-empty column name"):
        tcri.pp.from_mudata(mdata, covariate_cols=("tissue", "site"))


def test_from_mudata_missing_covariate_column_raises():
    mdata = _toy_mudata(include_site=False)
    with pytest.raises(KeyError, match="covariate_cols"):
        tcri.pp.from_mudata(mdata, covariate_cols="site")
