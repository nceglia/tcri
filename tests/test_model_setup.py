import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scvi import REGISTRY_KEYS

from tcri.model._model import TCRIModel


def _tiny_adata():
    obs = pd.DataFrame(
        {
            "unique_clone_id": ["c1", "c1", "c2", "c2"],
            "phenotype_col": ["A", "B", "A", "B"],
            "timepoint": ["T1", "T1", "T2", "T2"],
            "patient": ["P1", "P1", "P2", "P2"],
        },
        index=[f"cell_{i}" for i in range(4)],
    )
    X = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    return AnnData(X=X, obs=obs)


def test_setup_anndata_defaults_to_x_matrix():
    adata = _tiny_adata()

    TCRIModel.setup_anndata(adata)

    manager = TCRIModel._get_most_recent_anndata_manager(adata)
    assert manager.registry["setup_args"]["layer"] is None
    assert "tcri_layer" not in adata.uns
    # Phase 4: the AnnDataManager is no longer stashed in uns (tcri_manager retired).
    assert "tcri_manager" not in adata.uns
    np.testing.assert_array_equal(
        manager.get_from_registry(REGISTRY_KEYS.X_KEY),
        adata.X,
    )


def test_setup_anndata_explicit_none_uses_x_matrix_with_layers_present():
    adata = _tiny_adata()
    adata.layers["counts"] = np.full(adata.shape, 7, dtype=np.float32)
    adata.layers["other"] = np.full(adata.shape, 11, dtype=np.float32)

    TCRIModel.setup_anndata(adata, layer=None)

    manager = TCRIModel._get_most_recent_anndata_manager(adata)
    assert manager.registry["setup_args"]["layer"] is None
    assert "tcri_layer" not in adata.uns
    np.testing.assert_array_equal(
        manager.get_from_registry(REGISTRY_KEYS.X_KEY),
        adata.X,
    )


def test_setup_anndata_default_uses_x_matrix_with_one_layer_present():
    adata = _tiny_adata()
    adata.layers["counts"] = np.full(adata.shape, 7, dtype=np.float32)

    TCRIModel.setup_anndata(adata)

    manager = TCRIModel._get_most_recent_anndata_manager(adata)
    assert manager.registry["setup_args"]["layer"] is None
    assert "tcri_layer" not in adata.uns
    np.testing.assert_array_equal(
        manager.get_from_registry(REGISTRY_KEYS.X_KEY),
        adata.X,
    )


def test_setup_anndata_default_uses_x_matrix_with_layers_present():
    adata = _tiny_adata()
    adata.layers["counts"] = np.full(adata.shape, 7, dtype=np.float32)
    adata.layers["other"] = np.full(adata.shape, 11, dtype=np.float32)

    TCRIModel.setup_anndata(adata)

    manager = TCRIModel._get_most_recent_anndata_manager(adata)
    assert manager.registry["setup_args"]["layer"] is None
    assert "tcri_layer" not in adata.uns
    np.testing.assert_array_equal(
        manager.get_from_registry(REGISTRY_KEYS.X_KEY),
        adata.X,
    )


def test_setup_anndata_accepts_single_explicit_layer():
    adata = _tiny_adata()
    adata.layers["counts"] = np.full(adata.shape, 7, dtype=np.float32)

    TCRIModel.setup_anndata(adata, layer="counts")

    manager = TCRIModel._get_most_recent_anndata_manager(adata)
    assert manager.registry["setup_args"]["layer"] == "counts"
    assert adata.uns["tcri_layer"] == "counts"
    np.testing.assert_array_equal(
        manager.get_from_registry(REGISTRY_KEYS.X_KEY),
        adata.layers["counts"],
    )


@pytest.mark.parametrize(
    ("layer", "value"),
    [
        ("counts", 7),
        ("other", 11),
    ],
)
def test_setup_anndata_accepts_explicit_layer_when_multiple_layers_present(
    layer, value
):
    adata = _tiny_adata()
    adata.layers["counts"] = np.full(adata.shape, 7, dtype=np.float32)
    adata.layers["other"] = np.full(adata.shape, 11, dtype=np.float32)

    TCRIModel.setup_anndata(adata, layer=layer)

    manager = TCRIModel._get_most_recent_anndata_manager(adata)
    assert manager.registry["setup_args"]["layer"] == layer
    assert adata.uns["tcri_layer"] == layer
    np.testing.assert_array_equal(
        manager.get_from_registry(REGISTRY_KEYS.X_KEY),
        np.full(adata.shape, value, dtype=np.float32),
    )


def test_setup_anndata_rejects_missing_explicit_layer():
    adata = _tiny_adata()

    with pytest.raises(ValueError, match="other is not a valid key in adata.layers"):
        TCRIModel.setup_anndata(adata, layer="other")
