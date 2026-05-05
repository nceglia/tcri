import numpy as np
import pandas as pd
import pytest

from tcri.metrics._metrics import (
    clonotypic_entropy,
    phenotypic_entropy,
    mutual_information,
    clonality,
)


def test_clonotypic_entropy(trained_model):
    _, adata = trained_model
    covariate = adata.uns["tcri_covariate_categories"][0]
    phenotypes = adata.uns["tcri_phenotype_categories"]

    s = clonotypic_entropy(adata, covariate, n_samples=20)
    assert isinstance(s, pd.Series)
    assert list(s.index) == list(phenotypes)
    vals = s.to_numpy()
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0)

    arr = clonotypic_entropy(
        adata, covariate, point_estimate=False, n_samples=5
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, len(phenotypes))
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0)

    with pytest.raises(ValueError):
        clonotypic_entropy(adata, covariate, n_samples=0)


def test_phenotypic_entropy(trained_model):
    _, adata = trained_model
    covariate = adata.uns["tcri_covariate_categories"][0]

    meta = adata.uns["tcri_metadata"]
    expected_clones = (
        adata.obs.loc[adata.obs[meta["covariate_col"]] == covariate, meta["clone_col"]]
        .unique()
        .tolist()
    )

    s = phenotypic_entropy(adata, covariate, n_samples=20)
    assert isinstance(s, pd.Series)
    assert set(s.index) == set(expected_clones)
    vals = s.to_numpy()
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0)

    arr = phenotypic_entropy(
        adata, covariate, point_estimate=False, n_samples=5
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, len(expected_clones))
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0)

    with pytest.raises(ValueError):
        phenotypic_entropy(adata, covariate, n_samples=0)


def test_mutual_information(trained_model):
    _, adata = trained_model
    covariate = adata.uns["tcri_covariate_categories"][0]

    mi = mutual_information(adata, covariate=covariate, temperature=1.0, verbose=False)
    assert np.isfinite(mi)
    assert mi >= 0

    mi_alt = mutual_information(adata, covariate=covariate, temperature=0.5, verbose=False)
    assert np.isfinite(mi_alt)
    assert mi_alt >= 0
    assert mi_alt != mi


def test_clonality(mock_adata):
    """Test clonality function."""
    clonality_dict = clonality(mock_adata)

    for val in clonality_dict.values():
        assert 0 <= val <= 1

    assert set(clonality_dict.keys()) == set(mock_adata.uns["tcri_phenotype_categories"])
