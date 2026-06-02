"""Tests for joint_distribution_posterior view/subset handling (Notion #4 / T7).

Filtered AnnData (views or subset copies) keep their per-cell `.uns` arrays in the
original full-cell space while `.obs`/`.obsm` are subset. The function used to index
one with positions derived from the other, silently returning misaligned results.
It must now raise instead.
"""

import numpy as np
import pandas as pd
import pytest

from tcri.preprocessing._preprocessing import joint_distribution_posterior


def _covariate(adata):
    return adata.uns["tcri_covariate_categories"][0]


def test_jd_posterior_full_adata_ok(trained_model):
    """Baseline: the full registered AnnData returns a valid distribution."""
    _, adata = trained_model
    df = joint_distribution_posterior(adata, _covariate(adata), silent=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == len(adata.uns["tcri_phenotype_categories"])
    assert np.all(np.isfinite(df.to_numpy()))


def test_jd_posterior_rejects_filtered_view(trained_model):
    """A cell-filtered view must raise, not silently misalign (Notion #4)."""
    _, adata = trained_model
    cov_col = adata.uns["tcri_metadata"]["covariate_col"]
    mask = np.asarray(adata.obs[cov_col] == _covariate(adata))
    view = adata[mask]
    assert view.n_obs < adata.n_obs

    with pytest.raises(ValueError, match=r"filtered|subset|register_model"):
        joint_distribution_posterior(view, _covariate(adata), silent=True)


def test_jd_posterior_rejects_filtered_copy(trained_model):
    """A cell-filtered copy is also misaligned: .uns stays full-length."""
    _, adata = trained_model
    cov_col = adata.uns["tcri_metadata"]["covariate_col"]
    mask = np.asarray(adata.obs[cov_col] == _covariate(adata))
    sub = adata[mask].copy()
    assert sub.n_obs < adata.n_obs

    with pytest.raises(ValueError, match=r"filtered|subset|register_model"):
        joint_distribution_posterior(sub, _covariate(adata), silent=True)


def test_jd_posterior_allows_gene_subset(trained_model):
    """Var-only subsetting keeps n_obs intact and must NOT trip the guard."""
    _, adata = trained_model
    sub = adata[:, : adata.n_vars // 2]
    assert sub.n_obs == adata.n_obs

    df = joint_distribution_posterior(sub, _covariate(adata), silent=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == len(adata.uns["tcri_phenotype_categories"])
