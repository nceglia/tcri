import contextlib
import io

import pytest
import numpy as np
import pandas as pd
import pyro
import torch
from anndata import AnnData
from scipy import sparse


def pytest_configure(config):
    """Deterministic RNG for all tests (Notion T13)."""
    np.random.seed(42)
    torch.manual_seed(42)


def _seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object with the necessary structure for testing."""
    n_cells = 100
    n_genes = 50
    X = sparse.random(n_cells, n_genes, density=0.1, format='csr')

    obs = pd.DataFrame({
        'unique_clone_id': [f'clone_{i}' for i in range(20)] * 5,
        'phenotype_col': ['A', 'B', 'C'] * 33 + ['A'],
        'timepoint': ['T1', 'T2'] * 50,
        'patient': ['P1', 'P2'] * 50,
    })

    adata = AnnData(X=X, obs=obs)

    adata.uns["tcri_metadata"] = {
        "covariate_col": "timepoint",
        "clone_col": "unique_clone_id",
        "phenotype_col": "phenotype_col",
        "batch_col": "patient"
    }

    adata.uns["tcri_phenotype_categories"] = ['A', 'B', 'C']
    adata.uns["tcri_clone_key"] = "unique_clone_id"
    adata.uns["tcri_phenotype_key"] = "phenotype_col"

    n_clones = 20
    n_phenotypes = 3
    mock_jd = np.random.dirichlet(np.ones(n_phenotypes), size=n_clones)
    mock_jd = pd.DataFrame(
        mock_jd,
        columns=['A', 'B', 'C'],
        index=[f'clone_{i}' for i in range(n_clones)]
    )
    adata.uns["mock_joint_distribution"] = mock_jd

    n_ct_pairs = 40
    adata.uns["tcri_p_ct"] = np.random.dirichlet(
        np.ones(n_phenotypes), size=n_ct_pairs
    ).astype(np.float32)
    adata.uns["tcri_ct_to_cov"] = torch.tensor([0, 1] * 20)
    adata.uns["tcri_ct_to_c"] = torch.tensor([i for i in range(20)] * 2)
    adata.uns["tcri_covariate_categories"] = ['T1', 'T2']
    adata.uns["tcri_clonotype_categories"] = [f'clone_{i}' for i in range(20)]
    adata.uns["tcri_ct_array_for_cells"] = np.random.randint(0, n_ct_pairs, size=n_cells)
    adata.uns["tcri_cov_array_for_cells"] = np.random.randint(0, 2, size=n_cells)
    adata.uns["tcri_local_scale"] = 5.0

    return adata


@pytest.fixture
def mock_joint_distribution():
    """Create a mock joint distribution DataFrame."""
    n_clones = 20
    n_phenotypes = 3
    base_dist = np.random.dirichlet(np.ones(n_phenotypes), size=n_clones)
    temp_dist = base_dist ** (1.0 / 0.5)
    temp_dist = temp_dist / temp_dist.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        temp_dist,
        columns=['A', 'B', 'C'],
        index=[f'clone_{i}' for i in range(n_clones)]
    )


@pytest.fixture(scope="session")
def synthetic_adata():
    """Tiny deterministic AnnData for model fixtures (Notion T1)."""
    _seed_all(0)
    rng = np.random.default_rng(0)

    n_cells, n_genes, n_clones = 200, 50, 20
    phenotypes = ["A", "B", "C"]
    covariates = ["T1", "T2"]
    batches = ["P1", "P2"]

    X = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)

    obs = pd.DataFrame({
        "unique_clone_id": rng.choice(
            [f"clone_{i}" for i in range(n_clones)], size=n_cells
        ),
        "phenotype_col": rng.choice(phenotypes, size=n_cells),
        "timepoint": rng.choice(covariates, size=n_cells),
        "patient": rng.choice(batches, size=n_cells),
    })
    for col in obs.columns:
        obs[col] = obs[col].astype("category")

    return AnnData(X=X, obs=obs)


@pytest.fixture(scope="session")
def trained_model(synthetic_adata):
    """TCRIModel fit for 50 epochs on synthetic_adata, with register_model applied."""
    _seed_all(0)

    from tcri.model._model import TCRIModel
    from tcri.preprocessing._preprocessing import register_model

    adata = synthetic_adata.copy()
    TCRIModel.setup_anndata(
        adata,
        layer=None,
        clonotype_key="unique_clone_id",
        phenotype_key="phenotype_col",
        covariate_key="timepoint",
        batch_key="patient",
    )
    model = TCRIModel(
        adata,
        n_latent=8,
        n_hidden=16,
        n_layers=1,
        classifier_n_layers=1,
        classifier_hidden=16,
        K=3,
        n_pseudo_obs=3,
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.train(
            max_epochs=50,
            batch_size=64,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        register_model(adata, model, clonotype_key="unique_clone_id")
    return model, adata