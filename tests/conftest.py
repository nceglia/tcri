import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
import torch

@pytest.fixture
def mock_adata():
    """Create a mock AnnData object with the necessary structure for testing."""
    # Create random data
    n_cells = 100
    n_genes = 50
    X = sparse.random(n_cells, n_genes, density=0.1, format='csr')
    
    # Create mock observations with all arrays having length 100
    obs = pd.DataFrame({
        'unique_clone_id': [f'clone_{i}' for i in range(20)] * 5,  # 20 unique clones
        'phenotype_col': ['A', 'B', 'C'] * 33 + ['A'],  # 3 phenotypes (100 total)
        'timepoint': ['T1', 'T2'] * 50,  # 2 timepoints
        'patient': ['P1', 'P2'] * 50,  # 2 patients
    })
    
    # Create AnnData object
    adata = AnnData(X=X, obs=obs)
    
    # Add required metadata
    adata.uns["tcri_metadata"] = {
        "covariate_col": "timepoint",
        "clone_col": "unique_clone_id",
        "phenotype_col": "phenotype_col",
        "batch_col": "patient"
    }
    
    adata.uns["tcri_phenotype_categories"] = ['A', 'B', 'C']
    adata.uns["tcri_clone_key"] = "unique_clone_id"
    adata.uns["tcri_phenotype_key"] = "phenotype_col"
    
    # Add mock joint distribution data
    n_clones = 20
    n_phenotypes = 3
    mock_jd = np.random.dirichlet(np.ones(n_phenotypes), size=n_clones)
    mock_jd = pd.DataFrame(
        mock_jd,
        columns=['A', 'B', 'C'],
        index=[f'clone_{i}' for i in range(n_clones)]
    )
    adata.uns["mock_joint_distribution"] = mock_jd
    
    # Add required model outputs
    n_ct_pairs = 40  # 20 clones * 2 timepoints
    n_phenotypes = 3
    adata.uns["tcri_p_ct"] = torch.randn(n_ct_pairs, n_phenotypes)
    adata.uns["tcri_ct_to_cov"] = torch.tensor([0, 1] * 20)  # Alternating timepoints
    adata.uns["tcri_ct_to_c"] = torch.tensor([i for i in range(20)] * 2)  # Clone indices
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
    # Create base distribution
    base_dist = np.random.dirichlet(np.ones(n_phenotypes), size=n_clones)
    # Create temperature-scaled distribution
    temp_dist = base_dist ** (1.0 / 0.5)  # Scale by temperature 0.5
    temp_dist = temp_dist / temp_dist.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        temp_dist,
        columns=['A', 'B', 'C'],
        index=[f'clone_{i}' for i in range(n_clones)]
    ) 