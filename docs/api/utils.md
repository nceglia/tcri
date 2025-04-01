# Utils API

The `tcri.utils` module (imported as `tcri.ut`) provides utility functions for working with TCRi data.

## get_clones

```python
def get_clones(adata, clone_key="clone_id"):
    """
    Get unique clone IDs from the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with TCR information
    clone_key : str, default="clone_id"
        Key in adata.obs containing clone IDs
        
    Returns
    -------
    list
        List of unique clone IDs
    """
    pass
```

## get_phenotypes

```python
def get_phenotypes(adata, phenotype_key="phenotype"):
    """
    Get unique phenotype labels from the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with phenotype information
    phenotype_key : str, default="phenotype"
        Key in adata.obs containing phenotype labels
        
    Returns
    -------
    list
        List of unique phenotype labels
    """
    pass
```

## normalize_distribution

```python
def normalize_distribution(distribution):
    """
    Normalize a distribution to sum to 1.
    
    Parameters
    ----------
    distribution : numpy.ndarray
        Distribution to normalize
        
    Returns
    -------
    numpy.ndarray
        Normalized distribution
    """
    pass
```

## Usage Examples

```python
import tcri
import scanpy as sc

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Get unique clones
clones = tcri.ut.get_clones(adata, clone_key="clone_id")
print(f"Found {len(clones)} unique clones")

# Get unique phenotypes
phenotypes = tcri.ut.get_phenotypes(adata, phenotype_key="phenotype")
print(f"Found {len(phenotypes)} unique phenotypes: {phenotypes}")

# Normalize a distribution
import numpy as np
dist = np.random.rand(10)
normalized_dist = tcri.ut.normalize_distribution(dist)
print(f"Sum of normalized distribution: {normalized_dist.sum():.6f}")
```
