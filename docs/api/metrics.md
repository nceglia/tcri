# Metrics API

The `tcri.metrics` module (imported as `tcri.tl`) provides functions for calculating information-theoretic metrics on paired single-cell RNA and TCR sequencing data.

## clonotypic_entropy

```python
def clonotypic_entropy(adata, covariate, phenotype, temperature=1.0):
    """
    Calculate the clonotypic entropy for each value of the covariate.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    covariate : str
        Name of the covariate in adata.obs
    phenotype : str
        Name of the phenotype in adata.obs
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
        
    Returns
    -------
    dict
        Dictionary mapping covariate values to entropy values
    """
    pass
```

## phenotypic_entropy

```python
def phenotypic_entropy(adata, covariate, clonotype, temperature=1.0):
    """
    Calculate the phenotypic entropy for each value of the covariate.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    covariate : str
        Name of the covariate in adata.obs
    clonotype : str
        Name of the clonotype field in adata.obs
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
        
    Returns
    -------
    dict
        Dictionary mapping covariate values to entropy values
    """
    pass
```

## mutual_information

```python
def mutual_information(adata, covariate, temperature=1.0, weighted=False):
    """
    Calculate the mutual information between phenotypes and TCR clonotypes
    for each value of the covariate.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    covariate : str
        Name of the covariate in adata.obs
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
    weighted : bool, default=False
        Whether to weight by clone size
        
    Returns
    -------
    dict
        Dictionary mapping covariate values to mutual information values
    """
    pass
```

## clonality

```python
def clonality(adata):
    """
    Calculate clonality metrics for the data.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with TCR information
        
    Returns
    -------
    dict
        Dictionary containing clonality metrics
    """
    pass
```

## flux

```python
def flux(adata, from_this, to_that, clones=None, temperature=1.0):
    """
    Calculate phenotypic flux between two covariate values.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    from_this : str
        Starting covariate value
    to_that : str
        Ending covariate value
    clones : list, optional
        List of clone IDs to include
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
        
    Returns
    -------
    numpy.ndarray
        Flux matrix of shape (n_phenotypes, n_phenotypes)
    """
    pass
```

## Usage Examples

```python
import tcri
import scanpy as sc

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize and train model
model = tcri.TCRIModel(adata)
model.train()
tcri.pp.register_model(adata, model)

# Calculate metrics
mi = tcri.tl.mutual_information(adata, "timepoint")
entropy = tcri.tl.clonotypic_entropy(adata, "timepoint", "phenotype")
clonality = tcri.tl.clonality(adata)

# Calculate flux between timepoints
flux_matrix = tcri.tl.flux(adata, from_this="T1", to_that="T2")
```
