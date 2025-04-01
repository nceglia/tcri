# Preprocessing API

The `tcri.preprocessing` module (imported as `tcri.pp`) provides functions for preprocessing data and preparing it for analysis with TCRi.

## register_model

```python
def register_model(adata, model, phenotype_prob_slot="X_tcri_phenotypes", phenotype_assignment_obs="tcri_phenotype", latent_slot="X_tcri", batch_size=256):
    """
    Register model outputs in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to register model outputs in
    model : TCRIModel
        Trained TCRIModel
    phenotype_prob_slot : str, default="X_tcri_phenotypes"
        Key in adata.obsm where phenotype probabilities will be stored
    phenotype_assignment_obs : str, default="tcri_phenotype"
        Key in adata.obs where phenotype assignments will be stored
    latent_slot : str, default="X_tcri"
        Key in adata.obsm where latent representations will be stored
    batch_size : int, default=256
        Batch size for model inference
        
    Returns
    -------
    AnnData
        Updated AnnData object with model outputs registered
    """
    pass
```

## joint_distribution

```python
def joint_distribution(adata, covariate_label, temperature=1.0, n_samples=0, clones=None, weighted=False):
    """
    Compute joint distribution of phenotypes and TCR clonotypes conditioned on a covariate.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    covariate_label : str
        Label of the covariate to condition on (must be in adata.obs)
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
    n_samples : int, default=0
        Number of samples to draw from the posterior
    clones : list, optional
        List of clone IDs to include in the analysis
    weighted : bool, default=False
        Whether to weight the distribution by clone size
        
    Returns
    -------
    dict
        Dictionary mapping covariate values to joint distribution matrices
    """
    pass
```

## global_joint_distribution

```python
def global_joint_distribution(adata, temperature=1.0, n_samples=0):
    """
    Compute global joint distribution of phenotypes and TCR clonotypes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with model results registered
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
    n_samples : int, default=0
        Number of samples to draw from the posterior
        
    Returns
    -------
    numpy.ndarray
        Joint distribution matrix of shape (n_clones, n_phenotypes)
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

# Register model outputs in AnnData
tcri.pp.register_model(
    adata,
    model,
    phenotype_prob_slot="X_tcri_phenotypes",
    phenotype_assignment_obs="tcri_phenotype",
    latent_slot="X_tcri"
)

# Compute joint distributions
joint_dist = tcri.pp.joint_distribution(
    adata,
    covariate_label="timepoint",  # Replace with your covariate
    temperature=1.0
)

# Compute global joint distribution
global_dist = tcri.pp.global_joint_distribution(
    adata,
    temperature=1.0
)
```
