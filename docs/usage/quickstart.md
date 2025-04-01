# Quickstart

This guide will help you get started with TCRi by walking through a basic analysis workflow using sample data.

## Loading Data

TCRi works with AnnData objects that contain both gene expression and TCR information. Here's how to load and set up your data:

```python
import tcri
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Make sure your AnnData object has the right fields for TCR information
# Typically, TCR information should be in obs under 'clone_id' or similar
```

## Setting up the Model

```python
# Initialize the model
model = tcri.TCRIModel(
    adata,
    n_latent=10,      # Dimension of latent space
    n_hidden=128,     # Size of hidden layers
    global_scale=10.0,
    local_scale=5.0
)

# Train the model
model.train(
    max_epochs=50,
    batch_size=128,
    lr=1e-3,
    reconstruction_loss_scale=1e-2
)

# Get latent representations
latent_z = model.get_latent_representation(adata)
```

## Preprocessing

```python
# Register model outputs back to the AnnData object
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
    covariate_label="timepoint"  # Replace with your covariate of interest
)
```

## Computing Metrics

```python
# Calculate mutual information
mi = tcri.tl.mutual_information(
    adata,
    "timepoint",  # Replace with your covariate of interest
    temperature=1.0
)

# Calculate clonotypic entropy
entropy = tcri.tl.clonotypic_entropy(
    adata,
    "timepoint",   # Covariate
    "phenotype"    # Phenotype field
)

# Calculate clonality
clonality = tcri.tl.clonality(adata)
```

## Visualization

```python
# Mutual information plot
tcri.pl.mutual_information(
    adata,
    splitby="timepoint",  # Replace with your covariate
    temperature=1.0,
    figsize=(8,4)
)

# Polar plot of phenotype distributions
tcri.pl.polar_plot(
    adata,
    statistic="distribution",
    method="joint_distribution"
)

# Phenotype probability ternary plot
tcri.pl.probability_ternary(
    adata,
    ["Phenotype1", "Phenotype2", "Phenotype3"],  # Replace with your phenotype names
    splitby="condition"  # Optional: split by a condition
)
```

## Next Steps

Explore the API documentation for more detailed information on each function and additional functionality:

- [Model API](../api/model.md)
- [Preprocessing API](../api/preprocessing.md)
- [Metrics API](../api/metrics.md)
- [Plotting API](../api/plotting.md)
