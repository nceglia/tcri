# **TCR**i
Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing

![alt text](https://github.com/nceglia/tcri/blob/main/framework.png?raw=true)

https://www.biorxiv.org/content/10.1101/2022.10.01.510457v1

## Installation

```bash
python3 -m venv tvenv
source tvenv/bin/activate
python3 setup.py install
```

## Overview

TCRi is a comprehensive framework for analyzing paired single-cell RNA and TCR sequencing data. It provides tools for:
- Joint distribution analysis
- Information theoretic metrics
- Visualization capabilities
- Deep learning model for phenotype prediction

## Core Components

### 1. Model (`tcri.model`)

The `TCRIModel` class implements a hierarchical Bayesian model for analyzing TCR and gene expression data:

```python
import tcri

# Initialize model
model = tcri.TCRIModel(
    adata,
    n_latent=10,
    n_hidden=128,
    global_scale=10.0,
    local_scale=5.0,
    sharp_temperature=1.0,
    sharpness_penalty_scale=0.0,
    use_enumeration=False,
    device=None
)

# Train model
model.train(
    max_epochs=50,
    batch_size=128,
    lr=1e-3,
    margin_scale=0.0,
    margin_value=2.0,
    adaptive_margin=False,
    reconstruction_loss_scale=1e-2,
    n_steps_kl_warmup=1000
)

# Get latent representations
latent_z = model.get_latent_representation(adata)
```

### 2. Preprocessing (`tcri.preprocessing`)

Key preprocessing functions:

```python
# Register model outputs in AnnData
tcri.pp.register_model(
    adata,
    model,
    phenotype_prob_slot="X_tcri_phenotypes",
    phenotype_assignment_obs="tcri_phenotype",
    latent_slot="X_tcri",
    batch_size=256
)

# Compute joint distributions
joint_dist = tcri.pp.joint_distribution(
    adata,
    covariate_label="timepoint",
    temperature=1.0,
    n_samples=0,
    clones=None,
    weighted=False
)

# Global joint distribution
global_dist = tcri.pp.global_joint_distribution(
    adata,
    temperature=1.0,
    n_samples=0
)
```

### 3. Metrics (`tcri.metrics`)

Information theoretic metrics for analyzing TCR and phenotype relationships:

```python
# Clonotypic entropy
clonotypic_entropy = tcri.tl.clonotypic_entropy(adata, covariate, phenotype, temperature=1.0)

# Phenotypic entropy
phenotypic_entropy = tcri.tl.phenotypic_entropy(adata, covariate, clonotype, temperature=1.0)

# Mutual information
mutual_info = tcri.tl.mutual_information(adata, covariate, temperature=1.0, weighted=False)

# Clonality
clonality = tcri.tl.clonality(adata)

# Phenotypic flux
flux = tcri.tl.flux(adata, from_this="T1", to_that="T2", clones=None, temperature=1.0)
```

### 4. Plotting (`tcri.plotting`)

Visualization tools for all metrics:

```python
# Polar plots
tcri.pl.polar_plot(
    adata,
    phenotypes=None,
    statistic="distribution",
    method="joint_distribution",
    splitby=None,
    color_dict=None,
    temperature=1.0
)

# Ternary plots
tcri.pl.probability_ternary(
    adata,
    phenotype_names,
    splitby=None,
    conditions=None,
    top_n=None
)

# Mutual information plots
tcri.pl.mutual_information(
    adata,
    splitby=None,
    temperature=1.0,
    n_samples=0,
    normalized=True,
    palette=None,
    save=None,
    legend_fontsize=6,
    bbox_to_anchor=(1.15,1.),
    figsize=(8,4),
    rotation=90,
    weighted=True,
    return_plot=True
)

# Clonality plots
tcri.pl.clonality(
    adata,
    groupby=None,
    splitby=None,
    s=10,
    order=None,
    figsize=(12,5),
    palette=None
)
```

## Example Usage

```python
import tcri
import scanpy as sc

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Setup model
model = tcri.TCRIModel(adata)
model.train()

# Register model outputs
tcri.pp.register_model(adata, model)

# Compute metrics
mi = tcri.tl.mutual_information(adata, "timepoint")
entropy = tcri.tl.clonotypic_entropy(adata, "timepoint", "phenotype")

# Visualize results
tcri.pl.mutual_information(adata, splitby="timepoint")
tcri.pl.polar_plot(adata, statistic="entropy")
```

## Citation

If you use TCRi in your research, please cite:
```
@article{nceglia2022tcri,
  title={TCRi: An Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing},
  author={Nceglia, Nicholas and others},
  journal={bioRxiv},
  year={2022}
}
```

