# Quickstart

A minimal end-to-end TCRi workflow. For the concepts behind these objects see
[Data model & concepts](../concepts/data-model.md); for full signatures see the
API reference.

## Loading data

TCRi works on an `AnnData` with paired gene expression and TCR information. The
`.obs` table needs columns identifying each cell's **clonotype**, **phenotype**,
**covariate** (e.g. timepoint), and **batch** (e.g. patient).

```python
import scanpy as sc
import tcri
from tcri.model import TCRIModel

adata = sc.read_h5ad("your_data.h5ad")
```

## Setting up and training the model

```python
TCRIModel.setup_anndata(
    adata,
    clonotype_key="clone_id",     # your .obs column names
    phenotype_key="phenotype",
    covariate_key="timepoint",
    batch_key="patient",
)

model = TCRIModel(adata)          # defaults are sensible; tune n_latent, n_hidden, ...
model.train(max_epochs=200, batch_size=128)
```

## Registering model outputs

`register_model` writes the learned distributions, latent embedding, per-cell
phenotype posteriors, and indexing arrays back onto the `AnnData`
(`.uns` / `.obsm` / `.obs`) so the metric and plotting functions can read them.

```python
tcri.pp.register_model(adata, model, clonotype_key="clone_id")
```

## Computing metrics

```python
covariate = adata.uns["tcri_covariate_categories"][0]

# Mutual information between clonotype and phenotype (point estimate -> float)
mi = tcri.tl.mutual_information(adata, covariate)

# Clonotypic entropy: a value per phenotype (Series). n_samples > 0 draws from
# the posterior; phenotypic_entropy is the per-clone analogue.
ce = tcri.tl.clonotypic_entropy(adata, covariate, n_samples=50)
pe = tcri.tl.phenotypic_entropy(adata, covariate, n_samples=50)

# Clonality per phenotype
clonality = tcri.tl.clonality(adata)
```

## Visualization

Plotting helpers live under `tcri.pl`:

```python
# Mutual-information summary across covariate groups
tcri.pl.mutual_information(adata, splitby=covariate)
```

Other helpers — phenotype-flux Sankey diagrams (`plot_pheno_sankey`), per-cell
phenotype probabilities, and more — are documented in the
[Plotting API](../api/plotting.md).

## Next steps

- [Data model & concepts](../concepts/data-model.md) — how the model and objects fit together
- [Model API](../api/model.md) · [Preprocessing API](../api/preprocessing.md) · [Metrics API](../api/metrics.md) · [Plotting API](../api/plotting.md)
