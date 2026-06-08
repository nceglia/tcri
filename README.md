<p align="center">
  <img src="https://github.com/nceglia/tcri/blob/main/tcri_logo.png?raw=true" alt="TCRi logo" width="400">
</p>

# **TCR**i
[![tests](https://github.com/nceglia/tcri/actions/workflows/tests.yml/badge.svg)](https://github.com/nceglia/tcri/actions/workflows/tests.yml)
Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing

![alt text](https://github.com/nceglia/tcri/blob/main/framework.png?raw=true)

https://www.biorxiv.org/content/10.1101/2022.10.01.510457v1

## Installation

```bash
python3 -m venv tvenv
source tvenv/bin/activate
pip install .
```

## Overview

TCRi is a comprehensive framework for analyzing paired single-cell RNA and TCR sequencing data.
It provides tools for:

- Joint distribution analysis
- Information theoretic metrics
- Visualization capabilities
- Deep learning model for phenotype prediction

## Quick start

```python
import scanpy as sc
import tcri
from tcri.model import TCRIModel

# AnnData with paired gene expression and, in .obs, columns identifying each
# cell's clonotype, phenotype, covariate (e.g. timepoint), and batch (e.g. patient).
adata = sc.read_h5ad("your_data.h5ad")

# 1. Register the fields and fit the hierarchical model
TCRIModel.setup_anndata(
    adata,
    clonotype_key="clone_id",     # <- your .obs column names
    phenotype_key="phenotype",
    covariate_key="timepoint",
    batch_key="patient",
)
model = TCRIModel(adata)          # defaults are sensible; tune n_latent, n_hidden, ...
model.train(max_epochs=200, batch_size=128)

# 2. Write learned distributions, latent embedding, and per-cell phenotype
#    posteriors back onto the AnnData (.uns / .obsm / .obs)
tcri.pp.register_model(adata, model, clonotype_key="clone_id")

# 3. Information-theoretic metrics at a covariate value
covariate = adata.uns["tcri_covariate_categories"][0]

mi = tcri.tl.mutual_information(adata, covariate)                 # clone–phenotype coupling (float)
ce = tcri.tl.clonotypic_entropy(adata, covariate, n_samples=50)  # Series over phenotypes
pe = tcri.tl.phenotypic_entropy(adata, covariate, n_samples=50)  # Series over clones
```

Plotting helpers live under `tcri.pl` — phenotype-flux Sankey diagrams
(`plot_pheno_sankey`), per-cell phenotype probabilities, mutual-information
summaries, and more. See the API reference for the full set.

## Documentation

Full documentation — a conceptual overview of the data model plus the complete
API reference — lives on [Read the Docs](https://tcri.readthedocs.io):

- **Concepts:** the hierarchical model and the objects `register_model` writes onto your AnnData
- **API reference:** `tcri.model`, `tcri.pp`, `tcri.tl`, `tcri.pl`, `tcri.ut`

## Citation

If you use TCRi in your research, please cite:
```
@article{nceglia2022tcri,
  title={TCRi: An Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing},
  author={Nceglia, Nicholas, Salehi, Sohrab and others},
  journal={bioRxiv},
  year={2022}
}
```
