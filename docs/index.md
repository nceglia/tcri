# TCRi

```{image} _static/tcri_logo_hero.png
:alt: TCRi
:width: 360px
:align: center
```

**Information-theoretic analysis of paired single-cell RNA + TCR sequencing.**

TCRi quantifies how T-cell **clonotypes** relate to transcriptional **phenotypes** in paired
single-cell data. It fits a hierarchical Bayesian model (on top of
[scvi-tools](https://scvi-tools.org) / [Pyro](https://pyro.ai)) that ties each clone's
phenotype distribution to gene expression, then reads out information-theoretic summaries —
clonotypic and phenotypic entropy, clone↔phenotype mutual information, and phenotypic flux —
over the learned clone × phenotype joint distribution.

New here? Start with [Installation](usage/installation.md) and the
[Quickstart](usage/quickstart.md), then the [Tutorials](tutorials/index.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

usage/installation
usage/quickstart
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/data-model
concepts/model
concepts/metrics
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/model
api/preprocessing
api/metrics
api/plotting
api/diagnostics
api/datasets
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Governance & notes

contracts/index
nmi_temperature_bias
```

## Citation

If you use TCRi, please cite:

> **TCRi: An Information Theoretic Framework for Paired Single Cell Gene Expression and TCR
> Sequencing.** Ceglia N., Salehi S., _et al._ _bioRxiv_ 2022.
> doi: [10.1101/2022.10.01.510457](https://doi.org/10.1101/2022.10.01.510457)

## Links

- [GitHub repository](https://github.com/nceglia/tcri)
- [Issue tracker](https://github.com/nceglia/tcri/issues)
- [Paper](https://www.biorxiv.org/content/10.1101/2022.10.01.510457v1)
