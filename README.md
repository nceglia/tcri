<p align="center">
  <a href="https://tcri.readthedocs.io">
    <img src="https://github.com/nceglia/tcri/blob/main/tcri_logo_hero.png?raw=true" alt="TCRi" width="420">
  </a>
</p>

<p align="center">
  <b>Information-theoretic analysis of paired single-cell RNA + TCR sequencing.</b>
</p>

<p align="center">
  <a href="https://tcri.readthedocs.io">Documentation</a> ·
  <a href="https://tcri.readthedocs.io/en/latest/tutorials/index.html">Tutorials</a> ·
  <a href="https://www.biorxiv.org/content/10.1101/2022.10.01.510457v1">Paper</a> ·
  <a href="https://github.com/nceglia/tcri/issues">Issues</a>
</p>

<p align="center">
  <a href="https://github.com/nceglia/tcri/actions/workflows/tests.yml"><img src="https://github.com/nceglia/tcri/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://tcri.readthedocs.io"><img src="https://readthedocs.org/projects/tcri/badge/?version=latest" alt="docs"></a>
  <a href="https://github.com/nceglia/tcri/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python 3.10+">
  <img src="https://img.shields.io/badge/built%20on-scverse-%231f9e16.svg" alt="built on scverse">
</p>

---

TCRi quantifies how **T-cell clonotypes** relate to **transcriptional phenotypes** in
paired single-cell data. It fits a hierarchical Bayesian model (on top of
[scvi-tools](https://scvi-tools.org) / [Pyro](https://pyro.ai)) that ties each clone's
phenotype distribution to gene expression, then reads out **information-theoretic
summaries** — clonotypic and phenotypic entropy, clone↔phenotype mutual information, and
phenotypic flux across conditions — over the learned clone × phenotype joint distribution.

It is built on the [scverse](https://scverse.org) stack
([AnnData](https://anndata.readthedocs.io), [scanpy](https://scanpy.readthedocs.io),
scvi-tools) and works on a standard `AnnData` carrying both a clonotype label and a
gene-expression matrix.

## Key capabilities

- **Joint model of expression + repertoire** — a hierarchical VAE (`TCRIModel`) that
  learns each clone's phenotype distribution while reconstructing counts with a ZINB
  decoder.
- **Information-theoretic metrics** — clonotypic entropy `H(c|φ)`, phenotypic entropy
  `H(φ|c)`, normalized mutual information `I(c;φ)`, and phenotypic flux between conditions,
  all in bits over the posterior joint.
- **Uncertainty-aware** — every metric can be reported as a posterior mean ± HDI by
  drawing from the fitted Dirichlet posterior (`n_samples > 0`).
- **Group comparisons** — paired/unpaired contrasts across cohorts (e.g. responders vs
  non-responders) with direction probabilities.
- **Diagnostics** — posterior-predictive checks, phenotype calibration, and permutation
  nulls for the metrics.
- **Contract-governed** — the public API, the generative model, the metric definitions,
  and the training plan are each frozen by a machine-checked contract (see below).

## Installation

TCRi targets **Python ≥ 3.10**. Install from source:

```bash
git clone https://github.com/nceglia/tcri.git
cd tcri
pip install .
```

The heavy scientific stack (PyTorch, Pyro, scvi-tools, scanpy) is pulled in
automatically. A GPU is optional but speeds up model fitting. See the
[installation guide](https://tcri.readthedocs.io/en/latest/usage/installation.html) for
details.

## Quickstart

```python
import tcri

# 1. register clonotype / phenotype / covariate / batch columns on your AnnData
tcri.ml.TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="timepoint",
    batch_key="patient",
)

# 2. fit the model and write learned quantities back onto the AnnData
model = tcri.ml.TCRIModel(adata)
model.train(max_epochs=200, batch_size=512)
model.to_anndata(adata)

# 3. read out information-theoretic metrics (bits)
mi = tcri.tl.mutual_information(adata, covariate="pre", normalize_mode="average")
ce = tcri.tl.clonotypic_entropy(adata, covariate="pre")
flux = tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post")
```

No paired data yet? `tcri.datasets.simulate_tcri()` returns a synthetic AnnData whose
true mutual information is known in closed form — the basis for the
[tutorials](https://tcri.readthedocs.io/en/latest/tutorials/index.html).

## Framework

TCRi models the joint distribution of clonotype and phenotype hierarchically: a
clonotype-level prior `p_c` over phenotypes, a covariate-specific `p_ct` anchored to it,
and a per-cell phenotype that fuses a classifier on the latent embedding `z` with the
clone's local prior. The metrics are then pure functions of the learned clone × phenotype
joint. The full generative model (Supplementary Note 1) and its plate diagram are
documented under [The model](https://tcri.readthedocs.io/en/latest/concepts/model.html).

<p align="center">
  <img src="https://github.com/nceglia/tcri/blob/main/framework.png?raw=true" alt="TCRi framework" width="760">
</p>

## The contracts

TCRi is governed by four frozen, machine-checked contracts — the manuscript
(Supplementary Note 1) is upstream of all of them:

| Contract | Freezes | Prose |
|----------|---------|-------|
| **API** | the public interface | [`API_CONTRACT.md`](docs/contract/API_CONTRACT.md) |
| **Model** | the generative mathematics | [`MODEL_CONTRACT.md`](docs/contract/MODEL_CONTRACT.md) |
| **Metrics** | what the metrics compute | [`METRICS_CONTRACT.md`](docs/contract/METRICS_CONTRACT.md) |
| **Training** | how the model is fit | [`TRAINING_CONTRACT.md`](docs/contract/TRAINING_CONTRACT.md) |

A failing conformance test means *stop and decide* — never loosen a contract to make it
pass. See [The contracts](https://tcri.readthedocs.io/en/latest/contracts/index.html).

## Related tools

| Tool | Relationship |
|------|-------------|
| [scirpy](https://scirpy.readthedocs.io) | AIRR/TCR repertoire handling and analysis |
| [scvi-tools](https://scvi-tools.org) | probabilistic modelling backbone TCRi builds on |
| [scanpy](https://scanpy.readthedocs.io) / [AnnData](https://anndata.readthedocs.io) | single-cell data structures and workflows |

## Citation

If you use TCRi in your research, please cite:

> **TCRi: An Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing**
>
> Nicholas Ceglia, Sohrab Salehi, et al.
>
> _bioRxiv_ 2022. doi: [10.1101/2022.10.01.510457](https://doi.org/10.1101/2022.10.01.510457)

<details>
<summary>BibTeX</summary>

```bibtex
@article{ceglia2022tcri,
  title   = {TCRi: An Information Theoretic Framework for Paired Single Cell Gene Expression and TCR Sequencing},
  author  = {Ceglia, Nicholas and Salehi, Sohrab and others},
  journal = {bioRxiv},
  year    = {2022},
  doi     = {10.1101/2022.10.01.510457}
}
```

</details>

## License

TCRi is released under the [MIT License](LICENSE).
