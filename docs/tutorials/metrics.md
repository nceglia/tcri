# Metrics

TCRi's information-theoretic metrics are pure functions of the learned clone × phenotype
joint. Because `simulate_tcri` carries the **true** mutual information in
`adata.uns["tcri_truth"]`, this tutorial can compare the recovered metrics against ground
truth. The definitions are in [the metrics concepts](../concepts/metrics.md).

## Setup

```python
import numpy as np
import tcri

adata = tcri.datasets.simulate_tcri(n_clones=30, n_phenotypes=4, n_cells=4000, seed=0)

tcri.ml.TCRIModel.setup_anndata(
    adata, layer="counts",
    clonotype_key="clone_id", phenotype_key="phenotype",
    covariate_key="covariate", batch_key="batch",
)
model = tcri.ml.TCRIModel(adata, n_latent=32, seed=0)
model.train(max_epochs=200, batch_size=256)
model.to_anndata(adata)
```

## The joint distribution

Every metric reads the joint produced by {func}`tcri.tl.joint_distribution` — a tidy
clone × phenotype table for a covariate:

```python
jd = tcri.tl.joint_distribution(adata, covariate="cov_0")
jd.head()
```

## Mutual information vs. ground truth

```python
mi = tcri.tl.mutual_information(
    adata, covariate="cov_0", weighted=True, normalize_mode="average"
)
truth = adata.uns["tcri_truth"]
print(f"recovered NMI (average) = {float(mi):.3f}")
print(f"true NMI (average)      = {truth['true_nmi_average']:.3f}")
```

```{important}
Pass `normalize_mode="average"` to compare against the manuscript's NMI (eq 6). The default
`"min"` is a deliberate, group-comparable deviation and will **not** match `true_nmi_average`
— it matches `true_nmi_min` instead. See [the metrics concepts](../concepts/metrics.md).
```

## Entropies

```python
# one value per phenotype: how clonally diverse each phenotype is
ce = tcri.tl.clonotypic_entropy(adata, covariate="cov_0")

# one value per clone: how phenotypically plastic each clone is
pe = tcri.tl.phenotypic_entropy(adata, covariate="cov_0")

ce_finite = np.asarray(ce, float)[np.isfinite(ce)]
print(f"mean clonotypic entropy = {ce_finite.mean():.3f} bits (normalized to [0, 1])")
```

Both entropies are **support-only** and return `NaN` for an empty phenotype column or a
zero-mass clone — never a spurious `1.0`.

## Uncertainty via posterior samples

Passing `n_samples > 0` draws from the fitted Dirichlet posterior and returns a mean ± HDI
instead of a point estimate:

```python
mi_hdi = tcri.tl.mutual_information(
    adata, covariate="cov_0", n_samples=200, normalize_mode="average"
)
mi_hdi  # includes the posterior mean and highest-density interval
```

## Grouped comparison

The real use case is contrasting a metric across cohorts. With multiple patients you would
compute per-group values and compare:

```python
mi_df = tcri.tl.mutual_information(
    adata, covariate="cov_0", groupby="batch", normalize_mode="average"
)
# result = tcri.tl.compare_groups(mi_df, value="MI", splitby="response")
```

```{note}
`compare_groups` needs a `splitby` column that is constant within each group (e.g. a
response label that is one value per patient). The single-batch synthetic set has nothing to
split on; on real data this is where responder vs non-responder contrasts are made.
```

## Phenotypic flux

With two or more covariates, flux measures how a clone's phenotype mix shifts between them
(default `distance_metric="kl"`):

```python
multi = tcri.datasets.simulate_tcri(n_covariates=2, n_cells=4000, seed=0)
# ... setup_anndata + train + to_anndata as above ...
# flux = tcri.tl.phenotypic_flux(multi, cov_from="cov_0", cov_to="cov_1")
```
