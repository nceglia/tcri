# Diagnostics

`tcri.diag` provides model-checking and metric-validation tools: posterior-predictive checks
on the learned joint, phenotype-probability calibration, reconstruction PPCs, permutation
nulls for the metrics, and training-curve inspection. This tutorial runs each on a fitted
model. See the [diagnostics API](../api/diagnostics.md) for full signatures.

## A fitted model

```python
import tcri

adata = tcri.datasets.simulate_tcri(n_clones=30, n_cells=2000, seed=0)
tcri.ml.TCRIModel.setup_anndata(
    adata, layer="counts",
    clonotype_key="clone_id", phenotype_key="phenotype",
    covariate_key="covariate", batch_key="batch",
)
model = tcri.ml.TCRIModel(adata, n_latent=32, seed=0)
model.train(max_epochs=200, batch_size=256)
model.to_anndata(adata)
```

## Joint-distribution posterior predictive

Does the learned joint reproduce the observed clone × phenotype structure? `distance_metric`
selects how the observed and predicted tables are compared:

```python
ppc = tcri.diag.joint_distribution_ppc(adata, distance_metric="l1")
ppc.head()
```

## Phenotype calibration

Are the per-cell phenotype probabilities calibrated — when the model says 0.7, is it right
~70% of the time? `phenotype_calibration` bins predictions against observed accuracy:

```python
cal = tcri.diag.phenotype_calibration(adata, n_bins=5)
cal
```

## Reconstruction posterior predictive

Simulate counts from the fitted decoder and compare summary statistics (library size,
dropout) to the observed data. This is the check that re-calibrated
`reconstruction_loss_scale` on real data:

```python
recon = tcri.diag.reconstruction_ppc(model, adata, n_sims=1, random_state=0)
recon
```

## Permutation null for the metrics

Is the observed clone↔phenotype coupling more than chance? `permutation_null` shuffles labels
to build a null distribution for a metric:

```python
null = tcri.diag.permutation_null(adata, n_perm=30, random_state=0)
null
```

## Training curves

`loss` plots the ELBO trajectory (and the monitored per-cell objective); `archetypes`
visualizes the learned clone archetypes:

```python
tcri.diag.loss(model)
tcri.diag.archetypes(model)
```

```{tip}
If `model.train` warned that the KL ramp did not complete, the loss curve is the quickest way
to see it — the objective is still in its warmup regime rather than at a stationary value.
Increase `max_epochs` and refit. See [the training contract](../contracts/index.md).
```
