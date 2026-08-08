# Training

This tutorial fits `TCRIModel` on a synthetic dataset, then on a realistic small
configuration matching TCRi's own end-to-end test. The mathematics behind the model are in
[The model](../concepts/model.md); how fitting is governed is in
[the training contract](../contracts/index.md).

## Fit on synthetic data

```python
import tcri

adata = tcri.datasets.simulate_tcri(n_clones=30, n_cells=2000, seed=0)

tcri.ml.TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="covariate",
    batch_key="batch",
)

model = tcri.ml.TCRIModel(adata, n_latent=128, seed=0)
model.train(max_epochs=200, batch_size=256)
model.to_anndata(adata)
```

After `to_anndata`, the learned latent, per-cell phenotype posterior, and clone phenotype
distributions are written under the canonical `tcri_*` keys (see
[The data model](../concepts/data-model.md)).

## A realistic small configuration

The knobs below match TCRi's real-data (Yost et al.) end-to-end test — a compact model that
trains in minutes on CPU. `K` is the number of phenotypes.

```python
import pyro

pyro.clear_param_store()  # own the process-global Pyro store

tcri.ml.TCRIModel.setup_anndata(
    adata, layer="counts",
    clonotype_key="clone_id", phenotype_key="phenotype",
    covariate_key="covariate", batch_key="batch",
)

model = tcri.ml.TCRIModel(
    adata,
    n_latent=16, n_hidden=32, n_layers=1,
    classifier_n_layers=1, classifier_hidden=32,
    K=adata.obs["phenotype"].nunique(),
    seed=0,
)
model.train(
    max_epochs=120,
    batch_size=512,
    accelerator="cpu",
    enable_progress_bar=False,
    enable_model_summary=False,
)
model.to_anndata(adata)
```

## Mind the KL warmup

```{important}
`n_steps_kl_warmup` (default 2000) counts **optimizer steps**, not epochs. On a small
dataset an epoch is only a few steps, so the KL ramp can need many epochs to finish — and
until it does, no early-stopping checkpoint is selected. `model.train` **warns** if the ramp
did not complete. Increase `max_epochs` (or the batch size) so the ramp finishes. This is
pinned by the [training contract](../contracts/index.md).
```

## Inspecting the fit

```python
# posterior-mean clone x phenotype table, shape (n_ct, P)
p_ct = model.get_p_ct()

# per-cell phenotype probabilities (index = obs_names, columns = phenotypes)
probs = model.predict(adata)

# latent embedding, shape (n_cells, n_latent)
z = model.get_latent_representation(adata)
```

See the [diagnostics tutorial](diagnostics.md) for posterior-predictive checks and training
curves, and [save/load](../api/utils.md) for persisting a fitted model with its `AnnData`.
