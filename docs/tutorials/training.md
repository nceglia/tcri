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
trains in minutes on CPU. (`K` is the number of clone *archetypes*; here it is set to the
phenotype count for a small, fast model.)

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

## Model construction knobs

The [five-line quickstart](index.md) spotlights `n_latent`, `n_layers`, `global_scale` (α),
`local_scale` (β), and `K`. The full set of construction knobs — including the four
**deferred** from that page (`gate_prob`, `phenotype_kl_weight`, `n_pseudo_obs`, `seed`) — is
below. The Greek symbols are the ones used in [the model](../concepts/model.md).

| Knob | Default | Symbol | What it controls |
| --- | --- | --- | --- |
| `n_latent` | `128` | — | latent-space dimension |
| `n_hidden` / `n_layers` | `128` / `3` | — | width and depth of the encoder/decoder |
| `classifier_hidden` / `classifier_n_layers` / `classifier_dropout` | `128` / `3` / `0.1` | — | size, depth, and dropout of the phenotype classifier head |
| `global_scale` | `5.0` | α | concentration of the clonotype→archetype prior (higher ⇒ clones pulled harder toward shared archetypes) |
| `local_scale` | `3.0` | β | concentration of the per-clone phenotype Dirichlet (higher ⇒ cells pulled harder toward their clone's phenotype prior) |
| **`gate_prob`** | `0.5` | π | mixes classifier logits with the clone's phenotype prior as `π·classifier + (1−π)·log prior`. `None` switches to the additive rule `classifier + log prior`. Must be in `[0, 1]` or `None`. |
| **`phenotype_kl_weight`** | `1.0` | γ | weight of the phenotype-alignment surrogate `KL(cell posterior ‖ clone prior)`; `0.0` disables the alignment factor |
| `K` | `10` | — | number of clone archetypes (k-means centroids over clonotypes); capped at the clonotype count |
| **`n_pseudo_obs`** | `10` | — | number of VampPrior pseudo-inputs forming the mixture prior over the latent `z` |
| `patience_epochs` | `300` | — | early-stopping patience, counted in validation checks |
| **`seed`** | `None` | — | seeds network initialization **and** minibatch order; set it for reproducible fits |
| `prior_temperature`, `guide_temperature`, `classifier_temperature` | `1.0` | — | sampling temperatures — advanced; leave at `1.0` |
| `kl_weight_max`, `guide_init_scale`, `use_enumeration` | `1.0`, `10.0`, `False` | — | advanced; leave at defaults |

```{note}
`gate_prob` (π) and `phenotype_kl_weight` (γ) change the model's mathematics, so they are
governed by the [model contract](../contracts/index.md). Setting `gate_prob=None` or
`phenotype_kl_weight=0.0` is a deliberate ablation, not a tuning default.
```

## Training knobs

`model.train(...)` takes a handful of optimizer knobs; everything else is forwarded to
scvi-tools / Lightning.

| Knob | Default | What it controls |
| --- | --- | --- |
| `max_epochs` | `1000` | maximum passes over the data |
| `batch_size` | `1000` | minibatch size — prefer 256–1024 (warns if `≥ n_obs`) |
| `lr` | `1e-3` | Adam learning rate |
| `reconstruction_loss_scale` | `1e-2` | weight on the ZINB reconstruction vs the latent prior (β-VAE-style; a sanctioned deviation from Note 1) |
| `n_steps_kl_warmup` | `2000` | optimizer steps to anneal the latent KL 0 → 1 (see below) |
| `**kwargs` | — | forwarded to scvi's `TrainRunner` / Lightning: `accelerator`, `devices`, `enable_progress_bar`, `enable_model_summary`, `check_val_every_n_epoch`, `early_stopping_monitor` / `_mode` / `_patience` |

```{note}
The KL-warmup schedule lives on the module, so a **second `model.train(...)` call continues
the ramp** rather than restarting it — construct a fresh model for a clean schedule. After
fitting, `model.training_record_` records the epochs run, the warmup progress, and the
selected checkpoint.
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
