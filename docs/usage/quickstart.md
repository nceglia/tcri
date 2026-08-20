# Quickstart

This guide walks through a full TCRi analysis: register your data, fit the model, write
the learned quantities back onto the `AnnData`, and read out the information-theoretic
metrics. For the concepts behind each step, see [The data model](../concepts/index.md),
[The model](../concepts/index.md), and [the metrics](../concepts/index.md).

## 1. Prepare your AnnData

TCRi works on an `AnnData` where every cell carries **both** a gene-expression profile and
a TCR **clonotype** label, plus a **covariate** (e.g. timepoint) and a **batch** (e.g.
patient). Raw counts should live in a layer.

```python
import scanpy as sc
import tcri

adata = sc.read_h5ad("your_data.h5ad")
# expected: adata.layers["counts"] (raw counts)
#           adata.obs["clone_id"], adata.obs["phenotype"],
#           adata.obs["timepoint"], adata.obs["patient"]
```

No paired data yet? Generate a synthetic dataset with a known ground-truth mutual
information:

```python
adata = tcri.datasets.simulate_tcri(seed=0)
```

## 2. Register the columns

`setup_anndata` records which columns hold the clonotype, phenotype, covariate, and batch,
and which layer holds counts.

```python
tcri.ml.TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="timepoint",
    batch_key="patient",
)
```

## 3. Fit the model

```python
model = tcri.ml.TCRIModel(adata, n_latent=128, seed=0)
model.train(max_epochs=200, batch_size=512)
```

```{tip}
`max_epochs` interacts with the KL warmup (`n_steps_kl_warmup`, default 2000 **optimizer
steps**). On small datasets a step is only a few cells, so the ramp can need many epochs to
complete — `model.train` warns if it did not. See [the training contract](../contracts/index.md).
```

## 4. Write results onto the AnnData

`to_anndata` materializes the learned latent, per-cell phenotype posterior, and clone
phenotype distributions under the canonical `tcri_*` keys.

```python
model.to_anndata(adata)
```

## 5. Compute metrics

All entropies and mutual information are in **bits**.

```python
# mutual information between clonotype and phenotype at one covariate
mi = tcri.tl.mutual_information(adata, covariate="pre", normalize_mode="average")

# how phenotypically diverse each phenotype's clones are
ce = tcri.tl.clonotypic_entropy(adata, covariate="pre")

# per-clone plasticity
pe = tcri.tl.phenotypic_entropy(adata, covariate="pre")

# how a clone's phenotype mix shifts between two covariates
flux = tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post")
```

```{important}
For the classical NMI, pass `normalize_mode="average"` explicitly. The default
(`"min"`) is group-comparable, which is usually what you want when clone counts differ
between groups. See [Concepts](../concepts/index.md).
```

## 6. Visualize

Each `tcri.pl` function is a plotting twin of the `tcri.tl` metric of the same name:

```python
# MI per patient, boxed by cohort
tcri.pl.mutual_information(adata, groupby="patient", splitby="response")

# per-clone phenotype flux from the first to the last covariate
tcri.pl.phenotypic_flux(adata, order=["pre", "post"])
```

## 7. Compare groups

Contrasting a metric across cohorts is **not a separate step**. Pass `splitby` and the
metric produces the contrast itself, with the replicate unit already resolved:

```python
res = tcri.tl.mutual_information(
    adata, covariate="pre", groupby="patient", splitby="response"
)

res["result"]   # one row per patient, carrying its response label
res["stats"]    # the R-vs-NR contrast: delta, U, p, stars, and direction probabilities
```

`groupby` is the replicate — one value per patient — and `splitby` is the cohort label,
which must be constant within each group. The contrast in `stats` is computed **over
groups**, not over rows, so a handful of patients cannot be inflated into significance by
the number of clones they happen to contain.

## Next steps

- [Tutorials](../tutorials/index.md) — runnable, end-to-end examples for preprocessing,
  training, metrics, and diagnostics.
- [Concepts](../concepts/index.md) — the model, the data it writes, and what the metrics mean.
- [API reference](../api/model.md) — exact call signatures.
