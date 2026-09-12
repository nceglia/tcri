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
#           adata.obs["condition"], adata.obs["patient"]
```

No paired data yet? {func}`simulate_cohort <tcri.datasets.simulate_cohort>` gives a synthetic
cohort with the shape the rest of this page assumes — patients as replicates, an ordered
condition axis within each patient, and a `disease_status` label between them:

```python
adata = tcri.datasets.simulate_cohort(seed=0)

# obs: clone_id, phenotype, condition ("pre"/"post"), patient,
#      disease_status ("disease"/"control")
```

{func}`simulate_tcri <tcri.datasets.simulate_tcri>` is the single-sample alternative, with a
mutual information known in closed form; it has one covariate level, so the flux and
group-comparison steps below do not apply to it.

## 2. Register the columns

`setup_anndata` records which columns hold the clonotype, phenotype, covariate, and batch,
and which layer holds counts.

If your clonotype column is Scirpy-style (`clone_id`, `cc_*`, or `<name>` paired with
`<name>_size`), you can pass `clonotype_key="auto"` to detect it instead of naming it.

```python
tcri.ml.TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="condition",
    batch_key="patient",
    replicate="patient",
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

Then build the permutation references once. Every scored metric reads its own against them, and
raises rather than reporting a bare number if they are not there.

```python
tcri.null.all(model, adata)
```

```{important}
**Upgrading from 0.11.** Every scored metric now computes a permutation reference by default, so
a script that recomputes a metric on a loaded 0.11 session raises until `tcri.null.all(model,
adata)` has run, or until it passes `null_model=None`. Rendering a cached 0.11 result is
unaffected: it draws with a "no reference" label. One phenotype fit serves both entropies,
mutual information and gene importance, so this is one prerequisite rather than four.
```

## 5. Compute metrics

All entropies and mutual information are in **bits**. Each carries its reference: `value` is the
number, `null_value` is what the same model gives on permuted labels, and
`excess = value - null_value` is the part the structure accounts for.

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

Each `tcri.pl` function is a plotting twin of the `tcri.tl` metric of the same name. A twin
takes **no metric arguments** — it renders what its `tl` counterpart cached, so run that first
and the figure cannot disagree with the frame in your hand:

```python
# MI per patient, boxed by cohort
tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", splitby="disease_status")
tcri.pl.mutual_information(adata)

# per-clone phenotype flux from the first to the last covariate
tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post", groupby="patient")
tcri.pl.phenotypic_flux(adata)
```

## 7. Compare groups

Contrasting a metric across cohorts is **not a separate step**. Pass `splitby` and the
metric produces the contrast itself, with the replicate unit already resolved:

```python
res = tcri.tl.mutual_information(
    adata, covariate="pre", groupby="patient", splitby="disease_status"
)

res["result"]   # one row per patient, carrying its disease_status label
res["stats"]    # the disease-vs-control contrast: mean_a/mean_b, delta, stat, p, stars
```

`groupby` is the replicate — one value per patient — and `splitby` is the cohort label,
which must be constant within each group. The contrast in `stats` is computed **over
groups**, not over rows, so a handful of patients cannot be inflated into significance by
the number of clones they happen to contain.

## 8. Which genes drive the calls

`tcri.perturb` asks the fitted model a counterfactual: silence a gene and see how far the
phenotype call moves, with every parameter held fixed. `gene_importance` scores each gene per
patient and contrasts the arms per gene; `knockout` returns the per-cell probabilities for a
gene program, the same frame as `predict()`.

```python
tcri.perturb.gene_importance(model, adata, splitby="disease_status")
tcri.pl.gene_importance(adata)                 # the top genes, one dot per patient
tcri.pl.gene_importance(adata, kind="shift")   # which phenotype each gene moves the call toward

probs = tcri.perturb.knockout(model, adata, genes=["GZMB", "PRF1"])
```

## Next steps

- [Tutorials](../tutorials/index.md) — runnable, end-to-end examples for preprocessing,
  training, metrics, and diagnostics.
- [Concepts](../concepts/index.md) — the model, the data it writes, and what the metrics mean.
- [API reference](../api/model.md) — exact call signatures.
