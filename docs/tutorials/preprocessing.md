# Preprocessing

Before fitting, TCRi expects an `AnnData` where each cell carries a **clonotype** label, a
**phenotype** label, a **covariate**, and a **batch**, with raw counts in a layer. This
tutorial covers the lightweight clone bookkeeping in `tcri.pp` and the one required
registration step. See [The data model](../concepts/data-model.md) for the concepts.

## A dataset to work with

`simulate_tcri` returns an `AnnData` with everything already in place: counts in
`layers["counts"]`, and `obs` columns `clone_id`, `phenotype`, `covariate`, `batch`.

```python
import tcri

adata = tcri.datasets.simulate_tcri(n_clones=30, n_cells=2000, seed=0)
print(adata)
print(adata.obs[["clone_id", "phenotype", "covariate", "batch"]].head())
```

With real data you would instead `sc.read_h5ad(...)` and make sure those four columns and
the counts layer exist.

## Clone sizes

{func}`tcri.pp.clone_size` counts the cells per clonotype and writes them to
`obs["clone_size"]`:

```python
tcri.pp.clone_size(adata, key_added="clone_size")
adata.obs["clone_size"].describe()
```

## Collapsing singletons

Real repertoires are dominated by singletons. {func}`tcri.pp.group_singletons` relabels
small clones (below `min_clone_size`) into a per-group "singleton" bucket, writing a new
clonotype column you can register instead of the raw one:

```python
tcri.pp.group_singletons(
    adata,
    clonotype_key="clone_id",
    groupby="batch",
    target_col="clone_id_grouped",
    min_clone_size=10,
)
adata.obs["clone_id_grouped"].nunique()
```

```{note}
Whether to collapse singletons is an analysis choice. Collapsing stabilizes per-clone
estimates but discards resolution on rare clones; keep the raw column if singletons matter
for your question.
```

## Register the columns

The one required step before modelling is `setup_anndata`, which records which columns hold
the clonotype, phenotype, covariate, and batch, and which layer holds counts:

```python
tcri.ml.TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",   # or "clone_id_grouped"
    phenotype_key="phenotype",
    covariate_key="covariate",
    batch_key="batch",
)
```

You are now ready to [train the model](training.md).
