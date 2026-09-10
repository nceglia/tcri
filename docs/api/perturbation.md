# Perturbation (`tcri.perturb`)

In-silico perturbation of a fitted model: silence genes in the expression matrix and read the
phenotype call back through the same per-cell rule `predict()` uses, with every parameter
held fixed. Exposed as ``tcri.perturb``.

Two functions on one kernel, mirroring the split between `predict()` (per cell) and the
`tl` metrics (per group, with replicates, a contrast and a cache):

| function | returns | stores |
|---|---|---|
| {func}`knockout <tcri.perturbation.knockout>` | per-cell probabilities with a gene, or a gene program, silenced — the `predict()` frame | `obsm[key_added]` only when asked |
| {func}`gene_importance <tcri.perturbation.gene_importance>` | one importance per gene and group: the L1 distance the mean phenotype call moves when the gene is silenced | `uns["tcri_gene_importance"]`, drawn by {func}`tcri.pl.gene_importance` |

The pass is deterministic. The head reads the encoder's posterior mean, so no encoder sample
is drawn; `n_samples` draws the clone × covariate prior from the fitted posterior, shared by
every gene, and enters only through the gate. `use_gate=True` (default) is the model's own
rule, so the importance is about the calls you see; `use_gate=False` isolates the expression
pathway from the clone structure. The definitions are in the
[metrics contract](../contracts/index.md).

```python
# which genes drive the calls, per patient, contrasted between arms
tcri.perturb.gene_importance(adata_model, adata, splitby="disease_status")
tcri.pl.gene_importance(adata)                 # top genes, one dot per patient
tcri.pl.gene_importance(adata, kind="shift")   # which phenotype each gene moves the call toward

# a gene program, per cell, next to the baseline probabilities
tcri.perturb.knockout(adata_model, adata, genes=["GZMB", "PRF1"], key_added="X_ko_cytotoxic")
```

## Knockout

```{eval-rst}
.. automodule:: tcri.perturbation._knockout
   :members:
```

## Gene importance

```{eval-rst}
.. automodule:: tcri.perturbation._importance
   :members:
```

The payload has the three slots every metric has plus a fourth:

| slot | one row per | holds |
|---|---|---|
| `table` | (gene, group, draw) | `value`, the importance per draw — the substrate |
| `result` | (gene, group) | `value` with `sd`/`hdi_low`/`hdi_high` over draws |
| `stats` | (gene, contrast) | the per-gene Mann–Whitney over groups, when `splitby` is set |
| `shift` | (gene, phenotype, group) | `baseline`, `perturbed`, `shift = baseline − perturbed`, averaged over draws |

`shift` sums to zero over phenotypes: a positive entry is mass the phenotype **loses** when
the gene is silenced. Read any slot back with `tcri.get.gene_importance(adata, which=...)`.
