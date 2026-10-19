# Perturbation (`tcri.perturb`)

In-silico perturbation of a fitted model: silence genes in the expression matrix and read the
phenotype call back through the same per-cell rule `predict()` uses, with every parameter
held fixed. Exposed as ``tcri.perturb``.

Two functions on one kernel, mirroring the split between `predict()` (per cell) and the
`tl` metrics (per group, with replicates, a contrast and a cache):

| function | computes | where it lands |
|---|---|---|
| {func}`knockout <tcri.perturbation.knockout>` | per-cell probabilities with a gene, or a gene program, silenced — the `predict()` frame | returned, and written to `obsm[key_added]` only when asked |
| {func}`gene_importance <tcri.perturbation.gene_importance>` | one importance per gene and group: the L1 distance the mean phenotype call moves when the gene is silenced, and that move per phenotype | `uns["tcri_gene_importance"]`, read with `tcri.get.replicates` and `tcri.get.items` and drawn by {func}`tcri.pl.gene_importance`; the call returns `None` |

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
tcri.pl.gene_importance(adata, view="item")    # which phenotype each gene moves the call toward

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

The result is stored by level, like the metrics, with the gene as a label on every row:

| slot | one row per | holds | read with |
|---|---|---|---|
| `replicate.draws` | (gene, group, draw) | `value`, the importance per draw | `tcri.get.draws` |
| `replicate.summary` | (gene, group) | `value` with `sd`/`hdi_low`/`hdi_high` over draws | `tcri.get.replicates` |
| `item.draws` | (gene, group, phenotype, draw) | `value`, the shift `baseline − silenced` per draw | `tcri.get.draws(items=True)` |
| `item.summary` | (gene, group, phenotype) | the shift with `sd`/`hdi_low`/`hdi_high` over draws | `tcri.get.items` |
| `stats` | (gene, contrast) | the per-gene Mann–Whitney over groups, when `splitby` is set | `tcri.get.stats` |

The shift sums to zero over phenotypes: a positive entry is mass the phenotype **loses** when
the gene is silenced, and the importance is the sum of its absolute values, per draw. A group with
no cell at `covariate` keeps its rows, NaN.

## The reference

Silencing a gene moves the phenotype call for every gene, so the importance is read against a
permutation reference like every scored metric. `null_model` defaults to `"auto"`: the phenotype
null, rebuilt from the parameter store and this object and run on the same genes, the same cells
and the same rule, so `adjusted = value - null_value` is the part of a gene's importance that
its relationship to phenotype accounts for rather than its expression level.

```python
tcri.null.all(model, adata)                          # once, after to_anndata
tcri.perturb.gene_importance(model, adata)
tcri.get.replicates(adata, "gene_importance")        # value, null_value, adjusted
tcri.get.items(adata, "gene_importance")             # the shift, the null's, the adjusted shift
tcri.pl.gene_importance(adata)                       # ranked by the adjusted importance
tcri.pl.gene_importance(adata, quantity="value")     # the bare ranking, for comparison
```

The correction matters most here. Silencing a gene is an intervention whose size scales with the
gene's counts, and the encoder responds to that whatever the gene says about phenotype, so the
bare ranking is not merely incomplete: it is dominated by something the question is not about.

Which genes are SHOWN and which quantity is DRAWN are separate decisions. The gene set is
ranked by the adjusted importance whenever the result carries one, unless you explicitly ask for
`quantity="value"`, so `view="replicate"` and `view="item"` always describe the same genes.

There is no `fit=` here. This is a query on a MODEL rather than on a stored substrate, so a
different fit is reached by handing it a different model — and a fitted `TCRIModel` passed as
`null_model`, or the list of them `tcri.null.phenotype` returned, is used directly, skipping the
rebuild. `"auto"` rebuilds every fit of the phenotype reference and pools them.

The shift is computed per draw on the fit and on each null, so the item level carries
`null_value` and `adjusted = value - null_value` on every pair, as the replicate level does.
`view="item"` draws the adjusted shift at the default `quantity="auto"`, like every item view.
The adjusted shift is a per-phenotype difference, not a decomposition of the adjusted
importance: the absolute values of a gene's adjusted shifts do not sum to its adjusted
importance, because the L1 norm of a difference is not the difference of the L1 norms.
