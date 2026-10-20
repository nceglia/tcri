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

The result is stored by level, like the metrics, with the gene as a label on every row, and only
the adjusted value is stored (see the reference, below):

| slot | one row per | holds | read with |
|---|---|---|---|
| `replicate.draws` | (gene, group, pair) | `adjusted`, the importance less its null's, per pair (`null`, `draw`) | `tcri.get.draws` |
| `replicate.summary` | (gene, group) | `adjusted` with `adjusted_sd` and `adjusted_hdi_low`/`adjusted_hdi_high` | `tcri.get.replicates` |
| `item.draws` | (gene, group, phenotype, pair) | `adjusted`, the shift `baseline − silenced` less its null's, per pair | `tcri.get.draws(items=True)` |
| `item.summary` | (gene, group, phenotype) | the adjusted shift with `adjusted_sd` and its HDI | `tcri.get.items` |
| `stats` | (gene, contrast) | the per-gene Mann–Whitney of the adjusted importance over groups, when `splitby` is set | `tcri.get.stats` |

The shift sums to zero over phenotypes, on the fit and on the null, so the adjusted shift does
too: a positive entry is mass the phenotype **loses** when the gene is silenced, beyond what it
loses on the null. The importance is the sum of the shift's absolute values, per draw. A group
with no cell at `covariate` keeps its rows, NaN.

## The reference

Silencing a gene moves the phenotype call for every gene, so the importance is read against a
permutation reference like every scored metric, and only the adjusted value,
`adjusted = value - null_value` per paired draw, is stored. `null_model` defaults to `"auto"`:
the phenotype null, rebuilt from the parameter store and this object and run on the same genes,
the same cells and the same rule. `null_model=None` raises: there is nothing to adjust against.
The null runs store nothing of their own.

```python
tcri.null.all(model, adata)                          # once, after to_anndata
tcri.perturb.gene_importance(model, adata)
tcri.get.replicates(adata, "gene_importance")        # the adjusted importance
tcri.get.items(adata, "gene_importance")             # the adjusted shift
tcri.pl.gene_importance(adata)                       # ranked by the adjusted importance
```

**Pairing.** At `n_samples=0` the one plug-in value is paired with every null of the reference:
`adjusted` is the mean over the nulls, `adjusted_sd` the spread between them, and there is no
HDI. At `n_samples > 0` draw `d` is paired with null `d mod n_nulls` only, so each null draws only
its share of the `n_samples`, and the summary is over the `n_samples` pairs. `n_samples` must then
be at least the number of nulls; a multiple of it gives each null an equal share. Each draws frame
has `max(n_samples, n_nulls)` rows per summary row. This pairing is gene importance's own; the
metrics pair every draw with every null.

Both views of `pl.gene_importance` show the same genes, ranked by the adjusted importance.

There is no `fit=` here. This is a query on a MODEL rather than on a stored substrate, so a
different fit is reached by handing it a different model — and a fitted `TCRIModel` passed as
`null_model`, or the list of them `tcri.null.phenotype` returned, is used directly, skipping the
rebuild. `"auto"` rebuilds every fit of the phenotype reference and pools them.

The shift is computed per draw on the fit and on its paired null, so the item level carries the
adjusted shift on every pair, as the replicate level carries the adjusted importance, and
`view="item"` draws it. The adjusted shift is a per-phenotype difference, not a decomposition of
the adjusted importance: the absolute values of a gene's adjusted shifts do not sum to its
adjusted importance, because the L1 norm of a difference is not the difference of the L1
norms.
