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

## The reference

Silencing a gene moves the phenotype call for every gene, so the importance is read against a
permutation reference like every scored metric. `null_model` defaults to `"auto"`: the phenotype
null, rebuilt from the parameter store and this object and run on the same genes, the same cells
and the same rule, so `excess = value - null_value` is the part of a gene's importance that its
relationship to phenotype accounts for rather than its expression level.

```python
tcri.null.all(model, adata)                       # once, after to_anndata
gi = tcri.perturb.gene_importance(model, adata)   # value, null_value, excess
tcri.pl.gene_importance(adata)                    # ranked by excess, by default
tcri.pl.gene_importance(adata, quantity="value")  # the bare ranking, for comparison
```

**This is the one twin whose default is the corrected quantity.** Everywhere else `quantity`
defaults to the value with its reference drawn behind it; here it defaults to the excess,
because the bare ranking is not merely incomplete. Silencing a gene is an intervention whose
size scales with the gene's counts, and the encoder responds to that whatever the gene says
about phenotype.

Measured on a real fit of 2,000 genes: the bare importance and its null are 0.901
rank-correlated, and the bare top ten is led by MALAT1, TMSB4X, MT-CO2 and three ribosomal
proteins. Ranked by excess the same fit gives CD8B, CD8A, GATA3, IKZF2, RTKN2 and KLRC4, and
only 17 of the top 50 genes are shared. A reader shown the first list reasonably concludes the
perturbation is broken.

Which genes are SHOWN and which quantity is DRAWN are separate decisions. The gene set is
ranked by the excess whenever the result carries one, unless you explicitly ask for
`quantity="value"`, so `kind="rank"` and `kind="shift"` always describe the same genes.

There is no `fit=` here. This is a query on a MODEL rather than on a stored substrate, so a
different fit is reached by handing it a different model — and a fitted `TCRIModel` passed as
`null_model` is used directly, skipping the rebuild.

`kind="shift"` accepts only `quantity="value"`. The heatmap decomposes an importance across
phenotypes and the reference has no such decomposition stored, so an "excess shift" would have
to be invented rather than read.
