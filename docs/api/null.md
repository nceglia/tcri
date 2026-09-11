# Permutation references (`tcri.null`)

A model-based number on its own says nothing about whether the structure it measures is there.
Mutual information is positive for any clone × phenotype table, an in-silico knockout moves the
phenotype call for any gene, and a flux between conditions is non-zero whenever the two fits
differ at all. The floor is what tells you how much of the number is the data.

`tcri.null` fits that floor. A null is **the same model, with the same knobs, the same seed and
the same training arguments, on one permuted label vector** -- so `value − null_value` is the
part of the observed number that the permuted structure accounts for, on the metric's own scale.

Three axes, one function each:

| function | shuffled within | destroys | the reference for |
|---|---|---|---|
| {func}`phenotype <tcri.null.phenotype>` | batch, covariate, replicate | which cell carries which phenotype | clonotypic and phenotypic entropy, mutual information, gene importance |
| {func}`clonotype <tcri.null.clonotype>` | batch, covariate, replicate | which cells share a clone | the complement: a metric that moves here and not under the phenotype null is reading clonal structure |
| {func}`condition <tcri.null.condition>` | clonotype, batch | which condition a cell sits at | phenotypic flux and the two deltas |

```python
model.train(max_epochs=200, batch_size=512)
model.to_anndata(adata)
nulls = tcri.null.all(model, adata)     # three fits, written beside the main one
```

Each call writes a named fit into the same `AnnData` (see
{doc}`One main fit, and as many named fits beside it <../concepts/index>`), together with the
integer permutation it used and a settings record: the kind, the resolved strata and their cell
counts, the permutation seed, the training arguments and the parent.

Once they are there, every scored metric picks them up on its own. `null_model` defaults to
`"auto"`, which is the metric's entry in the table above; `result` gains `null_value` and
`excess = value - null_value`; `null_model=None` reports the bare value and creates no column.

```python
mi = tcri.tl.mutual_information(adata, covariate="post")   # value, null_value, excess
tcri.pl.mutual_information(adata)                          # the reference drawn behind it
tcri.pl.mutual_information(adata, quantity="excess")       # the difference, on a zero rule
```

Nulls are explicit. No metric fits one for you: a metric reads its reference from the substrate
and raises, naming the call to run, when it is not there. Fitting is a decision with a cost, and
a function that quietly triples its own runtime is worse than one that says what it needs.

## Checking a null is the null you meant

A null is an ordinary fit, so every reader takes one: the model-first functions
({func}`tcri.diag.reconstruction_ppc`, {func}`tcri.diag.loss`, `predict`, `tcri.perturb.*`)
accept the model object, and the substrate readers accept `fit=`. Three checks say whether a
null is the one you meant. Measured at 30 epochs on a 200-cell, 3-phenotype, 8-clone fixture,
seeds 0/1/2, with the majority-class rate of the true phenotype labels in brackets:

| quantity | parent | phenotype null | clonotype null | condition null |
|---|---|---|---|---|
| head accuracy [0.405], seed 0 | 0.630 | 0.545 | 0.460 | 0.630 |
| head accuracy [0.535], seed 1 | 0.820 | 0.535 | 0.545 | 0.820 |
| head accuracy [0.480], seed 2 | 0.865 | 0.480 | 0.480 | 0.865 |
| mutual information, seed 0 | 0.2075 | 0.0075 | 0.0130 | 0.2069 |
| mutual information, seed 1 | 0.4744 | 0.0223 | 0.0331 | 0.4749 |
| mutual information, seed 2 | 0.4522 | 0.0386 | 0.0076 | 0.4520 |

Read it this way. Reconstruction is unchanged by every null, because no permutation touches the
expression matrix. Mutual information collapses under both label nulls and is untouched by the
condition null, which is the axis rule working. And a label null's phenotype call falls to the
**majority-class rate, not to chance**: the permutation preserves each stratum's label multiset,
so the marginal survives it and a head with nothing else to learn predicts the marginal. On
seeds 1 and 2 the accuracy lands on that rate to three decimals. Chance would be 0.333.

The clonotype null's call falls too, rather than staying at its parent's, because the stored
label is the argmax of the gated posterior and a clonotype permutation destroys the prior the
gate mixes in. The classifier itself is intact; the call is not.

```{important}
`within` **refines** the default strata; it never replaces them. A permutation that crosses the
covariate changes the clone × covariate index, so the null's rows stop corresponding to the main
fit's and the reference cannot be joined to it at all. A `within` that drops a required column
raises and names the consequence rather than silently adding the column back.
```

The clone × covariate index is the parent's under all three kinds -- same count, same order,
same maps, same per-pair cell counts -- which is what makes the join total. What moves per cell
differs: under the phenotype null nothing does, because neither the clone nor the covariate has
been touched.

## Functions

```{eval-rst}
.. automodule:: tcri.null._nulls
   :members: phenotype, clonotype, condition, all
```
