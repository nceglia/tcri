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

Nulls are explicit. No metric fits one for you: a metric reads its reference from the substrate
and raises, naming the call to run, when it is not there. Fitting is a decision with a cost, and
a function that quietly triples its own runtime is worse than one that says what it needs.

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
