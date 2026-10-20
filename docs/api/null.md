# Permutation references (`tcri.null`)

A model-based number on its own says nothing about whether the structure it measures is there.
Mutual information is positive for any clone × phenotype table, an in-silico knockout moves the
phenotype call for any gene, and a flux between covariate levels is non-zero whenever the two fits
differ at all. The floor is what tells you how much of the number is the data.

`tcri.null` fits that floor. A null is **the same model, with the same knobs, the same seed and
the same training arguments, on one permuted label vector** -- so `value − null_value` is the
part of the observed number that the permuted structure accounts for, on the metric's own scale.
A reference is `n_nulls` such fits (default 5), each on its own permutation, because a null moves
from one fit to the next by more than one fit's own draws show; every metric pairs its draws with
all of them.

Three axes, one function each:

| function | shuffled within | destroys | the reference for |
|---|---|---|---|
| {func}`phenotype <tcri.null.phenotype>` | batch, covariate, replicate | which cell carries which phenotype | clonotypic and phenotypic entropy and mutual information at one covariate level, gene importance |
| {func}`clonotype <tcri.null.clonotype>` | batch, covariate, replicate | which cells share a clone | the complement: a metric that moves here and not under the phenotype null is reading clonal structure |
| {func}`covariate <tcri.null.covariate>` | clonotype, batch | which covariate level a cell sits at | phenotypic flux, and the entropies and mutual information on a change between covariate levels (`cov_from` with `cov_to`) |

```python
model.train(max_epochs=200, batch_size=512)
model.to_anndata(adata)
nulls = tcri.null.all(model, adata)     # 3 x n_nulls fits (15), written beside the main one
```

Each call writes `n_nulls` named fits into the same `AnnData`, `null.phenotype[0]` to
`null.phenotype[4]` at the default (see
{doc}`One main fit, and as many named fits beside it <../concepts/index>`), each with the integer
permutation it used and a settings record: the kind, the resolved strata and their cell counts,
the permutation seed and its spawn key, the reference and index, the training arguments and the
parent. The cost scales with `n_nulls`: that many fits per kind, and a metric's per-draw frames
that many times longer. Calling again replaces the whole reference.

Once they are there, every scored metric picks them up on its own. `null_model` defaults to
`"auto"`, which is the metric's entry in the table above; every frame gains `null_value` and
`adjusted = value - null_value`, the quantity the metric reports; `null_model=None` reports the
bare value and creates no column. `null_model="null.phenotype[2]"` reads one fit alone, and the
list of models a `tcri.null.*` call returned can be passed back as `null_model`.

```python
tcri.tl.mutual_information(adata, covariate="post")
tcri.get.replicates(adata, "mutual_information")           # value, null_value, adjusted
tcri.pl.mutual_information(adata)                          # the adjusted value, on a zero rule
tcri.pl.mutual_information(adata, quantity="value")        # the value, reference drawn behind it
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

| quantity | parent | phenotype null | clonotype null | covariate null |
|---|---|---|---|---|
| head accuracy [0.405], seed 0 | 0.630 | 0.545 | 0.460 | 0.630 |
| head accuracy [0.535], seed 1 | 0.820 | 0.535 | 0.545 | 0.820 |
| head accuracy [0.480], seed 2 | 0.865 | 0.480 | 0.480 | 0.865 |
| mutual information, seed 0 | 0.2075 | 0.0075 | 0.0130 | 0.2069 |
| mutual information, seed 1 | 0.4744 | 0.0223 | 0.0331 | 0.4749 |
| mutual information, seed 2 | 0.4522 | 0.0386 | 0.0076 | 0.4520 |

Read it this way. Reconstruction is unchanged by every null, because no permutation touches the
expression matrix. Mutual information collapses under both label nulls and is untouched by the
covariate null, which is the axis rule working. And a label null's phenotype call falls to the
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
   :members: phenotype, clonotype, covariate, all
```
