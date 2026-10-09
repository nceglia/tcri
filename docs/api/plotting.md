# Plotting (`tcri.pl`)

Visualization twins that mirror the `tcri.tl` metrics by name — each renders the tidy
result of its `tl` counterpart (no metric math lives here) — plus the shared color
helpers. Exposed as ``tcri.pl``.

## One quantity, and `quantity`

Every twin takes `quantity`. The default draws the adjusted value, `value - null_value`, on y
against a zero rule when the cached result carries a reference; `quantity="adjusted"` asks for
that panel by name. `quantity="value"` draws the raw value with its permutation reference behind
it — the same mark, grey and hollow, at the same x positions.

```python
tcri.pl.mutual_information(adata)                    # the adjusted value, on a zero rule
tcri.pl.mutual_information(adata, quantity="value")  # the value, reference behind it
```

Three details are deliberate. The x order is computed once and given to every mark, so the
reference pass cannot re-sort the axis under marks already drawn. The adjusted panel carries no
error bar: an interval on a difference needs paired draws, and draws are not paired across two
fits. And the endpoint view of a delta draws its grey dots at a fixed size with no connector —
the null's matched clone count is provably its parent's, so sizing it would repeat one number,
and a line between grey dots would assert matched identity across a permuted fit.

A result computed with `null_model=None` has no reference: the default draws the value, with
"(no reference)" on the y label, and asking for `quantity="adjusted"` raises rather than drawing
an empty panel.

## Small multiples

A twin is often one panel in a row, and its defaults are the ones a row needs:

- The x axis of a metric twin is in the category order of the column on it (the `obs`
  categories, or the registered phenotype and clonotype categories), so a level sits at the same
  position in every panel. `order=` overrides it.
- `legend=False` draws no legend, so the row can carry one.
- Dots that are replicates are colored by replicate, with a legend naming them, unless a split is
  the hue. An item axis such as the phenotypes is drawn in one color, so the replicates within
  each item can be told apart.
- A y label longer than its axes is wrapped onto more lines.

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
tcri.pl.clonotypic_entropy(adata, key="entropy_pre", ax=axes[0], legend=False)
tcri.pl.clonotypic_entropy(adata, key="entropy_post", ax=axes[1])
```

## Entropy plots

```{eval-rst}
.. automodule:: tcri.plotting._entropy
   :members:
```

## Mutual information

```{eval-rst}
.. automodule:: tcri.plotting._mutual_information
   :members:
```

## Phenotypic flux

```{eval-rst}
.. automodule:: tcri.plotting._flux
   :members:
```

## Gene importance

The twin of {func}`tcri.perturbation.gene_importance`: a ranking of the most important genes,
or the signed gene × phenotype shift behind it.

```{eval-rst}
.. automodule:: tcri.plotting._perturbation
   :members:
```

## Colors

```{eval-rst}
.. automodule:: tcri.plotting._colors
   :members:
```
