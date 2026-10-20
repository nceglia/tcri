# Plotting (`tcri.pl`)

Visualization twins that mirror the `tcri.tl` metrics by name — each renders the tidy
result of its `tl` counterpart (no metric math lives here) — plus the shared color
helpers. Exposed as ``tcri.pl``.

## One quantity, and `quantity`

Every twin takes `quantity`. The default draws the adjusted value, `value - null_value`, on y
against a zero rule when the cached result carries a reference; `quantity="adjusted"` asks for
that panel by name. `quantity="value"` draws the raw value with its permutation reference behind
it — the same mark, gray and hollow, at the same x positions.

```python
tcri.pl.mutual_information(adata)                    # the adjusted value, on a zero rule
tcri.pl.mutual_information(adata, quantity="value")  # the value, reference behind it
```

Two details are deliberate. The x order is computed once and given to every mark, so the
reference pass cannot re-sort the axis under marks already drawn. The adjusted value is drawn
like the value: its per-draw differences make a violin where draws are the sample. Behind the
value's violins the null is one marker at its mean: its spread over draws is already inside the
adjusted value's interval.

A result computed with `null_model=None` has no reference: the default draws the value, with
"(no reference)" on the y label, and asking for `quantity="adjusted"` raises rather than drawing
an empty panel.

## Drawing a change

A twin draws a change result, computed with `cov_from` and `cov_to`, in the same two views as a
value. The default view draws one mark per replicate, its change, against a zero rule, since a
change can be negative. `view="item"` draws each item's change from the item summary: one per
persistent clone for phenotypic entropy, one per phenotype, recomputed on the persistent clones,
for clonotypic entropy. Mutual information has no item view. The y label of a change carries a
`Δ`, as in "adjusted Δ phenotypic entropy (bits)"; flux, a distance between the two levels,
keeps its own label.

```python
tcri.tl.phenotypic_entropy(adata, cov_from="pre", cov_to="post", splitby="disease_status")
tcri.pl.phenotypic_entropy(adata)                # each replicate's change, by split
tcri.pl.phenotypic_entropy(adata, view="item")   # each persistent clone's change
```

`pl.phenotypic_flux` takes the same `view=`: each replicate's flux, or each persistent clone's
with `view="item"`. Flux is a distance and never negative, so its zero rule is on the adjusted
panel only. Every plotted number is a stored one: the endpoints of a change stay in its
summaries, `value_from` and `value_to` with `adjusted_from` and `adjusted_to`, read with
`tcri.get.replicates` and `tcri.get.items`.

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

The twin of {func}`tcri.perturbation.gene_importance`: a ranking of the most important genes
(`view="replicate"`), or the signed gene × phenotype shift behind it (`view="item"`).

```{eval-rst}
.. automodule:: tcri.plotting._perturbation
   :members:
```

## Colors

```{eval-rst}
.. automodule:: tcri.plotting._colors
   :members:
```
