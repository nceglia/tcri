# Plotting (`tcri.pl`)

Visualization twins that mirror the `tcri.tl` metrics by name — each renders the tidy
result of its `tl` counterpart (no metric math lives here) — plus the shared color
helpers. Exposed as ``tcri.pl``.

## The reference mark, and `quantity`

Every twin takes `quantity`. The default, `"value"`, draws the metric with its permutation
reference behind it — the same mark, grey and hollow, at the same x positions — when the cached
result carries one. `quantity="excess"` puts `value - null_value` on y against a zero rule.

```python
tcri.pl.mutual_information(adata)                     # value, reference behind it
tcri.pl.mutual_information(adata, quantity="excess")  # the difference
```

Three details are deliberate. The x order is computed once and given to every mark, so the
reference pass cannot re-sort the axis under marks already drawn. The excess panel carries no
error bar: an interval on a difference needs paired draws, and draws are not paired across two
fits. And the endpoint view of a delta draws its grey dots at a fixed size with no connector —
the null's matched clone count is provably its parent's, so sizing it would repeat one number,
and a line between grey dots would assert matched identity across a permuted fit.

A result computed with `null_model=None` has no reference: the y label says so, and asking for
`quantity="excess"` raises rather than drawing an empty panel.

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
