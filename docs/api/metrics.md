# Metrics (`tcri.tl`)

Information-theoretic metrics over the clone × phenotype joint distribution: the joint
engine, the two entropies, mutual information, phenotypic flux, and group comparisons.
All entropies and MI are in **bits** (log base 2). Exposed as ``tcri.tl``.

The definitions are frozen by the [metrics contract](../contracts/index.md); the
conceptual reference is [Information-theoretic metrics](../concepts/index.md).

## Joint distribution

```{eval-rst}
.. automodule:: tcri.tools._joint
   :members:
```

## Entropies

```{eval-rst}
.. automodule:: tcri.tools._entropy
   :members:
```

## Mutual information

```{eval-rst}
.. automodule:: tcri.tools._mutual_information
   :members:
```

## Phenotypic flux

```{eval-rst}
.. automodule:: tcri.tools._flux
   :members:
```

## The `stats` slot

Every metric returns `{table, result, stats}`. When you pass `splitby`, the between-group
contrast lands in `stats` — you do not call a second function.

| column | meaning |
|---|---|
| *`<splitby>`* | the contrast as a label, e.g. `response` → `"R vs NR"` |
| `level_a`, `level_b` | the two levels contrasted |
| `replicate_unit` | the column `groupby` resolved to |
| `mean_a`, `sd_a`, `n_a` | first level's mean over replicates, its spread, and how many |
| `ci_low_a`, `ci_high_a` | between-replicate interval on that mean |
| `mean_b`, `sd_b`, `n_b`, `ci_low_b`, `ci_high_b` | the same for the second level |
| `delta` | `mean_a - mean_b` |
| `stat`, `p`, `stars` | Mann-Whitney statistic, p-value, significance marker |

The interval columns are `ci_*` — **between** replicates. The `hdi_*` columns that appear in
`result` are a different quantity, the **within**-group spread over posterior draws, and they
are deliberately absent here: an interval over patients and an interval over draws are not
interchangeable.

**`n` counts replicates, never items.** Items are collapsed to their group before the contrast,
so 18 clones from 4 patients give n=4. Handing the row-level frame to a rank test instead is
what produces a starred p-value off a handful of patients.

With more than two levels every pair is reported; multiplicity is yours to handle.
