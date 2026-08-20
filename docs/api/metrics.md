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
| `group_a`, `group_b` | the two levels contrasted |
| `mean_a`, `mean_b`, `delta` | group means and their difference |
| `U`, `p`, `stars` | Mann-Whitney statistic, p-value, significance marker |
| `p_gt`, `p_lt` | direction probabilities |
| `hdi_low`, `hdi_high` | interval on the difference |

**`n` counts replicates, never items.** Items are collapsed to their group before the contrast,
so 18 clones from 4 patients give n=4. Handing the row-level frame to a rank test instead is
what produces a starred p-value off a handful of patients.

With more than two levels every pair is reported; multiplicity is yours to handle.
