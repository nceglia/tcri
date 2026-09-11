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
| *`<splitby>`* | the contrast as a label, e.g. `disease_status` → `"disease vs control"` |
| `quantity` | which column was contrasted: `"value"`, or `"excess"` when a reference was computed |
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

**Both quantities are contrasted, and nothing switches on its own.** With a reference present
the frame carries one row per (contrast, `quantity`), and the plot selects which one to star, so
the marks and the star above them are always the same quantity. The two rows describe the same
replicates: the collapse runs once over both columns, so a replicate missing a reference is
dropped from both rather than from one.

## References

No model-based number here is reported bare. Every quantity above is positive for data with no
structure in it, so each is read against a **permutation reference**: the same model, refit on
one permuted label vector (see {doc}`tcri.null <null>`).

```python
mi = tcri.tl.mutual_information(adata, covariate="post")
mi["result"][["value", "null_value", "excess"]]
```

`null_model` defaults to `"auto"`: the phenotype null for the entropies, mutual information and
gene importance, the condition null for flux and the two deltas. `None` computes no reference
and creates no column; any fit on the object can be named instead. `result` and `table` gain
`null_v` and `excess` beside every native value column `v`, and `excess = value - null_value` on
the metric's own scale. There is no ratio: a ratio explodes on a near-zero reference and hides
effect size, and dividing two stored columns is yours.

The reference is **the caller's own call with the fit changed** — every other argument forwarded
verbatim, because `groupby`, `clones`, `weighted`, `normalized`, `normalize_mode`,
`n_clones_ref`, `distance_metric`, `temperature` and `n_samples` each change what is being
measured.

The `excess` carries no `sd` and no interval. It is a difference of two summaries, and there is
no correspondence between the parent's draw 7 and the null's, so nothing pairs them.

**Mutual information stores both denominators.** A normalised MI divides by a normaliser taken
from the same joint it normalises, and a null does not share it, so `result` carries `denom` and
`null_denom`. `value * denom` and `null_value * null_denom` recover both numbers in bits.
