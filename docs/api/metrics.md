# Metrics (`tcri.tl`)

Information-theoretic metrics over the clone × phenotype joint distribution: the joint
engine, the two entropies, mutual information, phenotypic flux, and the comparisons between
splits and between covariate levels.
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

Mutual information, the two entropies and flux store `{replicate, item, stats}`;
`joint_distribution` stores `{table, result}`. Every call returns `None`, and `tcri.get` reads the
stored result. The tests land in `stats`, read with `tcri.get.stats` — you do not call a second
function: the between-split contrast when you pass `splitby`, and the test of a change when you
pass `cov_from` and `cov_to`.

| column | meaning |
|---|---|
| `comparison` | what the row compares: `"split"`, two levels of `splitby`; `"covariate"`, the change from `cov_from` to `cov_to` |
| *`<splitby>`* | on a split row the contrast as a label, e.g. `disease_status` → `"disease vs control"`; on a covariate row the split level the row is within |
| `quantity` | which column was contrasted: `"adjusted"` when a reference was computed, `"value"` otherwise |
| `level_a`, `level_b` | the two levels contrasted: two split levels, or `cov_from` and `cov_to` |
| `replicate_unit` | the column `groupby` resolved to |
| `mean_a`, `sd_a`, `n_a` | first arm's mean over replicates, its spread, and how many |
| `ci_low_a`, `ci_high_a` | between-replicate interval on that mean |
| `mean_b`, `sd_b`, `n_b`, `ci_low_b`, `ci_high_b` | the same for the second arm |
| `delta` | `mean_b - mean_a` |
| `stat`, `p`, `stars` | the test statistic (Mann-Whitney U on a split row, Wilcoxon signed-rank W on a covariate row), p-value, significance marker; blank when no test could be run |

Every pair of split levels with a replicate has its split row. An arm whose replicates have no
finite value has `n` 0, and the row has NaN statistics and blank stars.

### Covariate rows

A change (`cov_from` with `cov_to`) asks whether the replicates changed, and its `stats` holds the
covariate rows before the split rows. Each is a two-sided Wilcoxon signed-rank test on each
replicate's change: with `splitby`, one row per split level (did disease change, did control
change) beside the split rows, which compare the changes between split levels with the
Mann-Whitney test; without `splitby`, one row over every replicate. A pooled row would mix the two
groups the split rows compare: when they move in opposite directions it reports no change
although every replicate changed.

On a covariate row:

- `level_a` and `level_b` are `cov_from` and `cov_to`;
- arm a and arm b are the change's endpoints of the tested quantity, `adjusted_from` and
  `adjusted_to`, or `value_from` and `value_to` without a reference, so `delta = mean_b - mean_a`
  as on a split row;
- a replicate enters the test when both endpoints are finite and its change is not exactly 0, and
  the arms, `delta` and `n` are all over the replicates the test used, so `n_a = n_b`;
- with fewer than two such replicates the row has a NaN `p` and blank stars;
- the adjusted endpoints are measured against the covariate null, the default reference of a
  change, so they are not the adjusted values of a single-covariate call.

The two-sided signed-rank test cannot reach $p<0.05$ with 5 or fewer replicates: its smallest p
over $n$ replicates is $2/2^n$.

Flux is a distance between the two levels and never negative, so a test of its value against 0
would only report that every flux is positive and how many replicates there are. Its covariate row tests the adjusted flux against 0 instead: its arms are
NaN, and `delta` is the mean adjusted flux over the replicates the test used. Without a reference
the row has no test.

The interval columns are `ci_*` — **between** replicates. The `hdi_*` columns that appear in
the summaries are a different quantity, the **within**-group spread over posterior draws, and they
are deliberately absent here: an interval over patients and an interval over draws are not
interchangeable.

**`n` counts replicates, never items.** Both tests run on one value per replicate, so 18
clones from 4 patients give n=4. Handing the row-level frame to a rank test instead is
what produces a starred p-value off a handful of patients.

With more than two levels every pair is reported; multiplicity is yours to handle.

**One quantity is contrasted, and `quantity` names it.** With a reference present the frame
carries one row per contrast, on the adjusted value; without one, on the value. The plot stars
only the quantity on its axis, so the marks and the star above them are always the same
quantity, and the value panel of a result whose contrast is on the adjusted value carries none.

## References

No model-based number here is reported bare. Every quantity above is positive for data with no
structure in it, so each is read against a **permutation reference**: the same model, refit on
permuted label vectors, five fits by default (see {doc}`tcri.null <null>`).

```python
tcri.tl.mutual_information(adata, covariate="post")
tcri.get.replicates(adata, "mutual_information")[["value", "null_value", "adjusted"]]
```

`null_model` defaults to `"auto"`, which resolves by the call:

| call | default reference |
|---|---|
| an entropy or mutual information at one covariate level, `covariate=` | the phenotype null |
| an entropy or mutual information on a change, `cov_from` with `cov_to` | the covariate null |
| `phenotypic_flux` | the covariate null |
| `perturb.gene_importance` | the phenotype null |

A change and a flux measure movement between covariate levels, so their reference is a fit in
which that movement is absent: the covariate null shuffles the level each cell sits at within its
clone and batch. `None` computes no reference and creates no column; any reference or fit on the
object can be named instead, or the fitted null models passed. Every frame gains `null_v` and
`adjusted` beside every native value column `v`, and `adjusted = value - null_value` is the
quantity the metric reports, on its own scale. The fit's draw *d* is paired with draw *d* of every fit of the reference and all the pairs
are summarized, so the interval carries how much the null moves from one fit to the next; with
`n_samples=0` a row has one pair per null, `adjusted_sd` is the sd between them and no HDI is
reported. There is no ratio: a ratio explodes on a
near-zero reference and hides effect size, and dividing two stored columns is yours.

The reference is **the caller's own call with the fit changed** — every other argument forwarded
verbatim, because `covariate`, `cov_from`, `cov_to`, `groupby`, `clones`, `exclude_pools`,
`weighted`, `normalized`, `normalize_mode`, `n_clones_ref`, `distance_metric`, `temperature` and
`n_samples` each change what is being measured. Its draws take their own seed, derived from your `random_state`.

**The adjusted value is computed per draw.** The null is the value expected under no
association, and the fit and the null are independent fits, so the fit's draw *d* is paired with
the null's draw *d*. The per-draw frames carry `null_value` and `adjusted` on every draw, and
the summaries give the adjusted value's mean with `adjusted_sd`, `adjusted_hdi_low` and `adjusted_hdi_high`
over those draws, so it has an interval like any value. At `n_samples=0` there is one draw on
each side, and `adjusted` is the difference of the two point values. On a change both endpoints
are paired within one null, so `adjusted = adjusted_to - adjusted_from`.

**Mutual information stores both denominators.** A normalized MI divides by a normalizer taken
from the same joint it normalizes, and a null does not share it, so its replicate summary
carries `denom` and `null_denom`. `value * denom` and `null_value * null_denom` recover both numbers in bits.
On a change each side is divided by its own normalizer, and the summary carries `denom_from`,
`denom_to` and their `null_` columns.
