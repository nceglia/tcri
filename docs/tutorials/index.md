# Walkthrough

One pass through the package on a synthetic cohort: **16 patients**, sampled **pre** and
**post** treatment, split into a **disease** and a **control** arm.

The order below is the order you would actually work in — build, fit, *check the fit*, then
measure, then plot. The diagnostics come before the metrics on purpose: none of the numbers
in section 4 mean anything if section 3 looks wrong.

Everything here runs from `tcri.datasets`, so there is no external data to download. The
complete notebook, with outputs, is in `examples/example.ipynb` in the repository.

## 1. Build a cohort

{func}`simulate_cohort <tcri.datasets.simulate_cohort>` gives the shape most analyses have:
patients as replicates, an ordered timepoint axis *within* each patient, and a
`disease_status` label *between* them.

```python
import tcri
from tcri.datasets import simulate_cohort

adata = simulate_cohort(
    n_patients=16,
    timepoints=("pre", "post"),
    disease_fraction=0.5,
    n_clones=(14, 24),                   # ragged, as real cohorts are
    n_phenotypes=4,
    n_genes=40,
    n_cells_per_sample=260,
    clone_size_distribution="powerlaw",
    clone_size_exponent=2.0,
    disease_enrichment=12.0,
    control_enrichment=1.1,
    seed=0,
)
```

Two properties matter downstream:

- **Clones are paired across timepoints.** Each patient is simulated once, fixing its
  clone→phenotype structure, and every timepoint is drawn from that one population. Without
  this, `phenotypic_flux` and every change between timepoints would have nothing to measure.
- **Clone sizes are heavy-tailed** — a power law, as real repertoires are: a few large
  expanded clones over a long tail of singletons.

Only the clone→phenotype *concentration* changes between timepoints. Disease clones
commit; control clones barely move. Nothing is relabeled, so each cell's phenotype still
matches the expression it was generated with.

```{note}
`n_patients=16` rather than a smaller number is deliberate. With 4 per arm a rank test over
patients cannot reach $p<0.05$ no matter how large the effect — the smallest achievable
p-value is bounded by the number of distinct orderings.
```

### Preprocessing

{func}`tcri.pp.pool_rare_clones` pools each patient's rare clones into one `pooled@{patient}`
label. A clone is rare when it has fewer than `min_cells` cells and all of them come from one
sample; every other clone is kept. On a heavy-tailed repertoire most clones are tiny, and a
clone seen through a cell or two says little about its own phenotype distribution while still
counting as a row in every entropy normalizer.

```python
tcri.pp.pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient")
```

{func}`tcri.pp.clone_size` also exists, but it reads the registry that `to_anndata` writes, so
it runs *after* fitting rather than here.

## 2. Register and fit

`setup_anndata` names the columns the model needs. Two of them are easy to confuse and mean
entirely different things, even when they are the same column:

For clonotypes, `clonotype_key="auto"` will resolve common Scirpy-style columns
(`clone_id`, `cc_*`, or `<name>` with `<name>_size`) if you do not want to name one explicitly.

- **`batch_key`** is a *modeling* choice — one-hot encoded into every hidden layer of the
  encoder and decoder, so the network can absorb batch structure.
- **`replicate`** is a *statistical* choice — the independent unit. Registering it once lets
  `groupby` be left implicit on every metric, and the effective value still lands in each
  result's provenance rather than a `None` placeholder.

```python
from tcri.model import TCRIModel

TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="timepoint",   # the axis a change is taken across
    batch_key="patient",         # one-hot into the networks
    replicate="patient",         # the independent unit for statistics
)

model = TCRIModel(adata, n_latent=10, n_hidden=64, n_layers=2,
                  classifier_n_layers=1, classifier_hidden=64, K=4, seed=0)
model.train(max_epochs=150, batch_size=256)
model.to_anndata(adata)
tcri.null.all(model, adata)
```

`to_anndata` materializes everything the rest of the package reads — see
[Concepts](../concepts/index.md) for the full table of what it writes. `tcri.null.all` fits the
three permutation references beside it, five fits each: the same model on permuted labels, which
is what every metric below is read against: the phenotype fits for both entropies and mutual
information at one timepoint and for gene importance, the covariate fits for flux and every change
between timepoints. Pass `null_model=None` to a metric to
report the bare number instead.

## 3. Diagnostics — before trusting any metric

`tcri.diag` asks whether the *fit* is worth interpreting. Every function returns a DataFrame,
so the numbers are inspectable and not only drawn.

| function | question |
|---|---|
| {func}`joint_distribution_ppc <tcri.diagnostics.joint_distribution_ppc>` | does the model's clone×phenotype table resemble the observed crosstab? Per-clone L1 distance, 0 perfect, 2 maximal |
| {func}`phenotype_calibration <tcri.diagnostics.phenotype_calibration>` | when the classifier says 0.8, is it right 80% of the time? `ECE` is expected calibration error, lower better |
| {func}`reconstruction_ppc <tcri.diagnostics.reconstruction_ppc>` | does the decoder generate counts like the real ones? |
| {func}`permutation_null <tcri.diagnostics.permutation_null>` | how large is the metric under shuffled labels? |

`permutation_null` is the reference any mutual information should be read against, and it is
**model-free** — it scores the empirical crosstab and draws no posterior samples.

## 4. `tl` — compute once, cached with provenance

Every `tl` call stores its result in `uns` and returns `None`; `tcri.get` reads it back.
Mutual information, the two entropies and flux store their result **by level**:

| frame | one row per | read with |
|---|---|---|
| `replicate.summary` | (covariate, replicate) | `tcri.get.replicates` |
| `replicate.draws` | (covariate, replicate, draw) | `tcri.get.draws` |
| `item.summary` | (covariate, replicate, item) | `tcri.get.items` |
| `item.draws` | (covariate, replicate, item, draw) | `tcri.get.draws(..., items=True)` |
| `stats` | test | `tcri.get.stats` |

A change between two timepoints (section 6), and flux, label their rows `cov_from` and `cov_to`
in place of `covariate`. A summary is built *from* its draws, so the two cannot drift. Mutual information has one value
per joint and no item level. `stats` appears when `splitby` is set or the call is a change
between timepoints (section 6); it tests the replicate values, so its **n counts replicates,
never items** — 18 clones from 4 patients give n=4, not n=18. `joint_distribution` stores
`table` and `result`, read with `tcri.get.result`. With `inplace=False` a call stores nothing and
returns the whole result instead.

```python
tcri.tl.mutual_information(
    adata, covariate="pre", groupby="patient", splitby="disease_status", n_samples=100
)
tcri.get.replicates(adata, "mutual_information")   # one row per patient
tcri.get.stats(adata, "mutual_information")        # the disease vs control contrast
```

```{important}
**`n_clones_ref` matters on ragged cohorts.** `clonotypic_entropy` normalizes by $\log_2(C)$
where $C$ is that group's own supported clone count, so two patients with different $C$ are on
different scales. Passing `n_clones_ref` pins one denominator for everyone.
```

### Reading results back

{mod}`tcri.get` is how anything reads a cached result without knowing the storage format — the
plots use it, and so should your own code.

```python
tcri.get.result(adata, "mutual_information")   # the whole stored result
tcri.get.params(adata, "mutual_information")   # every argument it ran with
```

`params` carries every argument the tool ran with, **including defaults that were never
passed**, so a figure from six months ago can still say what produced it.

## 5. `pl` — twins that render the cache

A `pl` twin takes **no metric arguments**. It reads what `tl` stored, so the covariate,
groupby, splitby and `n_samples` it draws are the ones actually used — a figure cannot
disagree with the frame in your hand. Run the `tl` twin first.

```python
tcri.tl.clonotypic_entropy(adata, covariate="pre", groupby="patient", splitby="disease_status")
tcri.pl.clonotypic_entropy(adata)
```

The mark follows one rule: **a mark shows one variance component.** Within an x position the
sample is the coarsest unit that varies there — replicate over item over draw. So the dots are
patients wherever a p-value over patients sits above them, and draws are never pooled across
replicates.

### Colors are a property of the level

{func}`tcri.pl.resolve_colors` caches under scanpy's `uns["<key>_colors"]`, so a level keeps
its color in every later figure — and `sc.pl.umap(color="disease_status")` matches too.

## 6. Changes between timepoints

Mutual information and both entropies take `cov_from` and `cov_to` in place of `covariate`, and
then report the change `cov_to − cov_from`. It is taken **within a posterior draw**, with both timepoints drawn from one
shared sample, so the reported interval is the interval of the *difference*.

```python
tcri.tl.phenotypic_entropy(
    adata, cov_from="pre", cov_to="post", groupby="patient", splitby="disease_status"
)
tcri.get.replicates(adata, "phenotypic_entropy")   # each patient's change
tcri.get.items(adata, "phenotypic_entropy")        # each persistent clone's change
tcri.get.stats(adata, "phenotypic_entropy")        # the tests
```

HDIs do not subtract — you cannot recover the interval of a difference from the endpoints'
intervals, which is why the change is computed rather than arithmetic you do yourself. Its rows
are labeled `cov_from` and `cov_to` instead of `covariate`, and each carries its endpoints,
`value_from` and `value_to`, with `adjusted_from` and `adjusted_to` beside the reference.

The change has the same two levels as the value:

- **Per patient** it is the difference of the patient's two values, each counting every clone
  the patient has at that timepoint. This is the number `stats` tests.
- **Per item** the rows hold the **persistent** clones only, those present at both timepoints
  within the patient. Phenotypic entropy has one row per persistent clone; clonotypic entropy
  recomputes each phenotype on the persistent clones, so both sides share one clone set and one
  normalizer. A clone seen at one timepoint has no item row.

A change warns when some patients have no counted clone at a timepoint — their values are NaN
and they leave the tests — and when no patient has a clone at both timepoints, whose usual cause
is a covariate that is constant within each patient.

The same call works on mutual information and clonotypic entropy, and flux always spans two
timepoints:

```python
tcri.tl.mutual_information(adata, cov_from="pre", cov_to="post", splitby="disease_status")
tcri.tl.clonotypic_entropy(adata, cov_from="pre", cov_to="post", splitby="disease_status")
tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post", splitby="disease_status")
```

Mutual information has no items, so its change has the patient level only; each side is divided
by its own normalizer, and both are stored, `denom_from` and `denom_to`. Flux is already a
distance between the two timepoints, so it has no endpoints: per item it is each persistent
clone's distance, and per patient those distances weighted by each clone's share of the
`cov_from` joint: equally at the default `weighted=False`, by its cells at `cov_from` with
`weighted=True`.

Each call overwrites that metric's stored result, so the clonotypic entropy change above takes
the place of the single-timepoint result of section 5; pass `key_added=` to keep both.

### Did it change, and did the change differ

A change result's `stats` holds two kinds of row, named in its `comparison` column:

| `comparison` | question | test | rows |
|---|---|---|---|
| `"covariate"` | did the patients change? | Wilcoxon signed-rank on each patient's change | one per split level, or one over every patient without `splitby` |
| `"split"` | did the change differ between arms? | Mann–Whitney between split levels on each patient's change | one per pair of split levels |

On a covariate row `level_a` and `level_b` are `cov_from` and `cov_to`, the two arms are the
endpoints, and `n` counts the patients the test used. The covariate rows are kept per arm because
pooling the arms mixes the two groups the split row compares: when disease and control move in
opposite directions, a pooled row reports no change although every patient changed. The note in
section 1 applies here too: a two-sided signed-rank test over $n$ patients cannot go below
$p = 2/2^n$, so an arm of 5 or fewer patients cannot reach $p<0.05$.

Flux is never negative, so a test of its value against 0 would only report that every flux is
positive and how many patients there are: its covariate row tests the adjusted flux against 0,
and without a reference it has no test.

```{note}
The default reference of a change is the **covariate null**, which shuffles the timepoint each
cell sits at within its clone and batch. The adjusted endpoints, `adjusted_from` and
`adjusted_to`, are therefore measured against that null, and are not the adjusted values a
single-timepoint call reports against its own default, the phenotype null.
```

### Drawing a change

The `pl` twins draw a stored change as they draw a value:

```python
tcri.pl.phenotypic_entropy(adata)                # each patient's change, against a zero line
tcri.pl.phenotypic_entropy(adata, view="item")   # each persistent clone's change
tcri.pl.phenotypic_flux(adata, view="item")      # each persistent clone's flux
```

The default view draws one mark per patient, the number `stats` tests; `view="item"` draws the
item summary. Mutual information has no item view.

## 7. `perturb` — which genes the calls rest on

`tcri.perturb` is a query on the fitted model with its parameters held fixed: zero a gene's
column, run the cells through the encoder again, and read the phenotype call back with the
same rule `predict()` uses. {func}`gene_importance <tcri.perturbation.gene_importance>`
does that one gene at a time and reports how far the mean call moved, per patient, with the
same `groupby`/`splitby` machinery and the same cache as every metric.

```python
tcri.perturb.gene_importance(model, adata, splitby="disease_status")
tcri.pl.gene_importance(adata)                 # the top genes; one dot per patient, starred per gene
tcri.pl.gene_importance(adata, kind="shift")   # gene x phenotype: which way each gene moves the call
```

Two things to know before reading the ranking:

- **It is about the calls you see.** By default the head is combined with the clone ×
  covariate prior exactly as `predict()` does, so a gene can score low simply because the
  cells expressing it sit in clones with confident priors. `use_gate=False` scores the
  expression pathway alone, invariant to the clone structure.
- **The contrast is per gene**, over patients: `stats` has one row per gene, uncorrected
  across genes.

{func}`knockout <tcri.perturbation.knockout>` is the primitive underneath: the per-cell
probabilities with a gene, or a whole program, silenced — the `predict()` frame, written to
`obsm` when you pass `key_added`.

```python
tcri.perturb.knockout(model, adata, genes=["gene_3", "gene_7"], key_added="X_ko_program")
```

## 8. Saving and resuming

{func}`tcri.ut.save_tcri_session` writes the fitted model, its Pyro parameter store and the
`AnnData` together, so a later session picks up exactly where this one stopped — including
every cached `tl` result, since those live in `uns`.

```python
tcri.ut.save_tcri_session(model, adata, "run1/")
model, adata = tcri.ut.load_tcri_session("run1/")
```

Note the order: **model first, in both directions.** Passing them the other way round raises
`RuntimeError: Expected model.save ... to exist on TCRIModel`, because the AnnData arrives where
the model was expected — and unpacking `load` backwards binds each name to the wrong object
without any error at all.
