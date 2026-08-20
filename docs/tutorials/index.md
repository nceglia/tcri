# Walkthrough

One pass through the package on a synthetic cohort: **16 patients**, sampled **pre** and
**post** treatment, split into **responders** and **non-responders**.

The order below is the order you would actually work in — build, fit, *check the fit*, then
measure, then plot. The diagnostics come before the metrics on purpose: none of the numbers
in section 4 mean anything if section 3 looks wrong.

Everything here runs from `tcri.datasets`, so there is no external data to download. The
complete notebook, with outputs, is in `examples/example.ipynb` in the repository.

## 1. Build a cohort

{func}`simulate_cohort <tcri.datasets.simulate_cohort>` gives the shape most analyses have:
patients as replicates, an ordered condition axis *within* each patient, and a response label
*between* them.

```python
import tcri
from tcri.datasets import simulate_cohort

adata = simulate_cohort(
    n_patients=16,
    conditions=("pre", "post"),
    responder_fraction=0.5,
    n_clones=(14, 24),                   # ragged, as real cohorts are
    n_phenotypes=4,
    n_genes=40,
    n_cells_per_sample=260,
    clone_size_distribution="powerlaw",
    clone_size_exponent=2.0,
    responder_enrichment=12.0,
    nonresponder_enrichment=1.1,
    seed=0,
)
```

Two properties matter downstream:

- **Clones are paired across conditions.** Each patient is simulated once, fixing its
  clone→phenotype structure, and every condition is drawn from that one population. Without
  this, `phenotypic_flux` and the `delta_*` metrics would have nothing to measure.
- **Clone sizes are heavy-tailed** — a power law, as real repertoires are: a few large
  expanded clones over a long tail of singletons.

Only the clone→phenotype *concentration* changes between conditions. Responders' clones
commit; non-responders' barely move. Nothing is relabelled, so each cell's phenotype still
matches the expression it was generated with.

```{note}
`n_patients=16` rather than a smaller number is deliberate. With 4 per arm a rank test over
patients cannot reach $p<0.05$ no matter how large the effect — the smallest achievable
p-value is bounded by the number of distinct orderings.
```

### Preprocessing

{func}`tcri.pp.group_singletons` collapses clones below a size threshold into one pooled id
*per group*. On a heavy-tailed repertoire most clones are tiny, and a clone seen once carries
no information about its own phenotype distribution while still counting as a row in every
entropy normalizer. Pooling is one defensible answer; dropping them is another.

```python
tcri.pp.group_singletons(
    adata, clonotype_key="clone_id", groupby="patient",
    target_col="clone_id_grouped", min_clone_size=5,
)
```

{func}`tcri.pp.clone_size` also exists, but it reads the registry that `to_anndata` writes, so
it runs *after* fitting rather than here.

## 2. Register and fit

`setup_anndata` names the columns the model needs. Two of them are easy to confuse and mean
entirely different things, even when they are the same column:

- **`batch_key`** is a *modelling* choice — one-hot encoded into every hidden layer of the
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
    covariate_key="condition",   # the axis a delta is taken across
    batch_key="patient",         # one-hot into the networks
    replicate="patient",         # the independent unit for statistics
)

model = TCRIModel(adata, n_latent=10, n_hidden=64, n_layers=2,
                  classifier_n_layers=1, classifier_hidden=64, K=4, seed=0)
model.train(max_epochs=150, batch_size=256)
model.to_anndata(adata)
```

`to_anndata` materializes everything the rest of the package reads — see
[Concepts](../concepts/index.md) for the full table of what it writes.

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

Every `tl` function returns the **same three slots** and stores that object in `uns`:

| slot | one row per | reduced over |
|---|---|---|
| `table` | (covariate, group, item, draw) | nothing — the substrate |
| `result` | (covariate, group, item) | `draw` only |
| `stats` | (split_a, split_b) | items → groups, then contrast |

`result` is built *from* `table`, so the two cannot drift. `stats` appears only when `splitby`
is set, and its **n counts replicates, never items** — 18 clones from 4 patients give n=4, not
n=18.

```python
res = tcri.tl.mutual_information(
    adata, covariate="pre", groupby="patient", splitby="response", n_samples=100
)
res["result"]   # one row per patient
res["stats"]    # the responder vs non-responder contrast
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
tcri.get.result(adata, "mutual_information")   # exactly what tl returned
tcri.get.params(adata, "mutual_information")   # every argument it ran with
```

`params` carries every argument the tool ran with, **including defaults that were never
passed**, so a figure from six months ago can still say what produced it.

## 5. `pl` — twins that render the cache

A `pl` twin takes **no metric arguments**. It reads what `tl` stored, so the covariate,
groupby, splitby and `n_samples` it draws are the ones actually used — a figure cannot
disagree with the frame in your hand. Run the `tl` twin first.

```python
tcri.tl.clonotypic_entropy(adata, covariate="pre", groupby="patient", splitby="response")
tcri.pl.clonotypic_entropy(adata)
```

The mark follows one rule: **a mark shows one variance component.** Within an x position the
sample is the coarsest unit that varies there — replicate over item over draw. So the dots are
patients wherever a p-value over patients sits above them, and draws are never pooled across
replicates.

### Colours are a property of the level

{func}`tcri.pl.resolve_colors` caches under scanpy's `uns["<key>_colors"]`, so a level keeps
its colour in every later figure — and `sc.pl.umap(color="response")` matches too.

## 6. The paired entropies — what changed, per clone

`delta_*` takes `cov_from`/`cov_to` and subtracts **within a posterior draw**, so the reported
interval is the interval of the *difference*.

```python
tcri.tl.delta_phenotypic_entropy(
    adata, cov_from="pre", cov_to="post", groupby="patient", splitby="response"
)
```

HDIs do not subtract — you cannot recover the interval of a difference from the endpoints'
intervals, which is why this is a function rather than arithmetic you do yourself.

Support is the **intersection**: clones present at both conditions, within each patient. A
delta needs both endpoints, and the drop is warned about because it moves `n`.

There is deliberately **no `delta_mutual_information`**. MI has no item axis, so it is already
the repertoire-level number and its "delta" is a subtraction of two cached scalars. That one
belongs to you, not the package.

## 7. Saving and resuming

{func}`tcri.ut.save_tcri_session` writes the fitted model, its Pyro parameter store and the
`AnnData` together, so a later session picks up exactly where this one stopped — including
every cached `tl` result, since those live in `uns`.

```python
tcri.ut.save_tcri_session(adata, model, "run1/")
adata, model = tcri.ut.load_tcri_session("run1/")
```
