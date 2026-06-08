# Data model & concepts

This page is the end-to-end mental model for TCRi: the objects it learns, where
they live on your `AnnData`, and how the hierarchical model connects gene
expression, clonotypes, and phenotypes. Read it once before the API reference and
the rest will make sense.

## The problem

TCRi works on **paired single-cell data**: every cell has both a gene-expression
profile and a TCR **clonotype** label, and belongs to a **covariate** group
(e.g. timepoint, response status). The questions it answers are *distributional*:
for a given clonotype, what is its distribution over **phenotypes**, and how does
that distribution shift across covariates? From those distributions TCRi computes
information-theoretic summaries (entropies, mutual information, flux).

Three index sets recur throughout:

| Symbol | Meaning | Example |
|--------|---------|---------|
| $c$ | clonotype | a TCR clone |
| $m$ | covariate value | `T1`, `T2` |
| $\phi$ | phenotype | `A`, `B`, `C` |

A **`ct` pair** is a specific $(c, m)$ combination — clonotype $c$ observed at
covariate $m$. Every cell maps to exactly one `ct` pair.

## The hierarchical model

`TCRIModel` is a two-level Bayesian model fit with Pyro on top of an scVI-style
variational autoencoder. The generative story:

1. **Clonotype prior $p_c$.** Each clonotype draws a distribution over phenotypes
   from a mixture-Dirichlet prior: $p_c \sim \text{MixtureDirichlet}(\cdot)$,
   shape `(n_clonotypes, P)`. This is the *top* of the hierarchy — what a clone
   looks like overall.
2. **Local distribution $p_{ct}$.** For each `ct` pair, a covariate-specific
   distribution is drawn, anchored to its clone's prior:
   $p_{ct} \sim \text{Dirichlet}(\texttt{local\_scale}\cdot p_{c[ct]})$. This lets
   a clone's phenotype mix *shift per covariate* while staying tied to the clone
   prior. `local_scale` controls how tightly: large → stay near the prior.
3. **Latent $z$.** Each cell's expression $x$ is encoded to a latent $z$ under a
   [VampPrior](https://arxiv.org/abs/1705.07120) mixture, and a ZINB decoder
   reconstructs counts (the scVI-style expression model).
4. **Per-cell phenotype.** A classifier maps $z$ to phenotype **logits**
   (expression-based evidence). Combined additively with the log of the cell's
   local prior $\log p_{ct}$, the **per-cell posterior** is
   $\operatorname{softmax}(\text{logits} + \log p_{ct})$.

So a cell's phenotype call fuses two sources: *what its expression looks like*
(classifier logits) and *what its clone tends to be at this covariate*
($p_{ct}$). The variational **guide** learns approximate posteriors
$q(p_c)$ and $q(p_{ct})$; `TCRIModel`'s `get_p_ct()` returns the posterior-mean
$p_{ct}$.

## What `register_model` writes

After training, {func}`register_model <tcri.preprocessing._preprocessing.register_model>`
materializes the learned quantities onto your `AnnData` so every downstream
function can read them. This is the **data model** the rest of the library
assumes:

| Location | Key | Meaning | Shape |
|----------|-----|---------|-------|
| `.uns` | `tcri_p_ct` | posterior-mean local phenotype distribution per `ct` pair | `(n_ct, P)` |
| `.uns` | `tcri_ct_to_c` | clonotype index for each `ct` pair | `(n_ct,)` |
| `.uns` | `tcri_ct_to_cov` | covariate index for each `ct` pair | `(n_ct,)` |
| `.uns` | `tcri_ct_array_for_cells` | `ct`-pair index for each **cell** | `(n_cells,)` |
| `.uns` | `tcri_cov_array_for_cells` | covariate index for each **cell** | `(n_cells,)` |
| `.uns` | `tcri_local_scale` | Dirichlet concentration scale for posterior sampling | scalar |
| `.uns` | `tcri_{phenotype,clonotype,covariate}_categories` | category label lists (index ↔ name) | — |
| `.uns` | `tcri_metadata` | column-name mapping (`phenotype_col`, `clone_col`, …) | dict |
| `.obsm` | `X_tcri` | latent means $z$ | `(n_cells, n_latent)` |
| `.obsm` | `X_tcri_logits` | classifier phenotype logits (expression evidence) | `(n_cells, P)` |
| `.obsm` | `X_tcri_logposterior` | `logits + log p_ct` | `(n_cells, P)` |
| `.obsm` | `X_tcri_probabilities` | `softmax(logits + log p_ct)` — per-cell phenotype posterior | `(n_cells, P)` |
| `.obs` | `tcri_phenotype` | hard phenotype label (argmax of the posterior) | `(n_cells,)` |

```{important}
These per-cell `.uns` arrays (`tcri_ct_array_for_cells`, `tcri_cov_array_for_cells`)
are stored in the **original full-cell space**. Slicing the `AnnData` to a view or
subset shifts `.obs`/`.obsm` but **not** `.uns`, so the indices misalign. Re-run
`register_model` on a subset, or pass the full object and filter with the
`clones=` argument. The metric functions guard against this and raise rather than
return silently-wrong numbers.
```

### Indexing, concretely

The `ct_to_c` / `ct_to_cov` arrays are the join keys. To pull the phenotype
distribution of clonotype $c$ at covariate $m$:

```text
ct_to_c   :  ct → c        (which clone is this pair?)
ct_to_cov :  ct → m        (which covariate is this pair?)
ct_array  :  cell → ct     (which pair does this cell belong to?)
tcri_p_ct :  ct → p(φ)     (the pair's phenotype distribution)
```

A cell's local prior is therefore `tcri_p_ct[ct_array[cell]]`, and all cells of a
given covariate are `cov_array_for_cells == m`.

## Point estimate vs. posterior samples

Two ways to read the clone→phenotype distributions:

- **Point estimate** — use the posterior mean `tcri_p_ct` directly.
- **Posterior samples** — draw
  $p_{ct} \sim \text{Dirichlet}(\texttt{local\_scale}\cdot \bar p_{ct})$ to
  propagate uncertainty into the metrics.

{func}`joint_distribution <tcri.preprocessing._preprocessing.joint_distribution>`
and {func}`joint_distribution_posterior <tcri.preprocessing._preprocessing.joint_distribution_posterior>`
expose both; passing `n_samples > 0` returns a stack of samples instead of a single
point estimate.

## From distributions to metrics

The **joint distribution** $p(c, \phi)$ for a covariate — clonotypes × phenotypes,
optionally weighted by clone size — is the input to every information-theoretic
metric in `tcri.tl`:

- **Phenotypic entropy** $H(\phi \mid c)$ — how phenotypically mixed a clone is.
- **Clonotypic entropy** $H(c \mid \phi)$ — how clonally diverse a phenotype is.
- **Mutual information** $I(c; \phi)$ — how much knowing the clonotype tells you
  about phenotype (clone–phenotype coupling).
- **Flux** — change in a clone's phenotype distribution between two covariates.

## Temperatures

Three distinct "temperatures" sharpen ($T<1$) or flatten ($T>1$) distributions at
different stages — don't conflate them:

| Parameter | Acts on |
|-----------|---------|
| `prior_temperature` | the fixed clone→phenotype prior, at model setup |
| `guide_temperature` | the learned variational posteriors $q(p_c)$, $q(p_{ct})$ |
| `temperature` (in `joint_distribution*`) | the combined per-cell distribution at query time |

## Typical flow

```text
setup_anndata → TCRIModel(...) → model.train(...) → register_model(adata, model)
                                                          │
                          ┌───────────────────────────────┴───────────────┐
                   tcri.tl metrics                                  tcri.pl plots
            (entropy, mutual_information, flux)              (sankey, probabilities, …)
```

With the objects above on your `AnnData`, the API reference for
[preprocessing](../api/preprocessing.md), [metrics](../api/metrics.md), and
[plotting](../api/plotting.md) tells you the exact call signatures.
