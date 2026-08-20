# Concepts

The end-to-end mental model for TCRi: what it learns, how it learns it, where the results
live on your `AnnData`, and what the metrics computed from them mean. Read this once and the
API reference will make sense.

## The problem

You never observe the distribution you care about.

A repertoire is enormous and a sequencing run samples a sliver of it. What comes back is a
few hundred or few thousand cells, unevenly distributed: a handful of expanded clones supply
most of the cells, while the long tail of the repertoire shows up once or twice, or not at
all. The clone→phenotype distribution you can count directly from that sample is not the
distribution of the underlying repertoire — it is one noisy realisation of it, and the noise
is worst exactly where the data is thinnest.

**TCRi is an estimator for that unobserved joint distribution** $P(c, \phi)$ over clonotypes
and phenotypes. Everything else in the package is downstream of it: the entropies, the mutual
information, and the flux are all pure functions of the joint, so the estimate is the engine
and the metrics are readouts of it.

### Why not just count?

You can build the joint empirically — crosstab clone against phenotype and normalize. The
difference is what each does when the data runs out.

The empirical table treats every clone's counts at face value. A clone seen twice at one
timepoint and forty times at the next gets a phenotype distribution from two cells, and the
apparent change between timepoints is mostly the two-cell estimate being wrong.

TCRi instead ties each clone's covariate-level distribution to that **clone's own distribution
pooled across covariates**, and shrinks toward it. A clone with plenty of cells at a covariate
level barely moves; a clone with two cells is pulled most of the way back to its overall
behaviour, because two cells are not evidence that it changed.

The consequence is deliberate: **differences between covariates come out attenuated.** The
estimator understates change rather than overstating it.

### Why conservative is the right failure mode here

Two reasons, and they compound:

- **There is no ground truth to check against.** The true joint of the underlying repertoire
  is unobservable, so an estimator that over-reads cannot be caught by comparing it to
  anything. A method that errs toward "no change" produces claims that survive scrutiny; one
  that errs toward change produces claims nobody can falsify.
- **The sampling is biased toward exactly the clones that look most convincing.** Large
  expanded clones fill most of a run's cell budget, so they carry tight per-clone estimates,
  while the small clones — where most of the repertoire's diversity lives — carry almost none.
  Reading raw counts weights the confident-looking part of a biased sample most heavily.

So the shift TCRi reports for a clone is a lower bound on the shift in the repertoire it was
drawn from. When it does report a difference, the difference is not an artefact of having
sequenced two cells.

Three index sets recur throughout:

| Symbol | Meaning | Example |
|--------|---------|---------|
| $c$ | clonotype | a TCR clone |
| $m$ | covariate value | `T1`, `T2` |
| $\phi$ | phenotype | `A`, `B`, `C` |

A **`ct` pair** is a specific $(c, m)$ combination — clonotype $c$ observed at covariate $m$.
Every cell maps to exactly one `ct` pair.

## The model

`TCRIModel` is a hierarchical Bayesian model fit with [Pyro](https://pyro.ai) on top of an
[scvi-tools](https://scvi-tools.org) variational autoencoder. It ties each clone's phenotype
distribution to gene expression.

```{image} ../images/model_pgm.png
:alt: TCRi plate diagram, annotated with the arguments that tune each part
:width: 780px
:align: center
```

Shaded nodes are **observed**; open nodes are latent. The three nested plates are the three
index sets above: clonotypes $c$, clone × covariate pairs $ct$, and cells $i$. The right-hand
column names the argument that tunes each piece — so if you want a clone's covariate levels to
stay closer to its overall distribution, the figure tells you that `local_scale` is the knob.

The generative story has four steps:

**1 — Clonotype prior $p_c$.** Each clonotype draws a distribution over phenotypes from a
mixture-Dirichlet prior over archetypes $\psi_b$:

$$\omega_c \sim \tfrac{1}{B_c}\sum_b \mathrm{Dir}(\alpha\,\psi_b).$$

This is the top of the hierarchy — what a clone looks like overall, across every covariate.

**2 — Local distribution $p_{ct}$.** For each `ct` pair, a covariate-specific distribution is
drawn, anchored to its clone's prior:

$$\phi_m \mid \omega_{h(m)} \sim \mathrm{Dir}(\beta\,\omega_{h(m)}).$$

This lets a clone's phenotype mix *shift per covariate* while staying tied to the clone
prior. $\beta$ controls how tightly — large means stay near the prior.

**3 — Latent $z$.** Each cell's expression is encoded to a latent $z_i$ under a
[VampPrior](https://arxiv.org/abs/1705.07120) mixture over learnable pseudo-inputs, and a
zero-inflated negative binomial decoder reconstructs counts.

**4 — Per-cell phenotype.** A classifier maps $z$ to phenotype **logits** — expression-based
evidence — which are fused with the log of the cell's local prior through a gate $\pi$.

So a cell's phenotype call fuses two sources: *what its expression looks like* and *what its
clone tends to be at this covariate*. The variational guide learns approximate posteriors
$q(p_c)$ and $q(p_{ct})$; `get_p_ct()` returns the posterior-mean $p_{ct}$.

### What the two scales actually control

$\alpha$ (`global_scale`, default `5.0`) is the concentration of the clonotype prior and
$\beta$ (`local_scale`, default `3.0`) the concentration of the covariate-level one. $\beta$ is
the shrinkage knob described above: raise it and each covariate level is held
closer to its clone's overall distribution, so differences between covariates attenuate further.

Neither is a hyperparameter to sweep for a better fit, because **a Dirichlet changes shape, not
just width, as its concentration crosses 1.** Above 1 the distribution is peaked at its mean —
"this clone looks like its archetype, give or take". Below 1 it turns U-shaped, piling mass at
the corners of the simplex — "this clone is entirely one phenotype, we just don't know which".
Those are opposite claims about what a clone is. Lowering a scale past 1 to loosen the prior
does not weaken the assumption; it replaces it with a different one.

### Gating

The per-cell phenotype fuses the classifier on $z$ and the clone's local prior through a gate
$\pi$ = `gate_prob` (default **0.5**):

$$P(\phi \mid \text{cell}) = \operatorname{softmax}\!\big(\pi\, f_\text{cls}(z) +
(1-\pi)\log p_{ct}\big).$$

The endpoints behave as you would expect: $\pi=1$ makes `predict()` the pure classifier,
$\pi=0$ the pure clonotype prior, and `gate_prob=None` recovers the additive rule
$f_\text{cls} + \log p_{ct}$.

### What trains the classifier

The model is fit by stochastic variational inference maximizing the ELBO. The discrete
per-cell phenotype terms are replaced by a differentiable **alignment penalty** that pulls
the classifier's distribution toward the clone's local prior:

$$\gamma \sum_i \mathrm{KL}\!\big(\operatorname{softmax}(\ell_i)\;\|\;\phi_{g(i)}\big).$$

This penalty is **the only thing that trains the classifier**. Without it the logits never
enter the objective and phenotype recovery sits at chance. $\gamma$ is `phenotype_kl_weight`
(default `1.0`).

## What `to_anndata` writes

After training, {meth}`model.to_anndata <tcri.model._model.TCRIModel.to_anndata>` materializes
the learned quantities onto your `AnnData` under canonical `tcri_*` keys, so every downstream
function can read them:

| Location | Key | Meaning | Shape |
|----------|-----|---------|-------|
| `.uns` | `tcri_p_ct` | posterior-mean local phenotype distribution per `ct` pair | `(n_ct, P)` |
| `.uns` | `tcri_ct_to_c` | clonotype index for each `ct` pair | `(n_ct,)` |
| `.uns` | `tcri_ct_to_cov` | covariate index for each `ct` pair | `(n_ct,)` |
| `.uns` | `tcri_ct_array_for_cells` | `ct`-pair index for each **cell** | `(n_cells,)` |
| `.uns` | `tcri_cov_array_for_cells` | covariate index for each **cell** | `(n_cells,)` |
| `.uns` | `tcri_local_scale` | Dirichlet concentration scale for posterior sampling | scalar |
| `.uns` | `tcri_gate_prob` | classifier/prior gate weight (`NaN` if gating is off) | scalar |
| `.uns` | `tcri_classifier_temperature` | temperature applied to classifier logits | scalar |
| `.uns` | `tcri_{phenotype,clonotype,covariate}_categories` | category label lists (index ↔ name) | — |
| `.uns` | `tcri_metadata` | column-name mapping | dict |
| `.obsm` | `X_tcri` | latent means $z$ | `(n_cells, n_latent)` |
| `.obsm` | `X_tcri_logits` | classifier phenotype logits | `(n_cells, P)` |
| `.obsm` | `X_tcri_logposterior` | `logits + log p_ct` (additive, ungated) | `(n_cells, P)` |
| `.obsm` | `X_tcri_probabilities` | per-cell phenotype posterior (**gate-aware**) | `(n_cells, P)` |
| `.obs` | `tcri_phenotype` | hard phenotype label (argmax of the posterior) | `(n_cells,)` |

```{important}
The per-cell `.uns` arrays are stored in the **original full-cell space**. Slicing the
`AnnData` to a view or subset shifts `.obs`/`.obsm` but **not** `.uns`, so the indices
misalign. Re-run `model.to_anndata` on the subset, or pass the full object and filter with
the `clones=` argument. The metric functions guard against this and raise rather than return
silently-wrong numbers.
```

### Indexing, concretely

The `ct_to_c` / `ct_to_cov` arrays are the join keys connecting a cell to its clone, its
covariate, and its phenotype distribution. A cell's local prior is
`tcri_p_ct[ct_array[cell]]`, and all cells of a given covariate are
`cov_array_for_cells == m`.

## Point estimate vs. posterior samples

Two ways to read the clone→phenotype distributions:

- **Point estimate** — the posterior mean `tcri_p_ct` directly (`n_samples=0`, the default).
- **Posterior samples** — draw
  $p_{ct} \sim \mathrm{Dir}(\texttt{local\_scale}\cdot \bar p_{ct})$ to propagate uncertainty
  into the metrics.

{func}`joint_distribution <tcri.tools._joint.joint_distribution>` exposes both. Passing
`n_samples > 0` returns a stack of posterior draws instead of a single point estimate, and
every metric accepts the same argument to report a posterior mean and HDI.

## Temperatures

Three distinct temperatures sharpen ($T<1$) or flatten ($T>1$) distributions at different
stages. Don't conflate them:

| Parameter | Acts on |
|-----------|---------|
| `prior_temperature` | the fixed clone→phenotype prior, at model setup |
| `guide_temperature` | the learned variational posteriors |
| `temperature` (in `joint_distribution`) | the combined per-cell distribution at query time |

## The metrics

The **joint distribution** $P(c, \phi)$ for a covariate — clonotypes × phenotypes, optionally
weighted by clone size — is the input to every metric. All are computed in **bits**.

Every metric is a pure function of that joint, so each can be computed from a point estimate
or over posterior draws.

### Clonotypic entropy — one value per phenotype

$$H[P(c\mid\phi)] = -\sum_c P(c\mid\phi)\,\log_2 P(c\mid\phi)$$

How spread a phenotype is across clones. It is **support-only**: clones with zero mass in
that column are dropped *before* renormalizing. An epsilon clip would fabricate uniform mass
on absent clones and inflate the entropy. The normalizer is $\log_2(\#\text{supported
clones})$, or $\log_2(\texttt{n\_clones\_ref})$ when supplied, so values are comparable across
groups. An empty column returns **NaN**.

### Phenotypic entropy — one value per clone

$$H[P(\phi\mid c)] = -\sum_\phi P(\phi\mid c)\,\log_2 P(\phi\mid c)$$

Plasticity versus commitment of a clone. All $P$ phenotypes are in the sum with
$0\log 0 := 0$, and the normalizer is $\log_2(P)$. A clone with zero mass returns **NaN** —
never reindexed to zeros, which would report a spurious $H=1$ for a clone never observed.

### Mutual information — one value per joint

$$I(c;\phi) = \sum_{c,\phi} P(c,\phi)\,\log_2\frac{P(c,\phi)}{P(c)\,P(\phi)}$$

How much knowing the clonotype tells you about phenotype — the clone↔phenotype coupling.
Normalization is controlled by `normalize_mode`:

| `normalize_mode` | denominator |
|------------------|-------------|
| `"min"` *(default)* | $\min(H(c), H(\phi))$ — coefficient of constraint |
| `"average"` | $\tfrac12(H(c)+H(\phi))$ — classical NMI |

```{important}
The default is **`min`**. The mean denominator scales with $\log_2(C)$ and so is **not
comparable across groups with different clone counts** — which is usually what you want to
compare. Choose `"average"` explicitly when you need the classical NMI.
```

### Phenotypic flux

Flux measures how a clone's phenotype distribution $P(\phi\mid c)$ **shifts between two
covariates** — pre to post treatment, say. The default `distance_metric="kl"` is a
Kullback–Leibler divergence; `"l1"` is also available.

### How they fit together

The entropy and MI families are tied by a single decomposition:

$$I(c;\phi) = H(c) - \sum_\phi P(\phi)\,H[P(c\mid\phi)]$$

This is the keystone — it makes it impossible to redefine one family without breaking the
other, and it is enforced by test. The same tests pin the sanity limits: uniform over $k$
gives $\log_2 k$ (normalized $1.0$), all mass on one outcome gives $0$, an independent joint
gives $I=0$, and $I(c;\phi) = I(\phi;c) \ge 0$.

## Where to go next

The [walkthrough](../tutorials/index.md) runs this end to end on a simulated cohort, in the
order you would actually work: register and fit, **check the fit with `tcri.diag`**, then
compute metrics, then plot them. The diagnostics step is not optional — an estimate of an
unobservable joint is only worth reading if the model that produced it checks out.

The API reference gives exact signatures for [metrics](../api/metrics.md),
[plotting](../api/plotting.md), and [diagnostics](../api/diagnostics.md).
