# Methods Conformance — code ↔ Supplementary Note 1

> **This is the eq-by-eq map + deviation history.** The *enforced* contract is
> `governance/MODEL_CONTRACT.md` (prose) + `tcri/model/_model_contract.py`
> (manifest), checked by `tests/test_model_contract_conformance.py`. Model math
> changes require updating that contract **first**.

Maps the TCRi generative model in **Supplementary Note 1: Methods for Information
theoretic metrics for single cell RNA and T-cell receptor sequencing**
(`governance/source/supplementary_note_1_SS_2026-08-03.pdf`) to the implementation, and records
every known deviation. The PDF is the source of truth; this file is the living
conformance record (update it whenever the model changes).

Code: `tcri/model/_module.py` (`TCRIModule.model`/`.guide`), `tcri/model/_priors.py`
(`VampPrior`, `MixtureDirichlet`), `tcri/model/_classifier.py`
(`PhenotypeClassifier`), `tcri/model/_model.py` (`TCRIModel`, `.predict`).

## Symbols

| Note | Meaning | Code |
|---|---|---|
| `ω_c` | clonotype-level phenotype dist | `p_c` (sample site `"p_c"`) |
| `ϕ_m` | covariate-level phenotype dist | `p_ct` (sample site `"p_ct"`); `get_p_ct()` |
| `z_i` | continuous latent embedding | `z` (sample site `"latent"`) |
| `z^ϕ_i` | discrete phenotype — **LATENT**, replaced by the surrogate | not sampled; `pheno_kl` factor (DE-18 withdrawn) |
| `x_i` | gene expression | `x` (sample site `"obs"`) |
| `f_cls` | classifier `R^L → R^P` (η_cls) | `self.classifier` |
| `π` | gating weight | `gate_prob` (default **0.5**) |
| `α` | global Dirichlet scale | `global_scale` |
| `β` | local Dirichlet scale | `local_scale` |
| `γ` | surrogate KL weight | `phenotype_kl_weight` (default 1.0) |
| `g(i)` | covariate-group of cell `i` | `ct_array[i]` (clone×covariate index) |
| `h(m)` | clonotype of group `m` | `ct_to_c[m]` |

## Generative model

| Eq | Note | Code (`_module.py::model`) | Status |
|---|---|---|---|
| 1 | `ω_c ~ (1/B_c) Σ_b Dir(α ψ_b)` | plate `"clonotypes"` → `MixtureDirichlet(weights, global_scale * mixture_concentration)`, sampled `"p_c"`. `ψ_b` = archetype centroids (`build_archetypes`). | ✅ α = `global_scale` (**[G]** fixed) |
| 2 | `ϕ_m \| ω_h(m) ~ Dir(β ω_h(m))` | plate `"ct_plate"` → `conc_ct = clamp(local_scale * p_c[ct_to_c])`, sampled `"p_ct"` | ✅ β = `local_scale` |
| 3 | `z_i ~ (1/B_z) Σ_k q(z\|u_k)` | `VampPrior.get_mixture()` (mixture of encoder-posteriors at learnable pseudo-inputs), sampled `"latent"` | ✅ |
| 4 | `l_i = f_cls(z_i)`, `ℓ_i = π l_i + (1-π) log ϕ_g(i)`; `z^ϕ_i ~ Cat(softmax(ℓ_i))` | `cls_logits = classifier(z)`; `ell = gate_prob*cls_logits + (1-gate_prob)*log_phi`; z^ϕ marginalised, surrogate `pheno_kl` in its place | ✅ latent (Inference Details) |
| 5 | `x_i ~ ZINB(g'_i, r_i, μ_i)` | `DecoderSCVI` → `ZeroInflatedNegativeBinomial(gate, total_count, logits)`, sampled `"obs"` | ✅ (scaled — **[E]**) |

## Variational family (eq 6) — `_module.py::guide`

- `q(ω_c) = Dir(λ_c)` — `q_p_c_raw` → `conc_c_guide = clamp(magnitude · sharpened direction)`. ✅ magnitude free (**[I]** fixed).
- `q(ϕ_m) = Dir(λ'_m)` — `q_p_ct_raw` → `conc_ct_guide = clamp(magnitude · sharpened direction)`. ✅ magnitude free (**[I]** fixed).

> **How [I] survived, kept as the lesson.** These two lines previously read
> `Dir(λ'_m)` on one side and `β · (a row summing to 1)` on the other, with an arrow between
> them — a correspondence written down and never tested. The conformance test traces sites,
> families, plates and event dims; the *structure* of a concentration is not a traced property,
> so nothing checked the claim. A map is not a proof.
- `q(z_i\|x_i) = N(μ_i, σ_i²)` — `encoder(x, batch)` → `Normal(z_loc, z_scale)`, sampled `"latent"`.
- `q(z^ϕ_i\|z_i, ϕ) = Cat(softmax(ℓ_i))` — represented by the surrogate, not an explicit categorical sample.

## ELBO (eq 7) and the surrogate ("Inference Details")

Eq 7 is the standard SVI ELBO (`Trace_ELBO`; `TraceEnum_ELBO` when `use_enumeration`),
`E[log p(x|z)] + E[log p(Ω,Φ,z,z^ϕ)] − E[log q]`, maximized by Adam.

The note replaces the discrete `z^ϕ` terms with a surrogate:

> `L_new = L#  +  γ Σ_i KL(probs_i ‖ ϕ_g(i))`,  `probs_i = softmax(ℓ_i)`

where `L#` is eq 7 with the `z^ϕ` terms removed and `γ>0`. The KL is a **penalty**
(the note "penalizes misalignment"), i.e. the objective is to *minimize* it. Pyro's SVI
**maximizes** the ELBO / log-joint, so the penalty enters the factor with a **minus
sign** — `−γ·KL`. (Reading the note's `+γ·ΣKL` as something to maximize would push
`probs` *away* from `ϕ`; the sign below is the one that realizes the note's intent.)
Implemented in `model()`'s `"data"` plate:

```python
phi      = p_ct[ct_idx].detach()          # the surrogate's target must stay CONSTANT,
ell      = gate_prob*cls_logits + (1-gate_prob)*log(phi)   # ℓ_i  (eq 4)
probs    = softmax(ell)                   # or -γ·KL(probs‖ϕ) is self-referential
pheno_kl = (probs * (log(probs) - log(phi))).sum(-1)
pyro.factor("phenotype_alignment", -phenotype_kl_weight * pheno_kl)
```

`z^ϕ` is **latent** and is never conditioned on the input phenotype labels. DE-18 briefly added
`pyro.sample("phenotype", Categorical(ell), obs=...)` on the argument that x ⊥ z^ϕ | z leaves ϕ_m
unidentifiable; that has been **withdrawn**. The hierarchical branch ω_c → ϕ_m → z^ϕ is a prior
over phenotype composition and is meant not to see `x` directly — that separation is why the
model is a VAE. Data reaches the hierarchy through `z` alone. `_model_contract.FORBIDDEN_MODEL_SITES`
now blocks the site's return.

- `ct_idx = ct_array[indices]` uses **global** cell indices (threaded in via
  `_get_fn_args_from_batch`), never the local pyro plate index — indexing with the
  local index scrambles the per-cell target across shuffled minibatches.
- Optimum of the surrogate is `f_cls → log ϕ + const` (distinct per clone), i.e. the
  classifier learns to predict the clonotype-informed phenotype from expression.
- `predict()` applies the same `ℓ_i` rule with `z_loc` (encoder mean, dropout off).

## In-silico perturbation (eqs 8–12) — **[F] not implemented**

`I_j = Σ_p |ϕ̄_p^(0) − ϕ̃_p^(j)|` (L1 shift after zeroing gene `j`). Additive feature;
no code path yet.

## Deviations

| id | deviation | severity | status |
|---|---|---|---|
| A | classifier had no ELBO gradient (missing factor) | HIGH | **fixed** — `pyro.factor("phenotype_alignment", …)` |
| A2 | surrogate target indexed by local plate idx → scrambled labels → f_cls collapse | HIGH | **fixed** — global `indices` threaded into `model()`/`guide()`; the `indices=None` path now `assert`s instead of silently falling back |
| B | `gate_prob` default was `None`; note sets π=0.5 | LOW | **fixed** — default `0.5` (typed `Optional[float]`) |
| C | `classifier_dropout` constructed but not passed to `PhenotypeClassifier` | LOW | **fixed** — plumbed |
| D | `class_weights`/`log_class_weights` — not in the note; was dead (computed + plumbed through 3 signatures, never read) | LOW | **fixed** — removed (with `phenotype_weights`) from `_model`/`_module`/`_training` |
| H | dead per-cell `encoder(x)` forward in `model()` (result discarded; the VampPrior carries its own encoder) | INFO | **fixed** — removed |
| G | α (`global_scale`) not applied to the clonotype prior (eq 1) in `model()`; concentration = normalized archetype centroid (sum≈1, U-shaped), so the prior was far more diffuse than `Dir(α·ψ_b)` and scaled inconsistently with the guide `q(ω_c)` | MED | **fixed** — `expanded_conc = global_scale * centroids` (eq 1); classifier recovery unchanged (1.000), suite green |
| E | `reconstruction_loss_scale` down-weights ZINB vs eq-7 full weight | MED | **resolved** — default raised 1e-3 → 1e-2; real-data library ratio 1.40 → 0.99 (recovery/latent unchanged). The original ~6× over-generation was mostly the phantom optimizer shrinking the decoder. |
| F | in-silico perturbation (eqs 8–12) not implemented | — | deferred — additive feature |

**Training-only deviations from eq 7 (intentional, documented here):**
- **KL warmup + z-only scope.** `UnifiedTrainingPlan` ramps `kl_weight` over `n_steps_kl_warmup`, and it scales only the `latent` (z) KL — the two Dirichlet KLs (`p_c`, `p_ct`) are unscaled. A standard annealing schedule; symmetric (no correctness bug) but not part of eq 7's full-weight KL.
- **`num_particles`** on `UnifiedTrainingPlan` is honored only on the enumeration path (`TraceEnum_ELBO`); the default `Trace_ELBO` uses 1 MC particle regardless.

### [I] — guide concentration is pinned where eq 6 specifies a free parameter (OPEN)

| id | deviation | severity | status |
|---|---|---|---|
| I | `q(ϕ_m)`/`q(ω_c)` concentrations were `scale · (normalized row)`, so their TOTAL was fixed at β/α regardless of how many cells the group has. Note 1 eq 6 + the notation table specify λ'_m, λ_c ∈ ℝ^P_{>0} — free variational parameters with learned magnitude. | MED–HIGH | **fixed** — magnitude freed on both guide sites; `guide_temperature` now sharpens the direction only. Landed alongside DE-18, which was subsequently **withdrawn**; DE-5 stands on its own — eq 6 specifies a free λ'_m irrespective of that question. |

**Consequence.** The posterior cannot concentrate with data: a clone-covariate group with 3
cells and one with 3,000 get the same posterior width. So every credible interval reported at
`n_samples>0` is prior-set rather than data-informed, and comparisons between groups of very
different size are the worst case. Per-entry concentration is `scale/P` (0.3 at the shipped
`local_scale=3`, P=10), which is below 1 and therefore corner-seeking — measured to inflate
`E[NMI(J)]` by +0.10 at weak coupling versus +0.017 at strong coupling.

**Why the contract did not catch it.** The manifest was written by reading the code
(2026-07-27) two weeks after the code was refactored into place, and it recorded what the code
does. A spec validated only against the implementation converges on the implementation. The
conformance test then verified agreement between manifest and code — both of which disagreed
with the note.

**Settled: a straight divergence from a spec that was always clear.** Checked against the
2026-04-30 note — eq 6 and the definition of Λ are word-for-word identical to the 2026-08-03
version, and the *same* expanded notation table is present in both. Both list `λ_c` and `λ'_m`
as "Dirichlet concentration for q(ω_c)/q(ϕ_m)" with domain **ℝ^P_{>0}**, i.e. vectors, and
members of Λ — the set being optimized. Separately they list `α, β` as scalar "Dirichlet
concentration scales (global, local)", `> 0`, appearing only in the conditioning of
`p(· | x; α, β, {ψ_b}, {u_k})`.

The code uses the scalar β as the vector's total. That conflates a variational parameter with
a prior hyperparameter — two rows of the same table, distinguished by name and by domain. No
reading of either note makes them the same object, so this is not an ambiguity that was
resolved the wrong way.

**Expect the benchmark to get worse, not better.** The pinned concentration inflates the
estimate, and that inflation is currently compensating for a separate downward error in the
gate fold. Fixing one side alone moves the net error the wrong way — this must not be judged
by whether NMI error dropped, but by whether the posterior concentrates with cells-per-group.

A/A2/B/C/D/H were fixed in the model PR that introduced this file. **G and E are now
resolved too** — α is applied to the eq-1 prior, and `reconstruction_loss_scale` was
re-measured and recalibrated to `1e-2` (real-data library ratio 1.40 → 0.99). Both
change fitted results, so runs are not comparable across them. **F** (in-silico
perturbation) remains out of scope for this release.

A further training-only deviation was found and removed: a **second torch Adam over all
module parameters**, installed by overriding scvi's deliberate no-op `configure_optimizers`
shim. It stepped after `SVI.step()` had zeroed the gradients, so weight decay degenerated
to a scale-free `~lr·sign(p)` shrink (networks held ~2.4× small), and `train(lr=)` never
reached Pyro's optimizer. See `optimizer_weight_decay` in the model contract.
