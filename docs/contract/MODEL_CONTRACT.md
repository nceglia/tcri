# TCRI Model Contract (FROZEN)

**The model this package implements is Supplementary Note 1** (`docs/contract/source/supplementary_note_1_SS_2026-08-03.pdf`).
This document is the prose contract; `tcri/model/_model_contract.py` is its
machine-checkable form; `tests/test_model_contract_conformance.py` enforces it.

Sibling of the API contract: `tcri/_contract.pyi` freezes the public *interface*,
this freezes the *mathematics*. `docs/contract/METHODS_CONFORMANCE.md` is the
eq-by-eq code map and deviation history.

---

## THE RULE

> **Changing the model's mathematics requires updating this contract *first*.**

Concretely — adding/removing a stochastic site, changing a distribution family or
plate, altering the ELBO or the surrogate, or changing what a prior is scaled by:

1. **Update the contract first** — this file *and* `_model_contract.py`, citing the
   note equation and stating what changes in the joint distribution.
2. **Then change the code** so `test_model_contract_conformance` passes again.
3. **If the note itself is superseded**, say so explicitly here (with the new
   reference). The note is the source of truth; the contract tracks it.

**Never** make a conformance failure disappear by loosening the manifest to match
whatever the code now does. That silently rewrites the model the package claims to
implement, which is exactly what this guardrail exists to prevent. A failure means
*stop and decide*: is this an intended model change (update the contract) or a
regression (fix the code)?

This applies to human and AI contributors alike. If you are an agent and a model
change seems necessary, surface the contract implication to the user rather than
editing the manifest to fit.

---

## The generative model

`p(Ω, Φ, z, x) = Π_c p(ω_c) · Π_m p(ϕ_m|ω_h(m)) · Π_i p(z_i)·p(z^ϕ_i|z_i,ϕ_g(i))·p(x_i|z_i)`

| eq | site | distribution | plate | meaning |
|---|---|---|---|---|
| 1 | `p_c` | `MixtureDirichlet` | `clonotypes` (c=1..C) | `ω_c ~ (1/B_c) Σ_b Dir(α·ψ_b)` — clonotype-level phenotype distribution over archetypes `ψ_b` |
| 2 | `p_ct` | `Dirichlet` | `ct_plate` (m=1..M) | `ϕ_m \| ω_h(m) ~ Dir(β·ω_h(m))` — covariate-level, hierarchical under its clonotype |
| 3 | `latent` | `MixtureSameFamily` (VampPrior) | `data` (i=1..N) | `z_i ~ (1/B_z) Σ_k q(z\|u_k)` over learnable pseudo-inputs |
| 4 | *(surrogate)* | — | `data` | `ℓ_i = π·f_cls(z_i) + (1−π)·log ϕ_g(i)`; `z^ϕ_i ~ Cat(softmax(ℓ_i))` |
| 5 | `obs` | `ZeroInflatedNegativeBinomial` | `data` | `x_i ~ ZINB(g'_i, r_i, μ_i)` from the scVI decoder |

**Scales are semantics, not tuning.** α (`global_scale`) scales eq 1's concentration
and β (`local_scale`) scales eq 2's. Dropping either changes the prior's *shape* —
with concentration entries < 1 a Dirichlet becomes U-shaped (mass at the simplex
corners), the opposite of a prior peaked at the archetype — and desynchronizes the
prior from the guide. Both are asserted by the conformance test.

## The variational family (eq 6)

`q(Ω,Φ,z|x) = Π_c Dir(ω_c|λ_c) · Π_m Dir(ϕ_m|λ'_m) · Π_i q(z_i|x_i;η_enc) · Π_i q(z^ϕ_i|z_i,ϕ;η_cls)`

| site | distribution | learnable |
|---|---|---|
| `p_c` | `Dirichlet` | `q_p_c_raw` (λ_c), scaled by α |
| `p_ct` | `Dirichlet` | `q_p_ct_raw` (λ'_m), scaled by β |
| `latent` | `Normal(μ_i, diag(σ_i²))` | encoder `η_enc` |

`z^ϕ` is **not** sampled — see the surrogate below. A categorical `q(z^ϕ)` site
reappearing in the guide changes the objective and is rejected by the test.

## The objective

The ELBO (eq 7) is `E_q[log p(x|z)] + E_q[log p(Ω,Φ,z,z^ϕ)] − E_q[log q]`, maximized
by SVI (Adam, reparameterized continuous latents).

**The surrogate** ("Inference Details") replaces the discrete `z^ϕ` terms:

> `L_new = L# + γ·Σ_i KL(probs_i ‖ ϕ_g(i))`, `probs_i = softmax(ℓ_i)`

The KL is a **penalty** on misalignment (the note "penalizes misalignment"), so it is
*minimized*. Pyro's SVI **maximizes** the log-joint, therefore the factor is registered
with a **minus sign**:

```python
pyro.factor("phenotype_alignment", -phenotype_kl_weight * kl)   # −γ·KL ≤ 0
```

A positive factor would push `probs` *away* from `ϕ`. The test asserts the factor's
log-value is ≤ 0 and non-zero.

**This term is the only thing that trains `f_cls`.** Without it the classifier's
logits never enter the ELBO and it receives no gradient (recovery sits at chance).

**The alignment target must use global cell indices.** `ϕ_g(i) = p_ct[ct_array[indices]]`
where `indices` are *global* cell ids threaded in via `_get_fn_args_from_batch`. The
pyro data-plate index is local (`0..batch_size−1`); using it scrambles each cell's
target across shuffled minibatches and collapses `f_cls` to a constant.

## Gating (π)

`gate_prob` = π ∈ (0,1), default **0.5** per the note. Endpoints are contract-tested:
π=1 ⇒ `predict()` is the pure classifier; π=0 ⇒ the pure clonotype prior;
π=`None` ⇒ the additive rule `f_cls + log ϕ`.

---

## Sanctioned deviations

Accepted departures from the note. **Anything not listed here that departs from the
note is a defect.** Keys match `SANCTIONED_DEVIATIONS` in `_model_contract.py` (the
test asserts they stay in sync).

| key | departure | rationale |
|---|---|---|
| `E_reconstruction_loss_scale` | eq 7 weights `E[log p(x\|z)]` at 1; the `obs` site is scaled by `reconstruction_loss_scale` (default **`1e-2`**) | β-VAE-style reweighting, **re-measured and recalibrated** — see below. |

### On `reconstruction_loss_scale` (deviation [E], resolved)

Re-measured after the phantom optimizer was removed. Posterior-predictive library
ratio (simulated ÷ observed; 1.00 is calibrated):

| scale | real yost (2259×1000) | synthetic (3000×60) |
|---|---|---|
| `1e-3` (old default) | **1.40** | 1.00 |
| **`1e-2` (new default)** | **0.99** | 1.00 |
| `1e-1` | 1.00 | 1.00 |
| `1.0` (eq-7 full weight) | — | **0.91** (over-corrects) |

Dropout fraction matches observed at every setting (0.870 vs ~0.873 on real data).
Classifier recovery (1.000) and latent separation (7.19) are **unchanged** across
`1e-3`→`1e-1`, so the recalibration costs nothing.

The originally-reported **~6× over-generation was mostly the phantom second
optimizer** shrinking the decoder (see `optimizer_weight_decay`); removing it took the
ratio 6× → 1.40, and this default closes the remainder. Note the synthetic data could
not detect this — only the real 1000-gene, 87%-dropout data discriminates.

Three inconsistent defaults (`_model.train`=1e-3, `_module`=1e-3, `_training`=1e-2)
were unified to `1e-2`.
| `kl_warmup_z_only` | `kl_weight` anneals only the `latent` KL; the Dirichlet KLs are unscaled | Standard annealing; training-only, not part of eq 7. |
| `num_particles_enumeration_only` | `num_particles` applies only on the `TraceEnum_ELBO` path | Default `Trace_ELBO` uses 1 MC particle. |
| `F_perturbation_not_implemented` | in-silico perturbation (eqs 8–12) absent | Additive feature; explicitly out of scope for this release. |
| `optimizer_weight_decay` | the SVI optimizer applies Adam weight decay (default `1e-4`) to the network parameters | The note fixes the *objective* (eq 7 + the surrogate), not the optimizer. Applied inside Pyro's optimizer so it acts on the ELBO gradients. See the note below. |

### On the optimizer (history worth keeping)

Weight decay was previously applied by a **second** `torch.optim.Adam` over every
module parameter, installed by overriding `UnifiedTrainingPlan.configure_optimizers`.
That override replaced scvi's *deliberate no-op shim* — scvi returns
`Adam([self._dummy_param])` purely to advance Lightning's step counter — and it ran
**after** `SVI.step()` had already stepped and **zeroed the gradients**.

Stepping Adam on zero gradients is not a no-op. The weight-decay term becomes the
entire gradient (`g = wd·p`), and Adam's normalization `g/√(g²)` then strips its
magnitude, so the update degenerates to **≈ `lr·sign(p)`** — a *scale-free* shrink of
roughly `lr` per step rather than proportional L2. Measured in isolation: a weight of
0.1 is driven to 5e-5 within 1000 steps, where true L2 (`(1−lr·wd)^n`) would leave it
at 0.9998. In the fitted model SVI pushed back, but the equilibrium sat at **~2.4×
smaller network weights** (encoder 0.107 vs 0.240).

It also meant `train(lr=...)` never reached the optimizer that fits the model — Pyro
always used scvi's hard-coded `1e-3`, and `lr` only set the shrink rate.

The override is removed; `lr`/`weight_decay`/`betas`/`eps` now go to Pyro's optimizer
via `optim_kwargs`, which is where the original intent belongs. `lr` is consequently a
**live** knob for the first time (verified: it moves recovery, weight scale, and the
final ELBO), so fits are not comparable across this change.

## What the conformance test checks

`tests/test_model_contract_conformance.py` traces the live `model()`/`guide()`:

- every declared site exists with the right distribution family, plate, event-dim,
  observed-flag — **and no undeclared site exists** (an extra site changes the joint);
- the guide's variational family + learnable params (λ_c, λ'_m), and that `z^ϕ` is not sampled;
- α scales eq 1's concentration;
- **eq 2 is hierarchical**: `p_ct`'s concentration is asserted *elementwise* to equal
  `clamp(β·(ω_c[ct_to_c] + eps))` against the sampled `p_c` **from the same trace** —
  pinning the scale, the source tensor, and the index map h(m) together;
- the surrogate factor is a *negative*, non-zero KL;
- **the alignment target is verified behaviorally**: on a minibatch whose global
  indices differ from the local plate positions, the traced factor must equal the
  surrogate recomputed under the *global* map and must differ from the local one;
- π endpoints reduce `predict()` to classifier / prior;
- every sanctioned deviation is documented here.

**Assertions are behavioral, not textual.** Two earlier drafts of this test were
defeated in an adversarial audit: a scalar "concentration totals ≈ β" check passed
even with the clonotype→covariate hierarchy severed (every simplex row totals 1, so
any tensor under any permutation satisfies it), and a source-grep for
`ct_array[indices]` was defeated by routing the same wrong lookup through
`index_select`. Prefer assertions computed from a live trace over ones that read
source text or scalar summaries.
