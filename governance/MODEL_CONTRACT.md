# Model contract

What the package fits. This file is the definition. `tests/test_model_contract_conformance.py`
traces the live `TCRIModule.model()` and `.guide()` and holds them to the block at the end of
this file; nothing else binds the model. Equation numbers are this document's.

**Provenance.** The model descends from Supplementary Note 1 of the TCRi manuscript, whose
numbering equations 1 to 7 keep, and extends it in four places marked ▸: the head reads the
posterior mean; one σ is shared by the latent posterior and its prior; the alignment
surrogate's target is detached; and a noisy-label readout, equation 8, gives the latent
phenotype an observation. The note is not kept in the repository. Where the two differ, this
file is what the code does.

## Symbols

| symbol | code | symbol | code |
|---|---|---|---|
| ω_c | `p_c` | f_cls | `classifier` |
| φ_m | `p_ct` | π | `gate_prob` |
| z_i | `latent` | α, β | `global_scale`, `local_scale` |
| z^φ_i | never sampled (summed out) | γ | `phenotype_kl_weight` |
| x_i | `obs` | ε | `label_error_rate` |
| y_i | `phenotype_label` | g(i), h(m) | `ct_array`, `ct_to_c` |

## The generative model

```
p(Ω, Φ, z, y, x) = Π_c p(ω_c) · Π_m p(φ_m | ω_h(m)) · Π_i p(z_i) · p(y_i | z_i, φ_g(i)) · p(x_i | z_i)
```

| eq | site | statement |
|---|---|---|
| 1 | `p_c` | `ω_c ~ (1/B) Σ_b Dir(α ψ_b)`. Clonotype level. The K archetypes ψ_b are k-means centroids of the per-clone label composition; α scales the concentration. |
| 2 | `p_ct` | `φ_m \| ω ~ Dir(β ω_h(m))`. Clonotype × covariate group, hierarchical under its clonotype; β scales it. |
| 3 | `latent` | `z_i ~ (1/B_z) Σ_k q(z \| u_k)`. VampPrior: the encoder posterior evaluated at learnable pseudo-inputs u_k, through the same (μ, σ) map as eq 6. ▸ |
| 4 | — | `ℓ_i = π f_cls(μ_i) + (1 − π) log φ_g(i)`; `z^φ_i ~ Cat(softmax ℓ_i)`. The cell's latent phenotype. f_cls reads the posterior mean μ_i, not a sample. ▸ |
| 5 | `obs` | `x_i ~ ZINB(decoder(z_i, library_i, batch_i))`. |
| 8 | `phenotype_label` | `y_i \| z^φ_i ~ Cat(C[z^φ_i, ·])`, C = (1 − ε) on the diagonal and ε/(P − 1) off it. The input label is a noisy readout of the latent phenotype. z^φ_i is summed out in closed form, `p(y_i) = (softmax ℓ_i · C)[y_i]`, with φ live inside ℓ_i. `label_error_rate=None` removes the site. ▸ |

Scales are semantics, not tuning: with concentration entries below 1 a Dirichlet is U-shaped,
so α and β decide the prior's shape.

## The variational family (eq 6)

```
q(Ω, Φ, z | x) = Π_c Dir(ω_c | λ_c) · Π_m Dir(φ_m | λ'_m) · Π_i N(z_i | μ_i, diag σ_i²)
```

λ_c and λ'_m are free positive vectors (`q_p_c_raw`, `q_p_ct_raw`): magnitude and direction
are both learned, and `guide_temperature` sharpens the direction only. (μ_i, σ_i) come from
`encoder_posterior`: σ is the square root of the encoder's variance output, clamped to
[1e-3, 10], and the VampPrior components of eq 3 are built by the same function. ▸ z^φ has no
variational factor; it is summed out in eq 8 and stands behind the surrogate below.

The two variational parameters are registered under the module's `name`: `q_p_ct_raw` for an
unnamed module, `x.q_p_ct_raw` for one called `x`. The family is the same either way; the name
exists because Pyro's store is process-global and a second model would otherwise overwrite the
first. `GUIDE_PARAMS` names the parameters, not the keys, so the conformance test compares the
tail of each traced param site.

## The objective (eq 7)

The ELBO of the model above plus the alignment surrogate, maximised by SVI with Adam:

```
L = ρ·E_q[log p(x | z)] + E_q[log p(y | z, φ)] − KL(q(z) ‖ p(z)) − KL(q(Φ) ‖ p(Φ | Ω)) − KL(q(Ω) ‖ p(Ω))
    − γ Σ_i KL(probs_i ‖ φ_g(i)),   probs_i = softmax ℓ_i
```

- **The surrogate's target is detached.** The penalty trains f_cls toward its group's
  distribution and moves nothing else. With the target live, the head and the hierarchy
  converge to one shared vector at a lower objective; eq 8 is the hierarchy's data term
  instead. ▸
- ρ = `reconstruction_loss_scale`, default 1e-2, re-weights the ZINB term.
- The data plate is declared at the size of the training split with the minibatch as its
  subsample, so per-cell terms carry scale N/B and the two Dirichlet KLs enter once: a
  minibatch is an unbiased estimate of L.
- `kl_weight` anneals the latent KL only. The Dirichlet KLs are never annealed.
- The surrogate and the readout index cells by their global id (`ct_array[indices]`), never by
  position in the batch.

## Prediction

`predict()` returns softmax ℓ_i computed with μ_i: the gate applied to the head and the group
distribution. π = 1 is the pure head, π = 0 the pure group prior, π = None the additive rule.
This per-cell rule is the object `tcri.perturb` intervenes on (`METRICS_CONTRACT.md`, "The
perturbation"): the expression matrix changes, the parameters and the rule do not.

## The machine-checked part

The conformance test parses this block. Sites, families, plates, observed flags and event
dims must match the live trace exactly, and no undeclared stochastic site may exist.

```python contract
PLATES = {
    "clonotypes": "c = 1..C (clonotype)",
    "ct_plate": "m = 1..M (clonotype x covariate group)",
    "data": "i = 1..N (cell); size = the training split, minibatch subsampled, per-cell sites scaled by N/B",
}

# name, distribution class after unwrapping Independent/Expanded, plate, observed, event_dim
GENERATIVE_SITES = [
    {"name": "p_c", "dist": "MixtureDirichlet", "plate": "clonotypes", "observed": False, "event_dim": 1, "eq": "1"},
    {"name": "p_ct", "dist": "Dirichlet", "plate": "ct_plate", "observed": False, "event_dim": 1, "eq": "2"},
    {"name": "latent", "dist": "MixtureSameFamily", "plate": "data", "observed": False, "event_dim": 1, "eq": "3"},
    {"name": "phenotype_alignment", "dist": "Unit", "plate": "data", "observed": True, "event_dim": None, "eq": "surrogate"},
    {"name": "phenotype_label", "dist": "Categorical", "plate": "data", "observed": True, "event_dim": 0, "eq": "8"},
    {"name": "obs", "dist": "ZeroInflatedNegativeBinomial", "plate": "data", "observed": True, "event_dim": 1, "eq": "5"},
]

GUIDE_SITES = [
    {"name": "p_c", "dist": "Dirichlet"},
    {"name": "p_ct", "dist": "Dirichlet"},
    {"name": "latent", "dist": "Normal"},
]

GUIDE_PARAMS = ["q_p_c_raw", "q_p_ct_raw"]   # registered under the module's namespace; see the prose

# a q(z^phi) site, or a direct observation of z^phi, changes the objective
FORBIDDEN_GUIDE_SITES = ["z_phi", "z_phenotype", "phenotype", "phenotype_alignment"]
FORBIDDEN_MODEL_SITES = ["phenotype", "z_phi", "z_phenotype"]
```

## Also enforced from the trace

Each of these is a test in `tests/test_model_contract_conformance.py` or
`tests/test_label_readout.py`, and each test carries its own explanation.

- α scales eq 1; eq 2's concentration is β·ω_h(m) elementwise from the ω sampled in the same
  trace.
- The surrogate factor is a negative, non-zero KL and carries no gradient to λ'.
- The surrogate and the readout use global cell indices.
- The head reads the posterior mean.
- One σ: the guide's latent scale and every VampPrior component scale are sqrt(variance) under
  the same clamp.
- The readout's probability is (softmax ℓ · C)[y]; it carries gradient to λ', the head and the
  encoder; with `label_error_rate=None` the site is absent and the hierarchy has no per-cell
  gradient.
- The data plate scales per-cell sites by N/B and the Dirichlet sites by 1.
- π = 1 and π = 0 reduce `predict()` to the head and to the group prior.

## A null is this model

A permutation reference is not a second model and not a different generative story. `permutation`
reorders ONE of the three label vectors -- phenotype, clonotype, covariate -- at the point
`TCRIModel.__init__` reads it from `obs`, before anything is derived from it. Everything
downstream is then built as a fresh fit on permuted data would build it: the clone x phenotype
prior, the archetypes, `ct_array`, the target labels. Nothing in the generative model, the guide,
the site names, the plates or the objective changes, which is why the contract block above is the
same block for a null and for its parent, and why the same conformance test covers both.
