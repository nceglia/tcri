"""TCRI **model contract** (frozen) — the machine-checkable probabilistic structure.

The API contract (``tcri/_contract.pyi``) freezes the public *interface*; this file
freezes the *model* — the generative program and variational family of
**Supplementary Note 1** (``docs/contract/MODEL_CONTRACT.md``, prose; the PDF is the
source of truth). The conformance test
(``tests/test_model_contract_conformance.py``) traces ``TCRIModule.model``/``.guide``
and asserts the live program matches this manifest **exactly** — no missing sites and
**no extra sites**.

RULES this file encodes:
- Every stochastic site in ``model()``/``guide()`` is declared here, with the
  equation of the note it realizes. A site NOT declared here must NOT exist.
- Changing the model's mathematics **requires updating this contract first**
  (and ``docs/contract/MODEL_CONTRACT.md``). The conformance test is the forcing
  function: it fails until the contract and the code agree. Do not "fix" a failure
  by loosening the manifest to match new code — update the contract deliberately,
  with the note reference, so the change is reviewed as a model change.
- Sanctioned departures from the note live in ``SANCTIONED_DEVIATIONS`` with a
  rationale. An unlisted departure is a defect, not a feature.

Symbols (note -> code): ω_c->``p_c``, ϕ_m->``p_ct``, z_i->``latent``, x_i->``obs``,
f_cls->``classifier``, π->``gate_prob``, α->``global_scale``, β->``local_scale``,
γ->``phenotype_kl_weight``, g(i)->``ct_array``, h(m)->``ct_to_c``.
"""
from __future__ import annotations

from typing import Optional

__all__ = [
    "GENERATIVE_SITES",
    "GUIDE_SITES",
    "GUIDE_PARAMS",
    "FORBIDDEN_GUIDE_SITES",
    "PLATES",
    "SEMANTIC_INVARIANTS",
    "SANCTIONED_DEVIATIONS",
    "SiteSpec",
]


class SiteSpec:
    """One declared stochastic site: what distribution, in which plate, per the note.

    ``dist`` is the *unwrapped* distribution class name — the checker strips Pyro's
    ``Independent``/``ExpandedDistribution`` wrappers (``.to_event()``/``.expand()``
    are implementation details, not model semantics). ``event_dim`` pins how many
    trailing dims are dependent, which IS semantics.
    """

    def __init__(
        self,
        name: str,
        dist: str,
        plate: Optional[str],
        eq: str,
        observed: bool = False,
        event_dim: Optional[int] = None,
        note: str = "",
    ):
        self.name = name
        self.dist = dist
        self.plate = plate
        self.eq = eq
        self.observed = observed
        self.event_dim = event_dim
        self.note = note

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SiteSpec({self.name!r}, dist={self.dist!r}, plate={self.plate!r}, eq={self.eq!r})"


# ── plates ───────────────────────────────────────────────────────────────────
# name -> what one index of the plate is (the note's index set).
PLATES = {
    "clonotypes": "c = 1..C  (clonotype)",
    "ct_plate": "m = 1..M  (clonotype x covariate group)",
    "data": "i = 1..N  (cell; subsampled minibatch)",
}


# ── the generative program: p(Ω, Φ, z, x) ────────────────────────────────────
GENERATIVE_SITES = [
    SiteSpec(
        "p_c", "MixtureDirichlet", "clonotypes", eq="1", event_dim=1,
        note="ω_c ~ (1/B_c) Σ_b Dir(α·ψ_b); ψ_b = archetype vectors, α = global_scale.",
    ),
    SiteSpec(
        "p_ct", "Dirichlet", "ct_plate", eq="2", event_dim=1,
        note="ϕ_m | ω_h(m) ~ Dir(β·ω_h(m)); β = local_scale, h(m) = ct_to_c.",
    ),
    SiteSpec(
        "latent", "MixtureSameFamily", "data", eq="3", event_dim=1,
        note="z_i ~ (1/B_z) Σ_k q(z|u_k) — VampPrior over learnable pseudo-inputs.",
    ),
    SiteSpec(
        "phenotype_alignment", "Unit", "data", eq="Inference Details", observed=True,
        note=(
            "The surrogate replacing the discrete z^ϕ (eq 4): a pyro.factor carrying "
            "−γ·KL(probs_i ‖ ϕ_g(i)), probs_i = softmax(ℓ_i), "
            "ℓ_i = π·f_cls(z_i) + (1−π)·log ϕ_g(i). This term is what trains f_cls."
        ),
    ),
    SiteSpec(
        "obs", "ZeroInflatedNegativeBinomial", "data", eq="5", observed=True, event_dim=1,
        note="x_i ~ ZINB(g'_i, r_i, μ_i) from the scVI decoder (+ library size).",
    ),
]


# ── the variational family q(Ω, Φ, z | x)  (eq 6) ────────────────────────────
GUIDE_SITES = [
    SiteSpec("p_c", "Dirichlet", "clonotypes", eq="6", event_dim=1,
             note="q(ω_c) = Dir(λ_c); concentration = α · normalized q_p_c_raw."),
    SiteSpec("p_ct", "Dirichlet", "ct_plate", eq="6", event_dim=1,
             note="q(ϕ_m) = Dir(λ'_m); concentration = β · normalized q_p_ct_raw."),
    SiteSpec("latent", "Normal", "data", eq="6", event_dim=1,
             note="q(z_i|x_i) = N(μ_i, diag(σ_i²)) from the encoder."),
]

# Learnable variational parameters the guide must register (λ_c, λ'_m).
GUIDE_PARAMS = ["q_p_c_raw", "q_p_ct_raw"]

# The discrete phenotype latent z^ϕ is NOT sampled: the note's "Inference Details"
# replaces it with the surrogate KL penalty. A q(z^ϕ) categorical site reappearing
# in the guide means someone re-introduced the enumerated path without updating the
# contract — which changes the objective.
FORBIDDEN_GUIDE_SITES = ["z_phi", "z_phenotype", "phenotype", "phenotype_alignment"]


# ── semantic invariants (behavior the structure alone can't pin) ─────────────
# Each is asserted by the conformance test; the string is the failure explanation.
SEMANTIC_INVARIANTS = {
    "alpha_scales_clonotype_prior": (
        "eq 1: the p_c prior concentration must scale with α (global_scale). "
        "Without it the prior is Dir(ψ_b) (sums to ~1, U-shaped: mass at the simplex "
        "corners) and is scaled inconsistently with the guide, which does apply α."
    ),
    "beta_scales_covariate_prior": (
        "eq 2: the p_ct prior concentration must scale with β (local_scale), i.e. "
        "conc = β·ω_h(m), so the total concentration is ~β."
    ),
    "factor_is_negative_kl": (
        "Inference Details: the phenotype_alignment factor must be −γ·KL(probs‖ϕ) "
        "(≤ 0). SVI MAXIMIZES the log-joint and the note's γ·KL is a PENALTY, so the "
        "sign must be negative; a positive factor would push probs AWAY from ϕ."
    ),
    "gate_mixes_classifier_and_prior": (
        "eq 4: ℓ_i = π·f_cls(z_i) + (1−π)·log ϕ_g(i). π=1 ⇒ pure classifier; "
        "π=0 ⇒ pure clonotype prior; π=None ⇒ the additive rule f_cls + log ϕ."
    ),
    "alignment_target_uses_global_indices": (
        "The per-cell target ϕ_g(i) must be indexed by GLOBAL cell indices, never the "
        "local pyro plate index (0..batch_size−1), which scrambles targets across "
        "shuffled minibatches and collapses f_cls to a constant."
    ),
}


# ── departures from the note that are accepted, with rationale ───────────────
# Anything NOT listed here that departs from the note is a defect.
SANCTIONED_DEVIATIONS = {
    "E_reconstruction_loss_scale": (
        "eq 7 weights E[log p(x|z)] at 1; the code scales the `obs` site by "
        "`reconstruction_loss_scale` (default 1e-3), a β-VAE-style reweighting. "
        "Known deviation: under-weights the decoder (over-generates counts). "
        "Deferred pending a retrain + R/NR revalidation."
    ),
    "kl_warmup_z_only": (
        "Training-only: `kl_weight` anneals the `latent` (z) KL over "
        "`n_steps_kl_warmup`; the two Dirichlet KLs are unscaled. Standard annealing, "
        "not part of eq 7."
    ),
    "num_particles_enumeration_only": (
        "`num_particles` is honored only on the TraceEnum_ELBO path; the default "
        "Trace_ELBO uses a single MC particle."
    ),
    "F_perturbation_not_implemented": (
        "In-silico perturbation (eqs 8-12) is not implemented. Additive feature; "
        "explicitly out of scope for this release."
    ),
}
