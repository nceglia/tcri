"""The TCRI Pyro module: a CVAE (encoder/decoder over gene expression) coupled to
two-level hierarchical Dirichlet priors (clonotype -> clonotype x covariate) and a
phenotype classifier head."""
from typing import Dict, Optional

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from scvi import REGISTRY_KEYS
from scvi.nn import Encoder, DecoderSCVI
from scvi.module.base import PyroBaseModuleClass, auto_move_data

from ._classifier import PhenotypeClassifier
from ._priors import VampPrior, MixtureDirichlet

__all__ = ["TCRIModule"]


class TCRIModule(PyroBaseModuleClass):
    """
    Two-level model that incorporates hierarchical priors (clonotype-level)
    and a CVAE structure that explicitly conditions gene expression on the
    observed cell-level phenotype.
    """

    def __init__(  
        self,
        n_input: int,
        n_latent: int,
        P: int,
        n_batch: int,
        global_scale: float = 10.0,
        local_scale: float = 5.0,
        prior_temperature: float = 1.0,
        guide_temperature: float = 1.0,
        gate_prob: Optional[float] = 0.5,  # None = additive (no gating)
        mixture_concentration: torch.Tensor = None,
        n_pseudo_obs: int = 10,
        use_enumeration: bool = False,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.1,
        classifier_n_layers: int = 3,
        n_hidden: int = 128,
        n_layers: int = 3,
        kl_weight_max: float = 1.0,
        guide_init_scale: float = 10.0,
        classifier_temperature: float = 1.0,
        phenotype_kl_weight: float = 1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.P = P
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.prior_temperature = prior_temperature
        self.guide_temperature = guide_temperature
        # register_buffer (not a plain attribute) so module.to(device) moves it with
        # the rest of the module — as a bare Tensor it stays on CPU and the eq-1 prior
        # then needs an ad-hoc .to() at every use, or fails outright on GPU.
        if mixture_concentration is not None and not torch.is_tensor(mixture_concentration):
            mixture_concentration = torch.as_tensor(mixture_concentration)
        self.register_buffer("mixture_concentration", mixture_concentration)
        self.n_pseudo_obs = n_pseudo_obs
        self.gate_prob = gate_prob
        # Assert that it is not None
        assert (
            self.mixture_concentration is not None
        ), "mixture_concentration must be provided"
        self.use_enumeration = use_enumeration
        self.eps = 1e-6
        self.classifier_hidden = classifier_hidden
        self.classifier_dropout = classifier_dropout
        self.kl_weight_max = kl_weight_max
        self.classifier_n_layers = classifier_n_layers
        self.guide_init_scale = guide_init_scale
        self.classifier_temperature = classifier_temperature
        self.phenotype_kl_weight = phenotype_kl_weight  # γ (methods §Inference Details)

        # Defaults so model()/guide() work before train() sets them
        self.kl_weight = 1e-6
        self.reconstruction_loss_scale = 1e-3

        self.encoder = Encoder(
            n_input=n_input,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            use_layer_norm=True,
        )

        # VampPrior
        pseudo_inputs = torch.randn(self.n_pseudo_obs, self.n_input)
        self.vamp_prior = VampPrior(pseudo_inputs, self.encoder)

        self.decoder_input_dim = self.n_latent
        self.decoder = DecoderSCVI(
            self.decoder_input_dim,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            scale_activation="softplus",
            use_layer_norm=True,
        )

        self.px_r = torch.nn.Parameter(torch.ones(n_input))

        self.classifier = PhenotypeClassifier(
            n_latent=self.n_latent,
            classifier_hidden=self.classifier_hidden,
            P=self.P,
            num_layers=self.classifier_n_layers,
            dropout_rate=self.classifier_dropout,
            temperature=self.classifier_temperature,
        )

        self.register_buffer("clone_phen_prior", torch.empty(0))
        self.register_buffer("ct_to_c", torch.empty(0, dtype=torch.long))
        self.register_buffer("c_array", torch.empty(0, dtype=torch.long))
        self.register_buffer("ct_array", torch.empty(0, dtype=torch.long))
        self.register_buffer("ct_to_cov", torch.empty(0, dtype=torch.long))
        self.c_count = 0
        self.ct_count = 0
        self.n_cells = 0

        self.register_buffer("_target_phenotypes", torch.empty(0, dtype=torch.long))

    def prepare_two_level_params(
        self,
        c_count: int,
        ct_count: int,
        clone_phen_prior_mat: torch.Tensor,
        ct_to_c_array: torch.Tensor,
        c_array_for_cells: torch.Tensor,
        ct_array_for_cells: torch.Tensor,
        target_phenotypes: torch.Tensor,
        ct_to_cov_array: torch.Tensor = None,
    ):
        self.c_count = c_count
        self.ct_count = ct_count
        self.n_cells = c_array_for_cells.shape[0]

        prior_mat = clone_phen_prior_mat + self.eps
        prior_mat = prior_mat / prior_mat.sum(dim=1, keepdim=True)

        if self.prior_temperature != 1.0:
            prior_mat = prior_mat ** (1.0 / self.prior_temperature)
            prior_mat = prior_mat / prior_mat.sum(dim=1, keepdim=True)

        self.register_buffer("clone_phen_prior", prior_mat)
        self.register_buffer("ct_to_c", ct_to_c_array)
        self.register_buffer("c_array", c_array_for_cells)
        self.register_buffer("ct_array", ct_array_for_cells)
        self.register_buffer("_target_phenotypes", target_phenotypes)

        if ct_to_cov_array is not None:
            self.register_buffer("ct_to_cov", ct_to_cov_array)

    @property
    def use_gate(self) -> bool:
        return self.gate_prob is not None

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: Dict[str, torch.Tensor]):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        batch_idx = tensor_dict[REGISTRY_KEYS.BATCH_KEY].long()
        log_library = torch.log(torch.sum(x, dim=1, keepdim=True) + 1e-6)
        # Global cell indices for this minibatch. Needed so model()/guide() can map
        # each cell to its (clonotype x covariate) group via ``ct_array``; the pyro
        # data-plate index is LOCAL (0..batch_size-1) and must NOT be used for this.
        indices = tensor_dict["indices"].long().view(-1)
        return (x, batch_idx, log_library, indices), {}

    @auto_move_data
    def model(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        log_library: torch.Tensor,
        indices: torch.Tensor = None,
    ):
        pyro.module("scvi", self)

        kl_weight = self.kl_weight
        batch_size = x.shape[0]

        with pyro.plate("clonotypes", self.c_count):
            B = self.mixture_concentration.shape[0]
            mixture_weights = torch.ones(B, device=x.device) / B
            # Expand mixture parameters to add a leading dimension for clonotypes.
            # expanded_conc will have shape (self.c_count, B, K).
            # eq 1: ω_c ~ (1/B) Σ_b Dir(α·ψ_b) — scale the archetype vectors ψ_b by
            # α (global_scale), mirroring eq 2's β on the covariate prior. Without α
            # the concentration sums to ~1 (U-shaped, mass at the simplex corners) and
            # is scaled inconsistently with the guide q(ω_c), which does apply α.
            expanded_conc = self.global_scale * self.mixture_concentration.unsqueeze(
                0
            ).expand(self.c_count, -1, -1)
            # expanded_weights will have shape (self.c_count, B)
            expanded_weights = mixture_weights.unsqueeze(0).expand(self.c_count, -1)
            mixture_dist = MixtureDirichlet(expanded_weights, expanded_conc)
            p_c = pyro.sample("p_c", mixture_dist)
            # print("p_c shape:", p_c.shape)

        with pyro.plate("ct_plate", self.ct_count):
            base_p = p_c[self.ct_to_c] + self.eps
            conc_ct = torch.clamp(self.local_scale * base_p, min=1e-3)
            p_ct = pyro.sample("p_ct", dist.Dirichlet(conc_ct))

        with pyro.plate("data", batch_size) as idx:

            with poutine.scale(scale=kl_weight):
                # vamp_mixture = self.vamp_prior.get_mixture().to_event(1)
                vamp_mixture = self.vamp_prior.get_mixture()
                z = pyro.sample("latent", vamp_mixture)

            # Map each cell to its (clonotype x covariate) group using GLOBAL indices,
            # not the local plate index `idx` (which is 0..batch_size-1 and would
            # scramble the alignment target across shuffled minibatches). Fail loudly
            # rather than silently falling back to the (wrong) local index.
            assert indices is not None, (
                "model() requires global cell indices (supplied by "
                "_get_fn_args_from_batch); the local plate index must not be used "
                "for the clonotype x covariate lookup."
            )
            ct_idx = self.ct_array[indices]
            cls_logits = self.classifier(z)  # l_i = f_cls(z_i)  (eq. 4)

            # Phenotype-alignment surrogate (Supplementary Note, "Inference Details"):
            #   ℓ_i = π·f_cls(z_i) + (1-π)·log φ_{g(i)},  probs_i = softmax(ℓ_i);
            #   add -γ·KL(probs_i ‖ φ_{g(i)}) to the log-joint so the ELBO trains the
            #   classifier f_cls (η_cls). φ (the covariate-level distribution) is the
            #   DETACHED alignment target. Without this factor cls_logits never enters
            #   the ELBO and f_cls receives no gradient.
            phi = p_ct[ct_idx].detach()
            log_phi = torch.log(phi + 1e-8)
            if self.gate_prob is not None:
                ell = self.gate_prob * cls_logits + (1.0 - self.gate_prob) * log_phi
            else:
                ell = cls_logits + log_phi
            probs = torch.softmax(ell, dim=-1)
            pheno_kl = (probs * (torch.log(probs + 1e-8) - log_phi)).sum(dim=-1)
            pyro.factor("phenotype_alignment", -self.phenotype_kl_weight * pheno_kl)

            px_scale, px_r_out, px_rate, px_dropout = self.decoder(
                "gene", z, log_library, batch_idx
            )

            zi_gate_probs = torch.sigmoid(px_dropout).clamp(min=1e-3, max=1.0 - 1e-3)
            nb_logits = (px_rate + self.eps).log() - (self.px_r.exp() + self.eps).log()
            nb_logits = torch.clamp(nb_logits, min=-10.0, max=10.0)
            total_count = self.px_r.exp().clamp(max=1e4)

            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate=zi_gate_probs,
                total_count=total_count,
                logits=nb_logits,
                validate_args=False,
            )
            # plain float, not torch.tensor(..., device=x.device): poutine.scale takes a
            # scalar, and materializing one on the device is a host->device copy (and a
            # sync point) on every SVI step for a value that never changes mid-epoch.
            with poutine.scale(scale=float(self.reconstruction_loss_scale)):
                pyro.sample("obs", x_dist.to_event(1), obs=x)

    @auto_move_data
    def guide(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        log_library: torch.Tensor,
        indices: torch.Tensor = None,
    ):
        pyro.module("scvi", self)
        batch_size = x.shape[0]

        with pyro.plate("clonotypes", self.c_count):
            # Start from a scaled version of the prior.
            init_mat_c = self.clone_phen_prior * self.guide_init_scale + 1e-3
            init_mat_c = init_mat_c.to(x.device)
            
            # Learnable raw parameters for q(p_c)
            if "q_p_c_raw" not in pyro.get_param_store():
                q_p_c_raw = pyro.param(
                    "q_p_c_raw",
                    init_mat_c.clone().detach(),
                    constraint=dist.constraints.positive
                )
            else:
                q_p_c_raw = pyro.param("q_p_c_raw")

            bad_c = ~torch.isfinite(q_p_c_raw)
            if bad_c.any():
                q_p_c_raw = torch.where(bad_c, init_mat_c.to(q_p_c_raw.device), q_p_c_raw)

            # Apply a sharpening transformation controlled by guide_temperature.
            q_p_c_sharp = q_p_c_raw ** (1.0 / self.guide_temperature)
            q_p_c_sharp = torch.clamp(q_p_c_sharp, min=1e-8)  # ← add this
            q_p_c_sharp = q_p_c_sharp / q_p_c_sharp.sum(dim=1, keepdim=True)
            conc_c_guide = torch.clamp(self.global_scale * q_p_c_sharp, min=1e-3)
            
            # Sample p_c from a single learned Dirichlet per clonotype.
            pyro.sample("p_c", dist.Dirichlet(conc_c_guide))

        with pyro.plate("ct_plate", self.ct_count):
            init_mat = self.clone_phen_prior[self.ct_to_c, :]
            init_mat = init_mat * self.guide_init_scale + 1e-3
            init_mat = init_mat.to(x.device)
            if "q_p_ct_raw" not in pyro.get_param_store():
                q_p_ct_raw = pyro.param(
                    "q_p_ct_raw",
                    init_mat.clone().detach(),  # Make sure it's not a leaf
                    constraint=dist.constraints.positive,
                )
            else:
                q_p_ct_raw = pyro.param("q_p_ct_raw")

            bad_ct = ~torch.isfinite(q_p_ct_raw)
            if bad_ct.any():
                q_p_ct_raw = torch.where(bad_ct, init_mat.to(q_p_ct_raw.device), q_p_ct_raw)

            q_p_ct_sharp = q_p_ct_raw ** (1.0 / self.guide_temperature)
            q_p_ct_sharp = torch.clamp(q_p_ct_sharp, min=1e-8)
            q_p_ct_sharp = q_p_ct_sharp / q_p_ct_sharp.sum(dim=1, keepdim=True)
            conc_ct_guide = torch.clamp(self.local_scale * q_p_ct_sharp, min=1e-3)
            pyro.sample("p_ct", dist.Dirichlet(conc_ct_guide))

        z_loc, z_scale, _ = self.encoder(x, batch_idx)
        z_scale = torch.clamp(z_scale, min=1e-3, max=10.0)

        with pyro.plate("data", batch_size) as idx:
            latent_posterior = dist.Normal(z_loc, z_scale)
            with pyro.poutine.scale(scale=self.kl_weight):
                z = pyro.sample("latent", latent_posterior.to_event(1))

    @auto_move_data
    def get_latent(self, tensor_dict: Dict[str, torch.Tensor]):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        batch_idx = tensor_dict[REGISTRY_KEYS.BATCH_KEY].long()
        z_loc, _, _ = self.encoder(x, batch_idx)
        if z_loc.ndim == 3:
            z_loc = z_loc.mean(dim=1)
        return z_loc.cpu()

    @torch.no_grad()
    def get_p_ct(self):
        from pyro import get_param_store

        param_store = get_param_store()
        q_p_ct_raw = param_store["q_p_ct_raw"]
        bad = ~torch.isfinite(q_p_ct_raw)
        if bad.any():
            n_phen = q_p_ct_raw.shape[1]
            q_p_ct_raw = torch.where(bad, torch.ones_like(q_p_ct_raw) / n_phen, q_p_ct_raw)
        if self.guide_temperature != 1.0:
            q_p_ct_sharp = q_p_ct_raw ** (1.0 / self.guide_temperature)
            q_p_ct_sharp = q_p_ct_sharp / q_p_ct_sharp.sum(dim=1, keepdim=True)
        else:
            q_p_ct_sharp = q_p_ct_raw / q_p_ct_raw.sum(dim=1, keepdim=True)
        return q_p_ct_sharp
