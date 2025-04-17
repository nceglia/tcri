import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch.nn as nn


from typing import Dict, Optional
from anndata import AnnData
import os

from scvi.data.fields import CategoricalObsField, LayerField
from scvi.data import AnnDataManager
from scvi.model.base import BaseModelClass
from scvi.train import PyroTrainingPlan, TrainRunner
from scvi import REGISTRY_KEYS
from scvi.nn import Encoder, DecoderSCVI
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.utils import setup_anndata_dsp
from scvi.dataloaders import DataSplitter
from torch.nn.functional import cosine_similarity
from pyro.infer import TraceEnum_ELBO, Trace_ELBO
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Found auxiliary vars")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*enumerate.*TraceEnum_ELBO.*"
)

os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_NTASKS_PER_NODE", None)
pyro.clear_param_store()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_archetypes(c2p_mat, K=4):
    """
    c2p_mat: shape (c_count, P)
        Each row is the global distribution of phenotypes for that clone
        (already row-stochastic).

    K: number of archetypes to find.

    Returns
    -------
    centers: shape (K, P)
        Each row is the centroid distribution of a cluster (archetype).
    labels: shape (c_count,)
        Which archetype each clone ended up in.
    """
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(c2p_mat)
    centers = kmeans.cluster_centers_
    # Ensure the centers are positive and row-stochastic
    centers = np.clip(centers, 1e-8, None)
    centers = centers / centers.sum(axis=1, keepdims=True)
    return centers, labels

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Shape becomes (batch_size, 1, embed_dim)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2).squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.ln2(out)
        return out + residual

class PhenotypeClassifier(nn.Module):
    def __init__(self, n_latent, classifier_hidden, P, num_layers=3, num_heads=2, dropout_rate=0.1):
        super(PhenotypeClassifier, self).__init__()
        self.attention_layer = AttentionLayer(embed_dim=n_latent, num_heads=num_heads, dropout_rate=dropout_rate)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(n_latent, classifier_hidden, dropout_rate) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(n_latent, P)

    def forward(self, x):
        x = self.attention_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

###############################################################################
# -1) VampPrior
###############################################################################
class VampPrior(torch.nn.Module):
    def __init__(self, pseudo_inputs, encoder):
        """
        Args:
            pseudo_inputs (torch.Tensor): Initial pseudo-inputs of shape (K, input_dim).
            encoder (torch.nn.Module): Encoder that takes an input and returns (mean, log_var)
                for the approximate posterior q(z|x).
        """
        super(VampPrior, self).__init__()
        # Learnable pseudo-inputs; these are optimized during training.
        self.pseudo_inputs = torch.nn.Parameter(pseudo_inputs)
        self.encoder = encoder

    def get_mixture(self):
        """
        Constructs the VampPrior as a uniform mixture of q(z|u_k) for each pseudo-input u_k.
        """
        # Compute the approximate posterior parameters for each pseudo-input.
        # Expected output shapes: means and log_vars: (K, latent_dim)
        K = self.pseudo_inputs.size(0)
        # Create a dummy categorical argument; unsqueeze to have shape (K, 1)
        dummy_batch = torch.zeros(K, dtype=torch.long, device=self.pseudo_inputs.device).unsqueeze(1)
        means, log_vars, _ = self.encoder(self.pseudo_inputs, dummy_batch)
        scales = torch.sqrt(torch.exp(log_vars))
        component_dist = dist.Independent(dist.Normal(means, scales), 1)
        mixture_weights = torch.ones(K, device=self.pseudo_inputs.device) / K
        mixture = dist.MixtureSameFamily(
            dist.Categorical(mixture_weights),
            component_dist
        )
        return mixture

    def log_prob(self, z):
        """
        Computes log p(z) under the VampPrior.
        """
        return self.get_mixture().log_prob(z)

    def sample(self, sample_shape=torch.Size()):
        """
        Draws samples from the VampPrior.
        """
        return self.get_mixture().sample(sample_shape)


###############################################################################
# 0) Mixture of Dirichlet Distributions - TODO: Refactor
###############################################################################
class MixtureDirichlet(dist.TorchDistribution):
    """
    Mixture of Dirichlet distributions parameterized by mixture weights and concentration parameters."
    """
    arg_constraints = {
        "mixture_weights": dist.constraints.simplex,  # shape: batch_shape + (B,)
        "concentration": dist.constraints.positive,  # shape: batch_shape + (B, K)
    }
    support = dist.constraints.simplex  # each sample is a simplex over K categories
    has_rsample = False

    def __init__(
        self,
        mixture_weights: torch.Tensor,
        concentration: torch.Tensor,
        validate_args=None,
    ):
        """
        mixture_weights: Tensor of shape batch_shape + (B,), with each row summing to 1.
        concentration: Tensor of shape batch_shape + (B, K), where K is the number of categories.
        """
        self.mixture_weights = mixture_weights
        # Clamp concentrations to ensure positivity.
        self.concentration = torch.clamp(concentration, min=1e-3)
        # Determine batch shape, B, and K.
        batch_shape = self.mixture_weights.shape[:-1]
        self.B = self.mixture_weights.size(-1)
        self.K = self.concentration.size(-1)
        event_shape = (self.K,)
        super(MixtureDirichlet, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def sample(self, sample_shape=torch.Size()):
        """
        Returns a sample of shape: sample_shape + batch_shape + (K,).
        For each batch element, first sample a mixture component, then sample from the corresponding Dirichlet.
        """
        # Create categorical for mixture weights.
        cat = dist.Categorical(self.mixture_weights)
        # Sample mixture indices; shape: sample_shape + batch_shape.
        mixture_idx = cat.sample(sample_shape)
        full_shape = mixture_idx.shape  # sample_shape + batch_shape

        # Expand concentration to shape: sample_shape + batch_shape + (B, K).
        target_shape = sample_shape + self.concentration.shape
        expanded_concentration = self.concentration.expand(target_shape)

        # Flatten the sample and batch dimensions.
        flat_shape = (-1, self.B, self.K)
        flat_concentration = expanded_concentration.reshape(flat_shape)
        flat_idx = mixture_idx.reshape(-1)  # shape: (num_samples,)

        # Select the concentration parameters corresponding to the sampled mixture index.
        selected_concentration = flat_concentration[
            torch.arange(flat_idx.size(0)), flat_idx
        ]

        # Sample from the Dirichlet for each sample.
        flat_samples = dist.Dirichlet(selected_concentration).sample()
        # Reshape to sample_shape + batch_shape + (K,).
        return flat_samples.reshape(full_shape + (self.K,))

    def log_prob(self, value):
        device = value.device  # get the device from input tensor
        
        # Move tensors explicitly to the same device
        value_expanded = value.unsqueeze(-2).to(device)
        expanded_concentration = self.concentration.expand(
            value.shape[:-1] + (self.B, self.K)
        ).to(device)
        
        d = dist.Dirichlet(expanded_concentration)
        
        component_log_probs = d.log_prob(
            value_expanded.expand(expanded_concentration.shape)
        )
        
        expanded_weights = self.mixture_weights.expand(value.shape[:-1] + (self.B,)).to(device)
        mixture_log = torch.log(expanded_weights)
        
        return torch.logsumexp(mixture_log + component_log_probs, dim=-1)


    def score_parts(self, value):
        # Compute log probability.
        lp = self.log_prob(value)
        # Return dummy zeros for the score function and entropy terms.
        zeros = torch.zeros_like(lp)
        return lp, zeros, zeros

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


###############################################################################
# 1) Pairwise Margin Loss Helper
###############################################################################
def pairwise_centroid_margin_loss(
    z: torch.Tensor,
    phenotypes: torch.Tensor,
    margin: float = 1.0,
    adaptive_margin: bool = False,
) -> torch.Tensor:
    """
    Margin-based separation penalty on latent z for different phenotype groups.
    If adaptive_margin=True, we scale the 'margin' by the standard deviation of
    the phenotype centroids.
    """
    phenotypes = phenotypes.view(-1)
    unique_phen = phenotypes.unique()
    if len(unique_phen) < 2:
        return z.new_zeros(1)

    centroids = []
    for p in unique_phen:
        mask = phenotypes == p
        z_p = z[mask]
        centroids.append(z_p.mean(dim=0))
    centroids = torch.stack(centroids, dim=0)

    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(torch.norm(centroids[i] - centroids[j], p=2))
    if not dists:
        return z.new_zeros(1)
    dists = torch.stack(dists)

    if adaptive_margin:
        margin = margin * torch.std(centroids)

    penalty = torch.clamp(margin - dists, min=0.0).mean()
    return penalty


###############################################################################
# 2) Pyro Module with CVAE + Hierarchical Priors
###############################################################################
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
        sharp_temperature: float = 1.0,
        gate_prob: float = 0.5,
        mixture_concentration: torch.Tensor = None,
        n_pseudo_obs: int = 10,
        use_enumeration: bool = False,
        gate_nn_hidden: int = 32,
        classifier_hidden: int = 32,
        classifier_dropout: float = 0.1,
        classifier_n_layers: int = 3,
        classifier_num_heads: int = 2,
        n_hidden: int = 128,
        n_layers: int = 3,
        class_weights: torch.Tensor = None,
        kl_weight_max: float = 1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.P = P
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.sharp_temperature = sharp_temperature
        self.mixture_concentration = mixture_concentration
        self.n_pseudo_obs = n_pseudo_obs
        self.classifier_num_heads = classifier_num_heads
        self.gate_prob = gate_prob
        # Assert that it is not None
        assert (
            self.mixture_concentration is not None
        ), "mixture_concentration must be provided"
        self.use_enumeration = use_enumeration
        self.eps = 1e-6
        self.gate_nn_hidden = gate_nn_hidden
        self.classifier_hidden = classifier_hidden
        self.classifier_dropout = classifier_dropout
        self.kl_weight_max = kl_weight_max
        self.classifier_n_layers = classifier_n_layers

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
            num_layers=self.classifier_n_layers,  # Use the existing configurable number of layers
            num_heads=self.classifier_num_heads  # You can make this configurable as well if needed
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

        # Store or compute log of class weights if provided
        if class_weights is not None:
            # Expect a tensor of shape (P,)
            self.register_buffer("log_class_weights", torch.log(class_weights))
        else:
            self.log_class_weights = None

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

        if self.sharp_temperature != 1.0:
            prior_mat = prior_mat ** (1.0 / self.sharp_temperature)
            prior_mat = prior_mat / prior_mat.sum(dim=1, keepdim=True)

        self.register_buffer("clone_phen_prior", prior_mat)
        self.register_buffer("ct_to_c", ct_to_c_array)
        self.register_buffer("c_array", c_array_for_cells)
        self.register_buffer("ct_array", ct_array_for_cells)
        self.register_buffer("_target_phenotypes", target_phenotypes)

        if ct_to_cov_array is not None:
            self.register_buffer("ct_to_cov", ct_to_cov_array)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict: Dict[str, torch.Tensor]):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        batch_idx = tensor_dict[REGISTRY_KEYS.BATCH_KEY].long()
        log_library = torch.log(torch.sum(x, dim=1, keepdim=True) + 1e-6)
        return (x, batch_idx, log_library), {}

    @auto_move_data
    def model(
        self, x: torch.Tensor, batch_idx: torch.Tensor, log_library: torch.Tensor
    ):
        pyro.module("scvi", self)

        # initial_confusion = torch.eye(self.P) + 1e-3
        # initial_confusion = initial_confusion / initial_confusion.sum(-1, keepdim=True)

        # confusion_matrix = pyro.param(
        #     "confusion_matrix", initial_confusion, constraint=dist.constraints.simplex
        # )

        kl_weight = self.kl_weight
        batch_size = x.shape[0]

        # Hierarchical priors
        # Assume self.mixture_concentration is a tensor of shape (B, K) provided as input.
        B = self.mixture_concentration.shape[0]
        # Create equal weights for each of the B mixture components.
        mixture_weights = torch.ones(B, device=self.mixture_concentration.device) / B

        with pyro.plate("clonotypes", self.c_count):
            # Assume self.mixture_concentration is a tensor of shape (B, K)
            B = self.mixture_concentration.shape[0]
            mixture_weights = torch.ones(B, device=x.device) / B
            # Expand mixture parameters to add a leading dimension for clonotypes.
            # expanded_conc will have shape (self.c_count, B, K)
            expanded_conc = self.mixture_concentration.unsqueeze(0).expand(
                self.c_count, -1, -1
            )
            # expanded_weights will have shape (self.c_count, B)
            expanded_weights = mixture_weights.unsqueeze(0).expand(self.c_count, -1)
            mixture_dist = MixtureDirichlet(expanded_weights, expanded_conc)
            p_c = pyro.sample("p_c", mixture_dist)
            # print("p_c shape:", p_c.shape)

        with pyro.plate("ct_plate", self.ct_count):
            base_p = p_c[self.ct_to_c] + self.eps
            conc_ct = torch.clamp(self.local_scale * base_p, min=1e-3)
            p_ct = pyro.sample("p_ct", dist.Dirichlet(conc_ct))

        # Encoder
        z_loc, z_scale, _ = self.encoder(x, batch_idx)
        z_scale = torch.clamp(z_scale, min=1e-3, max=10.0)

        with pyro.plate("data", batch_size) as idx:
            # latent_prior = dist.Normal(
            #     torch.zeros_like(z_loc), torch.ones_like(z_scale)
            # )
            # with poutine.scale(scale=kl_weight):
            #     z = pyro.sample("latent", latent_prior.to_event(1))
            with poutine.scale(scale=kl_weight):
                # vamp_mixture = self.vamp_prior.get_mixture().to_event(1)
                vamp_mixture = self.vamp_prior.get_mixture()
                z = pyro.sample("latent", vamp_mixture)

            ct_idx = self.ct_array[idx]
            prior_log = torch.log(p_ct[ct_idx] + 1e-8)  # log of local p_ct
            cls_logits = self.classifier(z)# + self.phenotype_decoder(z)
        

            gate_probs = torch.full((batch_size, self.P), self.gate_prob, device=x.device)
            # Weighted combination: gate_probs * classifier + (1 - gate_probs) * local prior
            local_logits_model = (
                gate_probs * cls_logits + (1.0 - gate_probs) * prior_log
            )

            # ======== ADD THIS TO WEIGHT CLASSES ========
            if self.log_class_weights is not None:
                local_logits_model = local_logits_model + self.log_class_weights
            # ===========================================

            # # sample the discrete phenotype from these logits
            # z_i_phen = pyro.sample(
            #     "z_i_phen",
            #     dist.Categorical(logits=local_logits_model),
            #     infer={"enumerate": "parallel"} if self.use_enumeration else {},
            # )
            
            obs_phen = self._target_phenotypes[
                idx
            ]  # This is your true label for each cell

            obs_phen = self._target_phenotypes[idx]

            pyro.sample(
                "obs_label",
                dist.Categorical(logits=local_logits_model),
                obs=obs_phen,
            )
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
            scale_val = torch.tensor(self.reconstruction_loss_scale, device=x.device)
            with poutine.scale(scale=scale_val):
                pyro.sample("obs", x_dist.to_event(1), obs=x)

    @auto_move_data
    def guide(
        self, x: torch.Tensor, batch_idx: torch.Tensor, log_library: torch.Tensor
    ):
        pyro.module("scvi", self)
        batch_size = x.shape[0]

        with pyro.plate("clonotypes", self.c_count):
            # Start from a scaled version of the prior.
            init_mat_c = self.clone_phen_prior * 10.0 + 1e-3
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
            
            # Apply a sharpening transformation controlled by sharp_temperature.
            q_p_c_sharp = q_p_c_raw ** (1.0 / self.sharp_temperature)
            # Normalize each row to be a valid probability vector.
            q_p_c_sharp = q_p_c_sharp / q_p_c_sharp.sum(dim=1, keepdim=True)
            # Scale to obtain Dirichlet concentration parameters.
            conc_c_guide = torch.clamp(self.global_scale * q_p_c_sharp, min=1e-3)
            
            # Sample p_c from a single learned Dirichlet per clonotype.
            pyro.sample("p_c", dist.Dirichlet(conc_c_guide))

        with pyro.plate("ct_plate", self.ct_count):
            init_mat = self.clone_phen_prior[self.ct_to_c, :]
            init_mat = init_mat * 10.0 + 1e-3
            init_mat = init_mat.to(x.device)
            if "q_p_ct_raw" not in pyro.get_param_store():
                q_p_ct_raw = pyro.param(
                    "q_p_ct_raw",
                    init_mat.clone().detach(),  # Make sure it's not a leaf
                    constraint=dist.constraints.positive,
                )
            else:
                q_p_ct_raw = pyro.param("q_p_ct_raw")
            q_p_ct_sharp = q_p_ct_raw ** (1.0 / self.sharp_temperature)
            q_p_ct_sharp = q_p_ct_sharp / q_p_ct_sharp.sum(dim=1, keepdim=True)
            conc_ct_guide = torch.clamp(self.local_scale * q_p_ct_sharp, min=1e-3)
            pyro.sample("p_ct", dist.Dirichlet(conc_ct_guide))

        z_loc, z_scale, _ = self.encoder(x, batch_idx)
        z_scale = torch.clamp(z_scale, min=1e-3, max=10.0)

        with pyro.plate("data", batch_size) as idx:
            latent_posterior = dist.Normal(z_loc, z_scale)
            with pyro.poutine.scale(scale=self.kl_weight):
                z = pyro.sample("latent", latent_posterior.to_event(1))

            ct_idx = self.ct_array[idx]
            prior_log_guide = torch.log(q_p_ct_sharp[ct_idx] + 1e-8)
            cls_logits = self.classifier(z) #+ self.phenotype_decoder(z)

            gate_probs = torch.full((batch_size, self.P), self.gate_prob, device=x.device)
            local_logits_guide = (
                gate_probs * cls_logits + (1 - gate_probs) * prior_log_guide
            )

            # ======== ADD THIS TO WEIGHT CLASSES IN THE GUIDE ========
            if self.log_class_weights is not None:
                local_logits_guide = local_logits_guide + self.log_class_weights
            # =========================================================

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
        if self.sharp_temperature != 1.0:
            q_p_ct_sharp = q_p_ct_raw ** (1.0 / self.sharp_temperature)
            q_p_ct_sharp = q_p_ct_sharp / q_p_ct_sharp.sum(dim=1, keepdim=True)
        else:
            q_p_ct_sharp = q_p_ct_raw / q_p_ct_raw.sum(dim=1, keepdim=True)
        return q_p_ct_sharp


###############################################################################
# 3) Unified Training Plan with Validation Step for scvi Early Stopping
###############################################################################
class UnifiedTrainingPlan(PyroTrainingPlan):
    """
    Training plan that includes margin, classification, reconstruction losses,
    KL warmup, plus a validation_step that logs 'elbo_validation' so scvi's
    early stopping can monitor it.
    """

    def __init__(
        self,
        module: TCRIModule,
        margin_scale: float = 0.0,
        margin_value: float = 2.0,
        n_steps_kl_warmup: int = 1000,
        adaptive_margin: bool = False,
        reconstruction_loss_scale: float = 1e-2,
        num_particles: int = 8,
        optimizer_config: dict = None,
        gate_saturation_weight: float = 0.0,
        clone_alignment_scale: float = 1.0,
        class_weights: torch.Tensor = None,
        **kwargs,
    ):
        self.num_particles = num_particles
        if module.use_enumeration:
            print("Using Enumeration")
            self._loss_fn = TraceEnum_ELBO(
                max_plate_nesting=3, num_particles=self.num_particles
            )
        else:
            self._loss_fn = Trace_ELBO()

        super().__init__(module, n_steps_kl_warmup=n_steps_kl_warmup, **kwargs)

        self.margin_scale = margin_scale
        self.margin_value = margin_value
        self.adaptive_margin = adaptive_margin
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self.clone_alignment_scale = clone_alignment_scale
        self._my_global_step = 0
        self.kl_sigmoid_midpoint = 4000
        self.kl_sigmoid_speed = 0.001
        self.gate_saturation_weight = gate_saturation_weight
        self.class_weights = class_weights
        if optimizer_config is None:
            optimizer_config = {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-5,
                "weight_decay": 1e-4,
            }
        self.optimizer_config = optimizer_config

    @property
    def loss(self):
        return self._loss_fn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=self.optimizer_config["lr"],
            betas=self.optimizer_config["betas"],
            eps=self.optimizer_config["eps"],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        kl_weight = self.module.kl_weight_max / (1.0 + np.exp(-0.005 * (self._my_global_step - 2000)))
        self.module.kl_weight = kl_weight

        loss_dict = super().training_step(batch, batch_idx)
        device = next(self.module.parameters()).device

        if not isinstance(loss_dict["loss"], torch.Tensor):
            loss_dict["loss"] = torch.tensor(
                loss_dict["loss"], device=device, requires_grad=True
            )
        else:
            loss_dict["loss"] = loss_dict["loss"].to(device)

        z_batch = self.module.get_latent(batch).to(device)
        idx = batch["indices"].long().view(-1).to(device)
        target_phen = self.module._target_phenotypes[idx].to(device)

        margin_val = 0.0
        if self.margin_scale > 0.0:
            margin_loss_val = pairwise_centroid_margin_loss(
                z_batch,
                target_phen,
                margin=self.margin_value,
                adaptive_margin=self.adaptive_margin,
            ).to(device)
            margin_loss = self.margin_scale * margin_loss_val
            loss_dict["loss"] += margin_loss
            margin_val = margin_loss_val.item()

        x = batch[REGISTRY_KEYS.X_KEY].float().to(device)
        batch_idx_tensor = batch[REGISTRY_KEYS.BATCH_KEY].long().to(device)
        log_library = torch.log(torch.sum(x, dim=1, keepdim=True) + 1e-6).to(device)

        px_scale, px_r_out, px_rate, px_dropout = self.module.decoder(
            "gene", z_batch, log_library, batch_idx_tensor
        )

        gate_probs = torch.sigmoid(px_dropout).clamp(min=1e-3, max=1 - 1e-3)

        nb_logits = (px_rate + self.module.eps).log() - (
            self.module.px_r.exp() + self.module.eps
        ).log()
        nb_logits = torch.clamp(nb_logits, min=-10.0, max=10.0)
        total_count = self.module.px_r.exp().clamp(max=1e4)
        x_dist = dist.ZeroInflatedNegativeBinomial(
            gate=gate_probs,
            total_count=total_count,
            logits=nb_logits,
            validate_args=False,
        )
        alpha = 10.0
        cls_logits = self.module.classifier(z_batch)
        gamma = 2.0
        alpha = 0.25

        p = F.softmax(cls_logits, dim=-1)
        ce_loss = F.nll_loss(F.log_softmax(cls_logits, dim=-1), target_phen, reduction="none")
        focal_loss = alpha * (1 - p[range(len(target_phen)), target_phen]) ** gamma * ce_loss
        focal_loss = focal_loss.mean()
        loss_dict["loss"] += focal_loss
        loss_dict["classification_loss"] = focal_loss.item()
 
        entropy_penalty_weight = 50.0
        T = max(0.5, 1.5 * np.exp(-0.001 * self._my_global_step))
        probs = F.softmax(cls_logits / T, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        loss_dict["loss"] += entropy_penalty_weight * entropy  # try 0.1 to 1.0
        loss_dict["classifier_entropy"] = entropy.item()

        loss_dict["margin_loss"] = margin_val

        fn_args, fn_kwargs = self.module._get_fn_args_from_batch(batch)
        trace = poutine.trace(self.module.model).get_trace(*fn_args, **fn_kwargs)
        trace.compute_log_prob()
        recon_log_prob = trace.nodes["obs"]["log_prob_sum"]  # reconstruction site
        cls_log_prob   = trace.nodes["obs_label"]["log_prob_sum"]  # classification site

        # Negative log-likelihood components:
        recon_loss = -recon_log_prob.item()
        cls_loss   = -cls_log_prob.item()
        print("Reconstruction Loss (batch):", recon_loss)
        print("Classification Loss (batch):", cls_loss)

        self._my_global_step += 1
        return loss_dict
    
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.module.eval()
            val_dict = super().training_step(batch, batch_idx)
            self.module.train()
        device = next(self.module.parameters()).device
        if not isinstance(val_dict["loss"], torch.Tensor):
            val_dict["loss"] = torch.tensor(val_dict["loss"], device=device)
        else:
            val_dict["loss"] = val_dict["loss"].to(device)
        self.log("elbo_validation", val_dict["loss"], prog_bar=True, on_epoch=True)
        return val_dict


###############################################################################
# 4) High-Level scVI Model with scvi Early Stopping
###############################################################################
class TCRIModel(BaseModelClass):
    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str = "X",
        clonotype_key: str = "unique_clone_id",
        phenotype_key: str = "phenotype_col",
        covariate_key: str = "timepoint",
        batch_key: str = "patient",
        **kwargs,
    ):
        for col in [clonotype_key, phenotype_key, covariate_key, batch_key]:
            if col not in adata.obs:
                raise ValueError(f"{col} not in adata.obs!")
        adata.obs["indices"] = list(range(len(adata.obs.index)))
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField("clonotype_col_in_registry", clonotype_key),
            CategoricalObsField("phenotype_col_in_registry", phenotype_key),
            CategoricalObsField("covariate_col_in_registry", covariate_key),
            CategoricalObsField("indices", "indices"),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        setup_method_args = cls._get_setup_method_args(**locals())
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        adata_manager.registry["clonotype_col"] = clonotype_key
        adata_manager.registry["phenotype_col"] = phenotype_key
        adata_manager.registry["covariate_col"] = covariate_key
        adata_manager.registry["batch_col"] = batch_key
        cls.register_manager(adata_manager)
        adata.uns["tcri_manager"] = adata_manager
        return adata

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        n_hidden: int = 128,
        global_scale: float = 10.0,
        local_scale: float = 5.0,
        sharp_temperature: float = 1.0,
        use_enumeration: bool = False,
        patience: int = 50,
        gate_saturation_weight: float = 0.0,
        gate_nn_hidden: int = 32,
        classifier_hidden: int = 32,
        classifier_dropout: float = 0.1,
        K: int = 10,
        phenotype_weights: Optional[Dict[str, float]] = None,
        gate_prob: float = 0.5,
        kl_weight_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(adata)
        n_vars = self.summary_stats["n_vars"]
        clonotype_col = self.adata_manager.registry["clonotype_col"]
        phenotype_col = self.adata_manager.registry["phenotype_col"]
        covariate_col = self.adata_manager.registry["covariate_col"]
        batch_col = self.adata_manager.registry["batch_col"]

        ph_series = self.adata.obs[phenotype_col].astype("category")
        P = len(ph_series.cat.categories)
        target_codes = torch.tensor(ph_series.cat.codes.values, dtype=torch.long)

        cvals = self.adata.obs[clonotype_col].astype("category")
        c_count = len(cvals.cat.categories)
        c_array_np = cvals.cat.codes.values
        pvals_np = ph_series.cat.codes.values
        c2p_mat = np.zeros((c_count, P), dtype=np.float32)
        for i in range(len(c_array_np)):
            c2p_mat[c_array_np[i], pvals_np[i]] += 1
        c2p_mat += 1e-6
        c2p_mat = c2p_mat / c2p_mat.sum(axis=1, keepdims=True)
        self.c2p_mat = c2p_mat
        self.centers, self.labels = build_archetypes(self.c2p_mat, K=K)
        cov_series = self.adata.obs[covariate_col].astype("category")
        cov_array_np = cov_series.cat.codes.values
        df_ct = pd.DataFrame({"c": c_array_np, "t": cov_array_np})
        combos = df_ct.drop_duplicates().sort_values(["c", "t"])
        ct_list = combos.values.tolist()
        ct_map = {}
        ct_to_c_list = []
        ct_to_cov_list = []
        for idx, (c_val, t_val) in enumerate(ct_list):
            ct_map[(c_val, t_val)] = idx
            ct_to_c_list.append(c_val)
            ct_to_cov_list.append(t_val)
        ct_count = len(ct_list)
        ct_array_np = np.empty(len(c_array_np), dtype=np.int64)
        for i in range(len(c_array_np)):
            ct_array_np[i] = ct_map[(c_array_np[i], cov_array_np[i])]

        batch_series = self.adata.obs[batch_col].astype("category")
        n_batch = len(batch_series.cat.categories)

        if phenotype_weights is None:
            # Automatically compute inverse-frequency weights for each phenotype
            freq_count = ph_series.value_counts(sort=False) 
            class_weights_arr = []
            for cat_name in ph_series.cat.categories:

                c = freq_count[cat_name]
                # inverse-frequency weight
                weight = 1.0 / c
                class_weights_arr.append(weight)
            class_weights = torch.tensor(class_weights_arr, dtype=torch.float32)
        else:
            class_weights_arr = []
            for cat_name in ph_series.cat.categories:
                weight = phenotype_weights.get(cat_name, 1.0)
                class_weights_arr.append(weight)
            class_weights = torch.tensor(class_weights_arr, dtype=torch.float32)

        self.class_weights = class_weights
        self.module = TCRIModule(
            n_input=n_vars,
            n_latent=n_latent,
            P=P,
            n_batch=n_batch,
            n_hidden=n_hidden,
            global_scale=global_scale,
            local_scale=local_scale,
            mixture_concentration=torch.from_numpy(self.centers),
            sharp_temperature=sharp_temperature,
            use_enumeration=use_enumeration,
            gate_nn_hidden=gate_nn_hidden,
            classifier_hidden=classifier_hidden,
            classifier_dropout=classifier_dropout,
            class_weights=self.class_weights,
            gate_prob=gate_prob,
            kl_weight_max=kl_weight_max,
        )
        self.init_params_ = self._get_init_params(locals())
        self.gate_saturation_weight = gate_saturation_weight
        c2p_torch = torch.tensor(c2p_mat, dtype=torch.float32)
        c_array_torch = torch.tensor(c_array_np, dtype=torch.long)
        ct_array_torch = torch.tensor(ct_array_np, dtype=torch.long)
        ct_to_c_torch = torch.tensor(ct_to_c_list, dtype=torch.long)
        ct_to_cov_torch = torch.tensor(ct_to_cov_list, dtype=torch.long)
        self.patience = patience
        self.module.prepare_two_level_params(
            c_count=c_count,
            ct_count=ct_count,
            clone_phen_prior_mat=c2p_torch,
            ct_to_c_array=ct_to_c_torch,
            c_array_for_cells=c_array_torch,
            ct_array_for_cells=ct_array_torch,
            target_phenotypes=target_codes,
            ct_to_cov_array=ct_to_cov_torch,
        )
        logger.info(
            f"Unified model: c_count={c_count}, ct_count={ct_count}, P={P}, "
            f"global_scale={global_scale}, local_scale={local_scale}, use_enumeration={use_enumeration}, "
            f"sharp_temperature={sharp_temperature}."
        )

    def train(
        self,
        max_epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        margin_scale: float = 0.0,
        margin_value: float = 2.0,
        adaptive_margin: bool = False,
        reconstruction_loss_scale: float = 1e-2,
        n_steps_kl_warmup: int = 1000,
        clone_alignment_scale: float = 0.0,
        **kwargs,
    ):
        """
        We split the data into train/val, define a UnifiedTrainingPlan with
        validation_step, and let scvi handle early stopping automatically
        by passing early_stopping parameters to TrainRunner.
        """
        # Create a train/val split
        self.module.reconstruction_loss_scale = reconstruction_loss_scale
        splitter = DataSplitter(
            self.adata_manager,
            train_size=0.9,
            validation_size=None,
            batch_size=batch_size,
        )

        plan = UnifiedTrainingPlan(
            module=self.module,
            margin_scale=margin_scale,
            margin_value=margin_value,
            n_steps_kl_warmup=n_steps_kl_warmup,
            adaptive_margin=adaptive_margin,
            reconstruction_loss_scale=reconstruction_loss_scale,
            gate_saturation_weight=self.gate_saturation_weight,
            clone_alignment_scale=clone_alignment_scale,
            class_weights=self.class_weights,
            optimizer_config={
                "lr": lr,
                "betas": (0.9, 0.999),
                "eps": 1e-5,
                "weight_decay": 1e-4,
            },
        )

        runner = TrainRunner(
            self,
            training_plan=plan,
            data_splitter=splitter,
            max_epochs=max_epochs,
            early_stopping=True,
            early_stopping_monitor="elbo_validation",
            early_stopping_mode="min",
            early_stopping_patience=self.patience,
            check_val_every_n_epoch=5,
            accelerator="auto",
            devices="auto",
            **kwargs,
        )

        runner()
        return

    @torch.no_grad()
    def get_latent_representation(self, adata=None, indices=None, batch_size=None):
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latents = [self.module.get_latent(tensors) for tensors in scdl]
        return torch.cat(latents, dim=0).cpu().numpy()

    @torch.no_grad()
    def get_p_ct(self):
        return self.module.get_p_ct().cpu().numpy()

    @torch.no_grad()
    def get_cell_phenotype_probs(
        self, adata=None, batch_size: int = 256, eps: float = 1e-8, gate_prob: float = 0.5
    ) -> np.ndarray:
        """
        Computes the cell-level phenotype probabilities in the same way as training,
        using:
        cls_logits = classifier(z_loc) + phenotype_decoder(z_loc)
        if class weights exist, add them
        gate_probs = sigmoid(gate_nn(z_loc))
        local_logits = gate_probs*cls_logits + (1 - gate_probs)*log(prior)
        softmax(local_logits) -> probabilities

        Parameters
        ----------
        adata
            If None, defaults to the AnnData used in training.
        batch_size : int
            Mini-batch size for data loader.
        eps : float
            Small epsilon for numerical stability in logs.

        Returns
        -------
        probs : np.ndarray
            Array of shape (n_cells, P) of phenotype probabilities.
        """
        adata = self._validate_anndata(adata)
        device = next(self.module.parameters()).device
        scdl = self._make_data_loader(adata=adata, batch_size=batch_size)

        # The learned posterior p_ct -> shape (ct_count, P)
        p_ct = self.module.get_p_ct().to(device)
        # Map each cell to its clonotype-covariate index -> shape (n_cells,)
        ct_array = self.module.ct_array.to(device)

        all_probs = []
        current_idx = 0

        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
            this_batch_size = x.shape[0]

            # Which (clonotype, covariate) does each cell belong to?
            ct_indices = ct_array[current_idx : current_idx + this_batch_size]
            clone_cov_posterior = p_ct[ct_indices]  # (batch_size, P)

            # Encode to get latent z
            z_loc, _, _ = self.module.encoder(x, b)

            # ----- 1) Compute classifier logits (same as training) -----
            cls_logits = self.module.classifier(z_loc) #+ self.module.phenotype_decoder(z_loc)

            # Optionally add log class weights if defined
            if self.module.log_class_weights is not None:
                cls_logits = cls_logits + self.module.log_class_weights

            # ----- 2) Gating from gate_nn (rather than fixed 0.5) -----
            # gate_logits = self.module.gate_nn(z_loc)    # shape (batch_size, 1)
            # gate_probs = torch.sigmoid(gate_logits)     # in [0,1]
            # gate_probs = gate_probs.expand(-1, self.module.P)
            gate_probs = torch.full(
                (this_batch_size, self.module.P), gate_prob, device=x.device
            )

            # ----- 3) Combine with local prior in log space -----
            prior_log = torch.log(clone_cov_posterior + eps)
            local_logits = gate_probs * cls_logits + (1.0 - gate_probs) * prior_log

            # ----- 4) Convert to probabilities via softmax -----
            probs = torch.softmax(local_logits, dim=-1)

            all_probs.append(probs.cpu())
            current_idx += this_batch_size

        # Concatenate into final array of shape (n_cells, P)
        return torch.cat(all_probs, dim=0).numpy()



    def plot_loss(self, log_scale=False):
        loss_history = self.history_["elbo_train"]
        loss_validation = self.history_["elbo_validation"]
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        if log_scale:
            plt.yscale("log")
        plt.legend()
        plt.show()
