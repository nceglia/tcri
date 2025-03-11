import logging
import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.distributions.kl import kl_divergence
from typing import Dict
from anndata import AnnData

# scvi-tools imports
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

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Found auxiliary vars")
warnings.filterwarnings("ignore", category=UserWarning, message=".*enumerate.*TraceEnum_ELBO.*")

os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_NTASKS_PER_NODE", None)
pyro.clear_param_store()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        mask = (phenotypes == p)
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

import torch
from torch.nn.functional import cosine_similarity

def continuity_loss(z, labels, temperature=0.1, num_samples=10000):
    batch_size = z.size(0)

    num_samples = min(num_samples, batch_size * (batch_size - 1))
    idx = torch.combinations(torch.arange(batch_size, device=z.device), r=2)
    perm = torch.randperm(idx.size(0), device=z.device)[:num_samples]
    idx_i, idx_j = idx[perm, 0], idx[perm, 1]

    # Compute cosine similarities for selected pairs
    sim = cosine_similarity(z[idx_i], z[idx_j], dim=-1) / temperature

    # Labels for selected pairs
    labels_equal = labels[idx_i] == labels[idx_j]

    if labels_equal.sum() == 0 or (~labels_equal).sum() == 0:
        # Prevent division by zero if sampling misses positive or negative pairs
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    positives = sim[labels_equal]
    negatives = sim[~labels_equal]

    loss = -positives.mean() + negatives.mean()
    return loss

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
        sharpness_penalty_scale: float = 0.0,
        use_enumeration: bool = False,
        n_hidden: int = 128,
        n_layers: int = 3,
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
        self.sharpness_penalty_scale = sharpness_penalty_scale
        self.use_enumeration = use_enumeration
        self.eps = 1e-6

        # kl_weight updated each step
        self.kl_weight =5


        # Phenotype Embedding
        self.phenotype_embedding = torch.nn.Embedding(num_embeddings=P, embedding_dim=n_latent)

        # Encoder
        self.encoder = Encoder(
            n_input=n_input,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            use_layer_norm=True
        )

        # Decoder
        self.decoder_input_dim = self.n_latent + self.phenotype_embedding.embedding_dim
        self.decoder = DecoderSCVI(
            self.decoder_input_dim,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            scale_activation="softplus",
            use_layer_norm=True
        )

        self.px_r = torch.nn.Parameter(torch.ones(n_input))
        self.classifier = torch.nn.Linear(n_latent, P)

        # Buffers for hierarchical priors
        self.register_buffer("clone_phen_prior", torch.empty(0))
        self.register_buffer("ct_to_c", torch.empty(0, dtype=torch.long))
        self.register_buffer("c_array", torch.empty(0, dtype=torch.long))
        self.register_buffer("ct_array", torch.empty(0, dtype=torch.long))
        self.register_buffer("ct_to_cov", torch.empty(0, dtype=torch.long))
        self.c_count = 0
        self.ct_count = 0
        self.n_cells = 0

        # Phenotypes
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
        
        prior_mat = (clone_phen_prior_mat + self.eps)
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
    def model(self, x: torch.Tensor, batch_idx: torch.Tensor, log_library: torch.Tensor):
        pyro.module("scvi", self)
        kl_weight = self.kl_weight
        batch_size = x.shape[0]
        ct_array = self.ct_array
        # Top-level hierarchical priors (unchanged)
        with pyro.plate("clonotypes", self.c_count):
            conc_c = torch.clamp(self.global_scale * self.clone_phen_prior, min=1e-3)
            p_c = pyro.sample("p_c", dist.Dirichlet(conc_c))
            if self.sharpness_penalty_scale > 0:
                ent_c = dist.Dirichlet(conc_c).entropy()
                pyro.factor("sharpness_penalty_top", -self.sharpness_penalty_scale * ent_c.sum())

        with pyro.plate("ct_plate", self.ct_count):
            base_p = p_c[self.ct_to_c] + self.eps
            conc_ct = torch.clamp(self.local_scale * base_p, min=1e-3)
            p_ct = pyro.sample("p_ct", dist.Dirichlet(conc_ct))
            if self.sharpness_penalty_scale > 0:
                ent_ct = dist.Dirichlet(conc_ct).entropy()
                pyro.factor("sharpness_penalty_ct", -self.sharpness_penalty_scale * ent_ct.sum())
        # Encoder
        z_loc, z_scale, _ = self.encoder(x, batch_idx)
        z_scale = torch.clamp(z_scale, min=1e-3, max=10.0)
    
        with pyro.plate("data", batch_size) as idx:
            target_pheno = self._target_phenotypes[idx].long()
            ph_emb = self.phenotype_embedding(target_pheno)
    
            new_z_loc = z_loc + ph_emb
    
            # Flexible prior centered at new_z_loc, moderate variance
            latent_prior = dist.Normal(new_z_loc, torch.ones_like(z_scale))
            
            with poutine.scale(scale=kl_weight):
                z = pyro.sample("latent", latent_prior.to_event(1))
    
            # Phenotype classification from the phenotype-conditioned latent mean
            #local_logits = self.classifier(z_loc)
            local_logits = self.classifier(z_loc) + torch.log(p_ct[ct_array[idx]] + 1e-8)

            z_i_phen = pyro.sample(
                "z_i_phen",
                dist.Categorical(logits=local_logits),
                infer={"enumerate": "parallel", "is_auxiliary": True} if self.use_enumeration else {}
            )
    
            # Decoder conditioned on sampled latent and phenotype embedding
            ph_emb_sample = self.phenotype_embedding(z_i_phen)
            combined = torch.cat([z, ph_emb_sample], dim=1)
            px_scale, px_r_out, px_rate, px_dropout = self.decoder("gene", combined, log_library, batch_idx)
    
            gate_probs = torch.sigmoid(px_dropout).clamp(min=1e-3, max=1.0 - 1e-3)
            nb_logits = (px_rate + self.eps).log() - (self.px_r.exp() + self.eps).log()
            nb_logits = torch.clamp(nb_logits, min=-10.0, max=10.0)
            total_count = self.px_r.exp().clamp(max=1e4)
    
            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate=gate_probs,
                total_count=total_count,
                logits=nb_logits,
                validate_args=False
            )
            pyro.sample("obs", x_dist.to_event(1), obs=x)
    
    @auto_move_data
    def guide(self, x: torch.Tensor, batch_idx: torch.Tensor, log_library: torch.Tensor):
        pyro.module("scvi", self)
        kl_weight = self.kl_weight
        batch_size = x.shape[0]
        ct_array = self.ct_array
    
        # Guide distributions for top-level hierarchical parameters (unchanged)
        with pyro.plate("clonotypes", self.c_count):
            q_p_c_raw = pyro.param(
                "q_p_c_raw",
                torch.ones(self.c_count, self.P, device=x.device),
                constraint=dist.constraints.positive
            )
            q_p_c_sharp = q_p_c_raw ** (1.0 / self.sharp_temperature)
            q_p_c_sharp = q_p_c_sharp / q_p_c_sharp.sum(dim=1, keepdim=True)
            conc_c_guide = torch.clamp(self.global_scale * q_p_c_sharp, min=1e-3)
            if self.sharpness_penalty_scale > 0:
                entropy_c_guide = dist.Dirichlet(conc_c_guide).entropy()
                pyro.factor("sharpness_penalty_guide_top", -self.sharpness_penalty_scale * entropy_c_guide.sum())
            pyro.sample("p_c", dist.Dirichlet(conc_c_guide))
    
        with pyro.plate("ct_plate", self.ct_count):
            q_p_ct_raw = pyro.param(
                "q_p_ct_raw",
                torch.ones(self.ct_count, self.P, device=x.device),
                constraint=dist.constraints.positive
            )
            q_p_ct_sharp = q_p_ct_raw ** (1.0 / self.sharp_temperature)
            q_p_ct_sharp = q_p_ct_sharp / q_p_ct_sharp.sum(dim=1, keepdim=True)
            conc_ct_guide = torch.clamp(self.local_scale * q_p_ct_sharp, min=1e-3)
            if self.sharpness_penalty_scale > 0:
                entropy_ct_guide = dist.Dirichlet(conc_ct_guide).entropy()
                pyro.factor("sharpness_penalty_guide_ct", -self.sharpness_penalty_scale * entropy_ct_guide.sum())
            pyro.sample("p_ct", dist.Dirichlet(conc_ct_guide))
    
        # Encoder
        z_loc, z_scale, _ = self.encoder(x, batch_idx)
        z_scale = torch.clamp(z_scale, min=1e-3, max=10.0)
    
        with pyro.plate("data", batch_size) as idx:
            target_pheno = self._target_phenotypes[idx].long()
            ph_emb = self.phenotype_embedding(target_pheno)
    
            new_z_loc = z_loc + ph_emb
    
            with poutine.scale(scale=kl_weight):
                # Flexible posterior distribution around new_z_loc
                latent_posterior = dist.Normal(new_z_loc, z_scale)
                pyro.sample("latent", latent_posterior.to_event(1))
    
            # Phenotype classification from new_z_loc for consistency
            local_logits = self.classifier(z_loc) + torch.log(q_p_ct_sharp[ct_array[idx]] + 1e-8)
            pyro.sample(
                "z_i_phen",
                dist.Categorical(logits=local_logits),
                infer={"enumerate": "parallel", 'is_auxiliary': True} if self.use_enumeration else {}
            )
    
        
    @auto_move_data
    def get_latent(self, tensor_dict: Dict[str, torch.Tensor]):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        batch_idx = tensor_dict[REGISTRY_KEYS.BATCH_KEY].long()
        indices = tensor_dict["indices"].long()
        target_pheno = self._target_phenotypes[indices].long()
        ph_emb = self.phenotype_embedding(target_pheno)
        z_loc, _, _ = self.encoder(x, batch_idx)
        new_z_loc = z_loc + ph_emb
        if new_z_loc.ndim == 3:
            new_z_loc = new_z_loc.mean(dim=1)
        return new_z_loc.cpu()
        
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
        cls_loss_scale: float = 0.0,
        label_smoothing: float = 0.0,
        n_steps_kl_warmup: int = 1000,
        adaptive_margin: bool = False,
        reconstruction_loss_scale: float = 1e-2,
        consistency_scale = 0.1,
        optimizer_config: dict = None,
        **kwargs
    ):
        if module.use_enumeration:
            print("Using Enumeration")
            self._loss_fn = TraceEnum_ELBO(max_plate_nesting=module.ct_count + module.c_count)
        else:
            self._loss_fn = Trace_ELBO()

        super().__init__(module, n_steps_kl_warmup=n_steps_kl_warmup, **kwargs)

        self.margin_scale = margin_scale
        self.margin_value = margin_value
        self.cls_loss_scale = cls_loss_scale
        self.label_smoothing = label_smoothing
        self.adaptive_margin = adaptive_margin
        self.reconstruction_loss_scale = reconstruction_loss_scale

        self._my_global_step = 0
        self.kl_sigmoid_midpoint = 4000
        self.kl_sigmoid_speed = 0.001

        if optimizer_config is None:
            optimizer_config = {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-5,
                "weight_decay": 1e-4,
            }
        self.optimizer_config = optimizer_config
        self.consistency_scale = consistency_scale

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
        kl_weight = 5.0 / (1.0 + np.exp(-0.005 * (self._my_global_step - 2000)))

        self.module.kl_weight = kl_weight

        loss_dict = super().training_step(batch, batch_idx)
        device = next(self.module.parameters()).device

        # Ensure loss is on correct device
        if not isinstance(loss_dict["loss"], torch.Tensor):
            loss_dict["loss"] = torch.tensor(loss_dict["loss"], device=device, requires_grad=True)
        else:
            loss_dict["loss"] = loss_dict["loss"].to(device)

        z_batch = self.module.get_latent(batch).to(device)
        idx = batch["indices"].long().view(-1).to(device)
        target_phen = self.module._target_phenotypes[idx].to(device)

        # margin
        margin_val = 0.0
        if self.margin_scale > 0.0:
            margin_loss_val = pairwise_centroid_margin_loss(
                z_batch, target_phen,
                margin=self.margin_value,
                adaptive_margin=self.adaptive_margin
            ).to(device)
            margin_loss = self.margin_scale * margin_loss_val
            loss_dict["loss"] += margin_loss
            margin_val = margin_loss_val.item()

        # classification
        cls_val = 0.0
        if self.cls_loss_scale > 0.0:
            # Retrieve indices for cell-level clonotype-timepoint mapping
            ct_indices = self.module.ct_array[idx].to(device)

            # Retrieve the prior clonotype-to-phenotype probabilities for this batch
            prior_probs = self.module.get_p_ct()[ct_indices].to(device)

            # Compute the classifier logits
            cls_logits = self.module.classifier(z_batch)

            # Explicitly incorporate log-priors into the logits
            cls_logits_with_prior = cls_logits + torch.log(prior_probs + 1e-8)

            cls_loss_val = F.cross_entropy(
                cls_logits_with_prior, target_phen,
                label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0
            ).to(device)

            cls_loss = self.cls_loss_scale * cls_loss_val
            loss_dict["loss"] += cls_loss
            cls_val = cls_loss_val.item()

        # reconstruction
        x = batch[REGISTRY_KEYS.X_KEY].float().to(device)
        batch_idx_tensor = batch[REGISTRY_KEYS.BATCH_KEY].long().to(device)
        log_library = torch.log(torch.sum(x, dim=1, keepdim=True) + 1e-6).to(device)
        ph_emb_sample = self.module.phenotype_embedding(target_phen)
        combined = torch.cat([z_batch, ph_emb_sample], dim=1)
        px_scale, px_r_out, px_rate, px_dropout = self.module.decoder("gene", combined, log_library, batch_idx_tensor)

        gate_probs = torch.sigmoid(px_dropout).clamp(min=1e-3, max=1 - 1e-3)
        nb_logits = (px_rate + self.module.eps).log() - (self.module.px_r.exp() + self.module.eps).log()
        nb_logits = torch.clamp(nb_logits, min=-10.0, max=10.0)
        total_count = self.module.px_r.exp().clamp(max=1e4)
        
        # Define ZINB distribution
        x_dist = dist.ZeroInflatedNegativeBinomial(
            gate=gate_probs,
            total_count=total_count,
            logits=nb_logits,
            validate_args=False
        )
        
        # Compute negative log-likelihood reconstruction loss
        reconstruction_loss_val = -x_dist.log_prob(x).mean()
        
        # Scale and update loss
        total_recon_loss = self.reconstruction_loss_scale * reconstruction_loss_val
        loss_dict["loss"] += total_recon_loss
        recon_val = reconstruction_loss_val.item()

        
        with torch.no_grad():
            if self.cls_loss_scale > 0.0:
                preds = cls_logits_with_prior.argmax(dim=1)
                acc = (preds == target_phen).float().mean().item()
            else:
                acc = 0.0

        consistency_loss = F.kl_div(
            F.log_softmax(cls_logits_with_prior, dim=-1),
            prior_probs,
            reduction='batchmean'
        )

        loss_dict["loss"] += self.consistency_scale * consistency_loss

        cont_loss_val = continuity_loss(z_batch, target_phen)
        cont_loss_scale = 0.1  # moderate continuity encouragement
        loss_dict["loss"] += cont_loss_scale * cont_loss_val
        loss_dict["continuity_loss"] = cont_loss_val.item()
        
        loss_dict["margin_loss"] = margin_val
        loss_dict["cls_loss"] = cls_val
        loss_dict["reconstruction_loss"] = recon_val
        loss_dict["classification_accuracy"] = acc

        self._my_global_step += 1
        return loss_dict

    # ------------ VALIDATION STEP for scvi early stopping --------------
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

        # Explicitly ensure metric is always logged as "elbo_validation"
        self.log("elbo_validation", val_dict["loss"], prog_bar=True, on_epoch=True, sync_dist=True)

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
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
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
        sharpness_penalty_scale: float = 0.0,
        use_enumeration: bool = False,
        consistency_scale=0.1,
        **kwargs
    ):
        super().__init__(adata)
        n_vars = self.summary_stats["n_vars"]
        clonotype_col = self.adata_manager.registry["clonotype_col"]
        phenotype_col = self.adata_manager.registry["phenotype_col"]
        covariate_col = self.adata_manager.registry["covariate_col"]
        batch_col = self.adata_manager.registry["batch_col"]


        self.consistency_scale = consistency_scale
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
        
        self.module = TCRIModule(
            n_input=n_vars,
            n_latent=n_latent,
            P=P,
            n_batch=n_batch,
            n_hidden=n_hidden,
            global_scale=global_scale,
            local_scale=local_scale,
            sharp_temperature=sharp_temperature,
            sharpness_penalty_scale=sharpness_penalty_scale,
            use_enumeration=use_enumeration,
        )
        self.init_params_ = self._get_init_params(locals())
        
        c2p_torch = torch.tensor(c2p_mat, dtype=torch.float32)
        c_array_torch = torch.tensor(c_array_np, dtype=torch.long)
        ct_array_torch = torch.tensor(ct_array_np, dtype=torch.long)
        ct_to_c_torch = torch.tensor(ct_to_c_list, dtype=torch.long)
        ct_to_cov_torch = torch.tensor(ct_to_cov_list, dtype=torch.long)
        
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
            f"sharp_temperature={sharp_temperature}, sharpness_penalty_scale={sharpness_penalty_scale}."
        )

    def train(
        self,
        max_epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        margin_scale: float = 0.0,
        margin_value: float = 2.0,
        cls_loss_scale: float = 0.0,
        label_smoothing: float = 0.0,
        adaptive_margin: bool = False,
        reconstruction_loss_scale: float = 1e-2,
        n_steps_kl_warmup: int = 1000,
        **kwargs
    ):
        """
        We split the data into train/val, define a UnifiedTrainingPlan with
        validation_step, and let scvi handle early stopping automatically
        by passing early_stopping parameters to TrainRunner.
        """
        # Create a train/val split
        splitter = DataSplitter(
            self.adata_manager,
            train_size=0.9,        # e.g. 90% train, 10% validation
            validation_size=None,
            batch_size=batch_size,
        )

        plan = UnifiedTrainingPlan(
            module=self.module,
            margin_scale=margin_scale,
            margin_value=margin_value,
            cls_loss_scale=cls_loss_scale,
            label_smoothing=label_smoothing,
            n_steps_kl_warmup=n_steps_kl_warmup,
            adaptive_margin=adaptive_margin,
            reconstruction_loss_scale=reconstruction_loss_scale,
            consistency_scale=self.consistency_scale,
            optimizer_config={
                "lr": lr,
                "betas": (0.9, 0.999),
                "eps": 1e-5,
                "weight_decay": 1e-4,
            },
        )

        # Use scvi's TrainRunner with built-in early stopping
        runner = TrainRunner(
            self,
            training_plan=plan,
            data_splitter=splitter,
            max_epochs=max_epochs,
            # Pass scvi early stopping arguments:
            early_stopping=True,
            early_stopping_monitor="elbo_validation",
            early_stopping_patience=30,
            early_stopping_mode="min",
            check_val_every_n_epoch=5,
            accelerator="auto",  # if GPU available, use it
            devices="auto",
            **kwargs
        )

        runner()
        return

    @torch.no_grad()
    def get_latent_representation(self, adata=None, indices=None, batch_size=None):
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latents = [self.module.get_latent(tensors) for tensors in scdl]
        return torch.cat(latents, dim=0).cpu().numpy()

    @torch.no_grad()
    def get_p_ct(self):
        return self.module.get_p_ct().cpu().numpy()

    @torch.no_grad()
    def get_cell_phenotype_probs(self, adata=None, batch_size: int = 256, eps: float = 1e-8):
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, batch_size=batch_size)
        device = next(self.module.parameters()).device

        p_ct = self.module.get_p_ct().to(device)          # Learned clonotype-covariate posterior
        ct_array = self.module.ct_array.to(device)        # Cell clonotype-covariate indices

        all_probs = []
        current_idx = 0
        for tensors in scdl:
            batch_size_local = tensors[REGISTRY_KEYS.X_KEY].shape[0]

            ct_indices = ct_array[current_idx: current_idx + batch_size_local]
            clone_cov_posterior = p_ct[ct_indices]  # (batch_size_local, P)

            # Use pure latent representation (no phenotype embedding here)
            z_loc, _, _ = self.module.encoder(tensors[REGISTRY_KEYS.X_KEY].to(device),
                                            tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device))

            # Retrieve the prior clonotype-to-phenotype probabilities for this batch
            prior_probs = self.module.get_p_ct()[ct_indices].to(device)

            # Compute the classifier logits
            cls_logits = self.module.classifier(z_loc)

            # Explicitly incorporate log-priors into the logits
            cls_logits_with_prior = cls_logits + torch.log(prior_probs + 1e-8)
            cell_level_likelihood = F.softmax(cls_logits_with_prior, dim=-1)

            # Bayesian posterior explicitly (prior * likelihood)
            posterior_unnormalized = clone_cov_posterior * cell_level_likelihood
            probs = posterior_unnormalized / (posterior_unnormalized.sum(dim=-1, keepdim=True) + eps)

            all_probs.append(probs.cpu())
            current_idx += batch_size_local

        return torch.cat(all_probs, dim=0).numpy()

