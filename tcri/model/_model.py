"""User-facing scvi-tools model class :class:`TCRIModel`.

The generative model, priors, classifier, and training plan live in sibling
modules; this file holds only the high-level `BaseModelClass` API
(`setup_anndata`, `__init__`, `train`, `get_latent_representation`,
`get_cell_phenotype_probs`, `get_p_ct`, ...):

- :mod:`._module`     -- Pyro model/guide (:class:`TCRIModule`)
- :mod:`._priors`     -- :class:`MixtureDirichlet`, :class:`VampPrior`
- :mod:`._classifier` -- :class:`PhenotypeClassifier`
- :mod:`._training`   -- :class:`UnifiedTrainingPlan`, :func:`build_archetypes`
"""
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Dict, Optional
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from scvi.dataloaders import DataSplitter

from ._module import TCRIModule
from ._training import UnifiedTrainingPlan, build_archetypes
# re-exported so the public `tcri.model.*` surface is unchanged by the split
from ._priors import MixtureDirichlet, VampPrior  # noqa: F401
from ._classifier import PhenotypeClassifier  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning, message="Found auxiliary vars")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*enumerate.*TraceEnum_ELBO.*"
)

os.environ.pop("SLURM_NTASKS", None)
os.environ.pop("SLURM_NTASKS_PER_NODE", None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCRIModel(BaseModelClass):
    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
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
        if layer is None:
            adata.uns.pop("tcri_layer", None)
        else:
            adata.uns["tcri_layer"] = layer
        return adata

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 128,
        n_hidden: int = 128,
        n_layers: int = 3,
        classifier_n_layers: int = 3,
        global_scale: float = 5.0,
        local_scale: float = 3.0,
        prior_temperature: float = 1.0,
        guide_temperature: float = 1.0,
        use_enumeration: bool = False,
        patience: int = 300,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.1,
        n_pseudo_obs: int = 10,
        K: int = 10,
        phenotype_weights: Optional[Dict[str, float]] = None,
        gate_prob: Optional[float] = None,
        kl_weight_max: float = 1.0,
        guide_init_scale: float = 10.0,
        classifier_temperature: float = 1.0,
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
        # ---- TCRIModel.__init__ -------------
        if gate_prob is not None and not (0.0 <= gate_prob <= 1.0):
            raise ValueError("gate_prob must be in [0,1] or None")

        cvals = self.adata.obs[clonotype_col].astype("category")
        c_count = len(cvals.cat.categories)
        c_array_np = cvals.cat.codes.values
        pvals_np = ph_series.cat.codes.values
        clone_phenotype_prior = np.zeros((c_count, P), dtype=np.float32)
        for i in range(len(c_array_np)):
            clone_phenotype_prior[c_array_np[i], pvals_np[i]] += 1
        clone_phenotype_prior += 1e-6
        clone_phenotype_prior = clone_phenotype_prior / clone_phenotype_prior.sum(axis=1, keepdims=True)
        self.clone_phenotype_prior = clone_phenotype_prior
        self.centers, self.labels = build_archetypes(self.clone_phenotype_prior, K=K)
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
            n_layers=n_layers,
            classifier_n_layers=classifier_n_layers,
            global_scale=global_scale,
            local_scale=local_scale,
            mixture_concentration=torch.from_numpy(self.centers),
            prior_temperature=prior_temperature,
            guide_temperature=guide_temperature,
            use_enumeration=use_enumeration,
            classifier_hidden=classifier_hidden,
            classifier_dropout=classifier_dropout,
            class_weights=self.class_weights,
            gate_prob=gate_prob,
            kl_weight_max=kl_weight_max,
            n_pseudo_obs=n_pseudo_obs,
            guide_init_scale=guide_init_scale,
            classifier_temperature=classifier_temperature,
        )
        self.init_params_ = self._get_init_params(locals())
        c2p_torch = torch.tensor(clone_phenotype_prior, dtype=torch.float32)
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
            f"prior_temperature={prior_temperature}, guide_temperature={guide_temperature}."
        )

    def train(
        self,
        max_epochs: int = 1000,
        batch_size: int = 1000,
        lr: float = 1e-3,
        reconstruction_loss_scale: float = 1e-3,
        n_steps_kl_warmup: int = 2000,
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
            n_steps_kl_warmup=n_steps_kl_warmup,
            reconstruction_loss_scale=reconstruction_loss_scale,
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

    @property
    def use_gate(self) -> bool:
        return self.module.use_gate

    @torch.no_grad()
    def get_p_ct(self):
        return self.module.get_p_ct().cpu().numpy()

    @torch.no_grad()
    def get_cell_phenotype_probs(
        self, adata=None, batch_size: int = 256, eps: float = 1e-8
    ) -> np.ndarray:
        """
        Computes the cell-level phenotype probabilities in the same way as training.

        If ``self.module.gate_prob`` is set (i.e. ``use_gate`` is True), uses:
            local_logits = gate_prob * cls_logits + (1 - gate_prob) * log(prior)
        Otherwise uses the additive (Bayesian product) rule:
            local_logits = cls_logits + log(prior)

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
            cls_logits = self.module.classifier(z_loc)

            prior_log = torch.log(clone_cov_posterior + eps)
            if self.module.use_gate:
                local_logits = self.module.gate_prob * cls_logits + (1.0 - self.module.gate_prob) * prior_log
            else:
                local_logits = cls_logits + prior_log

            probs = F.softmax(local_logits, dim=-1) 

            all_probs.append(probs.cpu())
            current_idx += this_batch_size

        # Concatenate into final array of shape (n_cells, P)
        return torch.cat(all_probs, dim=0).numpy()

    def boost_phenotype_prior(
        self,
        phenotype_name       : str,
        boost_factor         : float = 5.0,
        *,
        affect_mixture       : bool  = True,
    ):
        GRN, YLW, MAG, RST = "\x1b[32m", "\x1b[33m", "\x1b[35m", "\x1b[0m"
        def _ok(m):   print(f"{GRN}✅ {m}{RST}")
        cats = self.adata.obs[ self.adata_manager.registry["phenotype_col"] ]\
                    .astype("category").cat.categories
        if phenotype_name not in cats:
            raise ValueError(f"phenotype '{phenotype_name}' not found. Choices: {list(cats)}")
        p_idx = list(cats).index(phenotype_name)

        # 2) clone-level prior  (numpy array stored in model.clone_phenotype_prior)
        mat = self.clone_phenotype_prior.copy()
        mat[:, p_idx] *= boost_factor
        mat /= mat.sum(axis=1, keepdims=True)
        self.clone_phenotype_prior = mat                                    # keep external copy

        with torch.no_grad():
            new_clone_prior = torch.tensor(mat, dtype=torch.float32,
                                        device=self.module.clone_phen_prior.device)
            self.module.clone_phen_prior.data = new_clone_prior

        _ok(f"Clone-level prior boosted ×{boost_factor:g} for '{phenotype_name}'")

        if affect_mixture:
            centres = self.centers.copy()
            centres[:, p_idx] *= boost_factor
            centres /= centres.sum(axis=1, keepdims=True)
            self.centers = centres

            with torch.no_grad():
                new_mix = torch.tensor(centres, dtype=torch.float32,
                                    device=self.module.mixture_concentration.device)
                self.module.mixture_concentration.data = new_mix

            _ok(f"Mixture prior boosted ×{boost_factor:g} for '{phenotype_name}'")

        _ok("Read to train.")

    def plot_archetypes(self):
        order = np.argsort(self.labels)
        ordered_mat = self.clone_phenotype_prior[order, :]

        # Plot heatmap of the clone phenotype distributions
        plt.figure(figsize=(10, 6))
        plt.imshow(ordered_mat, aspect='auto', cmap='viridis')
        plt.colorbar(label='Phenotype Distribution')
        plt.title('Heatmap of Clone Phenotype Distributions (Ordered by Cluster)')
        plt.xlabel('Phenotype')
        plt.ylabel('Clone (ordered by cluster)')
        plt.show()

        # Plot heatmap of the archetype centroids
        plt.figure(figsize=(6, 4))
        plt.imshow(self.centers, aspect='auto', cmap='viridis')
        plt.colorbar(label='Centroid Value')
        plt.title('Heatmap of Archetype Centroids')
        plt.xlabel('Phenotype')
        plt.ylabel('Archetype')
        plt.show()

    def plot_loss(self, log_scale=False):
        # Retrieve loss and accuracy history
        loss_history = self.history_.get("elbo_train", [])
        loss_validation = self.history_.get("elbo_validation", [])
        train_accuracy = self.history_.get("kl_divergence_with_prior_train_epoch", [])
        val_accuracy = self.history_.get("kl_divergence_with_prior_val", [])

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))

        # Plot ELBO loss
        axes[0].plot(loss_history, label="Training ELBO Loss")
        axes[0].plot(loss_validation, label="Validation ELBO Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("ELBO Loss")
        axes[0].set_title("ELBO Loss Over Epochs")
        axes[0].legend()

        # Plot Accuracy
        if len(train_accuracy) > 0 or len(val_accuracy) > 0:
            if len(train_accuracy) > 0:
                axes[1].plot(train_accuracy, label="Training Accuracy")
            if len(val_accuracy) > 0:
                axes[1].plot(val_accuracy, label="Validation Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("dKL")
            axes[1].set_title("DKL Over Epochs")
            axes[1].legend()

        # Apply log scale if requested
        if log_scale:
            for ax in axes:
                ax.set_yscale("log")

        plt.tight_layout()
        plt.show()
