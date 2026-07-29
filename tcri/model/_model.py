"""User-facing scvi-tools model class :class:`TCRIModel`.

The generative model, priors, classifier, and training plan live in sibling
modules; this file holds only the high-level `BaseModelClass` API
(`setup_anndata`, `__init__`, `train`, `get_latent_representation`,
`predict`, `to_anndata`, `get_p_ct`, ...):

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
import pyro
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from scvi.dataloaders import DataSplitter

from ._module import TCRIModule
from ._training import UnifiedTrainingPlan, build_archetypes

__all__ = ["TCRIModel"]

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
        *,
        layer: Optional[str] = None,
        clonotype_key: str = "unique_clone_id",
        phenotype_key: str = "phenotype_col",
        covariate_key: str = "timepoint",
        batch_key: str = "patient",
        **kwargs,
    ) -> None:
        """Register clonotype/phenotype/covariate/batch/count fields with scvi.

        Registration only. Writes ``obs['indices']`` (scvi glue that the
        training/validation steps consume via ``batch['indices']``) and records the
        layer, but performs **no** analysis/label ``obs`` mutation and does **not**
        stash the ``AnnDataManager`` in ``uns`` (the retired ``tcri_manager`` hack) —
        learned outputs are written solely by :meth:`to_anndata`.
        """
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
        if layer is None:
            adata.uns.pop("tcri_layer", None)
        else:
            adata.uns["tcri_layer"] = layer

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
        gate_prob: Optional[float] = 0.5,  # π (gating weight, methods §Generative Model); None = additive
        kl_weight_max: float = 1.0,
        guide_init_scale: float = 10.0,
        classifier_temperature: float = 1.0,
        phenotype_kl_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(adata)

        # Pyro's param store is PROCESS-GLOBAL: a second TCRIModel in the same session
        # silently inherits the first model's fitted q_p_c_raw/q_p_ct_raw and network
        # weights, so it starts from the previous fit instead of from scratch. We warn
        # rather than clearing, because clearing here would destroy the params of a
        # model loaded earlier in the session (load_tcri_session restores the store
        # after construction). Proper per-instance namespacing is a design change.
        _tcri_params = [k for k in pyro.get_param_store().keys() if k.startswith(("q_p_c", "q_p_ct", "scvi$$$"))]
        if _tcri_params:
            warnings.warn(
                "The global Pyro param store already holds TCRI parameters "
                f"({len(_tcri_params)} entries). This model will CONTINUE that fit "
                "rather than start fresh. Call `pyro.clear_param_store()` before "
                "constructing a new model (note this invalidates any model already "
                "loaded in this session).",
                UserWarning,
                stacklevel=2,
            )

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
        # K archetypes are k-means centroids over clonotypes, so K > n_clonotypes is
        # not satisfiable (sklearn raises "n_samples < n_clusters"). Clamp instead of
        # crashing — a dataset with few clones is legitimate (and is exactly what the
        # synthetic examples use).
        if K > c_count:
            warnings.warn(
                f"K={K} archetypes requested but the data has only {c_count} "
                f"clonotype(s); using K={c_count}.",
                UserWarning,
                stacklevel=2,
            )
            K = c_count
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
            gate_prob=gate_prob,
            kl_weight_max=kl_weight_max,
            n_pseudo_obs=n_pseudo_obs,
            guide_init_scale=guide_init_scale,
            classifier_temperature=classifier_temperature,
            phenotype_kl_weight=phenotype_kl_weight,
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

        # batch_size >= n_obs means ONE optimizer step per epoch, so the fixed
        # per-epoch overhead is paid per gradient update — the pathology behind the
        # "9-hour" synthetic run (1000 cells, batch_size=20000, max_epochs=1e6).
        n_obs = self.adata.n_obs
        if batch_size >= n_obs:
            warnings.warn(
                f"batch_size={batch_size} >= n_obs={n_obs}: each epoch is a SINGLE "
                "optimizer step, so per-epoch overhead dominates and `max_epochs` "
                "becomes the number of gradient updates. Use a smaller batch_size "
                "(e.g. 256-1024) for a comparable number of updates in far less time.",
                UserWarning,
                stacklevel=2,
            )

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
            optimizer_config={
                "lr": lr,
                "betas": (0.9, 0.999),
                "eps": 1e-5,
                "weight_decay": 1e-4,
            },
        )

        # Defaults the caller can override: setdefault (not hard-coded keywords) so
        # passing e.g. early_stopping_patience=10 or accelerator="gpu" through
        # train(**kwargs) works instead of raising "got multiple values for keyword".
        kwargs.setdefault("early_stopping", True)
        kwargs.setdefault("early_stopping_monitor", "elbo_validation")
        kwargs.setdefault("early_stopping_mode", "min")
        kwargs.setdefault("early_stopping_patience", self.patience)
        kwargs.setdefault("check_val_every_n_epoch", 5)
        kwargs.setdefault("accelerator", "auto")
        kwargs.setdefault("devices", "auto")

        runner = TrainRunner(
            self,
            training_plan=plan,
            data_splitter=splitter,
            max_epochs=max_epochs,
            **kwargs,
        )

        runner()
        return

    @torch.no_grad()
    def get_latent_representation(self, adata=None, indices=None, batch_size=4096):
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
    def predict(self, adata=None, *, batch_size: int = 4096, eps: float = 1e-8) -> pd.DataFrame:
        """Per-cell phenotype-probability ``DataFrame`` (index ``adata.obs_names``,
        columns = phenotypes) — the single source of phenotype probabilities.

        Combines classifier logits with ``log p_ct`` exactly as in training: if
        ``self.module.gate_prob`` is set (``use_gate``),
        ``gate_prob * cls_logits + (1 - gate_prob) * log(prior)``; otherwise the
        additive rule ``cls_logits + log(prior)``. Renamed from
        ``get_cell_phenotype_probs`` (which returned a bare ``ndarray``). The module
        is put in ``eval`` mode so classifier dropout is off and the result is
        deterministic; the sequential loader keeps the row order aligned to
        ``obs_names``.
        """
        adata = self._validate_anndata(adata)
        self.module.eval()
        device = next(self.module.parameters()).device
        scdl = self._make_data_loader(adata=adata, batch_size=batch_size)

        p_ct = self.module.get_p_ct().to(device)
        ct_array = self.module.ct_array.to(device)

        all_probs = []
        current_idx = 0
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
            n = x.shape[0]
            clone_cov_posterior = p_ct[ct_array[current_idx : current_idx + n]]
            z_loc, _, _ = self.module.encoder(x, b)
            cls_logits = self.module.classifier(z_loc)
            prior_log = torch.log(clone_cov_posterior + eps)
            if self.module.use_gate:
                local_logits = self.module.gate_prob * cls_logits + (1.0 - self.module.gate_prob) * prior_log
            else:
                local_logits = cls_logits + prior_log
            all_probs.append(F.softmax(local_logits, dim=-1).cpu())
            current_idx += n

        probs = torch.cat(all_probs, dim=0).numpy()
        phenotype_col = self.adata_manager.registry["phenotype_col"]
        pheno_cats = self.adata.obs[phenotype_col].astype("category").cat.categories.tolist()
        return pd.DataFrame(probs, index=adata.obs_names, columns=pheno_cats)

    @torch.no_grad()
    def to_anndata(self, adata=None, *, batch_size: int = 4096, compute_umap: bool = False) -> AnnData:
        """Write the model's learned state onto ``adata`` under the canonical
        ``tcri_*`` keys (from :mod:`tcri._keys`) and return it. Replaces the old
        ``preprocessing.register_model``; writes no manager stash.

        Writes — ``uns``: ``METADATA`` + covariate/clonotype/phenotype categories,
        ``P_CT`` (posterior-mean ``p_ct``), ``CT_TO_COV``/``CT_TO_C``, per-cell
        ``CT_ARRAY``/``COV_ARRAY``, ``LOCAL_SCALE``, ``GATE_PROB``,
        ``CLASSIFIER_TEMPERATURE``; ``obsm``: ``X_TCRI`` latent, ``X_LOGITS``,
        ``X_LOGPOSTERIOR``, ``X_PROBABILITIES`` (from :meth:`predict`); ``obs``:
        ``PHENOTYPE`` argmax hard label.
        """
        from .. import _keys as K

        adata = self._validate_anndata(adata)
        self.module.eval()
        device = next(self.module.parameters()).device
        reg = self.adata_manager.registry

        # 1) metadata + category orders (order = training) --------------------
        meta = {
            K.COVARIATE_COL: reg["covariate_col"],
            K.CLONE_COL: reg["clonotype_col"],
            K.PHENOTYPE_COL: reg["phenotype_col"],
            K.BATCH_COL: reg["batch_col"],
        }
        adata.uns[K.METADATA] = meta
        for col_key, cat_key in (
            (K.COVARIATE_COL, K.COVARIATE_CATEGORIES),
            (K.CLONE_COL, K.CLONOTYPE_CATEGORIES),
            (K.PHENOTYPE_COL, K.PHENOTYPE_CATEGORIES),
        ):
            adata.uns[cat_key] = adata.obs[meta[col_key]].astype("category").cat.categories.tolist()

        # 2) learned priors + per-cell index arrays --------------------------
        ct_arr = self.module.ct_array.cpu().numpy()
        adata.uns[K.P_CT] = self.module.get_p_ct().cpu().numpy()
        adata.uns[K.CT_TO_COV] = self.module.ct_to_cov.cpu().numpy()
        adata.uns[K.CT_TO_C] = self.module.ct_to_c.cpu().numpy()
        adata.uns[K.CT_ARRAY] = ct_arr
        adata.uns[K.COV_ARRAY] = self.module.ct_to_cov.cpu().numpy()[ct_arr]
        adata.uns[K.LOCAL_SCALE] = float(self.module.local_scale)
        gp = self.module.gate_prob
        adata.uns[K.GATE_PROB] = float(gp) if gp is not None else float("nan")
        adata.uns[K.CLASSIFIER_TEMPERATURE] = float(self.module.classifier_temperature)

        # 3) latent mean -----------------------------------------------------
        adata.obsm[K.X_TCRI] = self.get_latent_representation(
            adata=adata, batch_size=batch_size
        ).astype("float32")

        # 4) per-cell logits + additive log-posterior (folds _compute_logits_and_prior)
        loader = self._make_data_loader(adata=adata, batch_size=batch_size)
        p_ct_t = self.module.get_p_ct().to(device)
        ct_arr_t = self.module.ct_array.to(device)
        logits_buf, prior_buf = [], []
        start = 0
        for tensors in loader:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
            n = x.shape[0]
            z_loc, _, _ = self.module.encoder(x, b)
            logits_buf.append(self.module.classifier(z_loc).cpu())
            prior_buf.append(torch.log(p_ct_t[ct_arr_t[start : start + n]] + 1e-8).cpu())
            start += n
        cls_logits = torch.cat(logits_buf).numpy().astype("float32")
        prior_log = torch.cat(prior_buf).numpy().astype("float32")
        adata.obsm[K.X_LOGITS] = cls_logits
        adata.obsm[K.X_LOGPOSTERIOR] = cls_logits + prior_log

        # 5) probabilities (gate-aware, canonical) + argmax hard labels ------
        probs_df = self.predict(adata, batch_size=batch_size)
        adata.obsm[K.X_PROBABILITIES] = probs_df.values.astype("float32")
        adata.obs[K.PHENOTYPE] = pd.Categorical.from_codes(
            probs_df.values.argmax(1), categories=list(probs_df.columns)
        )

        # 6) legacy compat keys — retired in Phase 6/7 with their readers -----
        adata.uns[K.LEGACY_CLONE_KEY] = meta[K.CLONE_COL]
        adata.uns[K.LEGACY_PHENOTYPE_KEY] = K.PHENOTYPE

        if compute_umap:
            import umap
            adata.obsm[K.X_UMAP] = umap.UMAP(random_state=42).fit_transform(adata.obsm[K.X_TCRI])

        return adata

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


