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
import contextlib
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

from .._state import keys as K
from ._module import TCRIModule
from ._callbacks import BestObjectiveSnapshot, RampGatedEarlyStopping, ramp_is_complete
from ._training import UnifiedTrainingPlan, build_archetypes

#: The early-stopping criterion fixed by training-contract I3. NOT an ELBO -- it is the
#: per-cell block only, at a pinned kl_weight_max under a fixed evaluation seed. The name
#: says so on purpose; calling it an ELBO is what let the old monitor look well-posed.
MONITOR = "objective_validation_percell"

__all__ = ["TCRIModel"]

warnings.filterwarnings("ignore", category=UserWarning, message="Found auxiliary vars")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*enumerate.*TraceEnum_ELBO.*"
)

# NEW-5: neither of these belongs at import time.
#
# Deleting SLURM_NTASKS/SLURM_NTASKS_PER_NODE from os.environ stops Lightning's SLURM
# auto-detection from hijacking the trainer -- but doing it on `import tcri` mutates the
# environment for the WHOLE process, so anything else that sizes work from those variables
# (a joblib/dask pool, a subprocess srun, another Trainer) silently sees them missing. It is
# now scoped to train() and restored afterwards; see _slurm_autodetect_disabled.
#
# logging.basicConfig() configures the ROOT logger, which is the application's decision, not
# a library's. Importing tcri would switch on INFO logging for everything in the process. A
# library attaches a NullHandler and leaves configuration to the caller.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@contextlib.contextmanager
def _slurm_autodetect_disabled():
    """Hide SLURM_NTASKS* from Lightning for the duration of a fit, then put them back."""
    keys = ("SLURM_NTASKS", "SLURM_NTASKS_PER_NODE")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


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
        replicate: Optional[str] = None,
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
        # `replicate` names the independent unit for statistics -- the column a metric uses
        # when `groupby` is left implicit. Registering it once here means it is not retyped at
        # every call, and it is recorded as the EFFECTIVE value of groupby in each result's
        # provenance.
        #
        # Deliberately separate from batch_key. scvi's batch_key conditions the encoder and
        # decoder (one-hot into every hidden layer), which is a modelling decision about what
        # to correct for; `replicate` is a claim about what counts as an independent
        # observation. They coincide when batches are patients and diverge the moment they are
        # sequencing runs -- at which point deriving one from the other silently gives the
        # wrong n.
        if replicate is not None and replicate not in adata.obs:
            raise ValueError(
                f"replicate={replicate!r} is not a column of adata.obs. It names the "
                f"independent unit for statistics (usually the patient), and is used as the "
                f"default `groupby` for every metric."
            )
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
        adata_manager.registry[K.Config.REPLICATE] = replicate
        adata_manager.registry[K.Config.LAYER] = layer
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
        patience_epochs: int = 300,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.1,
        n_pseudo_obs: int = 10,
        K: int = 10,
        gate_prob: Optional[float] = 0.5,  # π (gating weight, methods §Generative Model); None = additive
        kl_weight_max: float = 1.0,
        guide_init_scale: float = 10.0,
        classifier_temperature: float = 1.0,
        phenotype_kl_weight: float = 1.0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(adata)

        # DE-19: network init and minibatch order were unseeded, so two fits with the same
        # nominal seed differed by ~1.8e-3 in reported NMI -- larger than the effect of several
        # defects being fixed in this stack, which makes those effects unmeasurable from a
        # paired fit. `seed` is on __init__ rather than train() deliberately: the networks are
        # built here, so seeding in train() would be too late, and adding it to train() would
        # be an API-contract change to _contract.pyi.
        self._seed = int(seed) if seed is not None else None
        self._n_train_calls = 0
        if self._seed is not None:
            self._apply_seed(self._seed)

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

        # re-seed immediately before the networks are constructed, so construction is
        # deterministic regardless of any RNG consumed by the setup above
        if self._seed is not None:
            self._apply_seed(self._seed)

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
        # B3: patience is counted in EPOCHS, guaranteed by check_val_every_n_epoch=1 in
        # train(). `patience` is accepted as a deprecated alias -- it appears in init_params_,
        # so dropping it outright breaks load_tcri_session() on every previously saved model.
        if "patience" in kwargs:
            legacy = int(kwargs.pop("patience"))
            if patience_epochs != 300:
                raise TypeError(
                    "pass either `patience_epochs` or the deprecated `patience`, not both"
                )
            patience_epochs = legacy
            warnings.warn(
                "`patience` is deprecated; use `patience_epochs`. The unit is now epochs, "
                "enforced by check_val_every_n_epoch=1. Previously it was counted in "
                "VALIDATION CHECKS, so patience=300 under the old cv=5 meant 1500 epochs -- "
                "more than the default max_epochs of 1000, which is why early stopping could "
                "never fire.",
                FutureWarning, stacklevel=2,
            )
        self.patience_epochs = int(patience_epochs)
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

    @staticmethod
    def _apply_seed(seed: int) -> None:
        """Seed every RNG that touches a fit: python, numpy, torch (incl. CUDA), and pyro.

        `lightning.seed_everything(workers=True)` also seeds dataloader workers, which is what
        makes minibatch order reproducible; `pyro.set_rng_seed` covers the param-store
        initialisers and the Dirichlet draws. Both are needed -- neither alone is sufficient.
        """
        import lightning.pytorch as _pl
        import scvi as _scvi

        _pl.seed_everything(seed, workers=True, verbose=False)
        pyro.set_rng_seed(seed)
        # DE-19: scvi builds the train/val split from `np.random.RandomState(scvi.settings.seed)`
        # (scvi/dataloaders/_data_splitting.py), and that setting defaults to None -- i.e. OS
        # entropy. Seeding lightning and pyro is NOT sufficient: which cells land in the
        # validation split stayed random, leaving a ~1.6e-3 spread between "identical" fits.
        # The determinism test only passed because tests/test_model_classifier.py happens to set
        # scvi.settings.seed process-globally earlier in the run.
        _scvi.settings.seed = seed

    def train(
        self,
        max_epochs: int = 1000,
        batch_size: int = 1000,
        lr: float = 1e-3,
        reconstruction_loss_scale: float = 1e-2,
        n_steps_kl_warmup: int = 2000,
        **kwargs,
    ):
        """
        We split the data into train/val, define a UnifiedTrainingPlan with
        validation_step, and let scvi handle early stopping automatically
        by passing early_stopping parameters to TrainRunner.
        """
        # Re-seed per call, offset by the call index: a second train() on the same model is
        # reproducible without being a bit-for-bit replay of the first.
        if self._seed is not None:
            self._apply_seed(self._seed + self._n_train_calls)
        self._n_train_calls += 1

        # DE-4: the KL ramp is carried on the module, so a second train() CONTINUES the
        # schedule instead of restarting it. There is deliberately no reset knob -- adding one
        # to train() would be an API-contract change to _contract.pyi, and restarting the ramp
        # is the behaviour this defect removes. Construct a new model for a fresh schedule.

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
        # passing e.g. accelerator="gpu" through train(**kwargs) works instead of raising
        # "got multiple values for keyword".
        #
        # B3: check_val_every_n_epoch=1 so that a check IS an epoch and `patience_epochs` means
        # what it says. Deliberately not solved by dividing patience at the call site -- two
        # units with a silent conversion between them is the same trap in a new place. Costs a
        # measured +7.6% wall clock on the worst-case fixture (2 training batches per epoch).
        #
        # Historic note: the old defaults made early stopping unreachable. patience=300 with
        # check_val_every_n_epoch=5 is 1500 epochs of non-improvement, against a max_epochs of
        # 1000, so every default run trained to the budget. That is why DE-2 and DE-3 never
        # produced a wrong number for anyone to notice.
        kwargs.setdefault("check_val_every_n_epoch", 1)
        kwargs.setdefault("accelerator", "auto")
        kwargs.setdefault("devices", "auto")

        # scvi's own early stopping is switched off and replaced. I3/B5 require the monitor to
        # be ignored until the KL ramp completes, and I4 requires a snapshot spanning both the
        # module state_dict AND the Pyro param store, which no stock callback carries.
        kwargs["early_stopping"] = False
        monitor = kwargs.pop("early_stopping_monitor", MONITOR)
        mode = kwargs.pop("early_stopping_mode", "min")
        patience = kwargs.pop("early_stopping_patience", self.patience_epochs)

        snapshot = BestObjectiveSnapshot(monitor=monitor, mode=mode)
        callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks += [
            RampGatedEarlyStopping(monitor=monitor, mode=mode, patience=patience),
            snapshot,
        ]
        kwargs["callbacks"] = callbacks

        runner = TrainRunner(
            self,
            training_plan=plan,
            data_splitter=splitter,
            max_epochs=max_epochs,
            **kwargs,
        )

        with _slurm_autodetect_disabled():
            runner()

        # B9: a fit records what actually happened. `steps_per_epoch` is read from the counter
        # rather than computed from batch_size, so a partial final batch cannot skew it.
        epochs_run = max(int(runner.trainer.current_epoch), 1)
        steps_per_epoch = max(self.module._kl_warmup_step / epochs_run, 1e-9)
        ramp_done = ramp_is_complete(plan)
        self.training_record_ = {
            "epochs_run": epochs_run,
            "warmup_steps_taken": int(self.module._kl_warmup_step),
            "n_steps_kl_warmup": int(n_steps_kl_warmup),
            "steps_per_epoch": steps_per_epoch,
            # The one number that says which regime a run was in. A ramp finishing early leaves
            # most of the fit at a stationary objective; one that never finishes means the prior
            # was substantially switched off throughout.
            "ramp_completes_at_epoch": (int(n_steps_kl_warmup) / steps_per_epoch
                                        if n_steps_kl_warmup > 0 else 0.0),
            "ramp_completed": ramp_done,
            "selection_criterion": (monitor if ramp_done
                                    else "last epoch (ramp incomplete)"),
            "selected_epoch": snapshot.best_epoch,
            "selected_score": snapshot.best_score,
            "seed": self._seed,
        }
        if not ramp_done:
            warnings.warn(
                f"the KL ramp did not complete: {self.module._kl_warmup_step} of "
                f"{n_steps_kl_warmup} warmup steps taken in {epochs_run} epochs "
                f"(~{steps_per_epoch:.1f} steps/epoch, so it needs "
                f"~{self.training_record_['ramp_completes_at_epoch']:.0f} epochs). No checkpoint "
                f"was selected and the final weights are kept, because no two checks in this run "
                f"came from the same objective. The fitted prior is scaled by "
                f"kl_weight={self.module.kl_weight:.3g}, not kl_weight_max="
                f"{self.module.kl_weight_max:.3g}.",
                UserWarning,
                stacklevel=2,
            )
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
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
            # NEW-1: bind each cell to ITS OWN clonotype x covariate group, via the global
            # cell id the loader carries -- never by position in this loader.
            #
            # This used to be `ct_array[current_idx : current_idx + n]` with a running offset.
            # `ct_array` is indexed by TRAINING cell id, so the offset is only the right index
            # when the passed adata is a contiguous prefix of the training data in its original
            # order. For any other subset, a reordered view, or a per-patient slice -- all legal
            # under the frozen contract -- cell i silently received the prior of the i-th
            # TRAINING cell. Measured on a reversed view of a 200-cell fixture: max |delta p|
            # = 0.3696 against the same cells predicted from the full object. It read 0.0000 on
            # a prefix, which is why it survived.
            #
            # model() already does exactly this (`ct_idx = self.ct_array[indices]`), and
            # test_alignment_target_uses_global_indices pins it there; predict() and
            # to_anndata() were the two places that did not.
            idx = tensors["indices"].long().view(-1).to(device)
            clone_cov_posterior = p_ct[ct_array[idx]]
            z_loc, _, _ = self.module.encoder(x, b)
            cls_logits = self.module.classifier(z_loc)
            prior_log = torch.log(clone_cov_posterior + eps)
            if self.module.use_gate:
                local_logits = self.module.gate_prob * cls_logits + (1.0 - self.module.gate_prob) * prior_log
            else:
                local_logits = cls_logits + prior_log
            all_probs.append(F.softmax(local_logits, dim=-1).cpu())

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
        # DE-5b: the guide's actual concentration, so credible intervals come from the
        # fitted posterior rather than from a reconstructed local_scale * mean.
        adata.uns[K.CONC_CT] = self.module.get_conc_ct().detach().cpu().numpy()
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
        for tensors in loader:
            x = tensors[REGISTRY_KEYS.X_KEY].to(device)
            b = tensors[REGISTRY_KEYS.BATCH_KEY].long().to(device)
            z_loc, _, _ = self.module.encoder(x, b)
            logits_buf.append(self.module.classifier(z_loc).cpu())
            # NEW-1, as in predict(): index by the cell's own global id, not by position.
            idx = tensors["indices"].long().view(-1).to(device)
            prior_buf.append(torch.log(p_ct_t[ct_arr_t[idx]] + 1e-8).cpu())
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

        if compute_umap:
            import umap
            adata.obsm[K.X_UMAP] = umap.UMAP(random_state=42).fit_transform(adata.obsm[K.X_TCRI])

        return adata

