"""Training plan for the TCRI model plus the archetype initializer.

`UnifiedTrainingPlan` adds classification/reconstruction diagnostics and a
`validation_step` (logs ``elbo_validation`` for scvi early stopping) on top of
Pyro's ELBO step. `build_archetypes` K-means-clusters the clone x phenotype
matrix to seed the Dirichlet mixture (returns centers AND labels).
"""
import numpy as np
import torch
import torch.nn.functional as F

from scvi.train import PyroTrainingPlan
from pyro.infer import TraceEnum_ELBO, Trace_ELBO
from sklearn.cluster import KMeans

from ._module import TCRIModule

__all__ = ["UnifiedTrainingPlan", "build_archetypes"]


def build_archetypes(clone_phenotype_prior, K=4):
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(clone_phenotype_prior)
    centers = kmeans.cluster_centers_
    centers = np.clip(centers, 1e-8, None)
    centers = centers / centers.sum(axis=1, keepdims=True)
    return centers, labels


class UnifiedTrainingPlan(PyroTrainingPlan):
    """
    Training plan that includes classification, reconstruction losses,
    KL warmup, plus a validation_step that logs 'elbo_validation' so scvi's
    early stopping can monitor it.
    """

    def __init__(
        self,
        module: TCRIModule,
        n_steps_kl_warmup: int = 2000,   # must match TCRIModel.train, which overrides it
        reconstruction_loss_scale: float = 1e-2,
        num_particles: int = 5,
        optimizer_config: dict = None,
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

        if optimizer_config is None:
            optimizer_config = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-5,
                                "weight_decay": 1e-4}
        self.optimizer_config = optimizer_config

        # Hand the optimizer settings to PYRO's optimizer — the one inside SVI that
        # actually descends the ELBO gradients. Previously this class overrode
        # `configure_optimizers` with a real torch Adam over every module parameter;
        # that override replaced scvi's deliberate no-op shim ("a shim optimizer ...
        # to keep Lightning happy") and ran AFTER SVI.step() had already stepped and
        # ZEROED the gradients. Stepping Adam on zero gradients is not a no-op: the
        # weight-decay term becomes the whole gradient, and Adam's own normalization
        # (g/sqrt(g^2)) strips its magnitude, so the update degenerates to ~lr*sign(p)
        # — a scale-free shrink, not proportional L2. Measured effect: network weights
        # held at ~2.4x smaller than without it. It also meant `lr` never reached the
        # real optimizer (Pyro always used scvi's hard-coded 1e-3).
        super().__init__(
            module,
            n_steps_kl_warmup=n_steps_kl_warmup,
            optim_kwargs={
                "lr": optimizer_config["lr"],
                "betas": optimizer_config["betas"],
                "eps": optimizer_config["eps"],
                "weight_decay": optimizer_config["weight_decay"],
            },
            **kwargs,
        )

        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self._my_global_step = 0

    @property
    def loss(self):
        return self._loss_fn

    # NOTE: configure_optimizers is deliberately NOT overridden — scvi's base class
    # returns a shim over a single dummy parameter purely to advance Lightning's step
    # counter. All real optimization happens in Pyro's SVI (see __init__).

    def training_step(self, batch, batch_idx):
        # ── KL warmup ────────────────────────────────────────────
        if self.n_steps_kl_warmup > 0 and self._my_global_step < self.n_steps_kl_warmup:
            kl_weight = max(1e-6, self.module.kl_weight_max * (self._my_global_step / self.n_steps_kl_warmup))
        else:
            kl_weight = self.module.kl_weight_max
        self.module.kl_weight = kl_weight

        # ── Pyro ELBO step ───────────────────────────────────────
        loss_dict = super().training_step(batch, batch_idx)
        device = next(self.module.parameters()).device

        if not isinstance(loss_dict["loss"], torch.Tensor):
            loss_dict["loss"] = torch.tensor(loss_dict["loss"], device=device, requires_grad=True)
        else:
            loss_dict["loss"] = loss_dict["loss"].to(device)

        # ── Diagnostics (no gradient contribution) ───────────────
        with torch.no_grad():
            z_diag = self.module.get_latent(batch).to(device)
            idx_diag = batch["indices"].long().view(-1).to(device)
            cls_logits = self.module.classifier(z_diag)
            ct_idx = self.module.ct_array[idx_diag]
            p_ct_prior = self.module.get_p_ct()[ct_idx].to(device)
            prior_log = torch.log(p_ct_prior + 1e-8)

            if self.module.use_gate:
                local_logits = self.module.gate_prob * cls_logits + (1.0 - self.module.gate_prob) * prior_log
            else:
                local_logits = cls_logits + prior_log

            probs = F.softmax(local_logits, dim=-1)
            kl_div = F.kl_div(probs.log(), p_ct_prior, reduction='batchmean')
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            confidence = (probs**2).sum(dim=-1).mean()

        self.log("kl_divergence_with_prior_train", kl_div, prog_bar=False, on_epoch=True)
        self.log("entropy_train", entropy, prog_bar=False, on_epoch=True)
        self.log("confidence_train", confidence, prog_bar=False, on_epoch=True)

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

        # ── Diagnostic only ──────────────────────────────────────
        z_batch = self.module.get_latent(batch).to(device)
        idx = batch["indices"].long().view(-1).to(device)
        cls_logits = self.module.classifier(z_batch)
        ct_idx = self.module.ct_array[idx]
        p_ct_prior = self.module.get_p_ct()[ct_idx].to(device)
        prior_log = torch.log(p_ct_prior + 1e-8)

        if self.module.use_gate:
            local_logits = self.module.gate_prob * cls_logits + (1.0 - self.module.gate_prob) * prior_log
        else:
            local_logits = cls_logits + prior_log

        probs = F.softmax(local_logits, dim=-1)
        kl_divergence = F.kl_div(probs.log(), p_ct_prior, reduction='batchmean')
        self.log("kl_divergence_with_prior_val", kl_divergence, prog_bar=False, on_epoch=True)

        self.log("elbo_validation", val_dict["loss"], prog_bar=True, on_epoch=True)
        return val_dict
