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

    @property
    def loss(self):
        return self._loss_fn

    # NOTE: configure_optimizers is deliberately NOT overridden — scvi's base class
    # returns a shim over a single dummy parameter purely to advance Lightning's step
    # counter. All real optimization happens in Pyro's SVI (see __init__).

    def training_step(self, batch, batch_idx):
        # ── KL warmup ────────────────────────────────────────────
        step = self.module._kl_warmup_step
        if self.n_steps_kl_warmup > 0 and step < self.n_steps_kl_warmup:
            kl_weight = max(1e-6, self.module.kl_weight_max * (step / self.n_steps_kl_warmup))
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

        self.module._kl_warmup_step += 1
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        """Evaluate the ELBO. Never step.

        DE-1: this used to call ``super().training_step()``, which reaches
        ``PyroTrainingPlan.training_step`` -> ``SVI.step()`` -> the Pyro optimizer. Every
        validation batch therefore applied an Adam update to ``q_p_c_raw``/``q_p_ct_raw`` — the
        exact guide parameters ``get_p_ct()`` and every metric read. Lightning zeroes ``.grad``
        on the LightningModule's parameters before the validation loop, so torch Adam skipped
        the networks; but those two tensors live only in Pyro's param store, are not reachable
        from ``parameters()``, and kept the zeroed grad Pyro left there. Adam then stepped them
        on ``weight_decay * theta`` in the UNCONSTRAINED log space of a positive-constrained
        parameter — every entry pulled toward ``log theta = 0``, i.e. every clone row pulled
        toward uniform. Measured 0.54 L1 per validation check.

        ``SVI.evaluate_loss`` computes the identical estimator through the same wrapped
        model/guide, with no ``param_capture``, no ``optim()`` and no ``zero_grads``.

        ``kl_weight`` is deliberately NOT set here: inheriting whatever the last training step
        left keeps ``elbo_validation`` on the same scale as ``elbo_train``. That makes the two
        series comparable but means the monitored quantity is not a fixed objective while the
        ramp is still climbing — invariant I3, which PR 4 (`stopping-policy`) resolves.
        """
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        device = next(self.module.parameters()).device
        with torch.no_grad():
            loss = self.svi.evaluate_loss(*args, **kwargs)
        val_dict = {"loss": torch.as_tensor(loss, dtype=torch.float32, device=device)}

        # ── Diagnostic only ──────────────────────────────────────
        # Stays in eval mode (Lightning's evaluation loop already set it) and under no_grad.
        # Previously the block above restored train mode before reaching here, so this ran with
        # classifier dropout ACTIVE, and it was outside no_grad — building an autograd graph
        # through get_latent, the classifier and get_p_ct() for values that are only logged.
        with torch.no_grad():
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
