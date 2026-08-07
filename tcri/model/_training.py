"""Training plan for the TCRI model plus the archetype initializer.

`UnifiedTrainingPlan` adds classification/reconstruction diagnostics and a
`validation_step` (logs ``objective_validation_percell``, the early-stopping
criterion fixed by training-contract I3) on top of Pyro's ELBO step. `build_archetypes` K-means-clusters the clone x phenotype
matrix to seed the Dirichlet mixture (returns centers AND labels).
"""
import numpy as np
import torch
import torch.nn.functional as F

from scvi.train import PyroTrainingPlan
from pyro import poutine
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
    Training plan that includes classification, reconstruction losses, KL warmup, plus a
    validation_step that logs 'objective_validation_percell' for scvi's early stopping.

    That monitor is NOT an ELBO and the name says so deliberately: it is the per-cell block
    only, evaluated at a pinned ``kl_weight_max`` under a fixed seed. See ``_objective_blocks``
    and training-contract I3.
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
        #: I3 clause 5. Fixed across every check, so the monitored series is a function of the
        #: parameters rather than of the draw. Not the fit seed: this only controls evaluation.
        self._validation_seed = 0

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
        # B9: without this, I3 and I5 are unfalsifiable once a run has finished -- nothing in
        # the record says whether the ramp ever completed.
        self.log("kl_weight", float(kl_weight), prog_bar=False, on_epoch=True)

        self.module._kl_warmup_step += 1
        return loss_dict
    
    #: Sites in the hierarchical branch. Both plates are declared at FULL size with no
    #: subsampling, so these contribute the same value no matter which cells are held out.
    GLOBAL_SITES = frozenset({"p_c", "p_ct"})

    def _objective_blocks(self, *args, **kwargs):
        """Split the log-joint minus log-guide into (per-cell, global) blocks.

        Returns a 2-tuple of floats whose SUM is the ELBO on this batch -- verified against
        ``Trace_ELBO().loss`` to ~3e-6 relative on a fixture. Splitting rather than calling
        ``SVI.evaluate_loss`` is the point: the monitored criterion is the per-cell block only.

        Why the global block is excluded from the monitor (contract I3, "scope"): ``p_c`` and
        ``p_ct`` live in ``pyro.plate("clonotypes", c_count)`` / ``pyro.plate("ct_plate",
        ct_count)``, neither subsampled. Their KL is therefore identical for every choice of
        validation split -- it is a function of the TRAINING-fitted guide and the prior, and of
        nothing that was held out. Monitoring it means selecting partly on prior-matching over
        data the criterion was supposed to exclude. Its share grows with clone count: ~27% on a
        6-clone fixture, and larger on real repertoires, which are singleton-dominated.

        The per-cell block still contains ``phenotype_alignment``, which scores held-out cells
        against ``phi = p_ct[ct_idx]``. The Dirichlet branch is therefore still covered by the
        criterion; what is dropped is only the prior-matching term held-out data cannot speak to.
        """
        guide_trace = poutine.trace(self.module.guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(self.module.model, trace=guide_trace)
        ).get_trace(*args, **kwargs)
        model_trace.compute_log_prob()
        guide_trace.compute_log_prob()

        per_cell = global_block = 0.0
        for trace, sign in ((model_trace, 1.0), (guide_trace, -1.0)):
            for name, site in trace.nodes.items():
                if site["type"] not in ("sample", "factor"):
                    continue
                log_prob = site.get("log_prob")
                if log_prob is None:
                    continue
                value = sign * float(log_prob.sum())
                if name in self.GLOBAL_SITES:
                    global_block += value
                else:
                    per_cell += value
        return per_cell, global_block

    def validation_step(self, batch, batch_idx):
        """Evaluate the selection criterion. Never step.

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

        I3. ``kl_weight`` used to be deliberately left unset here, inheriting whatever the last
        training batch happened to leave. The stated reason was that this keeps the validation
        series on the same scale as ``elbo_train`` — a property that was never real, since the
        two are computed on different splits and now on different site sets. The cost was that
        every check evaluated a *different* objective while the ramp climbed, and an argmin over
        a series of different functions is not an argmin.

        So the check now pins ``kl_weight`` to ``kl_weight_max``, runs in eval mode (Lightning's
        evaluation loop sets it), and draws under a forked, fixed seed. That last clause is
        load-bearing rather than fussy: a Monte-Carlo estimator redrawn each check is not a
        function of the parameters at all, so selecting its minimum selects noise.

        The pin is undone in ``finally``, so B1's monotone TRAINING schedule is untouched.
        """
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        device = next(self.module.parameters()).device

        # I3: pin, fork, evaluate. The pin is restored in `finally` so B1's monotone TRAINING
        # schedule is untouched -- kl_weight is raised for the duration of the check only.
        prev_kl = self.module.kl_weight
        fork_devices = [device.index] if device.type == "cuda" else []
        try:
            self.module.kl_weight = self.module.kl_weight_max
            with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
                torch.manual_seed(self._validation_seed)
                per_cell, global_block = self._objective_blocks(*args, **kwargs)
        finally:
            self.module.kl_weight = prev_kl

        n_cells = max(int(batch["indices"].shape[0]), 1)
        val_dict = {
            "loss": torch.as_tensor(-per_cell / n_cells, dtype=torch.float32, device=device),
            "global_block": torch.as_tensor(global_block, dtype=torch.float32, device=device),
        }

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
        # The monitored series (I3). NOT an ELBO -- see _objective_blocks.
        self.log("objective_validation_percell", val_dict["loss"], prog_bar=True, on_epoch=True)
        # The excluded half, logged so the exclusion is inspectable rather than merely asserted.
        self.log("global_block_validation", val_dict["global_block"], prog_bar=False,
                 on_epoch=True)
        return val_dict
