"""Callbacks implementing the stopping policy (training contract I3, I4, B5).

Two objects, one shared gate:

``RampGatedEarlyStopping``
    Lightning's ``EarlyStopping``, but blind until the KL ramp finishes.

``BestObjectiveSnapshot``
    Records the argmin weights and writes them back at ``on_fit_end``.

Why both read the SAME counter: a gate expressed twice, in two units, can disagree by a check
at the boundary — one callback would then be selecting from a series the other had already
started recording. scvi ships an ``early_stopping_warmup_epochs`` that counts EPOCHS while the
ramp counts OPTIMIZER STEPS; using it alongside this gate would be exactly that bug. So it is
deliberately not used.
"""
from __future__ import annotations

import warnings

import pyro
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping

__all__ = ["RampGatedEarlyStopping", "BestObjectiveSnapshot", "ramp_is_complete"]


def ramp_is_complete(pl_module) -> bool:
    """The single predicate. One counter, one unit (optimizer steps), read by both callbacks.

    ``n_steps_kl_warmup <= 0`` disables annealing entirely, in which case every check is already
    at ``kl_weight_max`` and selection may begin immediately.
    """
    n_warmup = int(getattr(pl_module, "n_steps_kl_warmup", 0) or 0)
    if n_warmup <= 0:
        return True
    return int(getattr(pl_module.module, "_kl_warmup_step", 0)) >= n_warmup


class RampGatedEarlyStopping(EarlyStopping):
    """Early stopping that ignores every check taken before the KL ramp completes.

    I3 makes each check well-posed; this makes the *series* comparable, by ensuring every entry
    in it came from the same objective ``L_(kl_weight_max)``. Without the gate, the argmin can
    land on an early check that scored well only because the KL term was still switched off.
    """

    def _run_early_stopping_check(self, trainer):
        if not ramp_is_complete(trainer.lightning_module):
            return
        super()._run_early_stopping_check(trainer)


class BestObjectiveSnapshot(Callback):
    """Keep the parameters that scored best, and restore them when the fit ends (I4).

    Early stopping has two outputs: a stop time and the argmin weights. Keeping the final
    weights implements the first and silently drops the second.

    The snapshot spans BOTH parameter stores, because neither alone is the model:

    * ``module.state_dict()`` — and ``state_dict()``, never ``named_parameters()``. The encoder
      and VampPrior carry ``BatchNorm1d`` running statistics (``FCLayers`` defaults
      ``use_batch_norm=True``), which are buffers: absent from ``named_parameters()``, absent
      from the Pyro store, and read by ``predict()``/``get_latent_representation()`` in eval
      mode. Restoring without them yields a model no check ever evaluated.
    * the Pyro param store — ``q_p_c_raw``/``q_p_ct_raw`` are not in ``state_dict()`` at all,
      and they are what every metric reads.

    The store is handled through ``named_parameters()`` in UNCONSTRAINED space. This is not a
    style preference. Those two are ``constraints.positive``, so ``store.items()`` yields a
    non-leaf ``ExpTransform`` output, and ``.data.copy_()`` on it is a silent no-op — verified
    directly: writing 5.0 leaves the store reading 2.0, with no error raised. A restore written
    the obvious way does nothing at all, which is DE-1's failure class: a real defect hiding
    behind an operation that looks like it worked.

    Index buffers are excluded. They are fixed for the life of the module and copying them each
    improvement is pure overhead.
    """

    #: int64 lookups, constant for the fit.
    _INDEX_BUFFERS = frozenset({
        "c_array", "ct_array", "ct_to_c", "ct_to_cov", "_target_phenotypes",
    })

    def __init__(self, monitor: str, mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self._module_state: dict | None = None
        self._store_state: dict | None = None

    # ── capture ──────────────────────────────────────────────────────────────
    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return score < self.best_score if self.mode == "min" else score > self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not ramp_is_complete(pl_module):
            return
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        score = float(score)
        if not self._is_better(score):
            return

        self.best_score = score
        self.best_epoch = int(trainer.current_epoch)
        self._module_state = {
            k: v.detach().cpu().clone()
            for k, v in pl_module.module.state_dict().items()
            if k not in self._INDEX_BUFFERS
        }
        self._store_state = {
            name: leaf.detach().cpu().clone()
            for name, leaf in pyro.get_param_store().named_parameters()
            if not name.startswith("scvi$$$")
        }

    # ── restore ──────────────────────────────────────────────────────────────
    def on_fit_end(self, trainer, pl_module):
        if self._module_state is None:
            # No gated check ever ran: the ramp did not finish inside this fit, or there was no
            # validation loop. B5 says warn, never raise -- raising here would break every short
            # fit in the test suite and the whole benchmark grid, whose 60-epoch default reaches
            # roughly 15% of a 2000-step ramp.
            return

        pl_module.module.load_state_dict(self._module_state, strict=False)

        store = pyro.get_param_store()
        live = {n for n, _ in store.named_parameters() if not n.startswith("scvi$$$")}
        missing = live - set(self._store_state)
        if missing:
            raise RuntimeError(
                f"the best-weight snapshot is missing param-store keys {sorted(missing)}. "
                f"Restoring would leave those at their final-epoch values while everything "
                f"else came from epoch {self.best_epoch} -- a model no check evaluated."
            )
        with torch.no_grad():
            for name, leaf in store.named_parameters():
                saved = self._store_state.get(name)
                if saved is not None:
                    leaf.data.copy_(saved.to(leaf.device))
