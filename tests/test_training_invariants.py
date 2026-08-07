"""Behavioural invariants for the training plan (DE-1, DE-4).

The training layer had no contract and no behavioural tests. The knob test verified that
``patience`` *arrives* at ``EarlyStopping.patience`` and marked it ✅ — while patience was
counted in validation checks, ``validation_step`` was taking optimizer steps, and the KL ramp
restarted on every ``train()`` call. Wiring-only verification is why the same defects kept
being rediscovered from new symptoms.

These assert what the layer *does*, not what it is wired to. Both fail on the parent commit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import pyro
import torch

warnings.filterwarnings("ignore")

pytestmark = pytest.mark.slow

STORE_KEYS = ("q_p_c_raw", "q_p_ct_raw")


def _snapshot():
    """The two guide tensors every metric reads. They live only in Pyro's param store —
    they are NOT in ``module.state_dict()`` and not reachable from ``parameters()``."""
    store = pyro.get_param_store()
    return {k: store[k].detach().clone() for k in STORE_KEYS if k in store}


@pytest.fixture
def adata():
    from tcri.datasets import simulate_tcri

    return simulate_tcri(n_clones=8, n_phenotypes=5, n_genes=40, n_cells=300,
                         omega_concentration=0.4, fuzziness=0.1, seed=0)


def _fresh(adata):
    from tcri.model._model import TCRIModel

    pyro.clear_param_store()
    TCRIModel.setup_anndata(
        adata, layer="counts", clonotype_key="clone_id", phenotype_key="phenotype",
        covariate_key="covariate", batch_key="batch",
    )
    return TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1,
                     classifier_n_layers=1, classifier_hidden=16, K=5, seed=0)


class _ValidationWatcher(torch.nn.Module):
    """Records the guide tensors either side of every validation loop."""

    def __init__(self):
        super().__init__()
        self.deltas = []
        self._before = None

    def on_validation_start(self, trainer, pl_module):
        self._before = _snapshot()

    def on_validation_end(self, trainer, pl_module):
        after = _snapshot()
        if self._before:
            self.deltas.append(
                sum(float((after[k] - self._before[k]).abs().sum()) for k in self._before)
            )


def test_validation_does_not_update_parameters(adata):
    """DE-1: a validation pass must not move a single parameter.

    ``validation_step`` used to call ``super().training_step()`` -> ``SVI.step()`` -> the Pyro
    optimizer. Lightning zeroes ``.grad`` on the LightningModule's own parameters before the
    validation loop, so the networks were spared; ``q_p_c_raw``/``q_p_ct_raw`` are not
    LightningModule parameters, kept the zeroed grad Pyro left behind, and were stepped on
    ``weight_decay * theta`` in the unconstrained log space of a positive-constrained
    parameter — pulling every clone row toward uniform. Measured 0.54 L1 per check.
    """
    import lightning.pytorch as pl

    watcher = _ValidationWatcher()

    modes = []

    class _Cb(pl.Callback):
        def on_validation_start(self, trainer, pl_module):
            watcher.on_validation_start(trainer, pl_module)

        def on_validation_end(self, trainer, pl_module):
            watcher.on_validation_end(trainer, pl_module)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            modes.append(bool(pl_module.module.training))

    m = _fresh(adata)
    m.train(max_epochs=6, batch_size=128, accelerator="cpu",
            check_val_every_n_epoch=1, callbacks=[_Cb()],
            enable_progress_bar=False, enable_model_summary=False)

    assert watcher.deltas, "no validation loop ran — the test asserted nothing"
    assert modes and all(modes), (
        f"only {sum(modes)}/{len(modes)} training batches ran with the module in train() mode. "
        f"validation_step no longer restores train mode by hand (Lightning owns that); if it "
        f"stops doing so, training silently runs with dropout disabled."
    )
    worst = max(watcher.deltas)
    assert worst == 0.0, (
        f"validation moved the guide parameters that every metric reads: max L1 {worst:.4f} "
        f"over {len(watcher.deltas)} checks. validation_step must evaluate, never step."
    )


def test_kl_ramp_is_monotone_across_resumed_training(adata):
    """DE-4: the warmup counter belongs to the model, not to a per-call training plan.

    ``train()`` builds a fresh ``UnifiedTrainingPlan`` each call. With the counter on the plan,
    a second ``train()`` restarted the ramp from zero, so a staged or resumed fit saw a sawtooth
    ``kl_weight`` instead of a monotone one. This invalidated two of our own diagnostic probes
    before it was found.
    """
    m = _fresh(adata)
    m.train(max_epochs=3, batch_size=128, accelerator="cpu",
            enable_progress_bar=False, enable_model_summary=False)
    after_first = m.module._kl_warmup_step
    weight_first = float(m.module.kl_weight)

    m.train(max_epochs=3, batch_size=128, accelerator="cpu",
            enable_progress_bar=False, enable_model_summary=False)
    after_second = m.module._kl_warmup_step
    weight_second = float(m.module.kl_weight)

    assert after_first > 0, "no optimizer steps were counted in the first call"
    assert after_second > after_first, (
        f"the warmup counter restarted: {after_first} -> {after_second}. A resumed fit must "
        f"continue the schedule, not begin a second ramp."
    )
    assert weight_second >= weight_first, (
        f"kl_weight went backwards across train() calls: {weight_first:.6g} -> "
        f"{weight_second:.6g}. The annealing schedule must be monotone."
    )
