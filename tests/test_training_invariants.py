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

# NOT slow-marked, deliberately. This whole file runs in ~3s, but it carried a module-level
# `slow` marker, and CI runs bare `pytest tests/` -- so the tests the training contract names as
# the enforcement for I2, I3, I4, I5, B1 and B5 never ran on a pull request. An invariant whose
# proof CI skips is an invariant nothing checks.

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


# ── the stopping policy: I3 and I4 ───────────────────────────────────────────

def _plan_and_batch(adata, n_steps_kl_warmup=8):
    """A fitted-enough model plus one validation batch, ready to evaluate."""
    from tcri.model._training import UnifiedTrainingPlan

    m = _fresh(adata)
    m.train(max_epochs=2, batch_size=128, n_steps_kl_warmup=n_steps_kl_warmup,
            accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    plan = UnifiedTrainingPlan(module=m.module, n_steps_kl_warmup=n_steps_kl_warmup)
    loader = m._make_data_loader(adata=m.adata, batch_size=128, shuffle=False)
    return m, plan, next(iter(loader))


def test_monitor_is_invariant_to_ramp_position(adata):
    """I3: the monitored quantity is a fixed function of the parameters.

    Evaluate the criterion at two different ramp positions with the parameters held EXACTLY
    fixed. A criterion that is a function of (Lambda, Theta) must return the same number; one
    that inherits the annealed kl_weight, or redraws its Monte-Carlo sample, will not.

    This fails on the parent commit, and it also fails a partial fix — pinning kl_weight without
    fixing the evaluation seed still leaves the estimator redrawing every check, so the two
    numbers differ in the low-order digits. Both clauses are required, so the assertion is
    exact rather than approximate.
    """
    m, plan, batch = _plan_and_batch(adata)
    plan.module.eval()

    before = {k: v.detach().clone() for k, v in plan.module.state_dict().items()}
    store_before = {n: p.detach().clone() for n, p in pyro.get_param_store().named_parameters()}

    plan.module._kl_warmup_step = 1          # early in the ramp
    plan.module.kl_weight = 1e-6
    first = float(plan.validation_step(batch, 0)["loss"])

    # identical parameters, a different point on the schedule
    plan.module.load_state_dict(before, strict=False)
    with torch.no_grad():
        for name, p in pyro.get_param_store().named_parameters():
            p.data.copy_(store_before[name])
    plan.module._kl_warmup_step = 10_000     # ramp long finished
    plan.module.kl_weight = plan.module.kl_weight_max
    second = float(plan.validation_step(batch, 0)["loss"])

    assert first == second, (
        f"the monitored quantity moved with ramp position while the parameters were held "
        f"fixed: {first!r} vs {second!r}. It is therefore not a function of (Lambda, Theta), "
        f"and an argmin over it is not an argmin (contract I3)."
    )


def test_validation_pin_restores_the_training_schedule(adata):
    """B1: the pin is scoped to the check.

    validation_step raises kl_weight to kl_weight_max to make the criterion well-posed. If it
    left it there, the next training step would read a kl_weight it never scheduled, and the
    ramp would jump to its endpoint the first time anything validated.
    """
    m, plan, batch = _plan_and_batch(adata)
    plan.module._kl_warmup_step = 3
    plan.module.kl_weight = 0.125

    plan.validation_step(batch, 0)

    assert plan.module.kl_weight == 0.125, (
        f"validation left kl_weight at {plan.module.kl_weight}; the pin must be undone so the "
        f"training schedule is the only thing that advances it (contract B1)."
    )


def test_selection_is_gated_until_the_ramp_completes(adata):
    """B5: no check is recorded before the ramp finishes.

    Every entry in the monitored series must come from the same objective. A run whose ramp
    never completes has no comparable pair, so it must select nothing, keep its final weights,
    and say so in the record rather than silently reporting an epoch.
    """
    m = _fresh(adata)
    with pytest.warns(UserWarning, match="KL ramp did not complete"):
        m.train(max_epochs=3, batch_size=128, n_steps_kl_warmup=10**6,
                accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)

    rec = m.training_record_
    assert rec["ramp_completed"] is False
    assert rec["selected_epoch"] is None, (
        "a checkpoint was selected from checks taken at different kl_weights; with the ramp "
        "incomplete no two checks share an objective (contract B5)."
    )
    assert rec["selection_criterion"] == "last epoch (ramp incomplete)"


def test_restored_model_is_the_selected_one(adata):
    """I4: what train() leaves behind is what the criterion chose.

    Records every gated check, then asserts that the parameters surviving the fit are the ones
    from the best-scoring check -- across all three places state lives:

      * the Pyro param store (``q_p_ct_raw``) -- not in state_dict() at all,
      * a network weight -- in both state_dict() and named_parameters(),
      * an encoder BatchNorm running statistic -- in state_dict() ONLY.

    That third one is the reason the snapshot uses state_dict(). A named_parameters() snapshot
    silently leaves the running stats at their final-epoch values, and predict() reads them in
    eval mode, so the restored model is one no check ever evaluated.
    """
    import lightning.pytorch as pl

    from tcri.model._callbacks import ramp_is_complete

    BN = "encoder.encoder.fc_layers.Layer 0.1.running_mean"
    NET = "classifier.mlp.0.weight"

    seen = []

    class _Spy(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking or not ramp_is_complete(pl_module):
                return
            score = trainer.callback_metrics.get("objective_validation_percell")
            if score is None:
                return
            sd = pl_module.module.state_dict()
            seen.append((
                float(score),
                pyro.get_param_store()["q_p_ct_raw"].detach().clone(),
                sd[NET].detach().clone(),
                sd[BN].detach().clone(),
            ))

    # lr=1e-2 over 60 epochs on this fixture puts the argmin at check ~47 of 58 with a clear
    # margin. At the default lr the criterion still descends at the last epoch, so "keep the
    # final weights" would pass by accident -- the assertion below guards against exactly that
    # if the fixture ever drifts back to monotone.
    m = _fresh(adata)
    m.train(max_epochs=60, batch_size=128, n_steps_kl_warmup=4, lr=1e-2, accelerator="cpu",
            callbacks=[_Spy()], enable_progress_bar=False, enable_model_summary=False)

    assert len(seen) > 2, f"only {len(seen)} gated checks ran; the test asserted nothing"
    best_score, best_ct, best_net, best_bn = min(seen, key=lambda r: r[0])
    assert best_score < seen[-1][0], (
        "the best check WAS the last one, so keeping final weights would pass by accident. "
        "This fixture no longer discriminates."
    )

    sd = m.module.state_dict()
    assert torch.equal(pyro.get_param_store()["q_p_ct_raw"].detach(), best_ct), (
        "q_p_ct_raw is not the selected checkpoint's. Every metric reads this tensor, and it "
        "lives only in the Pyro param store -- note that writing it through store.items() is a "
        "silent no-op, because the positive constraint makes that a non-leaf view (I4)."
    )
    assert torch.equal(sd[NET], best_net), "a network weight is not the selected checkpoint's"
    assert torch.equal(sd[BN], best_bn), (
        "the encoder's BatchNorm running_mean is not the selected checkpoint's. It is a buffer: "
        "in state_dict() but NOT in named_parameters(), and read by predict() in eval mode. "
        "Snapshotting named_parameters() restores a model no check evaluated (I4)."
    )


def test_monitor_excludes_the_global_block(adata):
    """I3 scope: the monitor is the per-cell block, not the ELBO.

    This is the deliberate departure the contract records, and it needs its own test: including
    the global sites does NOT break ramp-invariance (the pin fixes kl_weight either way), so
    ``test_monitor_is_invariant_to_ramp_position`` cannot see the difference. Without this,
    re-adding ``p_c``/``p_ct`` to the monitored number would be a silent change.

    Both global plates are declared at full size with no subsampling, so their KL is the same
    number whichever cells are held out — a training-set quantity a validation criterion must
    not contain.
    """
    m, plan, batch = _plan_and_batch(adata)
    plan.module.eval()

    out = plan.validation_step(batch, 0)
    n_cells = int(batch["indices"].shape[0])

    args, kwargs = plan.module._get_fn_args_from_batch(batch)
    prev = plan.module.kl_weight
    try:
        plan.module.kl_weight = plan.module.kl_weight_max
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(plan._validation_seed)
            per_cell, global_block = plan._objective_blocks(*args, **kwargs)
    finally:
        plan.module.kl_weight = prev

    assert abs(global_block) > 0.0, "the fixture has no global-block mass; nothing is asserted"
    assert float(out["loss"]) == pytest.approx(-per_cell / n_cells, rel=1e-6), (
        "the monitor is not the per-cell block alone (contract I3 'scope')"
    )
    full = -(per_cell + global_block) / n_cells
    assert float(out["loss"]) != pytest.approx(full, rel=1e-9), (
        f"the monitored value equals the FULL elbo ({full:.6f}); the global block is being "
        f"included, so selection is partly on prior-matching over the training data"
    )
    assert float(out["global_block"]) == pytest.approx(global_block, rel=1e-6), (
        "the excluded block must still be logged, so the exclusion stays inspectable"
    )


def test_hitting_the_epoch_cap_warns_and_is_recorded(adata):
    """Reaching max_epochs means the stopping rule never fired — say so.

    A truncated fit and a converged one are otherwise indistinguishable: same record shape,
    same outputs, no signal. That silence has a measured cost. On a 100k-cell dataset the
    mutual information kept climbing well past the default budget — 0.236 at 200 epochs,
    0.290 at 600, 0.328 at 1000, and 0.342 once early stopping finally engaged at epoch 1208
    of a 2000 budget. Every run at or below 1000 epochs hit its cap, and 1000 is the default,
    so a default fit understated MI by roughly 31% with nothing to indicate it.
    """
    import warnings as _w

    m = _fresh(adata)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        # a budget this small cannot converge, so the cap is certainly why training ended
        m.train(max_epochs=2, batch_size=128, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)

    assert m.training_record_["stopped_early"] is False, (
        "a fit that ran to max_epochs did not stop early, and the record must say so"
    )
    msgs = " ".join(str(c.message) for c in caught)
    assert "max_epochs" in msgs and "converge" in msgs, (
        f"the warning must say the cap, not convergence, ended training: {msgs}"
    )


def test_train_rejects_unknown_kwargs_itself(adata):
    """An unsupported argument must fail HERE, naming tcri, not four frames deep in scvi.

    train() forwards **kwargs to TrainRunner and on into Trainer.__init__, so before this
    guard a name train() does not accept produced

        TypeError: Trainer.__init__() got an unexpected keyword argument 'validation_size'

    from inside lightning, with nothing pointing at tcri or at what the caller should do.
    That is how `validation_size=0.1` killed a 17-minute run — after training had finished,
    on the first metric call.
    """
    m = _fresh(adata)
    with pytest.raises(TypeError) as exc:
        m.train(max_epochs=1, batch_size=128, accelerator="cpu", validation_size=0.1)
    msg = str(exc.value)
    assert "validation_size" in msg, f"the error must name the offending argument: {msg}"
    assert "train()" in msg, f"the error must come from train(), not lightning: {msg}"


def test_train_still_accepts_genuine_lightning_kwargs(adata):
    """The guard must not reject what lightning legitimately takes.

    The accepted set is introspected from the INSTALLED lightning and scvi, not hardcoded,
    precisely so an upgrade cannot turn this guard into a new source of errors.
    """
    from tcri.model._model import _accepted_train_kwargs

    accepted = _accepted_train_kwargs()
    for name in ("accelerator", "max_epochs", "callbacks", "enable_progress_bar"):
        assert name in accepted, f"{name} is a real option and must not be rejected"
    assert "validation_size" not in accepted, (
        "train() fixes the split at 0.9; accepting validation_size would imply otherwise"
    )
