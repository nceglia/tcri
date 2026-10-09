"""Behavioural invariants for the training plan.

Each test here asserts what the training layer *does*: that a validation pass takes no
optimizer step, that the annealing schedule is monotone across ``train()`` calls, that the
monitored quantity is a fixed function of the parameters, and that the weights a fit leaves
behind are the ones the criterion selected. Wiring checks -- does a value reach the object it
names? -- live in ``tests/test_model_knobs.py`` and cannot see a value that arrives and is
then ignored.

The numbered statements these enforce are in ``governance/TRAINING_CONTRACT.md``, which names
the test for each one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import pyro
import torch

warnings.filterwarnings("ignore")

# Deliberately NOT slow-marked. CI runs a bare `pytest tests/`, so a module-level `slow` marker
# would skip the tests governance/TRAINING_CONTRACT.md names as the enforcement for I2, I3, I4,
# I5, B1, B2, B3, B4 and B5 on every pull request. An invariant whose proof CI skips is unchecked.

STORE_KEYS = ("q_p_c_raw", "q_p_ct_raw")


def _store_keys(model):
    """The two guide tensors under the namespace of ``model``, a ``TCRIModel`` or its module."""
    module = getattr(model, "module", model)
    return tuple(module.pname(k) for k in STORE_KEYS)


def _snapshot(model):
    """The two guide tensors every metric reads. They live only in Pyro's param store —
    they are NOT in ``module.state_dict()`` and not reachable from ``parameters()``."""
    store = pyro.get_param_store()
    return {k: store[k].detach().clone() for k in _store_keys(model) if k in store}


@pytest.fixture
def adata():
    from tcri.datasets import simulate_tcri

    return simulate_tcri(n_clones=8, n_phenotypes=5, n_genes=40, n_cells=300,
                         omega_concentration=0.4, fuzziness=0.1, seed=0)


def _fresh(adata, **kw):
    from tcri.model._model import TCRIModel

    pyro.clear_param_store()
    TCRIModel.setup_anndata(
        adata, layer="counts", clonotype_key="clone_id", phenotype_key="phenotype",
        covariate_key="covariate", batch_key="batch",
    )
    return TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1,
                     classifier_n_layers=1, classifier_hidden=16, K=5, seed=0, **kw)


class _ValidationWatcher(torch.nn.Module):
    """Records the guide tensors either side of every validation loop."""

    def __init__(self):
        super().__init__()
        self.deltas = []
        self._before = None

    def on_validation_start(self, trainer, pl_module):
        self._before = _snapshot(pl_module.module)

    def on_validation_end(self, trainer, pl_module):
        after = _snapshot(pl_module.module)
        if self._before:
            self.deltas.append(
                sum(float((after[k] - self._before[k]).abs().sum()) for k in self._before)
            )


def test_validation_does_not_update_parameters(adata):
    """A validation pass must not move a single parameter (contract I2).

    ``q_p_c_raw``/``q_p_ct_raw`` are not LightningModule parameters, so the gradient zeroing
    Lightning does before a validation loop does not protect them. If ``validation_step``
    reaches an optimizer step, weight decay alone moves them -- in the unconstrained log space
    of a positive-constrained parameter, which pulls every clone row toward uniform. Every
    metric reads those two tensors, so a held-out evaluation would be changing the quantity it
    reports.

    The second assertion checks the module is in train() mode for each training batch, because
    a fit that silently runs with dropout disabled produces no other symptom.
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


def test_train_resets_module_mode(adata):
    """A fit must not depend on the module mode an earlier inference call left behind.

    ``predict()``, ``to_anndata()`` and a session load all leave the module in eval mode, and
    Lightning restores per-submodule ``training`` flags around validation but never forces train
    mode for a fit. A ``train()`` that follows any of them would therefore run the ENTIRE fit
    with classifier dropout off and encoder BatchNorm frozen at its running statistics -- and
    do so bit-reproducibly, so a seed check cannot see it. Contract B7: a fit is a function of
    (seed, data, knobs) and of nothing that ran before ``train()``.
    """
    import lightning.pytorch as pl

    m = _fresh(adata)
    m.module.eval()
    m.get_latent_representation(adata)      # the inference call that leaves eval mode behind
    assert not m.module.training, "fixture did not leave the module in eval mode"

    seen = []

    class _Spy(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            mod = pl_module.module
            bn = [s for s in mod.encoder.modules() if isinstance(s, torch.nn.BatchNorm1d)]
            do = [s for s in mod.classifier.modules() if isinstance(s, torch.nn.Dropout)]
            assert bn and do, "fixture has no BatchNorm/Dropout to observe"
            seen.append((bool(mod.training), all(s.training for s in bn), all(s.training for s in do)))

    m.train(max_epochs=2, batch_size=128, accelerator="cpu", callbacks=[_Spy()],
            enable_progress_bar=False, enable_model_summary=False)

    assert seen, "no training batch ran -- the test asserted nothing"
    assert all(mode for mode, _, _ in seen), (
        "train() ran the fit with the module in eval mode -- it must call module.train() "
        "before fitting; an earlier predict()/to_anndata()/session load otherwise "
        "silently disables dropout and freezes BatchNorm for the whole fit"
    )
    assert all(bn for _, bn, _ in seen), "encoder BatchNorm was frozen (eval mode) during training"
    assert all(do for _, _, do in seen), "classifier Dropout was off (eval mode) during training"


def test_kl_ramp_is_monotone_across_resumed_training(adata):
    """The warmup counter belongs to the model, not to a per-call training plan.

    ``train()`` builds a fresh ``UnifiedTrainingPlan`` on every call, so a counter owned by the
    plan restarts the ramp: a staged or resumed fit then sees a sawtooth ``kl_weight`` instead
    of a monotone one, and its two segments descend different objectives (contract I5 and B1).
    """
    m = _fresh(adata)
    m.train(max_epochs=3, batch_size=128, accelerator="cpu",
            enable_progress_bar=False, enable_model_summary=False)
    after_first = m.module._kl_warmup_epochs
    weight_first = float(m.module.kl_weight)

    m.train(max_epochs=3, batch_size=128, accelerator="cpu",
            enable_progress_bar=False, enable_model_summary=False)
    after_second = m.module._kl_warmup_epochs
    weight_second = float(m.module.kl_weight)

    assert after_first > 0, "no warmup epochs were counted in the first call"
    assert after_second > after_first, (
        f"the warmup counter restarted: {after_first} -> {after_second}. A resumed fit must "
        f"continue the schedule, not begin a second ramp."
    )
    assert weight_second >= weight_first, (
        f"kl_weight went backwards across train() calls: {weight_first:.6g} -> "
        f"{weight_second:.6g}. The annealing schedule must be monotone."
    )


# ── the stopping policy: I3 and I4 ───────────────────────────────────────────

def _kl_weight_spy():
    import lightning.pytorch as pl

    class Spy(pl.Callback):
        def __init__(self):
            super().__init__()
            self.weights, self.positions, self.batches = [], [], None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.positions.append(
                pl_module.module.kl_warmup_position(trainer.num_training_batches))

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.weights.append(float(pl_module.module.kl_weight))
            self.batches = int(trainer.num_training_batches)

    return Spy()


def test_schedule_arguments_are_counted_in_epochs(adata):
    """Contract B2: every schedule setting is counted in epochs.

    Neither ``train()`` nor the training plan takes a schedule argument counted in optimizer
    steps; the old step-counted warmup raises before any fit and names its replacement. Lightning's
    ``max_steps`` limits a run rather than setting the schedule, so it stays accepted.
    """
    import inspect

    from tcri.model._model import TCRIModel, _accepted_train_kwargs
    from tcri.model._training import UnifiedTrainingPlan

    for fn in (TCRIModel.train, UnifiedTrainingPlan.__init__):
        steps = [name for name in inspect.signature(fn).parameters if "step" in name]
        assert not steps, f"{fn.__qualname__} takes a step-counted schedule argument: {steps}"
    m = _fresh(adata)
    with pytest.raises(TypeError, match="n_epochs_kl_warmup"):
        m.train(max_epochs=1, batch_size=128, accelerator="cpu", n_steps_kl_warmup=10)
    assert getattr(m, "training_record_", None) is None, "the fit ran before the error"
    assert "max_steps" in _accepted_train_kwargs()


def test_kl_weight_rises_within_an_epoch(adata):
    """Contract B2: the warmup position is fractional within an epoch, so ``kl_weight`` rises at
    every step. Three batches per epoch and two warmup epochs: k/6 of the ceiling at step k, then
    the ceiling."""
    spy = _kl_weight_spy()
    m = _fresh(adata)
    m.train(max_epochs=4, batch_size=128, n_epochs_kl_warmup=2, early_stopping=False,
            accelerator="cpu", callbacks=[spy], enable_progress_bar=False,
            enable_model_summary=False)
    assert spy.batches == 3, f"the fixture has {spy.batches} batches per epoch, not 3"
    top = m.module.max_kl_weight
    expected = [max(1e-6, top * k / 6) for k in range(6)] + [top] * (len(spy.weights) - 6)
    np.testing.assert_allclose(spy.weights, expected, rtol=1e-12)


def test_the_ramp_ends_at_the_same_epoch_whatever_the_batches_per_epoch(adata):
    """Contract B2/B5: a warmup counted in epochs opens selection at the same epoch whether an
    epoch is 3 batches or 9, which a warmup counted in steps does not."""
    from tcri.model._callbacks import ramp_is_complete
    import lightning.pytorch as pl

    class FirstGated(pl.Callback):
        def __init__(self):
            super().__init__()
            self.epoch = None

        def on_validation_end(self, trainer, pl_module):
            if self.epoch is None and not trainer.sanity_checking and ramp_is_complete(pl_module):
                self.epoch = int(trainer.current_epoch)

    found = {}
    for batch_size in (128, 32):
        spy, gate = _kl_weight_spy(), FirstGated()
        m = _fresh(adata)
        m.train(max_epochs=4, batch_size=batch_size, n_epochs_kl_warmup=2, early_stopping=False,
                accelerator="cpu", callbacks=[spy, gate], enable_progress_bar=False,
                enable_model_summary=False)
        found[spy.batches] = (gate.epoch, m.training_record_["ramp_completes_at_epoch"])
    assert sorted(found) == [3, 9], f"the fixtures have {sorted(found)} batches per epoch"
    assert found[3] == found[9] == (1, 2), f"selection opened at different epochs: {found}"


def test_an_epoch_cut_short_counts_as_completed(adata):
    """Contracts B1/B2: a ``train()`` stopped mid-epoch leaves a partial epoch, which the next
    ``train()`` counts as completed, so the position and ``kl_weight`` only move up across the
    calls."""
    import warnings as _w

    first, second = _kl_weight_spy(), _kl_weight_spy()
    m = _fresh(adata)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        m.train(max_epochs=10, max_steps=4, batch_size=128, n_epochs_kl_warmup=6,
                accelerator="cpu", callbacks=[first], enable_progress_bar=False,
                enable_model_summary=False)
    assert (m.module._kl_warmup_epochs, m.module._kl_warmup_batch) == (1, 1), (
        "max_steps=4 at 3 batches per epoch should stop one batch into the second epoch")
    m.train(max_epochs=2, batch_size=128, n_epochs_kl_warmup=6, accelerator="cpu",
            callbacks=[second], enable_progress_bar=False, enable_model_summary=False)
    assert second.positions[0] == 2.0, f"the second call started at {second.positions[0]}"
    weights = first.weights + second.weights
    assert all(b >= a for a, b in zip(weights, weights[1:])), f"kl_weight went down: {weights}"


def _plan_and_batch(adata, n_epochs_kl_warmup=3):
    """A fitted-enough model plus one validation batch, ready to evaluate."""
    from tcri.model._training import UnifiedTrainingPlan

    m = _fresh(adata)
    m.train(max_epochs=2, batch_size=128, n_epochs_kl_warmup=n_epochs_kl_warmup,
            accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    plan = UnifiedTrainingPlan(module=m.module, n_epochs_kl_warmup=n_epochs_kl_warmup)
    loader = m._make_data_loader(adata=m.adata, batch_size=128, shuffle=False)
    return m, plan, next(iter(loader))


def test_svi_steps_on_the_plans_loss(adata):
    """Contract I1: SVI descends the estimator the plan declares, a one-particle Trace_ELBO.

    A loss the plan builds but does not hand to SVI is invisible to every other test, because
    the fit still runs on whatever SVI was given. scvi's own default is also a one-particle
    Trace_ELBO, so the hand-off is checked on a plan built with two particles, which only a
    declared loss can give SVI. The loss SVI evaluates on a batch must then equal a
    one-particle Trace_ELBO on the same batch and draw.
    """
    from pyro.infer import Trace_ELBO

    from tcri.model._training import UnifiedTrainingPlan

    m, plan, batch = _plan_and_batch(adata)
    assert plan.svi.loss.__self__ is plan.loss_fn, "SVI does not step on the plan's loss"
    assert isinstance(plan.loss_fn, Trace_ELBO) and plan.loss_fn.num_particles == 1
    two = UnifiedTrainingPlan(module=m.module, n_epochs_kl_warmup=3, num_particles=2)
    assert two.svi.loss.__self__.num_particles == 2, "the plan's loss did not reach SVI"

    args, kwargs = plan.module._get_fn_args_from_batch(batch)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(0)
        got = plan.svi.evaluate_loss(*args, **kwargs)
        torch.manual_seed(0)
        ref = Trace_ELBO(num_particles=1).loss(plan.module.model, plan.module.guide, *args, **kwargs)
    assert got == pytest.approx(ref, rel=1e-6), f"SVI evaluates {got}, a one-particle ELBO {ref}"


def test_num_particles_averages_that_many_draws(adata):
    """Contract I1: ``train(num_particles=n)`` makes each step's ELBO the mean of ``n`` draws.

    The fit's own SVI, evaluated on a batch from a fixed seed, must equal the mean of ``n``
    one-draw ELBOs taken in sequence from the same seed; a count that did not reach SVI gives
    one draw instead. The count is recorded with the other arguments, so a null replays it.
    """
    from pyro.infer import Trace_ELBO

    n = 3
    m = _fresh(adata)
    m.train(max_epochs=1, batch_size=128, n_epochs_kl_warmup=2, num_particles=n,
            accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    assert m.train_kwargs_["num_particles"] == n, "num_particles is not recorded for nulls"
    plan = m.trainer.lightning_module
    loader = m._make_data_loader(adata=m.adata, batch_size=128, shuffle=False)
    args, kwargs = plan.module._get_fn_args_from_batch(next(iter(loader)))
    one = Trace_ELBO(num_particles=1)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(0)
        got = plan.svi.evaluate_loss(*args, **kwargs)
        torch.manual_seed(0)
        draws = [one.loss(plan.module.model, plan.module.guide, *args, **kwargs) for _ in range(n)]
    assert len(set(draws)) == n, "the draws coincide; the comparison cannot tell 1 from n"
    assert got == pytest.approx(sum(draws) / n, rel=1e-5), (
        f"SVI evaluates {got}; the mean of {n} draws is {sum(draws) / n}, one draw {draws[0]}"
    )

    # The validation criterion takes one draw whatever the count (I1, I3), so early stopping
    # reads the same quantity at any num_particles.
    from tcri.model._training import UnifiedTrainingPlan

    batch = next(iter(loader))
    plan.module.eval()
    crit = {k: float(UnifiedTrainingPlan(module=plan.module, n_epochs_kl_warmup=2, num_particles=k)
                     .validation_step(batch, 0)["loss"]) for k in (1, n)}
    assert crit[n] == crit[1], f"the validation criterion depends on num_particles: {crit}"


def test_monitor_is_invariant_to_ramp_position(adata):
    """I3: the monitored quantity is a fixed function of the parameters.

    Evaluate the criterion at two different ramp positions with the parameters held EXACTLY
    fixed. A criterion that is a function of (Lambda, Theta) must return the same number; one
    that inherits the annealed kl_weight, or redraws its Monte-Carlo sample, will not -- and an
    argmin over a moving quantity is not an argmin, so early stopping would be selecting on the
    schedule rather than on the fit.

    Both clauses are needed, so the assertion is exact rather than approximate: pinning
    kl_weight without also fixing the evaluation seed leaves the estimator redrawing at every
    check, which shows up only in the low-order digits.
    """
    m, plan, batch = _plan_and_batch(adata)
    plan.module.eval()

    before = {k: v.detach().clone() for k, v in plan.module.state_dict().items()}
    store_before = {n: p.detach().clone() for n, p in pyro.get_param_store().named_parameters()}

    plan.module._kl_warmup_epochs = 0         # early in the ramp
    plan.module.kl_weight = 1e-6
    first = float(plan.validation_step(batch, 0)["loss"])

    # identical parameters, a different point on the schedule
    plan.module.load_state_dict(before, strict=False)
    with torch.no_grad():
        for name, p in pyro.get_param_store().named_parameters():
            p.data.copy_(store_before[name])
    plan.module._kl_warmup_epochs = 10_000    # ramp long finished
    plan.module.kl_weight = plan.module.max_kl_weight
    second = float(plan.validation_step(batch, 0)["loss"])

    assert first == second, (
        f"the monitored quantity moved with ramp position while the parameters were held "
        f"fixed: {first!r} vs {second!r}. It is therefore not a function of (Lambda, Theta), "
        f"and an argmin over it is not an argmin (contract I3)."
    )


def test_validation_pin_restores_the_training_schedule(adata):
    """Contract B1: the pin is scoped to the check.

    validation_step raises kl_weight to max_kl_weight to make the criterion well-posed. If it
    left it there, the next training step would read a kl_weight it never scheduled, and the
    ramp would jump to its endpoint the first time anything validated.
    """
    m, plan, batch = _plan_and_batch(adata)
    plan.module._kl_warmup_epochs = 1
    plan.module.kl_weight = 0.125

    plan.validation_step(batch, 0)

    assert plan.module.kl_weight == 0.125, (
        f"validation left kl_weight at {plan.module.kl_weight}; the pin must be undone so the "
        f"training schedule is the only thing that advances it (contract B1)."
    )


def test_selection_is_gated_until_the_ramp_completes(adata):
    """Contract B5: no check is recorded before the ramp finishes.

    Every entry in the monitored series must come from the same objective. A run whose ramp
    never completes has no comparable pair, so it must select nothing, keep its final weights,
    and say so in the record rather than silently reporting an epoch.
    """
    m = _fresh(adata)
    with pytest.warns(UserWarning, match="KL ramp did not complete"):
        m.train(max_epochs=3, batch_size=128, n_epochs_kl_warmup=10**6,
                accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)

    rec = m.training_record_
    assert rec["ramp_completed"] is False
    assert rec["selected_epoch"] is None, (
        "a checkpoint was selected from checks taken at different kl_weights; with the ramp "
        "incomplete no two checks share an objective (contract B5)."
    )
    assert rec["selection_criterion"] == "last epoch (ramp incomplete)"


def test_restored_model_is_the_selected_one(adata):
    """Contract I4: what train() leaves behind is what the criterion chose.

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
                pyro.get_param_store()[m.module.pname("q_p_ct_raw")].detach().clone(),
                sd[NET].detach().clone(),
                sd[BN].detach().clone(),
            ))

    # lr=1e-2 over 60 epochs puts the argmin well before the last check on this fixture. At the
    # default lr the criterion is still descending at the final epoch, so "keep the final
    # weights" would pass by accident -- the assertion below fails if the fixture ever drifts
    # back to monotone.
    # Stopping is switched off so that what is checked here is the restore alone, whatever the
    # stopping defaults are; test_fit_stops_once_improvements_fall_below_min_delta covers the
    # restore in a fit that stopped. The patience of 2 would end this fit long before epoch 60
    # (the argmin is well before the last check, asserted below), so reaching 60 shows the
    # switch removed the rule rather than the rule never firing.
    m = _fresh(adata)
    m.train(max_epochs=60, batch_size=128, n_epochs_kl_warmup=2, lr=1e-2, accelerator="cpu",
            early_stopping=False, early_stopping_patience=2, callbacks=[_Spy()],
            enable_progress_bar=False, enable_model_summary=False)
    assert m.training_record_["epochs_run"] == 60, "early_stopping=False must train to max_epochs"
    assert m.training_record_["stopped_early"] is False

    assert len(seen) > 2, f"only {len(seen)} gated checks ran; the test asserted nothing"
    best_score, best_ct, best_net, best_bn = min(seen, key=lambda r: r[0])
    assert best_score < seen[-1][0], (
        "the best check WAS the last one, so keeping final weights would pass by accident. "
        "This fixture no longer discriminates."
    )

    sd = m.module.state_dict()
    assert torch.equal(
        pyro.get_param_store()[m.module.pname("q_p_ct_raw")].detach(), best_ct), (
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


def _discriminating_min_delta(values, patience):
    """A threshold under which the rule stops strictly earlier than it does without one.

    Candidates are midpoints between the sizes of successive changes in the series, smallest
    first. One is taken only if the stop it gives is the same at 1% either side of it, so the
    float32 comparison inside the callback and the float64 replay here cannot disagree.
    """
    from ._stopping import stop_index

    stop_at_zero = stop_index(values, 0.0, patience)
    steps = sorted({abs(b - a) for a, b in zip(values, values[1:]) if b != a})
    for lo, hi in zip(steps, steps[1:]):
        delta = 0.5 * (lo + hi)
        stop = stop_index(values, delta, patience)
        if stop is None or (stop_at_zero is not None and stop >= stop_at_zero):
            continue
        if stop_index(values, 0.99 * delta, patience) == stop == stop_index(values, 1.01 * delta, patience):
            return delta
    return None


def test_fit_stops_once_improvements_fall_below_min_delta(adata):
    """Contract B3 and B4: a check restarts the patience count only when it beats the best check
    by more than ``early_stopping_min_delta``, and ``early_stopping_patience`` such checks in a
    row stop the fit.

    Two fits of one seed in one process. The first, with stopping switched off, records the
    gated series and picks a threshold under which the rule stops strictly earlier than it
    would with none: at that stop, some check inside the patience window was a new minimum by
    less than the threshold. The second fit runs with that threshold and patience and must run
    the same series (B7) and stop exactly where the rule says.
    The weights it ends on are still the lowest check's (I4), which lies after the last
    improvement the rule counted.
    """
    from ._stopping import GatedSeries, stop_index

    patience = 5
    fit = dict(max_epochs=60, batch_size=128, n_epochs_kl_warmup=2, lr=1e-2, accelerator="cpu",
               enable_progress_bar=False, enable_model_summary=False)

    probe = GatedSeries()
    _fresh(adata).train(early_stopping=False, callbacks=[probe], **fit)
    series = probe.values
    assert len(series) > 2 * patience, f"only {len(series)} gated checks; nothing is asserted"
    delta = _discriminating_min_delta(series, patience)
    assert delta is not None, (
        "no min_delta makes the rule stop earlier than it does without one on this series; "
        "the fixture no longer discriminates"
    )

    spy = GatedSeries()
    m = _fresh(adata)
    m.train(early_stopping_min_delta=delta, early_stopping_patience=patience, callbacks=[spy], **fit)
    got = spy.values

    assert got == series[:len(got)], "same seed, different series (B7); the comparison is void"
    assert len(got) - 1 == stop_index(series, delta, patience), (
        f"the fit stopped after {len(got)} gated checks; with min_delta={delta:.3g} and "
        f"patience {patience} the rule stops after {stop_index(series, delta, patience) + 1}"
    )
    rec = m.training_record_
    assert rec["stopped_early"] is True
    assert rec["selected_score"] == min(got), "the fit did not end on its lowest check (I4)"


def test_default_threshold_counts_every_new_low(adata):
    """Contract B4: at the default ``early_stopping_min_delta`` every new low restarts the count.

    A fit with the default threshold must stop where the zero-threshold rule stops on the
    series recorded with stopping off, or run to ``max_epochs`` if that rule never stops. The
    fixture must be one on which some positive threshold stops strictly earlier, so the
    comparison can tell a threshold from none.
    """
    from ._stopping import GatedSeries, stop_index

    patience = 5
    fit = dict(max_epochs=60, batch_size=128, n_epochs_kl_warmup=2, lr=1e-2, accelerator="cpu",
               enable_progress_bar=False, enable_model_summary=False)

    probe = GatedSeries()
    _fresh(adata).train(early_stopping=False, callbacks=[probe], **fit)
    series = probe.values
    assert _discriminating_min_delta(series, patience) is not None, (
        "no positive min_delta stops earlier than none on this series; the fixture no longer "
        "discriminates"
    )

    spy = GatedSeries()
    _fresh(adata).train(early_stopping_patience=patience, callbacks=[spy], **fit)
    got = spy.values
    assert got == series[:len(got)], "same seed, different series (B7); the comparison is void"
    stop = stop_index(series, 0.0, patience)
    expected = len(series) if stop is None else stop + 1
    assert len(got) == expected, (
        f"the default threshold ran {len(got)} gated checks; with no threshold and patience "
        f"{patience} the rule runs {expected}"
    )


@pytest.mark.parametrize("kw", [
    {"early_stopping_warmup_epochs": 10},
    {"early_stopping_monitor": "elbo_validation"},
    {"early_stopping_mode": "max"},
    {"trainer_config": {"check_val_every_n_epoch": 3}},
    {"learning_rate_monitor": True},
    {"enable_checkpointing": True},
    {"checkpointing_monitor": "objective_validation_percell"},
    {"check_val_every_n_epoch": 5},
    {"check_val_every_n_epoch": None},
    {"val_check_interval": 0.5},
])
def test_train_rejects_arguments_it_replaces(adata, kw):
    """Contract B11: a Trainer argument this model replaces raises instead of passing through.

    Each is accepted by scvi's or Lightning's Trainer, so the unknown-name guard alone would let
    it through; forwarded, it would be ignored, overwritten, or would change what patience
    counts. The error must come before any fit and name the argument.
    """
    m = _fresh(adata)
    with pytest.raises(TypeError, match=next(iter(kw))):
        m.train(max_epochs=1, batch_size=128, accelerator="cpu", **kw)
    assert getattr(m, "training_record_", None) is None, "the fit ran before the error"


def test_every_scvi_trainer_argument_is_forwarded_or_replaced():
    """Contract B11: each argument scvi's Trainer and TrainRunner add to Lightning's is either
    forwarded with its plain meaning, set by ``train()`` itself, or raises.

    scvi's own early-stopping and checkpointing arguments only act when scvi's stopping is on,
    which ``train()`` switches off, so an unreviewed one passes the unknown-name guard and does
    nothing. An scvi upgrade that adds an argument fails here until it is sorted into one of
    these sets.
    """
    import inspect

    from lightning.pytorch import Trainer as LightningTrainer
    from scvi.train import Trainer as ScviTrainer
    from scvi.train import TrainRunner

    from tcri.model._model import TCRIModel, _REPLACED_TRAIN_ARGS

    forwarded = {"accelerator", "devices", "benchmark", "max_epochs", "default_root_dir",
                 "num_sanity_val_steps", "enable_model_summary", "enable_progress_bar",
                 "progress_bar_refresh_rate", "simple_progress_bar", "logger",
                 "log_every_n_steps", "log_save_dir", "check_val_every_n_epoch"}
    set_by_train = {"model", "training_plan", "data_splitter", "early_stopping",
                    "early_stopping_patience", "early_stopping_min_delta"}
    lightning = set(inspect.signature(LightningTrainer.__init__).parameters)
    scvi_added = (set(inspect.signature(ScviTrainer.__init__).parameters)
                  | set(inspect.signature(TrainRunner.__init__).parameters)) - lightning
    scvi_added -= {"self", "kwargs", "trainer_kwargs"}
    unsorted = sorted(scvi_added - forwarded - set_by_train - set(_REPLACED_TRAIN_ARGS))
    assert not unsorted, (
        f"scvi Trainer/TrainRunner arguments nobody has sorted: {unsorted}. Forward each with its "
        f"plain meaning, or add it to _REPLACED_TRAIN_ARGS with what to use instead."
    )
    own = set(inspect.signature(TCRIModel.train).parameters)
    assert {"early_stopping", "early_stopping_patience", "early_stopping_min_delta"} <= own


def test_a_fit_ended_by_another_limit_is_not_an_early_stop(adata):
    """Contract B9: ``stopped_early`` says whether the stopping rule ended the fit.

    A Lightning limit such as ``max_steps`` also ends a fit before ``max_epochs``; recording that
    as an early stop would report a truncated fit as converged.
    """
    import warnings as _w

    m = _fresh(adata)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        m.train(max_epochs=20, max_steps=6, batch_size=128, n_epochs_kl_warmup=0,
                accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    rec = m.training_record_
    assert rec["epochs_run"] < 20, "max_steps did not end the fit; the test asserts nothing"
    assert rec["stopped_early"] is False, "a max_steps cut was recorded as an early stop"
    assert any("without the early-stopping rule firing" in str(c.message) for c in caught)


def test_monitor_excludes_the_global_block(adata):
    """Contract I3, scope: the monitor is the per-cell block, not the ELBO.

    This is the deliberate departure the contract records, and it needs its own test: including
    the global sites does NOT break ramp-invariance (the pin fixes kl_weight either way), so
    ``test_monitor_is_invariant_to_ramp_position`` cannot see the difference. Without this,
    re-adding ``p_c``/``p_ct`` to the monitored number would be a silent change.

    Both global plates are declared at full size with no subsampling, so their KL is the same
    number whichever cells are held out — a training-set quantity a validation criterion must
    not contain.

    The per-cell block comes out of the trace already scaled by ``plate_size()/B`` (the data
    plate carries the dataset size, I8), so "per cell" means dividing by ``plate_size()``,
    not by the batch size.
    """
    m, plan, batch = _plan_and_batch(adata)
    plan.module.eval()

    out = plan.validation_step(batch, 0)
    n_cells = int(plan.module.plate_size())

    args, kwargs = plan.module._get_fn_args_from_batch(batch)
    prev = plan.module.kl_weight
    try:
        plan.module.kl_weight = plan.module.max_kl_weight
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
    """Reaching max_epochs means the stopping rule never fired -- say so.

    A truncated fit and a converged one are otherwise indistinguishable: same record shape,
    same outputs, no signal to the caller. The objective can still be descending when the
    budget runs out, in which case every metric read off the fit understates its converged
    value, so the record must carry ``stopped_early`` and the warning must name the cap rather
    than convergence (contract B9).
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

    train() forwards ``**kwargs`` to TrainRunner and on into ``Trainer.__init__``, so without
    this guard a name train() does not accept surfaces as

        TypeError: Trainer.__init__() got an unexpected keyword argument 'validation_size'

    from inside lightning, with nothing pointing at tcri or at what the caller should do
    instead -- and it surfaces whenever that frame is first reached, which can be long after
    the caller has walked away from the fit.
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


# ── B8: weight decay stops at the guide concentrations ───────────────────────

def test_weight_decay_does_not_reach_the_guide_concentrations(adata):
    """Contract B8, tested on the mechanism rather than the wiring.

    Pyro's optimizer takes one ``weight_decay`` for every parameter in the store, and the guide
    concentrations' leaves are ``log θ``: L2 decay there is a flat ``Dirichlet(1, …, 1)`` prior
    applied through the optimizer, i.e. an undeclared optimizer setting acting as a prior. The
    guide is therefore routed through a per-parameter ``optim_args`` with ``weight_decay=0``.

    The discriminating step: give a guide leaf AND a network weight an all-zero gradient, step
    each once through the plan's Pyro optimizer, and compare. With decay the only force on the
    parameter, the leaf must be bit-identical and the weight must have moved. Reading
    ``weight_decay`` back off the optimizer instead would pass with a callable that returns
    ``base`` for every name.
    """
    from tcri.model._training import GUIDE_CONCENTRATION_PARAMS, UnifiedTrainingPlan

    m = _fresh(adata)
    plan = UnifiedTrainingPlan(
        module=m.module, n_epochs_kl_warmup=4, reconstruction_loss_scale=1e-2,
        optimizer_config={"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-5,
                          "weight_decay": 1e-4},
    )
    # one guide + model pass registers every parameter with the store
    tensors = next(iter(m._make_data_loader(adata=adata, batch_size=64)))
    args, kwargs = m.module._get_fn_args_from_batch(tensors)
    with torch.no_grad():
        m.module.guide(*args, **kwargs)
        m.module.model(*args, **kwargs)

    store = pyro.get_param_store()
    names = [m.module.pname(k) for k in sorted(GUIDE_CONCENTRATION_PARAMS)]
    assert set(names) <= set(store.keys())
    leaves = [pyro.param(k).unconstrained() for k in names]
    weight = next(p for n, p in m.module.encoder.named_parameters() if p.ndim == 2)

    before_leaves = [t.detach().clone() for t in leaves]
    before_weight = weight.detach().clone()
    for t in (*leaves, weight):
        t.grad = torch.zeros_like(t)
    plan.optim([*leaves, weight])

    for k, b, t in zip(sorted(GUIDE_CONCENTRATION_PARAMS), before_leaves, leaves):
        assert torch.equal(b, t.detach()), (
            f"{k} moved under a zero gradient: weight decay is still reaching the guide"
        )
    assert not torch.equal(before_weight, weight.detach()), (
        "the network weight did not move under a zero gradient, so the test is not "
        "exercising weight decay at all -- check the optimizer wiring"
    )


# ── I8: a minibatch is an unbiased estimate of eq 7 ──────────────────────────

def _conditioned_objective(mod, args, kwargs, *, fixed, z_all):
    """The batch objective with every stochastic site pinned, so it is a function of the
    parameters and the batch alone. Same site walk as ``UnifiedTrainingPlan._objective_blocks``."""
    import contextlib
    import io

    from pyro import poutine

    idx = args[3]
    data = {**fixed, "latent": z_all[idx]}
    cond_model = poutine.condition(mod.model, data=data)
    cond_guide = poutine.condition(mod.guide, data=data)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        gt = poutine.trace(cond_guide).get_trace(*args, **kwargs)
        mt = poutine.trace(poutine.replay(cond_model, trace=gt)).get_trace(*args, **kwargs)
    mt.compute_log_prob()
    gt.compute_log_prob()
    total = 0.0
    for tr, sign in ((mt, 1.0), (gt, -1.0)):
        for site in tr.nodes.values():
            if site["type"] in ("sample", "factor") and site.get("log_prob") is not None:
                total += sign * float(site["log_prob"].sum())
    return total


def test_minibatch_objective_is_unbiased_for_the_full_batch(adata):
    """Contract I8: the mean of the batch objectives over a partition of the cells equals the
    full-batch objective.

    eq 7 of ``governance/MODEL_CONTRACT.md`` sums the per-cell terms over the N cells and the
    two Dirichlet KLs once. A minibatch estimates that only if the data plate is declared at
    ``size = N`` with the batch as its subsample, so Pyro scales the per-cell sites by N/B and
    the Dirichlet KLs enter once per pass. Declared at ``size = B`` instead, each step counts
    the global KLs at full weight against B cells of data, so over an epoch of S steps the
    prior pull on ω_c and φ_m is S times eq 7's and the identity below fails by a wide margin.

    Every stochastic site is pinned (one z per cell from the encoder mean, one draw of p_c and
    p_ct shared by every batch) so the comparison is deterministic and the tolerance is float
    rounding, not Monte-Carlo noise.
    """
    import contextlib
    import io

    from pyro import poutine

    m = _fresh(adata)
    mod = m.module
    mod.eval()
    mod.kl_weight = mod.max_kl_weight
    n, B = adata.n_obs, 50
    assert n % B == 0, "the partition must be into equal batches for the identity to be exact"
    assert mod.plate_size() == n, "a fresh model's plate spans every cell"

    full_batch = next(iter(m._make_data_loader(adata=adata, batch_size=n, shuffle=False)))
    fargs, fkw = mod._get_fn_args_from_batch(full_batch)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        z_all = mod.encoder(fargs[0], fargs[1])[0]
        g = poutine.trace(mod.guide).get_trace(*fargs, **fkw)
    fixed = {k: g.nodes[k]["value"].detach() for k in ("p_c", "p_ct")}

    full = _conditioned_objective(mod, fargs, fkw, fixed=fixed, z_all=z_all)
    parts = [
        _conditioned_objective(mod, *mod._get_fn_args_from_batch(b), fixed=fixed, z_all=z_all)
        for b in m._make_data_loader(adata=adata, batch_size=B, shuffle=False)
    ]
    assert len(parts) == n // B
    assert float(np.mean(parts)) == pytest.approx(full, rel=1e-4), (
        f"mean of {len(parts)} batch objectives {np.mean(parts):.4f} != full-batch objective "
        f"{full:.4f}: the minibatch estimator is biased for eq 7 (contract I8)"
    )


def test_plate_size_tracks_the_training_split(adata):
    """The plate's ``size`` is the number of cells the training loader draws from, i.e. the
    split the runner actually made -- not the full object, and not a hardcoded 0.9."""
    import contextlib
    import io

    m = _fresh(adata)
    assert m.module.n_obs_training is None and m.module.plate_size() == adata.n_obs
    with contextlib.redirect_stdout(io.StringIO()):
        m.train(max_epochs=1, batch_size=128, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    assert m.module.n_obs_training == len(m.train_indices) == m.module.plate_size()
    assert m.module.plate_size() < adata.n_obs


# ── structural guards ────────────────────────────────────────────────────────

def test_no_reset_knob_on_train():
    """Contract B1: the KL schedule is non-decreasing within and across ``train()`` calls, so
    ``train()`` must expose no way to restart the ramp -- and a new keyword there would also
    change the signature ``governance/API_CONTRACT.md`` pins."""
    import inspect

    from tcri.model._model import TCRIModel

    assert "reset_schedule" not in inspect.signature(TCRIModel.train).parameters, (
        "a reset knob reappeared on train(); see TRAINING_CONTRACT.md B1"
    )


def test_warmup_counter_is_owned_by_the_module_not_the_plan():
    """Contract B1/I5: ``train()`` builds a fresh plan per call, so a plan-local counter
    restarts the ramp on every resumed fit. The counter must live on the module, which
    survives."""
    import inspect

    from tcri.model._training import UnifiedTrainingPlan

    assert not hasattr(UnifiedTrainingPlan, "_my_global_step"), (
        "the plan owns a warmup counter again — a fresh plan per train() call means the ramp "
        "restarts (DE-4)"
    )
    assert "self.module.kl_warmup_position" in inspect.getsource(UnifiedTrainingPlan.training_step)


def _body_source(fn) -> str:
    """Source with the docstring removed, so prose about a removed call cannot match."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    return ast.unparse(node)


def test_validation_step_does_not_call_training_step():
    """Contract I2, structurally; the behavioural proof is
    ``test_validation_does_not_update_parameters`` above. Catches a reintroduction at review."""
    from tcri.model._training import UnifiedTrainingPlan

    src = _body_source(UnifiedTrainingPlan.validation_step)
    assert "super().training_step" not in src, "validation_step reaches SVI.step() again (DE-1)"
    assert "_objective_blocks" in src, "validation_step must evaluate through _objective_blocks"
    assert "max_kl_weight" in src and "finally" in src, "the kl_weight pin and its restore (I3, B1)"
    assert "fork_rng" in src and "manual_seed" in src, "the fixed evaluation seed (I3)"


def test_every_test_the_contract_names_exists():
    """TRAINING_CONTRACT.md names the test that enforces each statement. A name that no test
    carries is a claim nothing checks."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "governance" / "TRAINING_CONTRACT.md").read_text()
    named = set(re.findall(r"`(test_[a-z0-9_]+)`", text))
    assert named, "the contract names no tests"
    defined = set()
    for py in (root / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_[a-z0-9_]+)\(", py.read_text(), re.M))
    missing = sorted(named - defined)
    assert not missing, f"TRAINING_CONTRACT.md names tests that do not exist: {missing}"
