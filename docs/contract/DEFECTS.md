# Defect register

Audited 2026-08-06. Each entry carries confirmation from `main`, the candidate fixes considered,
the chosen fix with an implementation plan, downstream dependencies including which metric
outputs move, and the tests that prove it landed.

Severity: `S1` changes reported numbers · `S2` wrong behaviour, numbers unaffected or small ·
`S3` correctness of record / harness only.

All 17 registered defects were confirmed on `main` at `46490e6`. The audit opened three more
(DE-18, DE-19, DE-20) and four open questions (Q-A…Q-D).

~~DE-18 blocks DE-3, DE-5, DE-6, DE-10 and DE-12.~~ **DE-18 is WITHDRAWN** (not a defect). DE-6 is closed as not-a-defect. DE-3 and DE-5 shipped on their own merits; DE-10 and DE-12 were re-derived without it and stand.

---

## DE-1 · S2 · FIXED (PR `read-only-validation`)

**Confirmed:** REAL, reproduced on main (46490e6).

Static chain: tcri/model/_training.py:141 `val_dict = super().training_step(batch, batch_idx)` inside `validation_step` -> .venv/.../scvi/train/_trainingplans.py:1501 `loss = torch.Tensor([self.svi.step(*args, **kwargs)])` -> .venv/.../pyro/infer/svi.py:153 `self.optim(params)` (a real optimizer step) followed by :156 `pyro.infer.util.zero_grads(params)`.

Measured (/tmp/de-training/de1_probe.py, 400 cells, 12 epochs, check_val_every_n_epoch=2, 6 validation checks):
```
   epoch   d_q_p_c  d_q_p_ct  d_nn_l1  n_nn_tensors_moved  n_train_outputs
       1  0.114189  0.177528      0.0                   0                7
      ...
      11  0.095686  0.138534      0.0                   0                7
total |Δ q_p_ct_raw| over validation: 0.8816   |Δ q_p_c_raw|: 0.6479   |Δ nn params|: 0.0
```
On a benchmark-shaped fixture (2000 cells / 200 genes / bs 1024, /tmp/de-training/de1_fix_and_cost.py): 8 checks, total drift 4.182, mean 0.52 per check — reproduces the register's 0.54.

Mechanism, nailed down (/tmp/de-training/de1_mechanism.py): at validation time all 22 nn tensors have `.grad is None`, because Lightning calls `on_validation_model_zero_grad` (lightning/pytorch/loops/training_epoch_loop.py:404 -> core/hooks.py:161 `self.zero_grad()`, set_to_none=True) — torch Adam then skips them. `q_p_c_raw`/`q_p_ct_raw` are NOT reachable from `LightningModule.parameters()` (they live only in the Pyro store; `pyro.get_param_store()` holds 24 keys = 22 `scvi$$$…` shadows + these two), so they keep the ZERO grad tensor pyro left there (pyro/infer/util.py:91 `p.grad = torch.zeros_like(p.grad)`). Adam therefore steps them on grad = weight_decay·θ (1e-4) plus stale momentum from the last training step, in the UNCONSTRAINED (log) space of a `constraints.positive` parameter — every entry pulled toward log θ = 0, i.e. rows pulled toward uniform. Systematic flattening, as the register says.

Second, undocumented consequence not in the register: `PyroTrainingPlan.training_step` appends to `self.training_step_outputs` (_trainingplans.py:1507), and Lightning runs the val loop inside `on_advance_end` BEFORE `on_train_epoch_end` averages and clears that list. Probe recorded 7 entries = 6 train batches + 1 validation batch, so `elbo_train` — the series `diag.loss` plots (tcri/diagnostics/_training.py:26) — is contaminated with validation-batch ELBOs.

**Candidates considered:** 1. **`self.svi.evaluate_loss(*args, **kwargs)`** (pyro/infer/svi.py:119-132) — evaluates the identical estimator with no `param_capture`, no `optim()`, no `zero_grads`. Verified identical: `Trace_ELBO.loss` (trace_elbo.py:64-80) returns `-(model_trace.log_prob_sum() - guide_trace.log_prob_sum())`, the same quantity `loss_and_grads` returns, through the same `self.model`/`self.guide` already wrapped in `block_fn(scale_fn(...))` on the SVI object; `pyro.factor("phenotype_alignment")` is a sample site and is included in `log_prob_sum()`. Returns a float, which the existing tensor-coercion branch at _training.py:146-149 already handles.
2. **Call the grandparent** `LowLevelPyroTrainingPlan.training_step` (_trainingplans.py:1345) — uses `differentiable_loss_fn` and does not step. But it builds an autograd graph for nothing and STILL appends to `training_step_outputs`, so it leaves the `elbo_train` contamination in place.
3. **Block the two params** via `PyroTrainingPlan(blocked=["q_p_c_raw","q_p_ct_raw"])` or a second `poutine.block`-wrapped SVI for validation. Heavier, changes the training SVI's construction, and blocks them in training too unless a second SVI is built.
4. **Snapshot-and-restore around the call.** Restores the values but still churns Adam's per-param state (`optim_objs`) with a bogus step; a patch over the symptom rather than a removal.

**Chosen fix:** **Candidate 1**, on both safety and speed: it is the smallest diff, removes the write rather than compensating for it, and provably preserves the monitored quantity.

File `tcri/model/_training.py`, method `UnifiedTrainingPlan.validation_step` (lines 138-169). Replace lines 139-149 with:
```
args, kwargs = self.module._get_fn_args_from_batch(batch)
was_training = self.module.training
self.module.eval()
try:
    loss = self.svi.evaluate_loss(*args, **kwargs)   # reads; does not step
finally:
    if was_training:
        self.module.train()
device = next(self.module.parameters()).device
val_dict = {"loss": torch.as_tensor(loss, dtype=torch.float32, device=device)}
```
Keep lines 151-168 verbatim: the `kl_divergence_with_prior_val` diagnostic is read by `diag.loss` (tcri/diagnostics/_training.py:29) and the `elbo_validation` log is what early stopping monitors. Do NOT set `self.module.kl_weight` in `validation_step` — inheriting the last training step's annealing weight is what makes `elbo_validation` comparable to `elbo_train`; add a comment saying so.

Add a comment above the new block pointing at `_model_contract.py`'s `optimizer_weight_decay` note, since this is the same class of bug it records as removed.

Prototyped and verified (/tmp/de-training/de1_fix_and_cost.py): with the fix, drift across 40 validation checks is **0.000000** exactly, and the `len(training_step_outputs)` assertion holds.

**Grouping.** DE-1..DE-4 are two coherent changes, not one, with a hard ordering constraint. Ship ONE branch off fresh `main` with TWO commits so each delta is attributable and the stale-base hazard in CLAUDE.md is not re-run: commit 1 = **DE-1 + DE-4** (pure mechanics, the code doing work it never intended; combined numeric effect below the 0.0018 fit-to-fit floor), commit 2 = **DE-3 + DE-2** (stopping policy and what is kept). DE-1 MUST precede DE-3, because today the state visible at `on_validation_end` has already absorbed that check's spurious optimizer step, so a best-weight snapshot would preserve the polluted state. DE-1 also makes DE-2's chosen fix affordable by removing the backward pass from every validation check.

**Contract change:** NONE. The model contract's `SANCTIONED_DEVIATIONS['optimizer_weight_decay']` (tcri/model/_model_contract.py:196-207) already states weight decay is "Applied inside Pyro's optimizer, where it acts on the ELBO gradients", and already records the identical "stepped AFTER SVI.step() had already zeroed the gradients" pathology as REMOVED. The validation path is a surviving instance of exactly that. Fixing it restores conformance with what the manifest already says; no manifest edit, no prose edit, no CODEOWNER review.

**Dependencies affected:** **Metric outputs that move, and direction.** Removing the flattening leaves `q_p_ct_raw` rows sharper. Through `get_p_ct()` -> `uns['tcri_p_ct']`: `tl.mutual_information` and `tl.compare_groups` move **UP**; `tl.clonotypic_entropy` and `tl.phenotypic_entropy` (conditional forms) move **DOWN**; `tl.phenotypic_flux` (D_KL) moves **UP**. Register's measured size: 0.5002 -> 0.5008 NMI (+0.0006, +0.12%) at 5000 cells / 2000 epochs. `model.predict`, and hence `obsm['X_tcri_probabilities']`, `obsm['X_tcri_logposterior']`, `obs['tcri_phenotype']`, shift with it.

**Published numbers: yes, but below the noise floor of a single fit.** +6e-4 sits under the measured 1.8e-3 unseeded fit-to-fit spread, so no individual reported figure changes visibly; the bias is systematic across fits, which is why it is worth removing.

**The larger effect is indirect.** `elbo_validation` at every check is now computed on parameters that were not perturbed by earlier checks, so the whole validation trajectory shifts and the early-stopping epoch can move. That changes which weights a run ends on — which is precisely why DE-3 must come after.

**`elbo_train` changes meaning** (stops including validation batches). `diag.loss` panel 1 changes shape slightly. No test reads it.

**Existing test expectations: none change.** `tests/test_model_guardrails.py::test_lr_and_weight_decay_reach_pyros_optimizer` asserts the neighbouring invariant ("nothing steps on zeroed gradients") and keeps passing — it is the natural home for the new test.

**Wall clock: neutral to slightly better.** Validation loses a backward pass and an optimizer step; measured 2.44 s -> 2.50 s at cv=5 on the 2000-cell fixture (within run-to-run noise).

**Verification tests:** **Proves it landed** (new, in `tests/test_model_guardrails.py`, which is not CODEOWNER-restricted):
1. `test_validation_step_does_not_step_the_optimizer` — build a plan, run >=1 `training_step` so pyro's Adam has momentum and non-None grads, snapshot `pyro.get_param_store()["q_p_c_raw"]` and `["q_p_ct_raw"]`, call `plan.validation_step(batch, 0)`, assert `torch.equal` on both. **Exact equality, not `approx`** — the tolerance is derived from the invariant "a validation pass reads, it does not write", not from any measured drift. Also assert every tensor in `module.state_dict()` is unchanged, so the test cannot pass by the fix moving the nn side instead.
2. `test_elbo_train_excludes_validation_batches` — `len(plan.training_step_outputs)` unchanged across a `validation_step` call.
3. `test_elbo_validation_is_the_same_elbo` — evaluate `plan.svi.evaluate_loss(*args)` k=20 times on one batch, take `4*std` as the MC-particle band, and assert the logged `elbo_validation` falls inside it. Derived from the estimator's own sampling spread, not from the current value.

**Proves dependencies are intact:**
4. Full suite green and unchanged in count: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q` -> 177 passed, 3 skipped.
5. `benchmarks/run_grid.py` at one fixed (seed, n_cells, temperature, fuzziness) cell, pre/post, recording ACTUAL epochs trained (DE-8 makes the recorded `max_epochs` unreliable — read `model.trainer.current_epoch`). Accept if ΔNMI is >= 0 and <= 1e-2; the upper bound comes from the parameter-space size of the perturbation (mean 0.52 L1 per check over ~600 checks against a q_p_ct_raw whose rows sum to O(10)), not from the observed 6e-4.
6. `tests/test_session_round_trip.py` unchanged — the param store still saves/loads the same two keys.

---

## DE-2 · S2 · effort XS for the change; S including tests · FIXED (PR #55 `stopping-policy`)

**Confirmed:** REAL, reproduced on main.

Static: `tcri/model/_model.py:111` `patience: int = 300`; `:300` `kwargs.setdefault("early_stopping_patience", self.patience)`; `:301` `kwargs.setdefault("check_val_every_n_epoch", 5)`. Lightning increments per validation check, not per epoch: `lightning/pytorch/callbacks/early_stopping.py:220` `on_validation_end` -> `:225 _run_early_stopping_check` -> `:278 self.wait_count += 1`, `:279 if self.wait_count >= self.patience`. Lightning's own docstring at :62-65 says so explicitly.

Measured (/tmp/de-training/de2_de4_probe.py, `patience=2`, `check_val_every_n_epoch=5`, `max_epochs=40`):
```
(epoch, wait_count, patience, best)
(4, 0, 2, inf) (9, 0, 2, 699.16) (14, 0, 2, 570.73) (19, 1, 2, 570.73)
epochs between consecutive validation checks: [5]
final epoch reached: 19
```
Last improvement at epoch 9; training stopped after epoch 19 = 2 checks x 5 epochs of non-improvement. Scaled to the shipped defaults that is 300 x 5 = **1500 epochs**, matching the register's measured best@1464 / stop@2964.

Undocumented: neither `TCRIModel.__init__`, `train`'s docstring, nor any doc in `docs/contract/` states the unit.

**Candidates considered:** 1. **Document only.** Zero numeric change, zero risk. Leaves an effective patience of 1500 that nobody chose.
2. **`check_val_every_n_epoch` default 5 -> 1** so checks and epochs coincide and `patience` is epoch-denominated by construction. Costs 5x the validation passes. Measured on the 2000-cell / 200-gene / bs-1024 fixture WITH the DE-1 fix in place: 2.50 s (cv=5) -> 2.69 s (cv=1), **+7.6%** — and that fixture has only 2 training batches per epoch, the worst possible ratio; on the benchmark shape (5000 cells / bs 1024 = 5 train batches/epoch) it is ~+3-4%. Bonus: gives DE-3's snapshot 5x finer granularity, and keeps `tests/test_model_knobs.py::test_max_epochs_and_patience_reach_the_trainer` (`es[0].patience == 4`) passing unchanged.
3. **Convert at the call site:** `early_stopping_patience = max(1, round(self.patience / cv))`. One arithmetic line, keeps the cheap 5-epoch cadence. But it leaves two parameters in two different units (`patience` in epochs, `early_stopping_patience` in checks) with a silent conversion between them — the same trap in a new place; it rounds badly at small values (`patience=4, cv=5` -> 1 check = 5 epochs, MORE patience than asked for); and it changes the expected value of an existing test (4 -> 1).
4. **Lower the default `patience` to 60** so 60 x 5 = 300 epochs is deliberate. A magic number that re-creates the same ambiguity at a different value.

**Chosen fix:** **Candidate 2**, on safety: it removes the unit mismatch structurally rather than converting between units, and it is the only option that leaves every existing test expectation intact. Speed cost is measured and small, and DE-1 (which strips a backward pass and an optimizer step from every check) is what pays for it.

File `tcri/model/_model.py`:
- Line 301: `kwargs.setdefault("check_val_every_n_epoch", 5)` -> `1`, with a comment stating that this is what makes `patience` epoch-denominated and that raising it re-introduces the multiplier.
- Line 111 (`patience: int = 300`): document in `TCRIModel.__init__`'s docstring that patience counts VALIDATION CHECKS, and that at the default cadence of one check per epoch this equals epochs; callers who raise `check_val_every_n_epoch` must divide `patience` themselves.
- Line 238 `logger.info(...)`: append the resolved stopping policy, e.g. `early_stopping: patience=300 checks x check_val_every_n_epoch=1 = 300 epochs`, so a run's own log states its effective value. This is the piece that would have caught the 1500 without anyone reading Lightning's source.

**Sequencing (this is the load-bearing part).** Ship in the SAME commit as DE-3, after DE-1. Alone, DE-2 moves published numbers, because it changes which epoch's weights the run ends on (2964 -> ~1764 on the register's case). With DE-3's best-weight restore in the same commit, the run keeps epoch-1464 weights either way, so DE-2 contributes ~0 to the reported number and becomes a pure compute saving (~40% of that run's wall clock). DE-3 is what de-risks DE-2; doing DE-2 first is the dangerous ordering.

**Contract change:** NONE. `check_val_every_n_epoch` is a `**kwargs` passthrough to scvi's Trainer, and `patience` is a `TCRIModel.__init__` argument — `__init__` is not declared in `tcri/_contract.pyi` (only `setup_anndata`, `train`, `get_latent_representation`, `predict`, `get_p_ct`, `to_anndata` are). No manifest or prose edit. NOTE: had the fix instead added a parameter to `train()`, that WOULD be an API-contract change to `tcri/_contract.pyi:35-38` and CODEOWNER-gated.

**Dependencies affected:** **Runs stop earlier.** Effective patience 1500 -> 300 epochs of non-improvement. On the register's own case (best 1464, stop 2964) the stop moves to ~1764.

**Which metric outputs move, and how much, depends entirely on whether DE-3 ships with it:**
- **DE-2 alone:** the kept weights change from epoch 2964 to ~1764, so every metric output moves. Direction is UP for MI/NMI and DOWN for the conditional entropies, because the register measured NMI 0.5020 at epoch 1464 vs 0.4979 at 2964 — earlier is higher. Size: some fraction of +0.004, above the 0.0018 fit-to-fit floor. This would be an S1 change to published numbers.
- **DE-2 + DE-3 together:** the restore selects epoch 1464 in both worlds, so DE-2's contribution to every metric output is **~0**. The number that moves is DE-3's, and it is attributable to DE-3.

**Cost:** 5x validation passes, measured +3-8% per epoch (the higher figure only on fixtures with very few training batches per epoch), against ~40% fewer epochs on a plateauing run. Net faster.

**Interaction to flag, not fix here:** with `n_steps_kl_warmup=2000` STEPS at ~5 steps/epoch, `module.kl_weight` is still ramping for the first ~400 epochs, so `elbo_validation` is a non-stationary objective over that window. Early stopping and any best-weight restore are only meaningful after it. That is DE-17/DUX-2; note it in the entry rather than silently relying on it.

**Interaction with DE-8:** `benchmarks/run_grid.py` records the REQUESTED `max_epochs`, so shortening the effective patience makes its records more misleading, not less. DE-8 should land first or in the same PR.

**Existing test expectations: none change.** `test_max_epochs_and_patience_reach_the_trainer` still sees `es[0].patience == 4`. `test_trainer_knobs_are_overridable` already passes `check_val_every_n_epoch=1` explicitly. Verified green on main: 7 passed in 3.04 s.

**Verification tests:** **Proves it landed** (new, `tests/test_model_knobs.py`):
1. `test_patience_is_denominated_in_epochs` — construct with `patience=2`; monkeypatch `UnifiedTrainingPlan.validation_step` to log a CONSTANT `elbo_validation` after the first check so non-improvement is forced deterministically; train with `max_epochs=40`; assert `model.trainer.current_epoch <= patience * model.trainer.check_val_every_n_epoch + model.trainer.check_val_every_n_epoch`. The bound is computed from Lightning's documented rule (patience counts checks) and the trainer's own attributes, not from an observed stopping epoch.
2. `test_default_check_val_cadence_is_one_epoch` — after a default `train()`, assert `model.trainer.check_val_every_n_epoch == 1` AND that the installed early-stopping callback has `patience == model.patience`. This is the invariant that makes the two units coincide; it fails loudly if anyone re-raises the cadence without re-deriving patience.
3. `test_stopping_policy_is_logged` — capture the `logger.info` line and assert it contains the resolved epoch count. Cheap, and it is what would have surfaced the 1500.

**Proves dependencies are intact:**
4. `tests/test_model_knobs.py::test_max_epochs_and_patience_reach_the_trainer` and `tests/test_model_guardrails.py::test_trainer_knobs_are_overridable` pass UNCHANGED — the point of choosing candidate 2 over 3.
5. Full suite: 177 passed, 3 skipped.
6. Wall-clock guard: `tests/test_model_smoke.py`'s default `train()` must not grow more than 15% (measure 3 runs, compare medians). The 15% comes from the measured worst-case +7.6% on a 2-batch-per-epoch fixture, doubled for headroom — derived from the cost model, not from post-fix behaviour.
7. With DE-3 in the same commit: `run_grid` at one fixed cell, pre/post, recording actual epochs and best epoch. `|ΔNMI|` must be below the 0.0018 unseeded fit-to-fit spread; if it is not, the restore is not selecting the same state and the two changes must be separated and re-measured.

---

## DE-3 · S1 · FIXED (PR #55 `stopping-policy`)

**Confirmed:** The DEFECT is real. Its STATED REASON is wrong, and the entry should be corrected rather than closed.

**Real:** `tcri/model/_model.py:296-303` sets no `enable_checkpointing` and installs no checkpoint callback; scvi's Trainer defaults `enable_checkpointing=False` (scvi/train/_trainer.py:103). Nothing restores weights; the run keeps its final state.

**Premise clause 1 — TRUE.** `/tmp/de-training/de3_probe.py`: `'q_p_ct_raw' in module.state_dict()` -> **False**. The store holds 24 keys (22 `scvi$$$…` shadows of `module.named_parameters()` plus `q_p_c_raw`, `q_p_ct_raw`); `grep -rn 'pyro.param\|pyro.module' tcri/model/` confirms `_module.py:188,280` (`pyro.module("scvi", self)`) and `_module.py:290,296,316,322` are the ONLY param sites, so those two are the complete set of store-only parameters.

**Premise clause 2 ("enabling it would not help") — FALSE.** `scvi/model/base/_base_model.py:871`: `model_state_dict["pyro_param_store"] = pyro.get_param_store().get_state()`. `scvi/train/_callbacks.py:161-167` (`SaveCheckpoint.on_train_end`, `load_best_on_end=True`): `pyro.get_param_store().set_state(pyro_param_store)`. Inspecting a checkpoint tcri actually wrote during the probe:
```
top keys: ['model_state_dict', 'var_names', 'attr_dict']
has pyro_param_store: True
params (non-scvi$$$): ['q_p_c_raw', 'q_p_ct_raw']   q_p_ct_raw shape: torch.Size([79, 3])
```

**The actual blocker is different and much smaller.** `SaveCheckpoint.on_train_end` reads `pl_module.module.device` (scvi/train/_callbacks.py:161); `TCRIModule` subclasses `PyroBaseModuleClass`, which — unlike `BaseModuleClass` (_base_module.py:159-164) — has no `device` property. Result: `AttributeError: 'TCRIModule' object has no attribute 'device'` at the end of training. With a three-line `device` property added (/tmp/de-training/de3_probe2.py) the scvi path works end to end:
```
validation checks: [(1,566.87),(3,649.53),(5,592.78),(7,579.52),(9,634.58),(11,576.57),(13,583.62),(15,524.64),(17,603.86),(19,572.32)]
best check: 15   last check: 19
|q_p_ct_raw(after train) - q_p_ct_raw(@epoch 15)|_1 = 0.000000
... @epoch 13 = 1.536980   @epoch 17 = 1.684894   @epoch 19 = 2.783001
```
So: keep the entry OPEN as a defect, rewrite its second sentence.

**Candidates considered:** 1. **scvi `SaveCheckpoint(monitor="elbo_validation", mode="min", load_best_on_end=True)` + a `device` property on `TCRIModule`.** Reuses library code that is already tested and already handles the Pyro store. But it calls `model.save()` — a full on-disk write of module state dict + param store + attrs — on EVERY improvement. On a 4000-epoch run at one check per epoch that is up to thousands of full saves; on a 2000-gene / 128-hidden / 3-layer model each is tens of MB.
2. **In-memory best-state callback in `tcri/model/_training.py`.** Snapshot `module.state_dict()` (CPU clones) plus the two store-only params on improvement; restore in place at `on_train_end`. No disk, no dependency on `model.save`/`_load_saved_files` round-trip fidelity for a non-standard model class, ~40 lines.
3. **Monitor a metric-relevant quantity instead of the ELBO** — e.g. L1 of `get_p_ct()` to the observed clone x phenotype crosstab, which the measured facts say bottoms at ~120 epochs. This is the only candidate that would actually select the metric optimum. Rejected here: it selects model state on a training-data statistic that `p_ct` is itself fitting (circular), and it replaces the note's objective with an ad-hoc one. That is a modelling decision for the authors, not a defect fix.
4. **Do nothing; fix the epoch budget instead** (DE-10, and the training-optimum finding). Addresses the larger effect but leaves early stopping meaning something other than what it says.

**Chosen fix:** **Candidate 2**, on both axes: no disk I/O in the training loop, no reliance on `model.save`/`load` round-tripping a model class with non-standard attributes, and exact control over what is snapshotted.

1. `tcri/model/_training.py` — add `RestoreBestState(pl.Callback)` and export it in `__all__`:
   - `on_validation_end`: read `trainer.callback_metrics[self.monitor]`; if improved under `self.mode`, store `(
     {k: v.detach().cpu().clone() for k, v in pl_module.module.state_dict().items()},
     {k: store[k].unconstrained().detach().cpu().clone() for k in ("q_p_c_raw", "q_p_ct_raw") if k in store})`.
   - `on_train_end`: `pl_module.module.load_state_dict(sd)`, then for each store key `store[k].unconstrained().data.copy_(v.to(...))` under `torch.no_grad()`.
   - **Restore must be in place, not `set_state`.** `pyro.module("scvi", self)` registers the module's own `nn.Parameter` objects, and `ParamStoreDict.set_state` (pyro/params/param_store.py:296-298) rebinds `self._params[name] = param` — that would leave the store holding clones while the `nn.Module` keeps the originals, silently desynchronising them. `nn.Module.load_state_dict` copies in place and preserves identity.
2. `tcri/model/_model.py::train` — when `kwargs["early_stopping"]` is true and `self.restore_best_weights`, append `RestoreBestState(monitor=kwargs["early_stopping_monitor"], mode=kwargs["early_stopping_mode"])` to `kwargs.setdefault("callbacks", [])`, so the restore criterion IS the stopping criterion by construction.
3. `tcri/model/_model.py::__init__` — add `restore_best_weights: bool = True` beside `patience` (line 111), which already carries the stopping policy. Not on `train()`; see contract_change.
4. `tcri/model/_module.py` — add `@property def device(self): return next(self.parameters()).device`. Three lines, no behaviour change, and it unblocks scvi's own `enable_checkpointing=True`, which currently crashes at `on_train_end` for every tcri user who passes it.

Prototyped end to end (/tmp/de-training/de3_inmemory_small.py), best at epoch 3 of 20:
```
 epoch    elbo   |Δq_p_ct|_1   |Δclassifier W|_1
     1   475.04     1.651214         0.533746
     3   453.97     0.000000         0.000000
     5   539.96     1.276184         0.434621
    19   515.81     5.530652         2.156899
```
Both the Pyro-store parameter and the nn side were restored exactly.

**Sequencing.** Ship WITH DE-2 (DE-3 is what makes DE-2 numerically inert), AFTER DE-1 (today the state visible at `on_validation_end` has already absorbed that check's spurious optimizer step, so the snapshot would preserve the polluted state).

**Contract change:** NONE, provided the escape hatch goes on `TCRIModel.__init__` (undeclared in the .pyi) and not on `train()`. If a reviewer prefers it on `train()`, that IS an API-contract change to `tcri/_contract.pyi:35-38`, gated by `.github/CODEOWNERS` and enforced by `tests/test_contract_conformance.py::test_signature_matches_contract` — flag it as such rather than editing the .pyi to fit the code. Also note `tests/test_shared_defaults.py` requires any knob declared in BOTH `train()` and `UnifiedTrainingPlan.__init__` to carry the same default; putting it on `__init__` sidesteps that entirely.

**Dependencies affected:** **Which metric outputs move, and by how much — this is the entry's most important qualification.** The kept weights change from "last epoch trained" to "epoch with the best `elbo_validation`". Everything downstream of `get_p_ct()` and of the encoder/classifier moves: `tl.mutual_information`, `tl.clonotypic_entropy`, `tl.phenotypic_entropy`, `tl.phenotypic_flux`, `tl.compare_groups`, `model.predict`, and `to_anndata`'s `uns['tcri_p_ct']`, `obsm['X_tcri']`, `obsm['X_tcri_logits']`, `obsm['X_tcri_probabilities']`, `obs['tcri_phenotype']`.

**Direction and size are regime-dependent:**
- Where validation ELBO plateaus then drifts, the restore recovers the earlier, better state. The register's own case: NMI **0.4979 -> 0.5020, +0.0041 (UP)**, above the 0.0018 fit-to-fit spread. Conditional entropies move DOWN correspondingly.
- Where validation ELBO improves monotonically to the epoch cap, the restore is a **no-op**. Measured directly on the 2000-cell fixture (/tmp/de-training/de3_inmemory.py): best = epoch 39 of 40, restore distance 0.000000.

**It does NOT fix the metric-vs-epoch degradation, and must not be described as one.** The supplied measured facts have validation ELBO still improving while the deterministic gate-0 read falls 0.1931 (epoch ~60) -> 0.1353 (epoch 4000). On that fixture the best-ELBO restore recovers none of the 0.058. Best-`elbo_validation` restore makes early stopping mean what it says; the epoch budget (DE-10) and the estimator defects (DE-5, DE-6) are the levers for the drift. This directly answers "is it worth it versus simply not training past the optimum": **both, and they are not substitutes** — the restore is cheap and makes the stated policy true, but the objective it selects on is not monotonically related to the metric it is being used to protect.

**Published numbers: yes.** Every figure regenerated after this lands moves by up to ~4e-3 NMI, upward where the ELBO plateaued.

**Enables DE-2 at zero numeric cost** (see DE-2).

**`model.history_` is untouched** — the restore does not rewrite logged metrics, so `diag.loss` still plots the full trajectory including the epochs after the best, which is the right thing to show.

**Existing test expectations: none change.** `tests/test_session_round_trip.py` still passes: the restore happens inside `trainer.fit`, before `TrainRunner` post-processing, so the saved store is the restored one.

**Memory:** one CPU copy of `module.state_dict()` plus two small tensors, overwritten (not accumulated). ~0.06 MB on the probe fixture; a few MB on a 2000-gene model.

**Verification tests:** **Proves it landed** (new, `tests/test_model_guardrails.py`):
1. `test_best_state_is_restored` — monkeypatch `validation_step` to log a SCRIPTED `elbo_validation` sequence with its minimum strictly interior (e.g. `[5,4,3,9,9,9,9]`); snapshot `get_p_ct()` and `module.classifier.mlp[0].weight` at every check via a spy callback; assert the post-`train()` values are **bitwise equal (`torch.equal`)** to the minimum check and **not equal** to the last. Exact equality follows from the mechanism (a restore copies, it does not recompute), so no numeric tolerance is invented and none is derived from current behaviour.
2. `test_restore_preserves_param_identity` — assert `pyro.get_param_store()['q_p_ct_raw'].unconstrained()` is the SAME object before and after restore, and likewise for `module.state_dict()` tensors. This is the guard against a future refactor switching to `set_state` and silently desynchronising the store from the `nn.Module`.
3. `test_snapshot_covers_all_store_only_params` — assert `{k for k in pyro.get_param_store() if not k.startswith('scvi$$$')} == {'q_p_c_raw', 'q_p_ct_raw'}`. If someone adds a third guide parameter, this fails rather than the restore silently dropping it.
4. `test_restore_best_weights_false_keeps_final_state` — the escape hatch is a real off switch.
5. `test_module_exposes_device` — `isinstance(model.module.device, torch.device)`; plus a smoke run with `enable_checkpointing=True, checkpointing_monitor='elbo_validation'` completing without `AttributeError` (it raises on main today).

**Proves dependencies are intact:**
6. Full suite: 177 passed, 3 skipped, `tests/test_session_round_trip.py` unchanged.
7. `benchmarks/run_grid.py` at a fixed (seed, n_cells, temperature, fuzziness) cell, pre/post, reporting actual epochs trained, the best epoch chosen, and NMI at both. **Expected: NMI moves UP or not at all.** A downward move means the monitored objective and the metric are anti-correlated on that cell — record it as a finding (it is the DE-3/DE-10 tension made concrete), do not suppress it by loosening the test.
8. A second cell chosen where validation ELBO is known to improve monotonically, asserting the restore is a no-op there (ΔNMI exactly 0), so the test suite carries BOTH regimes and nobody later reads the fix as a uniform improvement.

---

## DE-4 · S2 · effort XS · FIXED (PR `read-only-validation`)

**Confirmed:** REAL, reproduced on main.

Static: `tcri/model/_training.py:85` `self._my_global_step = 0` in `UnifiedTrainingPlan.__init__`; `:97-98` the ramp reads it (`kl_weight = max(1e-6, kl_weight_max * (step / n_steps_kl_warmup))`); `:135` `self._my_global_step += 1`. `tcri/model/_model.py:282` constructs a fresh `UnifiedTrainingPlan` on every `train()` call. `grep -rn '_my_global_step' tcri tests benchmarks docs` returns only those four lines plus the register entry — no other consumer, so the fix has no hidden callers.

Measured (/tmp/de-training/de2_de4_probe.py, two consecutive `train()` calls on one model, `n_steps_kl_warmup=60`):
```
max kl_weight run0: 1.0      | first kl_weight run1: 1e-06
my_global_step at end of run0: 120 | at start of run1: 1
```
The ramp restarts from the floor. `self.module.kl_weight` is a MODULE attribute that does persist, so the sawtooth is applied to a model that has already fully annealed — the model sees `1e-6 -> 1.0 -> 1e-6 -> 1.0`, not a monotone ramp.

**Candidates considered:** 1. **Move the counter onto the module** — `TCRIModule.__init__` gains `self._kl_warmup_step = 0`; the plan reads and increments `self.module._kl_warmup_step`. The counter then lives with the parameters it anneals and survives plan reconstruction.
2. **Seed the plan from the model** — `_model.py` passes `initial_kl_step=self._kl_steps_done` and reads it back after `runner()`. Same effect, more moving parts, and it leaks training state onto `TCRIModel` where nothing else lives.
3. **Use `self.trainer.global_step`.** Rejected: a fresh `Trainer` is built per `train()` call, so it resets identically — the same defect with a library name on it.
4. **Document the restart and leave it.** Rejected: it is not a behaviour anyone chose, and per the register it already invalidated two of the project's own diagnostic probes.
5. **Register the counter as a buffer so it rides `state_dict()`** and survives save/load. Attractive — resuming a saved model would keep the ramp — but it changes the checkpoint key set, so `load_state_dict(strict=True)` fails against every model saved before the change, including `tests/test_session_round_trip.py`'s artifacts and any existing user run. Not worth it; a plain `int` is enough, since the whole point is continuity within a session.

**Chosen fix:** **Candidate 1**, on safety and speed: three touched lines, no new plumbing, and it puts the counter where the thing it anneals lives.

- `tcri/model/_module.py`, `TCRIModule.__init__` — next to `self.kl_weight = 1e-6` (line 85), add `self._kl_warmup_step = 0`, with a comment that it is per-MODEL so a second `train()` call continues the ramp rather than restarting it. A plain `int`, deliberately not a buffer.
- `tcri/model/_training.py` — delete `self._my_global_step = 0` (line 85); at lines 97-98 read `step = self.module._kl_warmup_step`; at line 135 `self.module._kl_warmup_step += 1`.
- `tcri/model/_model.py::train` docstring — state that the warmup counter is per-model, so repeated `train()` calls continue the ramp.

This matches the package's already-stated stance that a second fit CONTINUES the first: `_model.py:131-141` warns in exactly those terms about the process-global Pyro param store.

**Grouping.** Ship in the same commit as DE-1. They are one thing — the training plan doing work it never intended (DE-1) and failing to carry state it should (DE-4) — they touch the same two files, and neither changes a documented behaviour. Their combined numeric effect on the shipped single-`train()`-call path is DE-1's ~6e-4 alone, below the 1.8e-3 fit-to-fit spread, so the commit is measurable as one delta.

**Contract change:** NONE. `SANCTIONED_DEVIATIONS['kl_warmup_z_only']` (tcri/model/_model_contract.py:183-187) says `kl_weight` anneals the `latent` KL "over `n_steps_kl_warmup`" without saying whose steps; making the counter per-model rather than per-`train()`-call is a strictly closer reading of it, not a departure. No manifest edit, no prose edit, no CODEOWNER review. Do NOT register the counter as a buffer — see candidate 5.

**Dependencies affected:** **Single `train()` call — the shipped path, `benchmarks/run_grid.py:129`, every test, every fixture: exactly zero change.** The counter starts at 0 either way and takes the same values step for step. **No metric output moves. No published number changes.** This is provable rather than argued: record the `module.kl_weight` sequence for one fixed-seed `train()` pre- and post-fix and compare bitwise.

**Multiple `train()` calls on one model** — staged or resumed training, the "train a bit more" idiom: `kl_weight` no longer drops to 1e-6 at the start of each call and stays at `kl_weight_max`. The `latent` (z) KL is therefore regularised throughout the second call, where before it was briefly unregularised. Consequence for those users only: a slightly tighter z-posterior, so `obsm['X_tcri']`, `predict`, and everything derived from the encoder shift. `q_p_ct_raw` is not directly scaled by `kl_weight` (the two Dirichlet KLs are unscaled — `_model_contract.py:183-187`), so `uns['tcri_p_ct']` moves only indirectly through the shared latent.

**Existing test expectations: none change.**
- `tests/test_model_knobs.py::test_n_steps_kl_warmup_ramps_the_kl_weight` builds a fresh model per call, so it still sees `seen[0] < 1e-3`, a monotone ramp, and `max(seen) == kl_weight_max`. Verified green on main (7 passed in 3.04 s together with the guardrails file).
- `tests/test_model_guardrails.py::test_lr_and_weight_decay_reach_pyros_optimizer` constructs a `UnifiedTrainingPlan` directly and never touches the counter.
- `tests/test_shared_defaults.py` compares only DEFAULT VALUES of shared signature parameters; no signature changes here.

**No benchmark re-run required**, which is the reason to pair this with DE-1 rather than with the policy changes: it contributes nothing to the delta being measured.

**Verification tests:** **Proves it landed** (new, `tests/test_model_knobs.py`, reusing the `training_step` spy already written for `test_n_steps_kl_warmup_ramps_the_kl_weight`):
1. `test_kl_warmup_counter_persists_across_train_calls` — train the SAME model twice with `n_steps_kl_warmup=K` and enough epochs in call 1 to exhaust the ramp; record `module.kl_weight` at every `training_step` across both calls; assert (a) the full concatenated sequence is non-decreasing, `all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))`, INCLUDING across the call boundary, and (b) the first step of call 2 equals `module.kl_weight_max`, not 1e-6. The `1e-12` is float-comparison slack; the assertion itself is the monotonicity the ramp is defined to have, taken from the definition at `_training.py:97-98`, not from any measured run.
2. `test_kl_warmup_counter_lives_on_the_module` — `model.module._kl_warmup_step > 0` after `train()`, and constructing a fresh `UnifiedTrainingPlan` over the same module does not reset it. This is the structural guard: it fails if someone moves the counter back onto the plan.

**Proves dependencies are intact:**
3. `test_n_steps_kl_warmup_ramps_the_kl_weight` passes UNCHANGED — it is the proof that the single-call path is untouched.
4. `test_single_train_call_kl_sequence_is_unchanged` — a one-off equivalence check run at review time rather than a permanent test: record the `module.kl_weight` sequence for a fixed-seed `train()` on the `tests/test_model_knobs.py` fixture before and after the change and assert element-wise equality. Bitwise equality is derivable from the mechanism (same counter, same initial value), so it is the correct tolerance.
5. Full suite: 177 passed, 3 skipped.

---

## DE-5 · S1 · FIXED (guide half PR #51; metric half DE-5b, PR #60)

**Confirmed:** REAL, and worse than the register states.

(1) The pin, from current code. `tcri/model/_module.py:328-331`: `q_p_ct_sharp = q_p_ct_raw ** (1/guide_temperature)`; clamp; **row-normalize**; `conc_ct_guide = clamp(local_scale * q_p_ct_sharp, min=1e-3)`. Identical shape at `:303-306` for `q_p_c_raw` with `global_scale`.

(2) Measured. Traced `TCRIModule.guide` on a fixture with clone sizes 3/6/12/30/90/300 cells (`/tmp/de-estimator/de5.py`):
```
 ct row   cells   sum(conc_ct)   min entry
      0       3       3.000000    1.000000
      5     300       3.000000    1.000000
spread of sum(conc_ct) across a 100x cell-count range : 0.000e+00
sum(conc_c) per clone : [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```
The total is exactly `local_scale`/`global_scale` for every row. Per-entry concentration at the shipped default is beta/P = 3/10 = 0.3 (<1, corner-seeking).

(3) The note. `docs/contract/source/supplementary_note_1_SS_2026-08-03.pdf`, Variational Inference: `Lambda = ((lambda_c)_{c=1..C}, (lambda'_m)_{m=1..M}, eta_enc, eta_cls)` collects **all variational parameters**; eq 6 is `q = prod_c Dir(omega_c | lambda_c) * prod_m Dir(phi_m | lambda'_m) * ...`. The notation list separately gives `alpha > 0, beta > 0` as "Dirichlet concentration scales (global, local)", appearing only in the conditioning `p(... | x; alpha, beta, {psi_b}, {u_k})`. The code uses the scalar prior hyperparameter as the variational vector's total. Confirmed divergence, not ambiguity.

(4) NEW AND LOAD-BEARING -- the ELBO carries **no data gradient at all** into `q_p_ct_raw`. Measured with `Trace_ELBO().differentiable_loss` on the same fixture:
```
full model()                        loss=126.558266  |grad q_p_ct_raw|_1=2.360682e+01
model() WITHOUT the surrogate       loss= 86.995209  |grad q_p_ct_raw|_1=2.360682e+01
model() WITHOUT the ZINB likelihood loss= 48.604061  |grad q_p_ct_raw|_1=2.360682e+01
|grad(full) - grad(no surrogate)|_1 = 0.000000e+00
|grad(full) - grad(no ZINB)|_1      = 0.000000e+00
```
The losses differ (both terms are in the objective) but the gradient into q(phi_m) is bit-identical. Cause is `tcri/model/_module.py:241`, `phi = p_ct[ct_idx].detach()` -- `log_phi`, `probs`, `pheno_kl` and the factor are all downstream of the detach, and nothing else in `model()` mentions `p_ct`. So the only ELBO term touching phi_m is `-KL(q(phi_m) || Dir(beta*omega_h(m)))`, whose argmax over a FREE lambda'_m is lambda'_m = beta*omega_h(m) -- total beta again.

Consequence for the fix: freeing the magnitude gives it nothing to concentrate on. This also explains the supplied measured facts (p_ct's L1 to the observed crosstab grows to 0.515 while validation ELBO improves; the read is best at 30-120 epochs): q_p_ct_raw is initialized from the empirical crosstab (`_module.py:312-320`) and thereafter relaxes toward the archetype prior, because the prior KL is the only force acting on it.

(5) The engine does not read the guide's concentration anyway. `tcri/tools/_joint.py:83-89` pulls the scalar `uns[LOCAL_SCALE]`, and `tcri/_compute/_joint.py:102` rebuilds `conc = clamp(local_scale * base, 1e-3)`. So even a free lambda' would not reach `n_samples>0` without plumbing.

**Candidates considered:** A. FREE PER-ROW MAGNITUDE (minimal; the task's suggestion). A positive vector factors uniquely as magnitude x simplex, so `conc = mag_m * normalize(raw_m)` with `mag_m > 0` learnable IS exactly eq 6's `lambda'_m in R^P_{>0}` -- no more, no less. Initialize `mag` at beta/alpha so step 0 is bit-identical to today. `guide_temperature` keeps acting on direction only. `get_p_ct()` is untouched (the Dirichlet mean is still `normalize(raw^(1/T))`). Cost: two new pyro params, so `GUIDE_PARAMS` grows.

B. LITERAL -- use `q_p_ct_raw` directly as lambda': `conc = clamp(raw ** (1/guide_temperature), 1e-3)`, deleting the normalize and the beta multiply. Fewest lines, `GUIDE_PARAMS` unchanged. Rejected: `guide_temperature` then rescales the MAGNITUDE (T=0.25 raises raw to the 4th power, blowing the total up by orders of magnitude), converting a read-out sharpening knob into a concentration knob with no test covering it; and the initial total becomes `guide_init_scale` (10), not beta (3), so no run before/after is comparable and the change cannot be verified as a reparameterization.

C. SIZE-AWARE PIN -- `conc = (beta + n_m) * simplex`, a conjugate-style pseudo-count. Rejected on governance: the note specifies a free parameter, not a rule; this invents a definition to make the posterior width look right, which is exactly the inference CLAUDE.md forbids. It would also need a new SANCTIONED_DEVIATION.

D. DECLARE THE PIN SANCTIONED and close [I]. Rejected: eq 6 and the Lambda definition are word-for-word identical across the 2026-04-30 and 2026-08-03 notes; there is nothing to resolve. The manuscript is upstream of the contract.

E. FIX THE DETACH ONLY, leave the pin. Would restore a data path to phi_m but the concentration total stays clamped at beta, so the posterior still cannot widen or narrow with n_m. Necessary, not sufficient -- it is the companion to A, not an alternative.

**Chosen fix:** A, plumbed to the engine, and sequenced AFTER DE-6. The detach is a prerequisite for A to have any effect and must be decided in the same review.

Why A on safety: it is a reparameterization with an exact-identity starting point, so a single test ("step 0 concentration equals clamp(beta*sharp)") proves it introduced no re-tuning. It changes the variational family and nothing else in the joint -- no site added or removed, no plate, no distribution family, so `test_model_contract_conformance.py`'s structural half is untouched. Why A on speed: ~8 lines in the guide, ~10 in the engine seam, plus contract text.

PLAN:
1. `tcri/model/_model_contract.py` FIRST. GUIDE_SITES notes for `p_c`/`p_ct`: "q(phi_m) = Dir(lambda'_m); lambda'_m free in R^P_{>0}, parameterized as magnitude x simplex (eq 6 + the Lambda definition). alpha/beta are PRIOR hyperparameters and appear only in eqs 1-2." `GUIDE_PARAMS = ["q_p_c_raw", "q_p_ct_raw", "q_p_c_mag", "q_p_ct_mag"]`. New `SEMANTIC_INVARIANTS["guide_concentration_magnitude_is_free"]`: the per-row concentration totals must not all equal beta/alpha after training. Mirror in `docs/contract/MODEL_CONTRACT.md`; flip [I] open -> fixed in `METHODS_CONFORMANCE.md:115-150` and delete the two false arrow lines at `:47-48`.
2. `tcri/model/_module.py::guide`. After the normalize at `:305`/`:330`:
   `q_p_c_mag = pyro.param("q_p_c_mag", torch.full((self.c_count,), float(self.global_scale)), constraint=dist.constraints.positive)`
   `conc_c_guide = torch.clamp(q_p_c_mag.unsqueeze(-1) * q_p_c_sharp, min=1e-3)`
   and the `ct_plate` twin with `local_scale`/`ct_count`. Keep the non-finite repair and the 1e-3 clamp.
3. `tcri/model/_model.py::to_anndata` (~`:414-421`). Write `uns["tcri_conc_ct"] = (mag * sharp)` -- the per-ct concentration VECTOR -- alongside the existing `uns[LOCAL_SCALE]` (kept, it is still the prior beta).
4. `tcri/tools/_joint.py:83-89` + `tcri/_compute/_joint.py:100-106`. Accept an optional `conc_ct` array; when present, draw `Dirichlet(clamp(conc_ct, 1e-3))` instead of `clamp(local_scale * base, 1e-3)`. Fall back to the scalar path when absent, so AnnData objects written by an older version still load. This step is what actually delivers a data-informed interval; without it the metric keeps drawing from the prior-set width regardless of the guide.
5. Separately, as its own reviewed model change: put the `.detach()` at `_module.py:241` to the authors. Removing it restores the note's per-cell dependence of the surrogate on phi_g(i) and is the only route by which lambda'_m can grow with cells-per-group.

**Contract change:** MODEL CONTRACT, first. `tcri/model/_model_contract.py` (GUIDE_SITES notes for p_c/p_ct, GUIDE_PARAMS, one new SEMANTIC_INVARIANT) + `docs/contract/MODEL_CONTRACT.md`, then close deviation [I] in `docs/contract/METHODS_CONFORMANCE.md:115-150`. CODEOWNER-restricted (@nceglia/@salehis). A SECOND contract question rides along and must be asked, not decided: the detached alignment target (below) is an undeclared departure from the note's Inference Details -- it is not in SANCTIONED_DEVIATIONS and the note's `L_new = L# + gamma*sum_i KL(probs_i || phi_g(i))` gives no basis for it. It needs either a sanctioned-deviation entry with a rationale or its own defect entry (suggest DE-18).

**Dependencies affected:** n_samples=0 (the deterministic read): UNCHANGED BY CONSTRUCTION. `get_p_ct()` returns `normalize(q_p_ct_raw^(1/T))`, which is the Dirichlet mean under both the pinned and the free parameterization. The fitted raw values do drift (the guide entropy term's gradient changes), so plug-in numbers shift slightly, but with no predictable sign. `uns[P_CT]`, `predict()`, `tl.*` at n_samples=0 -- no structural move.

n_samples>0 (every interval, and the note's headline benchmark number, which is 'the posterior mean NMI over 200 posterior draws'): THIS IS WHERE IT MOVES. For groups with more cells than beta implies, the concentration rises, draws tighten, the Jensen gap in E_s[NMI(J_s)] shrinks, and the reported posterior-mean NMI FALLS. Magnitude, from the register's own measurement of that gap: about -0.10 at weak coupling, -0.017 at strong. Small groups can move the other way if the learned magnitude falls below beta. HDI widths narrow, and stop being identical across groups of wildly different size -- which is the property to judge the fix by, not the NMI error.

WITHOUT the detach fix: essentially NOTHING moves. The argmax of the only phi-bearing term is lambda' = beta*omega, total beta. The family becomes eq-6-conformant (the point of the defect) but the register's stated consequence does not arrive. Predicted movement at convergence: ~0.

Tests that change:
- `tests/test_model_contract_conformance.py::test_guide_registers_variational_params` -- must FAIL before the manifest edit and pass after. That is the forcing function working.
- `tests/test_recovery.py::test_posterior_hdi_covers_the_truth` (`:242-289`) is the exposed one. It currently covers 8/8 at mean HDI width 0.103 -- a width that is prior-set, i.e. it passes for the wrong reason. Narrowing it while the DE-6 fold bias is still live puts a tight interval around a value biased low by 0.054-0.150, and the `covered >= 6` bar can fail BECAUSE a correct fix landed. This is the concrete reason DE-6 must land first.
- `tests/test_model_knobs.py::test_local_scale_controls_p_ct_draw_variance` (`:223-239`) hand-builds `Dirichlet(clamp(beta*base))` from `get_p_ct()`. It remains a true statement about the eq-2 PRIOR but stops describing the guide; re-aim it at the prior site or at the new magnitude parameter.
- `tests/test_session_round_trip.py:67` still holds; add a round-trip assertion for the new `tcri_conc_ct` key.
- `tests/test_shared_defaults.py` unaffected (beta stays the prior hyperparameter and the magnitude's init).
- `benchmarks/run_grid.py:115-116, 207` documents `local_scale` as "the TOTAL Dirichlet concentration on p_ct". That prose becomes false.

ORDER (asked explicitly): DE-6 FIRST, DE-5 SECOND.
- DE-5 first, effective (with the detach fixed): the deterministic read does not move, so the +0.076 metric-path error at 60 epochs stays at +0.076 and the fix looks inert; meanwhile the posterior read collapses onto NMI(fold(mean)), i.e. onto the ATTENUATED value, giving a single-sided low bias of 0.054 (flat) / 0.150 (sharp) with nothing opposing it. The already-measured fact that reporting NMI(E[J]) instead of E[NMI(J)] makes accuracy worse at every scale and both temperatures is the same experiment run a different way, and it confirms this prediction: removing the inflation while the fold is still there degrades accuracy. The next move after that is to re-tune the gate, which is how a cancellation gets re-established.
- DE-6 first: at n_samples=0 the read becomes exact-by-construction on a fixture where p_ct is exact (measured: gate=0 reproduces the base joint to -0.0000; on the supplied numbers 0.2688 -> 0.1933 against oracle 0.1927). The remaining error is then entirely the Jensen inflation -- one cause, one sign, and exactly what DE-5 removes. Errors become attributable instead of cancelling.

**Verification tests:** 1. `test_guide_concentration_total_is_free` -- fixture with cell counts spanning 100x; take N SVI steps; assert `max(sum(conc_ct)) / min(sum(conc_ct)) > 1.05`. Tolerance is categorical, not behavioural: under the pinned family the measured spread is exactly 0.000e+00, so any spread above float noise is a family change.
2. `test_guide_concentration_at_init_reproduces_the_pin` -- before any optimizer step, `conc_ct_guide == clamp(beta * q_p_ct_sharp, 1e-3)` to 1e-12. Proves the change is a reparameterization, not a re-tuning. Tolerance from the algebraic identity `mag=beta`.
3. `test_posterior_mean_is_parameterization_invariant` -- `get_p_ct()` equals `normalize(q_p_ct_raw^(1/guide_temperature))` for arbitrary injected magnitudes. Tolerance 1e-12, from the Dirichlet mean identity `E[Dir(lambda)] = lambda/sum(lambda)`.
4. `test_engine_draws_from_the_stored_concentration` -- with a hand-written `uns['tcri_conc_ct']`, the empirical variance of `joint_distribution(n_samples=20000)` matches `p(1-p)/(1 + sum(lambda'))` at rel 0.15. Tolerance from the analytic Dirichlet variance (the same derivation `test_local_scale_controls_p_ct_draw_variance:236-239` already uses), not from a recorded run.
5. `test_posterior_concentrates_with_cells_per_group` -- ONLY meaningful once the detach is resolved; the 3/6/12/30/90/300 fixture, assert Spearman rho(n_m, sum(lambda'_m)) > 0.8. Tolerance from the conjugate expectation that posterior concentration grows with observations, not from current behaviour. Until the detach is decided, ship this xfail with the gradient measurement above as its rationale.
6. Regression guard on the ELBO seam: re-run the `Trace_ELBO().differentiable_loss` probe (`/tmp/de-estimator/de5.py` part b) as a test asserting the surrogate DOES contribute a nonzero gradient to `q_p_ct_raw` once the detach is removed -- `|grad(full) - grad(surrogate blocked)|_1 > 0`. Tolerance: strict inequality against exact zero, which is the currently measured value.
7. `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q` green, with `test_model_contract_conformance.py` shown failing on the pre-manifest commit.

---

## DE-6 · S1 · CLOSED — NOT A DEFECT

> **Closed 2026-08-07.** The premise was that `use_logits=True` folds an already-folded
> quantity through eq 4 twice. It does not. `obsm[X_LOGITS]` holds RAW classifier logits
> (`_model.py`); the folded quantity goes to `X_LOGPOSTERIOR`. So `_compute/_joint.py`'s
> `g*ell + (1-g)*log_b` applies eq 4 exactly once, reproducing `predict()`. Flipping the
> default would move the metrics away from the estimand, not toward it.
>
> Part of the argument was also inherited from DE-18, which is withdrawn.

**Confirmed:** REAL, and larger than the register states.

Code. `tcri/_compute/_joint.py:150-156`:
```
log_b = torch.log(b_cell + _EPS)
combine = (g * ell + (1.0 - g) * log_b) if use_gate else (ell + log_b)
p_cell = torch.softmax(combine / float(temperature), dim=-1)
```
so `p_cell ~ exp(g*ell) * base^(1-g)`. `g` is `uns[GATE_PROB]` (`tcri/tools/_joint.py:103`), written straight from the model's pi at `tcri/model/_model.py:420-421`; default pi = 0.5.

The fold is NOT switchable from the metric API. None of `tl.mutual_information` (`_mutual_information.py:50,63`), `tl.clonotypic_entropy`/`tl.phenotypic_entropy` (`_entropy.py:80,96`) or `tl.phenotypic_flux` (`_flux.py:25,27`) has a `use_logits` parameter; all call `tcri/tools/_common.py:23-24` `joint_draws(..., use_logits=True)`. Only `tl.joint_distribution` exposes the switch.

Measured on current code, 47 clones x 10 phenotypes with Zipf abundances and an inert classifier (logits identically 0), `/tmp/de-estimator/de6_de7.py`:
```
sharp (T=0.35)  truth NMI(min)=0.5318   MI=1.6817 bits
   gate=0.0   NMI=0.5318  err=-0.0000
   gate=0.25  NMI=0.4446  err=-0.0872
   gate=0.5   NMI=0.3239  err=-0.2080
   gate=0.75  NMI=0.1581  err=-0.3737
   gate=1.0   NMI=0.0000  err=-0.5318
flat  (T=3.0)   truth NMI(min)=0.0701
   gate=0.0   NMI=0.0701  err=-0.0000
   gate=0.5   NMI=0.0224  err=-0.0477
```
At gate=0 the estimator reproduces the base joint EXACTLY, so on this fixture the whole metric-path error is the fold. That also means the fix is a removal, not a new code path: `use_logits=True, g=0` is algebraically identical to `use_logits=False`.

And the fold is not a one-signed bias. With an INFORMATIVE classifier (`logits = 4*log base`) it inflates instead:
```
sharp  truth 0.7031 -> gate0 0.7031, gate0.5 0.8883 (+0.185), gate1 0.9311 (+0.228)
flat   truth 0.0704 -> gate0 0.0704, gate0.5 0.2297 (+0.159), gate1 0.3741 (+0.304)
```
So the fold re-tempers each clone row by whatever the classifier's sharpness happens to be, in whichever direction. Combined with the register's finding that the optimal gate differs by regime (0.00 sharp, 0.75 flat), the shipped value is a tuned cancellation, not an estimate.

**Candidates considered:** 1. THE METRIC ESTIMATES phi_m -- flip the default at the single metric seam, `tcri/tools/_common.py:24`, `use_logits=True` -> `False`. `tl.joint_distribution` keeps `use_logits=True` (it is the general engine wrapper and the prediction-table path). One line + contract text.

2. FLIP BOTH -- same, plus `tl.joint_distribution`'s default. One visible answer to "what is the joint", but it changes the public engine's default output and `test_tools/test_joint.py` fixtures with it, for no estimator benefit.

3. EXPLICIT `metric_gate` PARAMETER defaulting to 0, keeping the fold reachable from the metrics. Same numbers as 1; adds a parameter to eight declarations in `tcri/_contract.pyi` and a knob whose only correct value is 0.

4. KEEP THE FOLD, DOCUMENT IT. Rejected: it makes I(c;phi) a function of pi, a MODEL hyperparameter that has no place in a metric definition -- the METRICS document (eq 5) defines I over p(c,phi) and nothing else. It also means changing `gate_prob` at fit time silently changes every published entropy and MI.

5. AGGREGATE THE CLASSIFIER ALONE (gate=1). Rejected: measured 0.0022/0.0000 on the benchmark fixture, and it discards phi_m entirely.

**Chosen fix:** 1, plus the contract clause that makes it non-recurring.

Why on safety: it is the removal of a second application of eq 4. Eq 4's `ell_i = pi*f_cls(z_i) + (1-pi)*log phi_g(i)` is the model's per-cell PREDICTION rule; `predict()` implements it and stays exactly as it is. The metric's estimand is the joint over (c, phi), whose conditional P(phi|c) is phi_m -- eq 2's variable, read back as `uns[P_CT]`. Applying eq 4 again at read time folds the prediction rule on top of a phi that the fit already accounts for. Verified as a removal: gate=0 reproduces the base joint to -0.0000. Why on speed: one default, one contract block, one test file.

PLAN:
1. `tcri/tools/_metrics_contract.py` FIRST. New exported `JOINT_CONSTRUCTION` dict (add to `__all__`) with a `rows` key: "P(phi|c) is phi_m, the covariate-level phenotype distribution (Note 1 eq 2; posterior mean of eq 6's q(phi_m)), read from uns[P_CT]. The per-cell classifier reaches the metric through the FIT only. Note 1 eq 4's gated combination is the per-cell prediction rule and belongs to predict() / joint_distribution(use_logits=True); applying it again at read time makes I(c;phi) a function of the model hyperparameter pi, which the METRICS document's eq 5 does not admit." Add `IDENTITIES["metric_joint_is_independent_of_the_gate"]`.
2. `docs/contract/METRICS_CONTRACT.md` -- a "What table" section placed UPSTREAM of the definitions, since every definition below it presumes one.
3. `tcri/tools/_common.py:23-24` -- `use_logits=False` default, with the citation in the docstring.
4. `docs/contract/API_CONTRACT.md` sec 7.1 and `tcri/_contract.pyi:17-20` -- record that the metric path and the engine wrapper now differ deliberately, and why.
5. `tests/test_metrics_contract_conformance.py` -- the gate-invariance test (below).
No signature changes, so `tests/test_contract_conformance.py` (which compares names/kinds/has_default, `:53-79`) is unaffected.

**Contract change:** METRICS CONTRACT, first. `tcri/tools/_metrics_contract.py` gains an exported `JOINT_CONSTRUCTION` block (what table the frozen reductions run on -- today the manifest freezes the reduction and says nothing about the table), plus a matching section in `docs/contract/METRICS_CONTRACT.md`. Also touches the API prose: `docs/contract/API_CONTRACT.md` sec 7.1 and the 'Decisions baked in' note in `tcri/_contract.pyi:17-20`. CODEOWNER-restricted. NOT a model-contract change -- `predict()` and eq 4 are untouched.

**Dependencies affected:** ALL FOUR METRIC FAMILIES MOVE, and every `pl.*` twin with them (`plotting/_entropy.py:11,34`, `_mutual_information.py:10`, `_flux.py:14`). Directions, measured on the 47x10 fixture at gate 0.5 -> 0 with an inert classifier:
- `tl.mutual_information` RISES: NMI(min) 0.3239 -> 0.5318 (sharp, +0.208); 0.0224 -> 0.0701 (flat, +0.048). Consistent with the register's 4000-epoch reads 0.3498 -> 0.4993 and 0.0831 -> 0.1367.
- `tl.phenotypic_entropy` FALLS (the fold flattens clone rows): mean normalized 0.5931 -> 0.3543 (sharp, -0.239); 0.9713 -> 0.9133 (flat, -0.058).
- `tl.clonotypic_entropy` FALLS: 0.7655 -> 0.6242 (sharp, -0.141); 0.9830 -> 0.9489 (flat, -0.034).
- `tl.phenotypic_flux` RISES (the fold tempers both conditionals toward uniform, shrinking the divergence): mean KL on random Dirichlet(0.4) rows 1.2574 -> 3.3512.
Yes, published numbers change -- every entropy and MI in any figure produced from this code.

Accuracy direction against truth: on the supplied 60-epoch measurement the shipped read is 0.2688 against a label oracle of 0.1927 while `det_g0` is 0.1933. Removing the fold at n_samples=0 takes the error from +0.0761 to +0.0006. At n_samples>0 the fold's attenuation stops opposing the Jensen inflation, so reported posterior-mean values RISE by roughly the register's measured gap (+0.10 weak coupling, +0.017 strong). That residual is single-signed and attributable, and it is exactly what DE-5 then removes.

A consequence to state plainly rather than hide: with the detach at `_module.py:241` still in place (see DE-5), phi_m carries no classifier information at all, so after this change `tl.*` is the plug-in statistic of the (slowly decaying) K-means initialization plus Dirichlet shrinkage. That is what the model currently knows. It makes the training-length defects DE-2/DE-3 visible instead of masked -- the supplied fact that the gate-0 read equals the oracle at 30-120 epochs and degrades to 0.1353 by 4000 is precisely that exposure.

Tests:
- `tests/test_recovery.py::test_model_mi_tracks_the_true_mi_across_difficulty` (`:201-238`) is ordinal -- survives, and should sit closer to truth.
- `tests/test_recovery.py::test_posterior_hdi_covers_the_truth` (`:242-289`) -- the point estimate moves UP toward the truth while the interval width (0.103) is unchanged, so coverage should hold or improve. This is what makes DE-6-first the safe order; confirm the 8/8 before merging.
- `tests/test_tools/test_joint.py::test_gate_aware_combine_direct` (`:136-159`) calls `_joint_draws` with an explicit gate and does NOT change -- keep it as the guard that eq 4 still exists for prediction tables.
- `tests/test_model_contract_conformance.py::test_gate_rule_endpoints` and `tests/test_model_knobs.py::test_gate_prob_endpoints_reduce_to_closed_forms` (`:283`) must both still pass unchanged -- that is the proof the change is confined to the metric path.
- Sweep `tests/test_tools/` and `tests/test_plotting/` for recorded metric values; any that exist must be re-derived from the definition, not from a fresh run.

**Verification tests:** 1. `test_gate_prob_does_not_enter_the_metric` -- set `uns[GATE_PROB]` to 0.0, 0.5, 1.0 and NaN in turn and assert all four metrics are bit-identical (atol 0). Exact, because the value is no longer read. This is the test that would have caught the defect.
2. `test_metric_joint_equals_p_ct` -- with `uns[X_LOGITS]` set to pure noise, `tl.mutual_information(adata, weighted=True)` must equal `_mi_from_joint(P_CT * counts[:, None])` computed independently in the test from the frozen eq-5 formula. Tolerance 1e-12, analytic.
3. `test_folded_read_cannot_reach_the_true_coupling` -- a base with row max 0.90 has an analytic NMI(min) of 0.7287, computed in the test from the definition; assert the metric returns it to 1e-9. The folded path returns 0.2355 on the same input, so the bound is 3x away from any float tolerance and the derivation is independent of current behaviour. (NB: an EXACT permutation joint does NOT discriminate -- measured gate0 0.999999 vs gate0.5 0.997152, because the 1e-8 floor survives the square root. Use a non-degenerate base.)
4. `test_joint_distribution_still_implements_eq4` -- keep `test_gate_aware_combine_direct` and add the `use_logits=True` public-wrapper equivalent, so the fold remains available and tested for prediction tables.
5. Re-run the metrics conformance suite: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_metrics_contract_conformance.py tests/test_tools -q`, then the full suite.

---

## DE-7 · S2 · FIXED (PR #63)

**Confirmed:** REAL as stated. `grep -c "use_logits\|weighted\|gate" tcri/tools/_metrics_contract.py` returns 0. `METRIC_SPECS` freezes formula / per / support / normalizer / empty for each metric but never says how the table's clone marginal P(c) arises -- and the same silence covers DE-6's `use_logits`.

What the sources say. The METRICS document is silent by omission: the archived `docs/contract/source/metrics_2026-08-05.docx` starts at the Entropy section and presumes p(c,phi) already defined. Note 1 IS explicit, for the benchmark: "For both TCRi and the GMM baseline, estimated NMI is computed using the observed clonotype abundances in each simulated dataset." That is P(c) proportional to n_c -- `weighted=True`. The repo's own oracle agrees: `tcri/datasets/_simulate.py:184-186` builds the truth from the realized crosstab (`np.add.at(counts, (z, phi), 1.0)`), i.e. abundance-weighted. So the shipped default `weighted=False` estimates a DIFFERENT estimand than the simulator's ground truth, which is why `tests/test_recovery.py:232` and `:286` both have to pass `weighted=True` explicitly.

Measured movement (`/tmp/de-estimator/de6_de7.py`, 47 clones x 10 phenotypes, Zipf abundances, on the truth joints so the model is not involved):
```
sharp (T=0.35) [min]      abundance 0.7345  uniform 0.6788   -7.6%
sharp (T=0.35) [average]  abundance 0.5966  uniform 0.4959  -16.9%
flat  (T=3.0)  [min]      abundance 0.0728  uniform 0.0862  +18.4%
flat  (T=3.0)  [average]  abundance 0.0618  uniform 0.0644   +4.2%
```
The SIGN REVERSES along the coupling axis. It is not a level shift, so it can reorder samples -- which is what the register's "the effect varying along the swept axis" refers to. I did not reproduce the register's specific 33%; on this fixture the largest move is 18.4% (min) / 16.9% (average). The 33% is plausible on a fixture with a heavier abundance tail and I have not contradicted it.

What is ALREADY correct and should not be disturbed: `benchmarks/run_grid.py:136` and `:151` already pass `weighted=True`, so the benchmark path follows the note's protocol today -- it just does so undocumented, with nothing preventing a future edit from dropping it.

**Candidates considered:** 1. DOCUMENT ONLY -- add `clone_mass` to `JOINT_CONSTRUCTION`, keep `weighted=False` as the package default (the repertoire-level statistic that `API_CONTRACT.md:109-113` sec 0.8 chose with a stated rationale), and make it a contract requirement that anything compared against a ground-truth MI passes `weighted=True`. Zero numeric change.

2. DOCUMENT + FLIP THE DEFAULT to `weighted=True`, so the out-of-the-box estimand matches Note 1's benchmark and the repo's oracle. Defensible, but it moves every default-call number a second time with a regime-dependent sign, on top of DE-6's shift, making DE-6's before/after unreadable. And it overrides a sec-0.8 decision that already has a written rationale -- that is the authors' call, not an audit's.

3. REMOVE THE DIAL, pick one. Rejected: both statistics answer real questions (repertoire structure vs clonal expansion) and sec 0.8 reasoned this out; the removal bar does not apply to a documented user choice.

4. RENAME TO NAME THE ESTIMAND -- `clone_mass="uniform"|"abundance"` replacing `weighted`. Same numbers as 1 and much harder to misuse, but it is an API-contract signature change across the eight declarations at `tcri/_contract.pyi:65,72,78,84,90,105,112,119,125` plus every `pl.*` twin. Right idea, wrong time.

**Chosen fix:** 1 now; 4 as a follow-up if the authors want the estimand in the name; 2 only on an explicit author decision.

Why on safety: option 1 moves no number, so it can land in the SAME PR as DE-6 without confounding DE-6's measured shift -- which matters, because DE-6 and DE-7 are the same manifest gap and splitting them leaves the manifest half-complete. Why on speed: contract text plus one introspection test.

And the governance read is unambiguous: there is no contract-versus-source conflict on the DEFAULT. Note 1 specifies abundance weighting for its benchmark, the METRICS excerpt is silent, and sec 0.8's `weighted=False` is a recorded API decision. The defect is that the metrics manifest does not say which estimand a metric call is answering.

PLAN:
1. `tcri/tools/_metrics_contract.py` -- `JOINT_CONSTRUCTION["clone_mass"]`: "P(c) is NOT specified by the METRICS document (the archived excerpt begins at the Entropy section). tcri exposes it as `weighted`. False (DEFAULT): uniform clone mass, each clone one unit -- a repertoire-level statistic (API contract sec 0.8). True: P(c) proportional to the observed cells of that clone at that covariate. Note 1's benchmark protocol specifies the latter -- 'estimated NMI is computed using the observed clonotype abundances' -- and `tcri.datasets.simulate_tcri`'s oracle is built from the realized crosstab, so ANYTHING compared against a ground-truth MI must pass weighted=True. The two are not a level shift: measured -7.6% at sharp coupling and +18.4% at flat on a 47x10 Zipf fixture, i.e. the sign reverses along the coupling axis, so they can reorder samples."
2. `OPEN_QUESTIONS["clone_mass_default"]` and `OPEN_QUESTIONS["metrics_source_is_an_excerpt"]`. The existing test that asserts no key appears in both `OPEN_QUESTIONS` and `SANCTIONED_EXTENSIONS` then keeps these from being quietly refiled as features.
3. `docs/contract/METRICS_CONTRACT.md` -- mirror, in the same "What table" section DE-6 adds.
4. `tcri/tools/_mutual_information.py:39`, `_entropy.py:106,118`, `_flux.py:51` and the four `pl.*` twins -- one docstring line each naming the estimand and pointing at the manifest.
5. The forcing function (below), so a future dial cannot appear undeclared.

**Contract change:** METRICS CONTRACT, and it is the same edit as DE-6 -- both are the one gap that the manifest freezes the REDUCTION and never the TABLE. `tcri/tools/_metrics_contract.py` `JOINT_CONSTRUCTION` gains a `clone_mass` key next to DE-6's `rows` key; `docs/contract/METRICS_CONTRACT.md` mirrors it. Two ASKs for the authors, recorded in `OPEN_QUESTIONS` rather than decided here: (a) should the shipped default follow the benchmark protocol; (b) the archived METRICS source is an EXCERPT -- it begins mid-document at 'Entropy ... With our various distributions defined', so the section defining p(c,phi) is not in the repo, while `SOURCES['METRICS']['owns']` claims 'eqs 2-7'. The source of truth for the estimand is not actually archived.

**Dependencies affected:** NO metric output moves under the chosen fix, and no published number changes. That is the point of choosing it -- DE-6 lands in the same PR and its measured shift stays attributable to DE-6 alone.

Unchanged: `tests/test_recovery.py:232,286` (already `weighted=True`), `tests/test_tools/test_joint.py::test_weighted_scales_rows_by_ct_cell_count` (`:88-103`, a mechanical assertion about row sums), `benchmarks/run_grid.py:136,151` (already correct). `tcri/_contract.pyi` untouched, so `tests/test_contract_conformance.py` is unaffected.

If the authors instead take option 2 (flip the default): every default `tl.*` and `pl.*` number moves, with a sign that depends on the coupling regime -- roughly -8% to -17% where clone rows are sharp and +4% to +18% where they are flat, on the measured fixture. Sample orderings can change, so any figure comparing groups would have to be regenerated, not rescaled. `tests/test_recovery.py` would be unaffected (it passes the flag explicitly) which is precisely why the suite would NOT catch the change -- another argument for landing the manifest entry first.

Cross-defect: DE-7 shares its manifest section with DE-6 and is independent of DE-5. It has no ordering constraint against either.

**Verification tests:** 1. `test_every_metric_knob_is_declared` -- the forcing function that makes this class of defect non-recurring. Introspect `inspect.signature` of `tl.mutual_information`, `tl.clonotypic_entropy`, `tl.phenotypic_entropy`, `tl.phenotypic_flux` and assert every parameter is declared either as a reduction parameter (in `METRIC_SPECS` / `SANCTIONED_EXTENSIONS`) or as a joint-construction parameter (in `JOINT_CONSTRUCTION`). Exact set equality; a new dial cannot appear without a manifest entry. This test fails on today's `main` for `weighted`, `use_logits` (via `_common`), `temperature` and `gate`, which is the confirmation restated as a guard.
2. `test_clone_mass_modes_match_their_definitions` -- synthetic joint with unequal clone sizes; assert `weighted=True` reproduces eq 5 on the raw count crosstab and `weighted=False` reproduces eq 5 on the row-normalized crosstab, both computed independently in the test from the formula. Tolerance 1e-12, analytic, derived from the definition rather than from a run.
3. `test_clone_mass_choice_is_not_a_level_shift` -- assert the two modes differ in SIGN across two fixtures with opposite coupling regimes (sharp: abundance > uniform; flat: abundance < uniform). This pins the property that makes the parameter contract-worthy, and its tolerance is a sign comparison, not a magnitude.
4. `test_ground_truth_comparisons_use_abundance_mass` -- assert `benchmarks/run_grid.py` calls `tcri.tl.mutual_information` with `weighted=True`, guarding Note 1's protocol against a future edit. Source-level assertion is acceptable here because the alternative (retraining a model in CI) is not.
5. Full suite green: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q` -- expect 177 passed / 3 skipped unchanged, since option 1 moves no number.

---

## DE-8 · S3 · effort XS · FIXED (PR `consolidate-and-record`)

**Confirmed:** REAL, on main. `benchmarks/run_grid.py:165` writes `epochs=epochs` into the result row, where `epochs` is the CLI value threaded unchanged from `run_grid.py:189` (`--epochs`, default 60) via `run_grid.py:232`. It is the REQUESTED value; `run_grid.py:129` passes it as `max_epochs=` and nothing reads back what the trainer did. The row records `t_train` (line 172) but no epoch count.

The actual count IS available and is being discarded. Ran (MPLBACKEND=Agg .venv/bin/python, simulate_tcri 400 cells / 80 genes, `patience=1`, `max_epochs=400`, batch_size=128, cpu):
```
requested max_epochs      : 400
trainer.current_epoch     : 25          # second run of the identical script: 15
trainer.max_epochs        : 400
earlystop stopped_epoch   : 24          # LoudEarlyStopping
check_val_every_n_epoch   : 5
history keys              : [... 'elbo_validation' ...]
n validation checks       : 5   best row label (epoch): 19
```
So `model.trainer.current_epoch`, `model.trainer.early_stopping_callback.stopped_epoch`, `model.trainer.global_step` and `model.history['elbo_validation']` (a DataFrame indexed BY EPOCH) are all populated after `train()` returns, and all are dropped. The two identical invocations stopped at 25 and 15 epochs, so the run-to-run spread this column would expose is not small.

**Candidates considered:** (1) Record `trainer.current_epoch` only. One line, but it cannot distinguish 'ran to max' from 'early-stopped at max-1', and gives no handle on where the best model was.

(2) Record the full training provenance: requested, actual, early-stopped flag, best-validation epoch, global step, number of validation checks. Four to six extra columns, all read from objects the harness already holds; zero runtime cost.

(3) Disable early stopping in the harness (`early_stopping=False`) so requested == actual by construction. Removes the discrepancy instead of recording it, and it would silently change every benchmark number while dropping the note's 'maximum of 2,000 epochs' semantics (a maximum implies stopping is permitted). Rejected: converts a reporting defect into a protocol change.

(4) Record epochs AND assert `actual == requested`, failing the cell otherwise. Too strict; early stopping is legitimate. The defect is that it was invisible, not that it happened.

**Chosen fix:** Candidate (2) — record the provenance, do not change the protocol. Safest (pure addition; run behaviour is untouched) and fastest (about ten lines).

In `benchmarks/run_grid.py::run_cell`, immediately after the `model.train(...)` / `model.to_anndata(adata)` block (currently lines 128-132):

```python
tr = model.trainer
hist = model.history.get("elbo_validation")
best_val_epoch = int(hist.idxmin().iloc[0]) if hist is not None and len(hist) else -1
esc = getattr(tr, "early_stopping_callback", None)
train_prov = dict(
    epochs_requested=int(epochs),
    epochs_run=int(tr.current_epoch),
    early_stopped=bool(getattr(esc, "stopped_epoch", 0)),
    stopped_epoch=int(getattr(esc, "stopped_epoch", -1)),
    best_val_epoch=best_val_epoch,
    global_step=int(tr.global_step),
    n_val_checks=(0 if hist is None else int(len(hist))),
)
```
Then in the `row = dict(...)` literal (lines 163-173) replace `epochs=epochs` with `**train_prov`. Extend the per-cell progress `print` at lines 236-238 with `ep={r['epochs_run']}/{r['epochs_requested']}`. Add a sentence to the module docstring's 'Two things this harness is careful about' list — making it three — naming the requested/actual distinction.

Order of work: land DE-8 FIRST, before DE-9/DE-10/DE-11. Every other entry's before/after comparison is uninterpretable without it.

**Contract change:** NONE. `benchmarks/run_grid.py` is not named in `tcri/_contract.pyi`, `tcri/model/_model_contract.py`, or `tcri/tools/_metrics_contract.py` (grep returns only incidental prose uses of the word 'benchmark' in the metrics manifest). No manifest or conformance test is touched.

**Dependencies affected:** NO metric output moves. This is additive instrumentation: `tcri_nmi`, `true_nmi`, `empirical_nmi`, every `ae_*` column and the printed MAE table are byte-identical. No published number changes. No existing test changes its expected value (nothing under `tests/` imports `run_grid`; grep found only a prose mention at `tests/test_recovery.py:257`).

One breaking detail for downstream readers: the `epochs` column is RENAMED to `epochs_requested`. Keep `epochs` as a duplicate alias for one release if any analysis script hardcodes it. Older CSVs will show NaN for the new columns when concatenated.

What this unblocks rather than affects: DE-2 (patience counted in validation checks), DE-3 (no best-weight restore) and DE-10 (the 60-epoch default) are currently unmeasurable from a results CSV. `best_val_epoch` vs `epochs_run` is exactly the DE-3 gap, made visible per cell for free.

**Verification tests:** 1. `tests/test_bench_harness.py::test_trainer_exposes_epoch_provenance` — guards the API this fix depends on so a scvi/Lightning upgrade cannot silently reintroduce the defect: after a 400-cell `train(max_epochs=8, early_stopping=False)`, assert `model.trainer.current_epoch == 8` and `'elbo_validation' in model.history`. Exact integer equality, derived from the loop structure (no early stopping implies epochs run == max_epochs), not from any measured value.

2. `test_early_stop_is_detectable` — with `patience=1`, `check_val_every_n_epoch=5`, `max_epochs=400`: assert `model.trainer.current_epoch < 400` and `model.trainer.early_stopping_callback.stopped_epoch > 0`. The inequality is structural: patience=1 means 10 epochs of non-improvement suffice, so 400 cannot be reached on a converging fit. No numeric tolerance.

3. `test_run_cell_records_actual_epochs` — call `run_cell` on the smallest synthetic cell with `epochs=400` and a small patience; assert the returned dict has `epochs_requested == 400` and `0 < epochs_run <= 400`, and that `epochs_run` appears in `pd.DataFrame([row]).columns`. Bounds derived from the argument, not from behaviour.

4. Dependency-intact check: `test_provenance_does_not_move_the_estimate` — run `run_cell` twice at a fixed pyro seed, once with the provenance block and once with it stubbed out; assert `tcri_nmi` is bitwise equal. Tolerance 0.0, justified because the block only reads attributes.

---

## DE-9 · S3 · effort S · open

**Confirmed:** REAL, on main. `benchmarks/run_grid.py:247` is `print(df.groupby("fuzziness")[cols].mean().round(4).to_string())`. `temperature` is a real grid axis (`PRESETS['published']` and `PRESETS['published_quick']` both carry `temperature=[0.1,0.5,1.0]`, expanded into `combos` at lines 222-223 and stored per row at line 165) and is absent from the grouping. `n_cells` and `k_infer` are equally absent — the header at line 244 at least names them ('mean over N, K, seeds') but never mentions temperature, so the printed caption is wrong as well as the aggregation.

Magnitude, measured on the actual `published_quick` grid (fit_params_K10 fixture, N ∈ {250,1000,5000}, T ∈ {0.1,0.5,1.0}, f=0.1, seeds 0-2, `normalize_mode='average'`), using the plug-in oracle so no training noise enters:
```
pooled AE over all 27 rows, as run_grid prints it : 0.0420
           true_nmi      ae
N    T
250  0.1      0.5196  0.0069
     0.5      0.3165  0.1093
     1.0      0.1816  0.1606
1000 0.1      0.5196  0.0004
     0.5      0.3165  0.0271
     1.0      0.1816  0.0534
5000 0.1      0.5196  0.0020
     0.5      0.3165  0.0088
     1.0      0.1816  0.0096
```
One printed number, 0.0420, stands in for per-cell values spanning 0.0000 to 0.1646. The pooled ground truth spans 0.5196/0.1816 = 2.862x, matching the register's 2.86x. `fuzziness` — the only grouping key — takes ONE value in that preset, so the table degenerates to a single row.

**Candidates considered:** (1) Add `temperature` to the groupby. Fixes the named symptom, leaves `n_cells` and `k_infer` pooled — and at fixed T=1.0 the N axis alone spans AE 0.0096 to 0.1606.

(2) Group by every axis that actually varies in this run, derived from the frame. The table then adapts to the preset and can never silently pool a swept axis again. Rows = product of varying levels: 1 for `smoke`, 9 for `published_quick`, 450 for `published` (too long to read on a console).

(3) (2) for the aggregation, plus a compact console view: print the full grouping when it is under ~40 rows, otherwise print one marginal table per axis and point at the CSV. Keeps the console usable on the full grid without misstating what was averaged.

(4) Report relative error `ae/true_nmi` instead of AE, so pooling across a 2.86x truth range is at least dimensionless. Rejected as a REPLACEMENT — the note is explicit: 'We use the absolute error between the estimated and ground-truth normalized mutual information (NMI) as the performance metric'. Viable as an ADDITIONAL column.

**Chosen fix:** Candidate (3), with (4)'s ratio as an extra column. Safest because the invariant becomes structural rather than a hand-maintained list of keys, and fastest because it is one small pure function.

Extract the summary out of `main()` into a testable function in `benchmarks/run_grid.py`:

```python
GRID_AXES = ("fuzziness", "n_cells", "k_infer", "temperature")

def summarize(df, cols, max_rows=40):
    """Group by EVERY swept axis. Pooling an axis that varies is the DE-9 defect."""
    varying = [a for a in GRID_AXES if a in df and df[a].nunique() > 1] or ["fuzziness"]
    return varying, df.groupby(varying)[cols].mean()
```
`main()` replaces lines 244-247 with a call to `summarize`, printing the full table when `len(full) <= max_rows` and otherwise one marginal table per axis in `varying`, each headed with an explicit 'marginal over the other axes' caption. Add `rel_ae_vs_true = ae_vs_true / true_nmi` to `row` in `run_cell` and include it in `cols`, giving the pooled view a scale-free companion to AE. Build the caption at line 244 from `varying` rather than hardcoding 'mean over N, K, seeds'.

**Contract change:** NONE. Harness-only; no manifest, no conformance test.

**Dependencies affected:** NO per-cell metric output moves. `tcri_nmi`, `true_nmi`, every `ae_*` value and the CSV rows are unchanged; only the console aggregation and one added derived column change. No published number changes on its own — but the MAE-vs-fuzziness reading a person would take from `published_quick` goes from a single 0.0420 to nine cell values ranging 0.0000-0.1646, so any narrative built on the pooled figure needs restating.

Interaction with DE-8: once `epochs_run` is a per-row column it becomes tempting as a grouping key. Do NOT add it to `GRID_AXES` (it is an outcome, not a design axis) — report it per group as mean and sd so a group whose members trained for wildly different durations is visible.

Interaction with DE-12: the interpretability flag must enter this same summary, so land DE-9's `summarize` extraction before DE-12 and let DE-12 add columns to it rather than re-forking the print.

No existing test changes its expected value.

**Verification tests:** 1. `test_summarize_groups_every_varying_axis` — build a synthetic frame with `fuzziness` constant at 0.1, `temperature` ∈ {0.1,0.5,1.0}, `n_cells` ∈ {250,5000}. Assert `varying == ['n_cells','temperature']`, `full.index.names == ['n_cells','temperature']`, `len(full) == 6`. Exact structural equality; no tolerance applicable.

2. `test_summarize_never_pools_a_swept_axis` — the property test, so this cannot regress for a NEW axis added later: for every column in `GRID_AXES`, if `df[col].nunique() > 1` then `col in full.index.names`. Asserted over a randomly generated frame; the invariant follows from the definition of an axis, not from behaviour.

3. `test_pooling_would_have_hidden_the_spread` — the anti-regression witness. On the frame from (1), assert `df.groupby('fuzziness')[['ae_vs_true']].mean()` has exactly one row while `summarize(...)` returns more than one, and that `full.ae_vs_true.max() / full.ae_vs_true.min() > 2`. The factor 2 comes from the ground-truth ratio of the pooled axis (2.862x, computed from `mi_from_joint_oracle` on the fixture omega at T=0.1 vs T=1.0), not from any measured estimate.

4. `test_caption_names_the_actual_axes` — assert the printed header contains every name in `varying`. Guards the specific way this defect stayed invisible: a caption that disagreed with the aggregation.

---

## DE-10 · S2 · effort XS · open

**Confirmed:** REAL, on main. `benchmarks/run_grid.py:189`: `ap.add_argument("--epochs", type=int, default=60)`. The note's protocol (Supplementary Note 1, 'Experimental Setup', extracted from `docs/contract/source/supplementary_note_1_SS_2026-08-03.pdf`): 'The semi-synthetic generator uses L = 5 NMF factors, and TCRi is trained for a maximum of 2,000 epochs.' The harness docstring at `run_grid.py:4` claims it 'Reproduces the design of Supplementary Note 1's Benchmarks section'; on this point it does not.

Measured the gap directly (fit_params_K10, N=1000, T=1.0, cpu, `normalize_mode='average'`, `n_samples=200`, seed 0; true_nmi = 0.1816):
```
f    init          epochs_req  epochs_actual  shipped(g=0.5)  det(g=0)  det(g=1)
0.1  true_labels          60             60          0.2987    0.2378    0.0001
0.1  true_labels        2000           2000          0.2939    0.2241    0.0002
0.9  true_labels          60             60          0.3001    0.2380    0.0000
0.9  true_labels        2000           2000          0.2913    0.2237    0.0000
0.1  kmeans               60             60          0.2969    0.2348    0.0000
0.1  kmeans             2000           2000          0.2838    0.2180    0.0000
0.9  kmeans               60             60          0.2304    0.1297    0.0000
0.9  kmeans             2000           2000          0.2222    0.1136    0.0000
```
Two facts. First, 60 -> 2000 moves the shipped estimate DOWN by 0.005-0.008 under the current true-label init and by 0.008-0.013 under K-Means init — small, and toward the truth in every cell. Second, `epochs_actual == 2000` in all four 2000-epoch runs: early stopping never fired inside the note's budget, because DE-2's patience is 300 validation checks x `check_val_every_n_epoch=5` = 1500 epochs of non-improvement. At the note's protocol the run is a fixed-length 2000-epoch fit and the early-stopping machinery is inert.

Separately, 60 epochs is where `p_ct` is still sitting on the crosstab it was INITIALISED at: the gate-0 deterministic read is 0.2378 against an initialisation crosstab NMI of 0.2383 (true labels, f=0.9) and 0.1297 against 0.1301 (K-Means, f=0.9). The harness default reports, to three decimals, its own starting value. That is the substantive reason to move it, independent of protocol conformance.

**Candidates considered:** (1) Set `default=2000`, matching the note. Correct by the source document; ~35x the wall clock (measured 2.5 s -> 90 s per cell at N=1000 on cpu, roughly linear in cells).

(2) Make `--epochs` REQUIRED. Forces every run to state its budget, so no number is produced by an unexamined default. Costs nothing and cannot be wrong, but loses the note-conformant convenience and breaks every existing invocation.

(3) Keep 60 as a documented shake-out value and bind the epoch budget to the preset: `smoke`/`reduced` -> 60, `full`/`published`/`published_quick` -> 2000, with `--epochs` overriding. The epoch budget is part of the protocol, so it belongs next to the grid it goes with.

(4) Set 2000 AND read the metric at the best-validation checkpoint rather than the final weights. Technically the right answer — but it is DE-3, it requires capturing the Pyro param store (`q_p_c_raw`, `q_p_ct_raw` are not in `module.state_dict()`), and it is a model-side change. Out of scope for a harness fix.

**Chosen fix:** Candidate (3), which subsumes (1) for every preset that claims to reproduce the note.

In `benchmarks/run_grid.py`: add an `epochs` key to each entry of `PRESETS` — 60 for `smoke` and `reduced` (relabelled in the docstring as shake-out grids that deliberately do NOT reproduce the note), 2000 for `full`, `published` and `published_quick`. Change `--epochs` to `default=None` and resolve in `main()` as `epochs = args.epochs if args.epochs is not None else grid['epochs']`, printing the resolved value in the `preset=... cells=...` banner at lines 225-226. Amend the module docstring's careful-about list with a bullet naming the note's 2,000-epoch maximum and stating that `smoke`/`reduced` do not honour it.

Do NOT bundle the checkpoint question in. Record in the DEFECTS entry that DE-10's fix leaves the estimate read from post-degradation weights and that DE-3 is the follow-up; with DE-8 landed, `best_val_epoch` vs `epochs_run` quantifies that gap per cell for free.

Sequencing: DE-8 first (so requested vs actual is recorded), then DE-10 together with DE-11 in a single re-run — the two interact and must not be stacked onto separate baselines.

**Contract change:** NONE for the default change itself (harness-only). The accompanying question is a MODEL one: whether the metric should be read at the best-validation checkpoint rather than the final weights is DE-3, and `q_p_ct_raw` living in the Pyro param store rather than `module.state_dict()` means a Lightning checkpoint cannot supply one. If DE-3 is resolved by adding param-store capture to the training plan, that touches `tcri/model/_training.py` and must be checked against `_model_contract.py` before code is written. Keep it out of DE-10.

**Dependencies affected:** EVERY benchmark number moves, by a small amount, in a consistent direction. Measured on the fixture above: `tcri_nmi` falls by 0.005-0.013, i.e. TOWARD the truth in all four cells, so `ae_vs_true` improves slightly. `ae_vs_empirical` moves the same way. `tcri_nmi_meanjoint` and `jensen_gap` shift correspondingly (same posterior). `t_train` and `t_total` rise ~35x. The clustering baseline columns are untouched (no training).

Any figure produced at 60 epochs must be regenerated. Under the CURRENT true-label init the shape of the MAE-vs-fuzziness curve does not change (it is flat at either budget — see DE-11), so a published figure is unlikely to flip qualitatively on DE-10 alone. Under the DE-11 correction they interact: at f=0.9 with K-Means init, 60 -> 2000 moves the deterministic read 0.1297 -> 0.1136, a 12% relative change.

Interaction with DE-2: at 2000 epochs early stopping is provably inert (effective patience 1500 epochs), so `early_stopped` will be False on every cell. That is a finding to record, not a bug in this fix. Interaction with DE-1: 35x the epochs means ~35x the validation checks, hence ~35x the spurious optimizer steps applied to `q_p_ct_raw`; DE-1's measured per-check delta of 0.54 compounds, so DE-1 should land before any 2000-epoch published run.

No existing test changes its expected value.

**Verification tests:** 1. `test_preset_epoch_budget_matches_the_note` — assert `PRESETS['published']['epochs'] == 2000`, same for `published_quick` and `full`. The value is transcribed from the source document ('trained for a maximum of 2,000 epochs'), not from measurement; the test's job is to make a later silent edit visible.

2. `test_shakeout_presets_are_labelled_non_conformant` — assert `PRESETS['smoke']['epochs'] == 60` and `PRESETS['reduced']['epochs'] == 60`, and that the module docstring contains a sentence marking these as not reproducing the note. Guards the exact failure mode: a fast default quietly becoming the published protocol.

3. `test_cli_epochs_overrides_preset` — parse `--preset published --epochs 60`, assert the resolved budget is 60 and that `epochs_requested` in the row equals 60. Exact equality from the argument.

4. `test_estimate_is_not_pinned_to_its_initialisation` — the substantive check behind this entry, and the one that would have flagged 60 epochs as a bad budget. Fit at the preset budget and assert `abs(det_g0_read - init_crosstab_nmi) > 0.01`, where `init_crosstab_nmi` is the DE-11 column. The 0.01 bound is roughly 5x the estimator's fit-to-fit spread at identical seed (0.0018), so it asserts the optimiser moved by more than its own noise — it is not fitted to the measured 0.0005 gap at 60 epochs (which is what fails today).

5. Dependency-intact check: `test_epoch_budget_leaves_the_oracles_alone` — run one cell at 60 and at 2000 epochs, assert `true_nmi` and `empirical_nmi` are bitwise identical. Tolerance 0.0, justified because both come from the generator's parameters and the realized `(z, phi)` counts, which training cannot touch.

6. Cross-defect record: `test_early_stopping_inert_at_the_note_budget` — at `max_epochs=2000` with default `patience=300` and `check_val_every_n_epoch=5`, assert `epochs_run == 2000` and `early_stopped is False`. Derived arithmetically (300 x 5 = 1500 epochs of required non-improvement, so a stop before 1500 is impossible and a stop between 1500 and 2000 requires the ELBO to have peaked before epoch 500). If this test ever starts failing, DE-2 has been fixed and this benchmark's protocol needs re-reading.

---

## DE-11 · S2 · effort M · open

**Confirmed:** REAL, on main, and its effect is larger than the register states. `benchmarks/run_grid.py:112` passes `phenotype_key="phenotype"`, and neither `simulate_tcri` nor `simulate_from_fit_params` is called with `label_error_rate`, which defaults to 0.0 (`tcri/datasets/_simulate.py:113`, `:256`), so `phi = phi_true.copy()` is left untouched (`_simulate.py:172-174`, `:311-314`). The note is explicit (Supplementary Note 1, 'Experimental Setup'): 'We use K-Means with the supplied value of K to obtain the initial phenotype assignments for TCRi.'

Where the labels land, traced: `tcri/model/_model.py:159-164` builds `clone_phenotype_prior` as the row-normalised clone x observed-phenotype crosstab; `_model.py:177` k-means-clusters THAT matrix into the VampPrior archetypes; `tcri/model/_module.py:285` and `:312` use it as the INITIAL VALUE of `q_p_c_raw` and `q_p_ct_raw`. The per-cell label tensor `_target_phenotypes` is registered (`_module.py:132,160`) and never read — grep over `tcri/` returns only those two sites. So the observed phenotype labels influence the fit through exactly one channel: the clone-level crosstab the benchmark is asking the model to estimate.

Measured (fit_params_K10, `normalize_mode='average'`), the initialisation point in metric units:
```
nmi(crosstab from true labels)  ==  empirical_nmi  to 1e-13, in all 27 published_quick cells
```
The model starts at the plug-in oracle exactly. At 60 epochs it has not left: gate-0 deterministic read 0.2378 vs init 0.2383 (true labels, N=1000, f=0.9); 0.1297 vs init 0.1301 (K-Means).

The consequence is that the benchmark's difficulty axis does nothing. `fuzziness` perturbs gene expression only, never the labels, so the initialisation is constant along it:
```
init-crosstab NMI, N=1000, T=1.0, mean of 3 seeds  (true = 0.1816)
f          0.0     0.1     0.3     0.5     0.7     0.9
true lab  0.2350  0.2350  0.2350  0.2350  0.2350  0.2350   <- flat, by construction
kmeans10  0.2344  0.2307  0.2332  0.2248  0.2056  0.1294
```
The trained, shipped read inherits it:
```
shipped NMI (g=0.5, n_samples=200), N=1000, T=1.0, true=0.1816
            f=0.1    f=0.9    delta over the whole axis
  60 ep  true lab   0.2987   0.3001   +0.0014
  60 ep  kmeans     0.2969   0.2304   -0.0665
2000 ep  true lab   0.2939   0.2913   -0.0026
2000 ep  kmeans     0.2838   0.2222   -0.0616
```
The fit-to-fit spread from unseeded network init is ~0.0018. Under the current harness the entire fuzziness sweep moves the reported number by 0.0014-0.0026 — at or below the noise. Under the note's K-Means init it moves by 0.062-0.067, roughly 35x the noise.

Second confirmed consequence: `k_infer` never reaches the phenotype dimension. `TCRIModel(K=k)` sets the number of VampPrior archetypes; `P` comes from the observed label column (`_model.py:148-149`). Verified:
```
k_infer= 8  -> module.P = 10   archetypes B = (8, 10)
k_infer=10  -> module.P = 10   archetypes B = (10, 10)
k_infer=12  -> module.P = 10   archetypes B = (12, 10)
```
The note's K ∈ {8,10,12} axis — 'both TCRi and the GMM baseline are supplied with the same value of K' — is not being exercised. K-Means init fixes this as a side effect: `P` becomes exactly `k_infer`.

Does the correction revive the classifier? NO. Classifier-only reads (gate=1) under both inits, both budgets: 0.0002, 0.0000, 0.0000, 0.0000. This is structural. The only term training `f_cls` is `pyro.factor('phenotype_alignment', -gamma * KL(softmax(pi*cls_logits + (1-pi)*log phi) || phi))` at `_module.py:241-249`, with `phi = p_ct[ct_idx].detach()` a per-(clone,covariate) constant. Its minimiser in `cls_logits` is any vector constant across phenotypes, so the classifier is trained toward carrying zero phenotype information regardless of where the labels came from, and `_target_phenotypes` never enters the ELBO. The inert classifier is a separate defect from DE-11 and DE-11's fix does not touch it.

**Candidates considered:** (1) Set `label_error_rate > 0` so the supplied labels are noisy true labels. Minimal code. Rejected: not the note's protocol (the note names K-Means specifically), and `P` stays pinned to the generator's phenotype count so the K axis remains dead.

(2) Add `--phenotype-init {kmeans,true}` to `run_grid`, default `kmeans`. Derive labels by the note's recipe (divide each cell by its gene sum, x1e4, standardize each gene to mean 0 / sd 1, `KMeans(n_clusters=k_infer)`), write to `adata.obs['phenotype_init']`, pass `phenotype_key='phenotype_init'`. Keeps `phenotype`/`true_phenotype` intact for the oracles and PPCs. `P` becomes `k_infer`, so the K axis becomes real.

(3) Promote the recipe into `tcri.datasets` as public `kmeans_phenotype_init(adata, k, *, seed)`. Reusable and testable inside the package, and the natural home for a protocol step the manuscript specifies — but it adds public surface and a CODEOWNER-restricted `_contract.pyi` change. Defer.

(4) Do (2) but default to `true`, K-Means opt-in. Rejected: it leaves the note-non-conformant configuration as the one people run by accident, which is how this got here.

**Chosen fix:** Candidate (2), with the `true` mode retained and explicitly LABELLED as an oracle rather than a benchmark configuration. Safest because it is additive and reversible per run; fastest because the normalize/standardize/cluster block already exists in the harness and only needs lifting out.

All in `benchmarks/run_grid.py`:

a. Extract lines 74-81 of `_baseline_nmi` into `def _kmeans_labels(adata, k, seed) -> np.ndarray`, implementing the note's recipe verbatim; `_baseline_nmi` then calls it.

b. Add `ap.add_argument('--phenotype-init', choices=['kmeans','true'], default='kmeans', help="'kmeans' == the note's protocol (default); 'true' hands the model the ground-truth labels and is an ORACLE, not a benchmark configuration")`. Thread into `run_cell`.

c. In `run_cell`, between the `simulate_*` call and `setup_anndata` (before line 111):
```python
if phenotype_init == "kmeans":
    lab = _kmeans_labels(adata, k_infer, seed)
    cats = [f"km_{i}" for i in range(k_infer)]
    adata.obs["phenotype_init"] = pd.Categorical([f"km_{i}" for i in lab], categories=cats)
    pheno_key = "phenotype_init"
else:
    pheno_key = "phenotype"
```
and pass `phenotype_key=pheno_key` at line 112. Naming `categories=` explicitly also closes DE-13 for this column.

d. Record provenance in `row`: `phenotype_init`, `P_supplied=int(model.module.P)`, and `init_crosstab_nmi` (NMI of the clone x supplied-label crosstab under the run's `normalize_mode`). The last makes the defect self-reporting: if `init_crosstab_nmi` does not move along the fuzziness axis, the axis is inert.

e. Change `--baseline` default from `kmeans` to `gmm`. The note says 'We compare TCRi to a Gaussian mixture model (GMM) as a baseline'; with (c) landed, a K-Means baseline is numerically identical to TCRi's own initialisation, so the comparison would be vacuous. `_baseline_nmi` already supports `method='gmm'`.

f. Update the module docstring: the careful-about list becomes four items (normalization, which oracle, epoch budget from DE-10, phenotype initialisation), and the 'Reproduces the design of Supplementary Note 1' claim at line 4 becomes accurate.

ADJACENT UNREGISTERED DEFECTS found while confirming this one — both belong in DEFECTS.md as their own entries, not folded in here:
- The fuzziness interpolation deviates from the note. The note (p. 11) specifies a CONCAVE mapping, 'we use g(f) = sqrt(f)', applied as `theta' = (1-g(f))*theta + g(f)*theta_bar`. `_simulate.py:321-325` (and the matching block in `simulate_tcri`) uses `fuzziness` directly, i.e. g(f) = f, which is not concave. Measured at N=1000, T=1.0: under g(f)=f the K-Means init NMI is flat 0.2344 -> 0.2248 from f=0 to f=0.5 and only falls at f>=0.7; under g(f)=sqrt(f) it declines monotonically 0.2344 / 0.2223 / 0.2011 / 0.1551 / 0.1005. The note's mapping spreads the difficulty axis; the code's bunches it at the top end.
- `--baseline` defaults to `kmeans` where the note specifies GMM (item (e)).

**Contract change:** NONE if the K-Means step lives in `benchmarks/run_grid.py` (chosen). A contract change ONLY if the recipe is promoted into `tcri.datasets` as public API — that adds a name to `tcri/_contract.pyi` and to the responsibilities prose, both CODEOWNER-restricted, and requires onboarding into `IMPLEMENTED` in `tests/test_contract_conformance.py` (which compares live signatures to the `.pyi` by parameter name and kind). Recommend keeping it in the harness precisely to avoid that.

**Dependencies affected:** EVERY benchmark number moves, and the shape of the headline figure changes. Directions, measured:

- `tcri_nmi` FALLS, by an amount that GROWS with fuzziness: -0.002 at f=0.1, -0.070 at f=0.9 (60 epochs); -0.010 at f=0.1, -0.069 at f=0.9 (2000 epochs). At low fuzziness the two inits nearly coincide because K-Means recovers the phenotypes almost perfectly there.
- `ae_vs_true` falls at high fuzziness (0.117 -> 0.045 at f=0.9, N=1000, T=1.0, 60 ep) and is roughly unchanged at low fuzziness (0.117 -> 0.113 at f=0.1). The improvement at f=0.9 is a CANCELLATION, not accuracy: the K-Means crosstab under-reads the truth (0.1294 vs 0.1816) while the estimator over-reads by ~+0.06, so the two errors partly annul. Report it as such.
- `ae_vs_empirical` moves MORE than `ae_vs_true`, because `empirical_nmi` is exactly the true-label crosstab the current harness hands the model. Under DE-11 the model no longer starts at that number, so this column stops being near-zero by construction.
- The MAE-vs-fuzziness curve goes from flat (total range 0.0014-0.0026, at or under the 0.0018 fit-to-fit noise) to sloped (range 0.062-0.067). The claim that the harness measures sensitivity to gene-expression-to-phenotype noise only becomes true after this fix.
- The K axis begins to exist: `module.P` becomes `k_infer` instead of a constant 10, so the k_infer ∈ {8,10,12} cells stop being three replicates of the same phenotype dimension. Expect K=8 and K=12 to separate from K=10 for the first time.
- The baseline column changes name and value under item (e) (`kmeans_nmi` -> `gmm_nmi`).
- `t_total` rises by the K-Means fit: ~0.2 s at N=250 to ~2 s at N=5000, negligible against training.

What does NOT move: `true_nmi` and `empirical_nmi`, computed from the generator's `omega`/`pi` and the true `(z, phi)` counts (`_simulate.py:307`, `:330-332`), untouched by which labels are supplied at inference. The estimand is unchanged; only the estimator's input.

What does NOT change: the classifier stays inert (gate=1 read 0.0000-0.0002 under both inits at both budgets). DE-6 is unaffected.

No published number in the PACKAGE changes; no metric definition changes; no existing test changes its expected value. Any benchmark figure already produced is superseded.

Sequencing: DE-8, then DE-9's `summarize` extraction, then DE-10 and DE-11 together in one re-run — they interact (60 -> 2000 epochs moves the K-Means f=0.9 deterministic read 0.1297 -> 0.1136, a 12% relative change).

**Verification tests:** 1. `test_supplied_K_reaches_the_phenotype_dimension` — the structural equality that is currently false. Run the `kmeans` init path with `k_infer=8` and `k_infer=12`, assert `model.module.P == k_infer` in both. Exact integer equality, derived from the note's 'both TCRi and the GMM baseline are supplied with the same value of K'.

2. `test_difficulty_axis_moves_the_initialisation` — the defect's signature, written so it cannot silently return. For `phenotype_init='true'`: assert `init_crosstab_nmi(f=0.0) == init_crosstab_nmi(f=0.9)` EXACTLY (equality holds by construction; `fuzziness` never touches `phi`) and keep it as documentation of why the axis was dead. For `phenotype_init='kmeans'`: assert `init_crosstab_nmi(f=0.0) - init_crosstab_nmi(f=0.9) > 0.05`. The 0.05 threshold is derived from the estimator's fit-to-fit spread at identical seed (0.0018): the axis must move the initialisation by at least an order of magnitude more than the estimator's own noise, or the sweep cannot be read. It is NOT taken from the measured 0.105.

3. `test_kmeans_recipe_matches_the_note` — unit-test `_kmeans_labels` against the note's three stated steps: row sums after the CPM step equal 1e4 (rtol 1e-9, from arithmetic); each gene column has mean 0 and sd 1 after standardization (atol 1e-9); `len(np.unique(labels)) == k`. All tolerances from float arithmetic, none from behaviour.

4. `test_benchmark_never_supplies_ground_truth` — the guard that would have caught this. When `phenotype_init='kmeans'`, assert `adata.obs[pheno_key]` differs from `adata.obs['true_phenotype']` on more than 1% of cells, and that `pheno_key != 'phenotype'`. The 1% floor is a triviality bound, not a measurement.

5. `test_true_init_is_labelled_an_oracle` — assert `run_cell` records `phenotype_init` in the row and that the `--phenotype-init` help text contains 'ORACLE'. Cheap, and it is what stops the oracle configuration being mistaken for a benchmark result again.

6. Dependencies intact: `test_oracles_are_init_invariant` — for the same `(fuzziness, n_cells, k_infer, temperature, seed)`, assert `true_nmi` and `empirical_nmi` are bitwise identical between `phenotype_init='true'` and `'kmeans'`. Tolerance 0.0, justified because both are computed from the generator's parameters and the true `(z, phi)` counts, which the init switch does not touch.

7. Cross-defect pin: `test_classifier_still_inert_under_both_inits` — assert the gate=1 deterministic read is < 0.01 under both inits, so a future DE-6 fix is attributed correctly and DE-11 is not credited with it. The 0.01 bound is roughly 5% of the smallest ground truth on the grid (0.1816) — a threshold for 'carries no usable information', not a fit to the observed 0.0000.

---

## DE-12 · S3 · effort M · open

**Confirmed:** REAL, on main. The result row in `benchmarks/run_grid.py:163-181` has no null, floor, or interpretability column of any kind, and the summary at lines 244-250 prints MAE and HDI coverage with nothing marking a cell as uninterpretable.

Reproduced the register's numbers exactly (fit_params_K10, `normalize_mode='average'`, f=0.1, seeds 0-2, 500 label permutations per cell):
```
            true_nmi  floor_mean  floor_p95  floor_p95 / true
N=250 T=0.1   0.5196     0.1376     0.1583    0.305
      T=0.5   0.3165     0.1814     0.2020    0.638
      T=1.0   0.1816     0.2110     0.2292    1.262   <- null exceeds the truth
N=1000 T=0.1  0.5196     0.0540     0.0610    0.117
       T=0.5  0.3165     0.0670     0.0744    0.235
       T=1.0  0.1816     0.0777     0.0854    0.470
N=5000 T=0.1  0.5196     0.0139     0.0155    0.030
       T=0.5  0.3165     0.0151     0.0169    0.053
       T=1.0  0.1816     0.0164     0.0183    0.101
```
Four of nine cells over 25% of truth (0.305, 0.638, 1.262, 0.470) — matches the register.

The design question — what the column should BE — turns on a measurement the register does not have. The above is the PLUG-IN floor; it is not the floor of the estimator that is actually reported. I measured the estimator's own floor by permuting the clone assignment across cells (abundances preserved, coupling destroyed), refitting, and reading the shipped metric (60 epochs, cpu, T=1.0, f=0.1, true = 0.1816):
```
        plugin_mean  plugin_p95  MODEL floor (shipped)  model/true   plugin cost   model cost
N=250      0.2144      0.2330            0.2799          1.542       0.004 s        1.7 s
N=1000     0.0763      0.0848            0.1703          0.938       0.006 s        3.0 s
N=5000     0.0165      0.0183            0.1380          0.760       0.019 s       14.6 s
```
The plug-in floor falls like 1/N (0.214 -> 0.076 -> 0.017); the model floor barely falls (0.280 -> 0.170 -> 0.138). At N=5000 a plug-in-only screen would pass the cell as clean (floor 10% of truth) while the estimator reports 0.138 — 76% of the truth — on data with NO clone-phenotype coupling at all. A plug-in-only column would mislabel the largest, most-trusted cells as interpretable.

Cost control: the model floor is invariant to the two largest axes. Six null fits at N=1000, f ∈ {0.0,0.5,0.9} x seeds {0,1}:
```
f=0.0: 0.1722, 0.1763   f=0.5: 0.1728, 0.1724   f=0.9: 0.1729, 0.1759
mean by f: 0.1743 / 0.1726 / 0.1744      overall sd: 0.0018
```
sd 0.0018 — the same magnitude as the unseeded-init fit-to-fit spread. The floor does not depend on fuzziness or seed, so one null fit amortises over both axes.

**Candidates considered:** (1) Plug-in floor only (permute phenotype labels, recompute NMI on the empirical crosstab; 200 permutations = 4-19 ms per cell). Free, and it correctly flags the small-N cells. Rejected as the sole column: it understates the estimator's floor by 0.06-0.12 NMI and would pass N=5000 cells whose real floor is 76% of truth.

(2) Model floor per cell (permute clone assignment, refit, read the shipped metric). Exactly the right quantity. Doubles the grid — 4500 extra fits on `published`, prohibitive at DE-10's 2000-epoch budget.

(3) Model floor amortised per configuration. The floor is invariant to fuzziness and seed (sd 0.0018, measured) and depends only on (n_cells, k_infer, temperature) — temperature enters via the phenotype MARGINAL, which survives permutation and sets the normalizer, so it must stay in the key. On `published` that is 5 x 3 x 3 = 45 configurations; with 3 null seeds each for a spread, 135 fits against 4500 cells = +3%.

(4) Exact analytic floor via the expected-mutual-information term of Adjusted MI (Vinh et al. 2010; `sklearn.metrics.cluster.expected_mutual_information`). Gives the plug-in MEAN exactly with no sampling, but no spread, and — decisively — it is a property of the contingency table, not of the estimator, so it inherits (1)'s understatement.

(5) Report nothing and document the small-N limitation in prose. Rejected: this is what allowed a cell whose null exceeds its truth to be read as a measurement.

**Chosen fix:** Candidate (3) as the reported floor, with candidate (1) alongside it as a free per-cell companion. Safest because the flag is derived from the estimator actually being scored; fast enough at +3% because the invariance measurement licenses the amortisation.

All in `benchmarks/run_grid.py`:

a. `def _plugin_floor(adata, *, normalize_mode, n_perm=200, seed)` — permute the phenotype-label code vector within each covariate, rebuild the clone x phenotype crosstab with `np.bincount` on the flattened key (cast codes to int64; pandas category codes are int8 and overflow the multiply — this bit me while measuring), score with `tcri.tools._mutual_information._mi_from_joint(J, normalized=True, mode=normalize_mode)`. Return mean, sd, p95. Note the `normalize_mode` argument: this is precisely what `diag.permutation_null` gets wrong at `_ppc.py:152`.

b. `def _model_floor(cfg, *, n_null_seeds=3, **fit_kwargs)` — for a configuration key `(n_cells, k_infer, temperature, phenotype_init, epochs)`, simulate as usual, permute `adata.obs['clone_id']` with `rng.permutation` keeping the category set, refit, and read `tcri.tl.mutual_information` with the run's exact `n_samples`/`weighted`/`normalize_mode`. Return mean and sd over `n_null_seeds`. Memoize on the configuration key in a module-level dict so each key is fitted once per invocation, and persist to a sidecar `<out>.null.json` so a re-run reuses it. Call `pyro.clear_param_store()` before and after each null fit.

c. Add to `row`: `null_model_mean`, `null_model_sd`, `null_plugin_mean`, `null_plugin_p95`, `floor_frac = null_model_mean / true_nmi`, and `interpretable = true_nmi > null_model_mean + 2*null_model_sd`. Keep both floors — the plug-in is free, and its DIVERGENCE from the model floor is itself the diagnostic for how much the estimator manufactures.

d. `--null {model,plugin,none}` defaulting to `model`, and `--null-seeds` defaulting to 3. `plugin` for cheap iteration; `none` only for smoke runs.

e. In DE-9's `summarize`, add `n_interpretable` / `n_cells` per group and print the MAE table twice — all cells, and interpretable cells only — with a one-line note when the two differ. Do not silently drop cells.

f. Docstring: state that the model floor is measured rather than assumed, that it is amortised over fuzziness and seed on the measured invariance (sd 0.0018 across f ∈ {0,0.5,0.9} x 2 seeds at N=1000), and that the plug-in floor is a lower bound on it.

Sequencing: after DE-9 (needs `summarize`), and after DE-10 and DE-11 (the floor must be measured at the epoch budget and phenotype init actually used — the memo key includes both).

**Contract change:** NONE for the chosen fix — the floor is computed inside `benchmarks/run_grid.py`, which no contract names.

A contract change WOULD be required for the rejected alternative of reusing `tcri.diag.permutation_null`: it hardcodes `mode="min"` at `tcri/diagnostics/_ppc.py:152` while the harness runs `normalize_mode='average'`, so it would compare a min-normalized null against an average-normalized truth. Giving it a `normalize_mode` parameter changes a frozen public signature — `tcri/_contract.pyi:142`, the prose row at `docs/contract/API_CONTRACT.md:721`, and the parameter-name check in `tests/test_contract_conformance.py`. All CODEOWNER-restricted. Recommend folding that into DE-15's fix, where `permutation_null`'s strength is already the subject.

**Dependencies affected:** NO existing metric output moves. `tcri_nmi`, `true_nmi`, `empirical_nmi`, every `ae_*` column and the HDI columns are unchanged; this is purely additive.

What changes is the READING of the existing numbers, materially. On `published_quick` at the measured floors: N=250/T=1.0 has a model floor of 0.280 against a truth of 0.182 — `interpretable=False`, and any MAE that averaged it in was averaging noise. N=1000/T=1.0 has floor 0.170 vs truth 0.182 — `interpretable=False` on the 2-sd rule. Even N=5000/T=1.0 sits at 0.138 vs 0.182, 76% of truth: it passes the rule but only barely, and should be reported with its `floor_frac`. Expect the interpretable-only MAE table to be built from noticeably fewer cells, concentrated at low temperature and high N.

Runtime: +3% at `--null model --null-seeds 3` on the `published` grid (135 null fits against 4500 cells); +0.02 s/cell for the plug-in floor at N=5000. Both dominated by DE-10's 35x epoch increase.

Cross-defect: the model floor is the natural measuring stick for DE-5 and DE-6. A floor of 0.138 at N=5000 with zero true coupling is the same order as the +0.076 metric-path error and DE-5's prior-set posterior width; once those land, this column is how you check they moved it. Do not tune anything against the floor — record it.

Relation to DE-15: that entry observes `diag.permutation_null` is weaker than the null actually used. This fix quantifies the gap (plug-in 0.018 vs model 0.138 at N=5000) and gives DE-15 a target. Recommend DE-15 add `normalize_mode` to `permutation_null` (contract change as noted) and document in its docstring that it is a plug-in floor and a LOWER BOUND on any model-based estimator's floor.

No existing test changes its expected value.

**Verification tests:** 1. `test_plugin_floor_calibrated_on_independent_data` — build an adata where clone and phenotype are drawn independently, so the observed NMI IS a draw from the null. Assert `abs(observed_nmi - null_plugin_mean) < 3 * null_plugin_sd`. The tolerance is the null's own sampling sd, so it holds with ~99.7% probability by construction; derived from the null distribution, not from any measured tcri behaviour.

2. `test_plugin_floor_scales_with_N` — same independent construction at N=250 and N=5000; assert `floor(5000) < floor(250) / 3`. The factor comes from the 1/N scaling of the plug-in MI bias ((r-1)(c-1)/(2N ln 2)), which predicts ~20x for a 20x N; the test asserts only 3x to stay robust to the normalizer.

3. `test_model_floor_exceeds_plugin_floor` — the finding that decided the design. On one small cell, assert `null_model_mean > null_plugin_p95`. The justification is structural, not empirical: the model floor contains the plug-in floor plus the estimator's own inflation (the Jensen term and the corner-seeking Dirichlet draws), both non-negative. If this ever fails, the estimator's inflation has been eliminated and the plug-in floor would then suffice — a result worth being told about.

4. `test_model_floor_invariant_to_fuzziness` — the invariance the amortisation rests on, so it cannot silently stop being true. Null fits at f=0.0 and f=0.9, fixed (N, K, T): assert `abs(floor(0.0) - floor(0.9)) < 0.02`. The 0.02 bound is ~10x the estimator's fit-to-fit spread at identical seed (0.0018), so it tests invariance rather than reproducing the measured 0.0018. If a future change makes the floor fuzziness-dependent, the memo key must gain `fuzziness`, and this test is what says so.

5. `test_interpretable_flag_on_a_known_clean_and_a_known_degenerate_cell` — a fixture with one clone per phenotype and N large must give `interpretable=True`; the N=250/T=1.0 configuration (where the null provably exceeds the truth) must give `interpretable=False`. The second needs no tolerance: assert `null_model_mean > true_nmi`, an ordering.

6. `test_null_is_read_with_the_run_normalize_mode` — assert the floor under `normalize_mode='min'` differs from the one under `'average'` on the same data, and that `run_cell` passes the run's mode through. Guards the specific way `diag.permutation_null` gets this wrong at `_ppc.py:152`.

7. Dependencies intact: `test_floor_columns_do_not_move_the_estimate` — run `run_cell` with `--null none` and `--null model` at a fixed pyro seed, assert `tcri_nmi` is bitwise equal. Tolerance 0.0; the null path fits a separate model and must not touch the cell's param store. This is also the regression test for forgetting `pyro.clear_param_store()` between the null fit and the real one — `tcri/model/_model.py:131-141` only WARNS on a dirty store, so a leak would silently continue the null fit.

---

## DE-13 · S3 · effort XS · FIXED (PR #61)

**Confirmed:** REAL. `tcri/datasets/_simulate.py:341` (`simulate_from_fit_params`) and `:190-192` (`simulate_tcri`) build `pd.Categorical(...)` with no `categories=`, so every label axis is data-derived.

Measured (`/tmp/de-generator-docs/de13.py`, `de13b.py`):
- `simulate_tcri(n_clones=40, n_phenotypes=10, n_genes=20, n_cells=100, omega_concentration=0.1, seed=5)` -> `obs['phenotype']` has **9** levels (`phen_1` missing) while `uns['tcri_truth']['omega']` is (40,10). `true_h_phenotype` = 3.0970 bits over 10 levels; the realized H(phi) in the data is 2.8827 over 9.
- `simulate_from_fit_params` with a fit-like peaked (47,10) omega: phenotype levels **7,7,7,8** at T=0.1/N=250; **8** at T=0.1/N=5000; **10** at T=1.0/N>=1000. Reproduces the register's "6-10 depending on (T, N, seed)".
- Same defect on the clone axis: **32-35 of 47** clone levels at N=250, **45-46 of 47** at N=5000.
- On `run_grid.py`'s own synthetic settings (`n_clones=40, n_phenotypes=5, omega_concentration=0.4`, seeds 0-2): phenotype is **5/5 at every grid size** — the phenotype loss does NOT bite the synthetic grid; clone levels are 36-39 of 40 at N<=1000.

So the mechanism is confirmed and the register's numbers reproduce; the bite is on the fit-params path (10 phenotypes, low T), which is the published-benchmark path.

**Candidates considered:** A. Declare `categories=` on all four label columns (`clone_id`, `phenotype`, `true_phenotype`, `covariate`) with the full generated space. Most faithful to "the generator knows its label space", but a declared-but-unobserved CLONE level propagates into the estimator: `_model.py:154-156` uses `.astype("category")`, which preserves empty categories, so `c_count` counts empty clones, `clone_phenotype_prior` gains an all-1e-6 (uniform after normalization) row, and the ct grid gains empty groups. Those clones then appear as pure-prior uniform rows in every `joint_distribution` at `weighted=False`, adding fictitious mass to the metric joint. That is a change to the estimator, not the generator.

B. Declare the PHENOTYPE space only (`phenotype` + `true_phenotype`), leave `clone_id`/`covariate` data-derived, and record the realized counts in `uns`. Gets the alignment that matters — phenotype axis == omega's column axis == the model's P — with no change to what the model sees on the clone axis, and makes the shortfall visible instead of silent.

C. Leave `obs` alone and instead restrict the recorded oracle to the realized support. Rejected: `true_*` is by construction the POPULATION value implied by (pi, omega); restricting it to a sample's support destroys the two-oracle distinction the module docstring (lines 17-31) exists to make.

**Chosen fix:** **B**, plus an explicit record of the shortfall. Safest (no estimator behaviour change on the clone axis) and fastest (one file, ~10 lines).

1. `tcri/datasets/_simulate.py::simulate_tcri`, obs block lines 188-199: build `pheno_levels = [f"phen_{p}" for p in range(n_phenotypes)]` and pass `categories=pheno_levels` to BOTH `phenotype` and `true_phenotype`. Leave `clone_id` / `covariate` / `batch` unchanged.
2. Same in `simulate_from_fit_params`, obs block lines 338-344, with `P = omega.shape[1]`.
3. Add to `uns['tcri_truth']` (both functions): `"realized_phenotypes": int(np.unique(phi).size)` and `"realized_clones": int(np.unique(z).size)`, next to `settings` — so a benchmark row can carry "this cell realized 33/47 clones" rather than that being discovered downstream.
4. Docstring Returns block: state that `phenotype`/`true_phenotype` declare the full `n_phenotypes` label space while `clone_id` carries only realized clones, and that `realized_clones` records the shortfall deliberately.

**Contract change:** NONE — `tcri/datasets` is absent from `tcri/_contract.pyi` (namespaces are ml/pp/tl/pl/diag/ut only) and from `tests/test_contract_conformance.py`. No manifest or prose contract is touched.

**Dependencies affected:** **No metric output moves on the current benchmark, and no published number changes.**

- `empirical_*` is unaffected: it is computed from a full-shape `counts` array (`_simulate.py:184-186` and `330-332`) in which an unrealized phenotype is a zero column contributing 0 to H(phi) and 0 to MI. Declaring the level changes nothing there.
- `true_*` is unaffected (closed form from pi, omega).
- Estimator side changes ONLY in runs that were losing a level. `TCRIModel.__init__` reads `P = len(ph_series.cat.categories)` (`_model.py:148-149`), so P goes 9 -> 10 exactly there. `clone_phenotype_prior` gains an all-1e-6 column, the classifier gains one output unit, `p_ct` gains a column. That column carries near-zero mass, so `tl.mutual_information` moves by <1e-3. Direction: **downward** at `normalize_mode="average"` (the H(phi) denominator can only grow); **unchanged** at the default `"min"` whenever H(phi) is not the minimum, which is the usual case at C >> P.
- `benchmarks/run_grid.py` synthetic path uses `n_phenotypes=5` and realizes all 5 at every grid size and seed measured, so its recorded `true_nmi` / `empirical_nmi` / `tcri_nmi` columns do not move. Nothing in `run_grid.py` indexes `omega` against the model axis (grep: omega appears only at lines 105-106, 201, 221), so the fit-params path's recorded numbers do not move either.
- No existing test changes an expected value: `tests/test_recovery.py` asserts inequalities and oracle identities, none referencing the category count.
- Interacts with existing task #40 ("generator self-consistency test") — that test should be written against the declared space.

**Verification tests:** New `tests/test_datasets/test_simulate.py`. Tolerances are exact equalities or derived from the closed-form oracle, never from a recorded run.

1. **Label space is declared** (the fix landed): for `n_phenotypes` in {4,10} x `n_cells` in {60,100,250} x seeds 0-7, assert `list(obs['phenotype'].cat.categories) == [f"phen_{p}" for p in range(n_phenotypes)]`, same for `true_phenotype`, and `len(...) == uns['tcri_truth']['omega'].shape[1]`. Exact set equality. Pre-fix this fails at (P=10, N=100, seed=5) — a confirmed reproducer.
2. **The two label columns always share one category set** at `label_error_rate` in {0.0, 0.3, 0.9}: `phenotype.cat.categories.equals(true_phenotype.cat.categories)`. Exact.
3. **Oracle invariance** (the dependency guard): recompute the oracle inside the test from `(z, phi)` via `mi_from_joint_oracle` on an independently built count table and assert `assert_allclose(uns_value, recomputed, rtol=0, atol=0)` for `true_mi`, `empirical_mi`, both nmi variants. Tolerance is exact-zero because both sides are the same closed form; it proves the fix did not move the oracle.
4. **`realized_clones` / `realized_phenotypes` are correct and non-vacuous**: recompute from `obs` and compare (integer equality), and assert a configuration exists where `realized_clones < n_clones` (C=47, N=250, seed=0 — measured 33).
5. **The rejected variant stays rejected**: assert `len(obs['clone_id'].cat.categories) == obs['clone_id'].nunique()`, so a future edit that also pins the clone axis trips a test and has to argue for it rather than sliding in.
6. **Suite regression**: full run stays at 177 passed / 3 skipped plus the new file's tests; all three conformance tests green (the datasets module is outside all of them, so they must be untouched).

---

## DE-14 · S3 · FIXED (PR #61)

**Confirmed:** REAL, and one step worse than the register states.

`tcri/datasets/_simulate.py:245-247`: `P = np.clip(P, eps, None); Pp = P ** (1.0/T); return Pp / Pp.sum(axis=1, keepdims=True)`.

Measured (`/tmp/de-generator-docs/de14.py`, `de14b.py`) on a fit-like (47,10) omega:
- Dead rows vs T: 0 down to T=0.002, **21/47 NaN rows at T=1e-3**, **47/47 at T=1e-4**. The only signal is a numpy `RuntimeWarning: invalid value encountered in divide`; no exception. (The register's "2/47 at T=0.002" was on the real fit; the threshold is matrix-dependent, the mechanism identical.)
- **NEW — the oracle does not catch it either.** `mi_from_joint_oracle`'s guard is `total = P.sum(); if total <= 0: raise` (`_simulate.py:54-56`). `NaN <= 0` is `False`, so it proceeds; `nz = P > 0` is then all-False and it returns `{'mi': 0.0, 'h_clone': -0.0, 'h_phenotype': -0.0, 'nmi_min': 0.0, 'nmi_average': 0.0}`. Both functions are public in `__all__`, so a caller using them together gets a silent, structurally-plausible all-zero oracle.
- End-to-end `simulate_from_fit_params(..., temperature=1e-3)` DOES eventually raise `ValueError: Probabilities contain NaN` from `rng.choice` (line 310). So a full simulate call crashes rather than returning garbage; the silent path is the direct `temperature_scale` + `mi_from_joint_oracle` use.
- **T->inf limit confirmed.** On a joint with structural zeros: shipped MI = 0.9225 / 0.0102 / 9.67e-5 / 9.60e-11 at T = 10 / 100 / 1e3 / 1e6 (-> 0). A support-preserving scaler holds 1.7245 / 1.7119 / 1.7112 / 1.7111. The eps floor turns every structural zero into full support, so every clone row converges to the SAME uniform and the coupling is destroyed rather than flattened within support.
- A log-space support-preserving implementation is NaN-free at T = 2e-3 / 1e-4 / 1e-8 with stable MI 3.138860 / 3.138873 / 3.138873.

**Candidates considered:** A. **Log-space, support-preserving.** `z = log(p)/T` on the support only, subtract the row max, exponentiate, renormalize; structural zeros stay zero. Exact wherever the current code is correct (`exp(log p / T)` and `p**(1/T)` agree to float rounding on the support), and cannot underflow for any finite `T>0`. Changes the T->inf limit for matrices with genuine zeros.

B. **Keep the power form, add validation.** Raise `ValueError` on `T <= 0` / non-finite `T`, and on any non-finite output, naming the offending T and the dead row count. Smallest diff, every current number bit-identical — but it converts a silent wrong answer into a hard stop rather than a right answer, and the T->inf support bug survives.

C. **Raise the eps floor.** Rejected: 1e-300 still underflows at 1/T=500; 1e-6 distorts the sharpened distribution in the usable T range. It moves the failure threshold without removing it.

**Chosen fix:** **A, with B's guards, plus a finiteness guard on the oracle.** Safety and speed point the same way — ~15 lines in one file, no contract surface, and it removes both the silent NaN and the destroyed-support limit instead of only the first.

1. `tcri/datasets/_simulate.py::temperature_scale` — rewrite the body:
   - validate `T` is finite and `> 0`, else `ValueError`;
   - `A = np.asarray(P, float)`; reject negative entries; `sup = A > 0`; `ValueError` naming the row indices if any row has `sup.sum() == 0`;
   - `z = np.where(sup, np.log(np.where(sup, A, 1.0)) / T, -np.inf)`; subtract the per-row max over the support; `out = np.where(sup, np.exp(z), 0.0)`; `return out / out.sum(axis=1, keepdims=True)`.
   - Drop the `eps` parameter (no in-repo caller passes it; the function is not contract-frozen).
2. Docstring: replace "Verbatim behaviour of `sc_simulator.temperature_scale_conditional`" with what it now guarantees — exact on the support, zeros preserved, no underflow for any `T>0` — and name the one deliberate difference from the reference sampler: the T->inf limit is uniform-on-support, not uniform-on-all-columns.
3. `tcri/datasets/_simulate.py::mi_from_joint_oracle` line 55: `if not np.isfinite(total) or total <= 0: raise ValueError(...)`; also reject negative entries. This is the guard that let the all-zero result through.

**Contract change:** NONE — `temperature_scale` and `mi_from_joint_oracle` live in `tcri/datasets`, which is not in `tcri/_contract.pyi`, not in `tcri/tools/_metrics_contract.py`, and not in `tcri/model/_model_contract.py`.

**Dependencies affected:** **No published number changes.**

- `simulate_from_fit_params` (line 297) is the only in-repo caller; `benchmarks/run_grid.py:99-101` reaches it via `--fit-params`. `temperature_scale` is skipped entirely at `T == 1.0` (line 296), so the T=1.0 anchor is untouched by construction.
- At the published anchors T = 0.1 / 0.5 the fitted omega is strictly positive with no entry within ~30 orders of magnitude of underflow, so the log-space path returns the same matrix to float rounding. `true_mi` / `true_nmi_*` / `empirical_*` do not move. Direction of any residual change: removing the eps clip lets sub-1e-12 entries stay small instead of being lifted to 1e-12, which can only RAISE MI — bounded at ~1e-11 bits at these magnitudes.
- The T->inf behaviour change is outside the published sweep entirely (anchors are T = 0.1 / 0.5 / 1.0), so nothing published moves; it only fixes what a user sweeping T>1 would get.
- `temperature_scale` currently has **zero test coverage** (grep finds it only inside `tcri/datasets/`), so no existing test has an expected value to change.
- The oracle guard converts a silent `{mi: 0.0, ...}` into a `ValueError`. Nothing in `tcri/`, `tests/`, or `benchmarks/` relies on the zero return; anything that did was relying on a bug.
- Removing the `eps` kwarg is a signature change on a public-but-uncontracted function. If any external notebook passes `eps=`, it raises `TypeError` loudly rather than silently — acceptable, and preferable to a silently-ignored parameter.

**Verification tests:** New `tests/test_datasets/test_temperature_scale.py`. Every tolerance derives from the closed form or from float rounding, none from current behaviour.

1. **No underflow**: for T in {1e-2, 1e-3, 1e-4, 1e-8} on a (47,10) Dirichlet omega, `np.isfinite(out).all()` and every row sums to 1 within `1e-12`. Pre-fix this fails at T=1e-3 (21 NaN rows) and T=1e-4 (47/47) — confirmed reproducers.
2. **Exact where the old code was right**: write the old power form INLINE in the test (not imported), and for T in {0.1, 0.5, 2.0} on a strictly-positive omega with no entry below 1e-6, `assert_allclose(new, old, rtol=1e-12)`. Tolerance = float rounding, derived from the algebraic identity `exp(log p / T) == p**(1/T)`.
3. **Sharpening limit**: `assert_allclose(temperature_scale(omega, 1e-8), one_hot(omega.argmax(1)), atol=1e-9)`. Derived from the definition of the T->0 limit.
4. **Support preservation**: on a matrix with structural zeros, `(out == 0) == (omega == 0)` elementwise for T in {0.1, 1.0, 10, 1e6}; and `mi_from_joint_oracle(pi[:,None]*out)['mi']` stays above `0.5 * MI_uniform_on_support`, where `MI_uniform_on_support` is computed IN-TEST from the support pattern (row-uniform on each clone's support), not from a run.
5. **Guards raise**: `pytest.raises(ValueError)` for `T=0`, `T=-1`, `T=inf`, `T=nan`, an all-zero row, a negative entry; and for `mi_from_joint_oracle` on a joint containing NaN, inf, or a negative entry.
6. **Dependency guard — the anchors do not move**: with a stored (pi, omega) fixture, assert `mi_from_joint_oracle(pi[:,None]*temperature_scale(omega, T))['nmi_min']` at T in {0.1, 0.5} matches the value from the inline old power form to `rtol=1e-9`. This is the test that proves the published anchors are untouched.
7. **Suite regression**: full run green, all three conformance tests green (datasets is outside all of them).

---

## DE-15 · S3 · FIXED (PR #63)

**Confirmed:** REAL, with three further specifics — and the decisive one is that the function CANNOT be strengthened in place.

- Model-free and MI-only: `tcri/diagnostics/_ppc.py:132-135` imports `_mi_from_joint` and raises for any `metric != "mutual_information"`; the null at line 164 permutes `pc` and recounts the empirical crosstab (lines 145-152). No `p_ct`, no draws.
- **`groupby` is dead.** `inspect.getsource` shows the token appears exactly once — in the signature at `_ppc.py:127`. Measured: `permutation_null(a, n_perm=200, random_state=0)` vs the same call with `groupby="grp"` -> `DataFrame.equals(...) is True`. It is nonetheless in the frozen `.pyi`.
- **The reported quantity is NMI(min), not MI.** Line 152 hard-codes `_mi_from_joint(J, normalized=True, mode="min")`. Measured on a 30-clone x 4-phenotype fixture: raw MI = 1.0413 bits, NMI-min = 0.5234, NMI-average = 0.3046; `permutation_null` reports `observed = 0.523414`. The column is `observed` under `metric="mutual_information"`, and neither `normalized` nor `normalize_mode` is exposed — so the denominator choice the metrics contract makes explicit and arguable for `tl.mutual_information` is hard-wired and undocumented here.
- **The doc claims a draw stack that does not exist.** `docs/contract/API_CONTRACT.md:613` (§7.8) lists `diag.permutation_null` among consumers of "one shared draw stack"; `:721` repeats it. The function makes no Dirichlet draw at all.
- **`p` has no floor.** Line 167 is `p = float(np.mean(null >= obs_mi))`, so `p == 0.0` is attainable and was measured (both covariates at `n_perm=200`). `docs/contract/REFACTOR_NOTES.md:128` records `p=0` from the live run. The standard estimator is `(1 + #{null >= obs}) / (R + 1)`.
- **DECISIVE: label permutation is inert for the reported metric.** `_compute/_joint.py::_joint_draws` consumes `uns[P_CT]`, `uns[CT_ARRAY]`, `uns[COV_ARRAY]` and `obsm[X_LOGITS]`; it never reads `obs[phenotype_col]`. Measured (`/tmp/de-generator-docs/de15b.py`): `tl.mutual_information(adata, covariate="cov_0", n_samples=0)` = **0.24184133781470923 both before and after permuting `obs['phenotype']` — bit-identical**, while `permutation_null`'s `observed` on the same object is **0.0400**. The diagnostic's observed statistic is a different quantity from the number the package reports, and no permutation of a column the estimator does not read can null it.
- The refit-based figure null is **not in the repo**: `grep -rn permut` over `benchmarks/`, `tcri/`, `dev/` finds only this function and its two callers (`dev/live_test_rnr.py:144`, `tests/test_diag/test_diag.py:35`).

**Candidates considered:** A. **Retire.** Delete the function, its two `__all__` entries, the contract stanza and the smoke test. Fits the repo's Removal-is-a-hard-bar rule. Cost: removes the only check that answers "is there clone<->phenotype structure in these labels at all" — the question that catches a mis-specified `phenotype_key` or a scrambled join BEFORE any model is fit, and the one it answered usefully on real data (`REFACTOR_NOTES.md:128`, z ~ 90-105).

B. **Rename + re-document + fix the three wrong things** (dead `groupby`, MI/NMI mislabel, zero p-value). Makes it an honest data-level test that says what it is.

C. **Strengthen in place** to null the reported number. **Ruled out by measurement**: the estimator is bit-identical under the permutation, so the only real strengthening is a model refit — which needs the training loop, a seed policy and per-cell cost, and belongs in the benchmark harness, not in a `diag` function the contract declares "adata only".

D. **B now, plus a separate refit null in `benchmarks/`** tracked under DE-12 (the noise floor already needs exactly that machinery).

The register frames this as "two things with similar names and very different strength". The measurement reframes it: they answer DIFFERENT QUESTIONS, and the weaker one is still worth answering. That makes it a naming and documentation problem, not a strength problem.

**Chosen fix:** **D** — B in this PR, refit null tracked under DE-12.

1. `tcri/diagnostics/_ppc.py` — rename `permutation_null` -> `label_permutation_test`. New signature: `(adata, *, covariate=None, normalized=True, normalize_mode="min", n_perm=1000, random_state=None)`. **Drop `metric`** (it admits exactly one value; the name now carries it) and **drop `groupby`** (dead). Thread `normalized` / `normalize_mode` into the `_mi_from_joint` call at line 152.
2. Rename the output column `observed` -> `observed_nmi` (`observed_mi` when `normalized=False`) and add a `statistic` column naming the quantity, so a frame read out of context cannot be mistaken for `tl.mutual_information`.
3. Line 167 -> `p = (1.0 + float(np.sum(null >= obs_mi))) / (n_perm + 1.0)`.
4. Docstring first line: this tests the **empirical label table**, is independent of the fitted model, and is **not** a null for `tl.mutual_information` — with the reason (the estimator reads `p_ct` and `obsm` logits, not the label column). Point at the benchmark refit null for that.
5. **Contract edits (part of this plan):** `tcri/_contract.pyi:142-145` (replace the stanza); `tests/test_contract_conformance.py:35` (update the key); `docs/contract/API_CONTRACT.md:721` (rewrite the row — new signature, "empirical label table", the NMI-not-MI statement) and **`:613` — delete `diag.permutation_null` from the §7.8 shared-draw-stack sentence, which is simply false**; plus the name occurrences at `:168`, `:211`, `:723`, `:732`, `docs/contract/REDO_LIST.md:41`, `docs/contract/REFACTOR_AGENDA.md:197`, `docs/contract/tcri_function_inventory.md:361`.
6. `dev/live_test_rnr.py:142-144` — update the call and the step label.

If a deprecation window is wanted, keep `permutation_null` as a three-line shim emitting `DeprecationWarning` and forwarding — but recommend the clean rename: two in-repo callers, no external release depends on the name, and the Removal Ledger is the house rule.

**Contract change:** REQUIRED. `diag.permutation_null` is frozen at `tcri/_contract.pyi:142-145` and keyed in `tests/test_contract_conformance.py:35`. A rename or signature change edits the API contract and its conformance test — both CODEOWNER-restricted per `.github/CODEOWNERS`, so @nceglia / @salehis review is part of the plan, not a follow-up. The prose row at `docs/contract/API_CONTRACT.md:721` and the false §7.8 claim at `:613` must change in the same commit. No model or metrics contract is touched.

**Dependencies affected:** - **No metric output moves.** Nothing in `tcri/tools/` or `tcri/plotting/` calls this function, and the estimator path is provably independent of the column it permutes.
- **`p` changes value for every caller**: any currently-`0.0` p becomes `1/(n_perm+1)` — at the shipped `n_perm=1000`, `0.0 -> 0.000999`. Direction is strictly upward, bounded by `1/(R+1)`. No published figure carries this number (the figures use the refit null); `docs/contract/REFACTOR_NOTES.md:128` records `p=0` from a dev run and should be annotated in the same commit.
- **`observed` -> `observed_nmi`** breaks `tests/test_diag/test_diag.py:36`, which asserts the column set. Its `p in [0,1]` assertion still holds. This is the one existing test whose expectation changes.
- **API contract is edited** — breaking rename, CODEOWNER review required. `tests/test_contract_conformance.py` fails until the key is updated; that failure is the intended forcing function, not something to route around.
- `dev/live_test_rnr.py` is a manual real-data harness, not CI; it must be updated or its step will `AttributeError`.
- The refit null itself is NOT delivered here. It is benchmark work (train under permuted clonotype labels, per grid cell) and belongs with DE-12's per-cell noise floor; delivering it inside `diag` would contradict the contract's "adata only" declaration for this namespace.

**Verification tests:** 1. **`groupby` cannot silently recur**: the parameter is gone, so `pytest.raises(TypeError)` on `label_permutation_test(adata, groupby="x")`. Exact.
2. **The quantity is what the column says**: on a fixed crosstab built in-test with `pd.crosstab`, assert `observed_nmi == _mi_from_joint(tab, normalized=True, mode="min")` exactly (`rtol=0`), and with `normalized=False` that it equals the raw-MI value. Same code path, same input — exact equality is the right tolerance.
3. **The independence fact, pinned so it cannot silently stop being true** (this is what the whole decision rests on): permute `obs[phenotype_col]`, assert `tl.mutual_information(adata, covariate=..., n_samples=0)` is **bit-identical** (`==`, not approx) and that `label_permutation_test(...)['observed_nmi']` **changes** (`!=`). Measured pre-emptively: 0.24184133781470923 both sides.
4. **p-value floor**: with `n_perm=R` on a perfectly-coupled fixture, `p == pytest.approx(1/(R+1))` and `p > 0` on every row. Tolerance from the estimator's definition, not a run.
5. **Null calibration — the thing that makes it a valid test at all**: on an INDEPENDENTLY generated fixture (`simulate_tcri(omega_concentration=1e6)` -> near-uniform rows -> true MI ~ 0), the observed statistic falls in the null's central mass: `abs(z) < 3` across 5 seeds. Threshold is the 3-sigma convention, not a measured value.
6. **Removal actually happened**: `tcri.diag.permutation_null` raises `AttributeError`; `tests/test_contract_conformance.py` green with the new key; `set(tcri.diagnostics.__all__)` matches the `.pyi`.
7. **Doc claim retracted**: a grep assertion that `permutation_null` / `label_permutation_test` no longer appears in the §7.8 shared-draw-stack sentence of `API_CONTRACT.md`.

---

## DE-16 · S3 · effort XS · FIXED

**Confirmed:** REAL, and the file also contradicts itself.

- `docs/contract/REFACTOR_AGENDA.md:207` reads: `- **Deferred:** **[E]** \`reconstruction_loss_scale=1e-3\` vs eq-7 full weight (over-generation symptom) — author deferred; may be an intentional beta-VAE reweighting, and raising it needs a retrain + R/NR revalidation. Tracked as a follow-up investigation.`
- Contradicted by `docs/contract/METHODS_CONFORMANCE.md:108`: `| E | ... | MED | **resolved** — default raised 1e-3 -> 1e-2 ... |`, and `:153` "**G and E are now resolved too**".
- Contradicted by `docs/contract/MODEL_CONTRACT.md:113` (default **`1e-2`**) and its "### On `reconstruction_loss_scale` (deviation [E], resolved)" section at `:115`.
- Contradicted by the manifest, `tcri/model/_model_contract.py:171-176`: "Default RAISED 1e-3 -> 1e-2 after re-measurement".
- Contradicted by the code: `tcri/model/_module.py:86`, `tcri/model/_model.py:249`, `tcri/model/_training.py:41` are all `1e-2`, pinned by `tests/test_shared_defaults.py:81` (`assert train["reconstruction_loss_scale"] == 1e-2`).
- The commit the register cites is real: `git log --oneline -1 19db68e` -> `fix(model): recalibrate reconstruction_loss_scale 1e-3 -> 1e-2 (deviation [E])`.

**Second stale spot in the same file, not in the register**: `REFACTOR_AGENDA.md:218` also lists "Deferred (author sign-off, change fitted results): [E] ...; [G] alpha not applied to the eq-1 clonotype prior" — [G] is likewise recorded fixed at `METHODS_CONFORMANCE.md:106` and at `REFACTOR_AGENDA.md:206` in the same diary section.

**The correction is already in the file**: `REFACTOR_AGENDA.md:216` (AUDIT LOG, ordered newest-first — PR0 sits at the bottom) records "**4) [E] re-measured** ... so the default was raised". So a reader scrolling the DIARY hits the wrong statement first and nothing on that line points forward.

**Candidates considered:** A. **Edit lines 207 and 218 to say "resolved".** Rewrites dated history. The file's own header calls the DIARY and AUDIT LOG "dated entries", and the [E] history is exactly the interesting case — the first measurement was confounded by the phantom optimizer, and erasing that erases why the number moved twice.

B. **Annotate in place**: leave the historical claim, append a superseded pointer. History intact, no reader misled. Leaves the duplication, so it can go stale again on the next deviation.

C. **Delete deviation-status bullets from the diary entirely** and have the agenda link to `METHODS_CONFORMANCE.md`. Kills the second source of truth for good, but also kills the record of what was believed when.

**Chosen fix:** **B for the two existing lines, plus the policy half of C** so it stops recurring without deleting anything.

1. `docs/contract/REFACTOR_AGENDA.md:207` — append to the bullet: `**[SUPERSEDED 2026-08]** [E] was re-measured and resolved (default 1e-2, commit 19db68e); see the AUDIT LOG "GOAL RUN" entry and METHODS_CONFORMANCE.md deviation [E]. [F] remains deferred.`
2. `docs/contract/REFACTOR_AGENDA.md:218` — same annotation covering **both** [E] and [G]: `**[SUPERSEDED 2026-08]** [E] resolved (default 1e-2); [G] fixed in the same PR (see line 206) — METHODS_CONFORMANCE.md deviations [E] and [G].`
3. `docs/contract/REFACTOR_AGENDA.md`, "How to use this doc" (after line 17) — add rule 6: *"The DIARY and AUDIT LOG are append-only history; they record what was believed at the time. **They are not the current state of any deviation.** Live deviation status lives in `docs/contract/METHODS_CONFORMANCE.md` (the table) and `tcri/model/_model_contract.py::SANCTIONED_DEVIATIONS` (the manifest). When a diary claim is overtaken, annotate it `[SUPERSEDED <date>]` with a pointer — never edit it to agree."*
4. `docs/contract/DEFECTS.md` — close DE-16, noting that line 218 carried the same staleness and is fixed with it.

**Contract change:** NONE. `docs/contract/REFACTOR_AGENDA.md` is not one of the three contracts (CLAUDE.md's table lists `_contract.pyi`, `_model_contract.py`, `_metrics_contract.py` and their prose) and is not CODEOWNER-restricted. The contract side of [E] — `MODEL_CONTRACT.md:113-115` and `_model_contract.py:171-176` — is already correct and is NOT edited.

**Dependencies affected:** Documentation only. No code path, no metric output, no published number, no fitted result, no test expectation changes. `tests/test_shared_defaults.py:81` and `tests/test_model_contract_conformance.py` stay green untouched. The only downstream effect is on readers and on any future agent that reads `REFACTOR_AGENDA.md` first — which CLAUDE.md instructs them to do, which is exactly why the stale line matters.

**Verification tests:** A doc-consistency check is the only thing that stops this recurring, and it is cheap.

1. **New `tests/test_docs_deviation_status.py`**: parse the deviation tables in `docs/contract/METHODS_CONFORMANCE.md` for `| <id> | ... | <status> |`; collect every id whose status contains "fixed" or "resolved"; assert no line of `docs/contract/REFACTOR_AGENDA.md` mentions that id inside a bullet beginning `**Deferred` UNLESS the same line contains `[SUPERSEDED`. Pure string containment — no numeric tolerance. **Pre-fix this fails on lines 207 and 218**, which is the confirmation that it tests the right thing.
2. **Extend the existing sync check**: `tests/test_model_contract_conformance.py` already asserts `SANCTIONED_DEVIATIONS` keys match the `MODEL_CONTRACT.md` table. Add the status-word direction so "resolved" in the manifest and "deferred" in the prose cannot coexist. Exact string comparison.
3. **Value guard stays green** (the dependency proof): `tests/test_shared_defaults.py:81` still asserts `1e-2` across all three declaration sites, and `tcri/model/_model_contract.py` still records the recalibration rationale — so the doc fix demonstrably did not touch the number.
4. **Suite regression**: 177 passed / 3 skipped plus the new file; all three conformance tests green.

---

## DE-17 · S3 · FIXED

**Confirmed:** **THE ANSWER: `n_steps_kl_warmup` counts optimizer STEPS (minibatches), not epochs. Unambiguously.**

- `tcri/model/_training.py:97-101`: `kl_weight = max(1e-6, self.module.kl_weight_max * (self._my_global_step / self.n_steps_kl_warmup))`; `:135` is `self._my_global_step += 1` at the end of `training_step` — one increment per minibatch. `validation_step` (`:138-169`) never touches it.
- **The parallel scvi schedule is inert.** `UnifiedTrainingPlan.__init__` forwards `n_steps_kl_warmup` to `PyroTrainingPlan` (`_training.py:71-81`) but never passes `n_epochs_kl_warmup`, so scvi's default `400` stands and WOULD override `n_steps` inside scvi's own `kl_weight`. It never fires: `LowLevelPyroTrainingPlan.__init__` sets `use_kl_weight = "kl_weight" in signature(module.model).parameters`, and `TCRIModule.model` (`_module.py:181-187`) takes `(x, batch_idx, log_library, indices)` — no `kl_weight`. Measured: `plan.use_kl_weight == False`, `plan.n_epochs_kl_warmup == 400`. Exactly one ramp is live and it is tcri's step counter.
- Measured `training_step` calls vs epochs (`/tmp/de-generator-docs/de17.py`): 500 cells / bs 1000 -> 4 calls in 4 epochs (1 step/epoch); 500 / bs 128 -> 16 in 4 (4); 5000 / bs 1024 -> 15 in 3 (5); 5000 / bs 1000 -> 15 in 3 (5). `_my_global_step` equals the call count in every case. Steps per epoch = `ceil(0.9 * n_cells / batch_size)` (`DataSplitter(train_size=0.9)`, `_model.py:275-280`).
- **The register is wrong that DUX-2 appears once.** It appears twice: `docs/contract/REFACTOR_AGENDA.md:182` AND `tests/test_model_knobs.py:183-186`, whose docstring already states the answer — "the warmup is counted in optimizer STEPS while max_epochs is in epochs; with batch_size >= n_obs that is one step per epoch (tracked as deviation DUX-2)". So the semantics were never open; the tracker item is stale, not unanswered.

**What it means at the shipped defaults** (`train(batch_size=1000, max_epochs=1000, n_steps_kl_warmup=2000)`):

| n_cells | steps/epoch | 2000 steps = | at max_epochs=1000 |
|---|---|---|---|
| 1000 | 1 | 2000 epochs | **never completes**; kl_weight peaks at 0.50 * kl_weight_max |
| 2000 | 2 | 1000 epochs | completes on the last epoch |
| 5000 | 5 | 400 epochs | completes |
| 20000 | 18 | 111 epochs | completes |

**At the benchmark's own settings** (`run_grid.py:129` `batch_size=1024`, default `--epochs 60`, `n_cells in {250,500,1000,2000,5000}`): steps/epoch = 1/1/1/2/5, so a 60-epoch run reaches only **3% / 3% / 3% / 6% / 15%** of `kl_weight_max`. The latent KL is effectively switched off for the whole of the default benchmark run. At 4000 epochs the warmup finishes at epoch 2000 / 2000 / 2000 / 1000 / 400.

**This settles DE-17's stated worry.** At 5000 cells — the configuration DE-2 measured — warmup ends at epoch 400, so the early-stopping best at epoch 1464 WAS found on a stationary objective. At n_cells <= 1000 it would not have been: epoch 1464 would still sit inside the ramp. The answer is size-dependent, and for the benchmark point that was measured it is yes.

**Candidates considered:** A. **Document only** — close DUX-2 by recording the answer in the contract and the docstrings; leave behaviour alone. Steps-based warmup is the scvi-native meaning of the parameter name. Cost: the 3%-of-warmup benchmark regime stays undetectable.

B. **Switch to epochs** — add `n_epochs_kl_warmup` and ramp on `trainer.current_epoch`. Makes the knob size-independent, but changes `kl_weight(t)` for every existing run and therefore every fitted result, and diverges from the convention the parameter name inherits from `PyroTrainingPlan`.

C. **Keep steps, make the impossible case loud** — warn when `n_steps_kl_warmup > max_epochs * steps_per_epoch`, naming the fraction of `kl_weight_max` the run will actually reach. Behaviour-preserving; removes the silent half-warmed regime. Delivers B's real benefit (size-awareness) at zero numeric cost.

D. **Remove the parallel dead schedule** — pass `n_epochs_kl_warmup=None` to `super().__init__`. Free, and it deletes a genuine trap: a reader who passes `n_epochs_kl_warmup` through `train(**kwargs)` today gets nothing and no error.

**Chosen fix:** **A + C + D.** No behaviour change to any existing configuration; the silent regime becomes loud; the dead parallel knob goes.

1. `tcri/model/_training.py::UnifiedTrainingPlan.__init__` (lines 71-81) — pass `n_epochs_kl_warmup=None` to `super().__init__`, with a comment naming `use_kl_weight=False` as the reason scvi's schedule never fired. **(D)**
2. Add `on_train_start` to `UnifiedTrainingPlan`: read `self.trainer.max_epochs` and `self.trainer.num_training_batches` (both populated by then) and, when `n_steps_kl_warmup > max_epochs * num_training_batches`, emit a `UserWarning` naming both numbers and the fraction of `kl_weight_max` the run will reach. **(C)**
3. `tcri/model/_model.py::train` docstring — state the unit: *"`n_steps_kl_warmup` is counted in optimizer steps (minibatches), not epochs. Steps per epoch = ceil(0.9 * n_obs / batch_size), so the ramp's length in epochs depends on dataset size and batch size."* Same sentence on `UnifiedTrainingPlan`.
4. `docs/contract/MODEL_CONTRACT.md` — the "Training-only deviations from eq 7" block already says the ramp is over `n_steps_kl_warmup`; add the unit, the `ceil(0.9*n_obs/batch_size)` formula and the shipped-defaults table. Mirror one sentence into `SANCTIONED_DEVIATIONS['kl_warmup_z_only']`'s rationale in `tcri/model/_model_contract.py`.
5. `docs/contract/REFACTOR_AGENDA.md:182` — replace "remain open as **DUX-2**" with the answer plus a pointer to the contract; `tests/test_model_knobs.py:185` — drop "(tracked as deviation DUX-2)".
6. `docs/contract/DEFECTS.md` — close DE-17 with the answer, and **cross-reference DE-10**: the 60-epoch default and the step-counted warmup interact, so fixing one without the other leaves the benchmark training with the latent KL near zero.

**Contract change:** YES, minor and tightening-only. `docs/contract/MODEL_CONTRACT.md` ("Training-only deviations from eq 7") and the rationale string of `SANCTIONED_DEVIATIONS['kl_warmup_z_only']` in `tcri/model/_model_contract.py` gain the unit and the size-dependence. This makes an already-sanctioned deviation MORE specific; it adds no deviation and loosens no constraint. Both are CODEOWNER-gated and `tests/test_model_contract_conformance.py` asserts manifest<->doc sync, so they must move in the same commit.

**Dependencies affected:** - **No metric output moves and no published number changes.** Items 1-6 are documentation, one warning, and the removal of a schedule that was never consulted. Verified inert: `plan.use_kl_weight is False`, so `n_epochs_kl_warmup` has no reader; setting it to `None` cannot change any trace.
- **The warning surfaces on short runs** — including every `run_grid` default-60-epoch cell at every `n_cells`, and the short fits in `tests/test_model_smoke.py` / `test_model_knobs.py`. Checked: `pyproject.toml:69-71` sets `filterwarnings = ["ignore::DeprecationWarning"]` only — there is no `error` filter, so a new `UserWarning` will not fail the suite. It will add output noise.
- **Anyone acting on the warning** by raising `max_epochs` or lowering `n_steps_kl_warmup` DOES change fitted results — a deliberate configuration change, not a side effect of this fix.
- **Contract**: `MODEL_CONTRACT.md` + `_model_contract.py` are edited; `tests/test_model_contract_conformance.py` asserts sync, so they move together or it goes red.
- **Re-scopes DE-10.** DE-10 currently reads "60 epochs sits in the steep part of the training curve". The measurement adds a second, independent reason: at 60 epochs the latent KL is at 3-15% of full weight for the entire run. A DE-10 fix that raises epochs also completes the warmup — so the two must be evaluated together, and any before/after comparison across an epoch change is confounded by the KL weight moving too.
- **Interacts with DE-4** (`_my_global_step` resets per `train()` call): the step semantics established here are what DE-4 breaks under staged training. DE-4's fix should preserve the step unit, not switch to epochs.

**Verification tests:** All tolerances derive from the dataloader arithmetic or the ramp formula, never from a recorded run.

1. **Unit semantics, exact**: reuse the spy pattern at `tests/test_model_knobs.py:186-198`; train `max_epochs=E`, `batch_size=B` on `n_obs=N` and assert `plan._my_global_step == E * ceil(floor(0.9*N)/B)` for at least (500,1000,4) -> 4, (500,128,4) -> 16, (5000,1024,3) -> 15. Integer equality; all three measured on main.
2. **The ramp is a function of steps, not epochs**: run the SAME `max_epochs` at two batch sizes and assert the final `module.kl_weight` differs by the step ratio — `kl_weight(B=128) / kl_weight(B=1000) == pytest.approx(4.0, rel=1e-6)` on the (N=500, E=4) configuration. Derived from the formula. This is the test that would fail if anyone ever switched to epochs, which is the point.
3. **scvi's schedule is dead and now explicitly disabled**: `assert plan.use_kl_weight is False` and `assert plan.n_epochs_kl_warmup is None`. Exact.
4. **The warning fires exactly when the ramp cannot finish**: `pytest.warns(UserWarning, match="kl.*warmup")` for `max_epochs=10, batch_size=1000, n_steps_kl_warmup=2000` on 500 cells (10 steps of 2000); and `warnings.catch_warnings` asserting NO such warning for `max_epochs=10, n_steps_kl_warmup=5`. The boundary is `total_steps >= n_steps_kl_warmup`, derived from the condition, not from behaviour.
5. **Dependency guard — nothing else moved**: `tests/test_model_knobs.py::test_n_steps_kl_warmup_ramps_the_kl_weight` stays green unchanged (its monotonicity / reaches-ceiling / starts-near-zero assertions are all still true); `tests/test_shared_defaults.py` still pins `n_steps_kl_warmup == 2000` in all three declaration sites; `tests/test_model_contract_conformance.py` green after the paired manifest+doc edit.
6. **Suite regression**: 177 passed / 3 skipped baseline preserved, plus the new tests; all three conformance tests green.

---
## DE-18 — `p_ct` has no data term in the ELBO · WITHDRAWN — NOT A DEFECT

> **WITHDRAWN 2026-08-07 by @nceglia. The premise was wrong, and the fix has been reverted.**
>
> The hierarchical branch (ω_c → φ_m → z^φ) is a **prior** over phenotype composition, and it
> is *supposed* to never see `x` directly — that separation is the reason the model is a VAE at
> all. Data reaches the hierarchy only through `z`. So "`p_ct` has no data term" describes the
> architecture; it is not a missing piece to be supplied.
>
> `z^φ` is **latent**. The step below from "x ⊥ z^φ | z, so φ_m is unidentifiable" to
> "therefore z^φ must be observed" does not follow, and conditioning the site on the input
> phenotype labels turns an unsupervised phenotype model into a supervised one — every metric
> then becomes partly a readout of the labels it was handed.
>
> Reverted in `_module.py` (back to `phi = p_ct[ct_idx].detach()`), the `phenotype` SiteSpec
> removed from `_model_contract.py`, and `FORBIDDEN_MODEL_SITES` added so the site cannot be
> re-introduced without deliberately editing the contract.
>
> **DE-5 is unaffected and stays.** It was landed in the same commit but is an independent fix
> to the guide's concentration (eq 6, λ'_m free), and its justification never depended on DE-18.
>
> Everything below is the original, superseded analysis, kept because five other defects were
> argued from it and those arguments have to be re-derived rather than silently inherited.

**Confirmed on `main` at `46490e6`, statically.** Inside `TCRIModule.model`, `p_c` is used at
exactly one place — `base_p = p_c[self.ct_to_c] + self.eps`, forming `p_ct`'s prior
concentration — and `p_ct` at exactly one place:

```python
phi = p_ct[ct_idx].detach()          # tcri/model/_module.py:241
```

Nothing else in `model()` mentions either. The `.detach()` severs every gradient path from the
observed phenotypes back to `p_ct`, and eq 7's discrete-phenotype term is replaced by the
surrogate (`L#` is "eq 7 with terms that involve z^φ removed"). So the clone×phenotype
distribution that **every metric reads** has no likelihood term at all. Its only forces are
`−KL(q(p_ct) ‖ Dir(β·p_c))` and `−KL(q(p_c) ‖ MixtureDir(α·ψ))`.

Training therefore does not *fit* `p_ct`. It relaxes `q_p_ct_raw` away from the crosstab it was
initialised at (`_module.py:312-320`) and toward the archetype prior.

Separately: `_target_phenotypes` is registered as a buffer at `_module.py:132` and `:160` and is
**never read anywhere in the package**. The name says what it was for.

**This one fact explains every measurement from the investigation**, with no further hypotheses:

| observation | explanation |
|---|---|
| gate-0 read equals the label oracle at 30–120 epochs | that is the initialisation, before it decays |
| decays to 0.1353 by 4000 epochs | relaxation toward the prior |
| `p_ct` L1 to the observed crosstab grows 0.284 → 0.515 | the same relaxation, measured directly |
| validation ELBO improves throughout | the ELBO has no term that wants `p_ct` near the data |
| gate=1 reads 0.0000 | the classifier's only training term targets a *detached constant*, minimised by any phenotype-constant logit vector |

**Why it blocks five other defects.** With the detach in place, the only φ-bearing ELBO term is
`−KL(q(φ_m) ‖ Dir(β·ω))`, whose argmax over a *free* `λ'_m` is `λ'_m = β·ω` — total exactly β,
every row. So DE-5's fix does not make the posterior data-informed; it makes it free to return
to the pin, and DE-5's proposed test would pass only by non-convergence. DE-3, DE-6, DE-10 and
DE-12 are each conditional in the same way.

**DECIDED 2026-08-07 — fix it (option C). @nceglia is taking the reading to @salehis; a
revised supplementary note is expected.**

### Why the implementation appears to work

`q_p_ct_raw` is initialised from `clone_phen_prior` — the observed crosstab — which is itself a
reasonable plug-in estimator of `P(φ|c)`. So at short training the numbers are approximately
right *for the wrong reason*: they are the empirical crosstab, lightly smoothed, not an
inference. They degrade toward the archetype prior as training continues. **The observed data
enters through initialisation instead of through the likelihood**, which is why the failure is
a slow drift rather than an obviously wrong answer, and why it stood for so long.

### Why removing the detach alone is not enough

The surrogate's target is `probs_i = softmax(π·cls_logits + (1−π)·log φ)` — a function of φ
itself. Un-detaching makes it self-referential: the optimum drives `cls_logits → log φ + const`
and admits a degenerate solution where `KL(probs ‖ φ) → 0` by mutual agreement, regardless of
data. No observed label enters that loop.

Structurally, under eqs 4–5, `x` depends on `z` only, so `x ⊥ z^φ | z`. A latent `z^φ`
marginalises out — the optimal `q(z^φ) = p(z^φ|z,φ)` makes the term exactly zero — so φ_m is
**unidentifiable** whether or not the surrogate is used. Either `z^φ` is observed, or φ carries
no information from the data.

### The fix

Inside the data plate, condition on the observed phenotype:

```python
pyro.sample("phenotype", dist.Categorical(logits=ell),
            obs=self._target_phenotypes[indices])
```

This is eq 4 with `z^φ` observed. Since `ℓ_i = π·cls_logits + (1−π)·log φ`, the label gives
gradient to both the classifier and φ. Enumeration is then unnecessary, so the surrogate's
original justification disappears — it is kept as a γ-weighted regulariser or dropped. If
partial labelling is wanted, the same site takes a mask.

Expected side effect: this plausibly resolves the classifier collapse, since `cls_logits` would
receive gradient from real labels for the first time.

### What the model is for, once φ is observed

Worth stating in the contract, because it will be asked: if φ is observed for every cell, the
crosstab is already a sufficient statistic, and what the model adds is shrinkage across clones
through the archetype → clone → covariate hierarchy, uncertainty quantification, and imputation
for unlabelled cells. The architecture is already built for exactly that. The missing likelihood
is what lets the shrinkage run to completion instead of balancing against data.

### On the note

The note is not ambiguous — the notation table declares `z^φ` a latent and the introduction
lists the inputs without phenotype labels, consistently. The issue is that what it specifies
leaves φ_m unidentified. The one constructive suggestion for the revision: declare which
quantities are **observed** and which are **inferred**; a single line would have made this
visible at writing time.

### Original framing (kept)

**This is a model-mathematics question, not a coding one.** Note 1 defines
`L_new = L# + γ Σ_i KL(probs_i ‖ φ_g(i))` and optimises Λ, which contains `λ'_m`. Read
literally, φ should receive gradient from the surrogate and the detach is a deviation. Whether
that is intended is for @nceglia / @salehis, and per `CLAUDE.md` the model contract is updated
first, citing the equation.

---

## DE-19 — the fit is unseeded · S2 · FIXED (PR 1 `seed-and-record`)

Network init and minibatch order are not seeded; `seed` reaches the simulator and the metric
draw only. Measured ~1.8e-3 fit-to-fit spread on the same nominal seed — larger than DE-1's
effect and comparable to DE-3's, so neither is measurable from a single paired fit without this.

**Fixed.** `seed` added to `TCRIModel.__init__` (not `train()` — the networks are built in
`__init__`, so seeding in `train()` would be too late, and it would be an API-contract change).
`_apply_seed` calls `lightning.seed_everything(workers=True)` — which is what makes minibatch
order reproducible — and `pyro.set_rng_seed`, which covers the param-store initialisers and the
Dirichlet draws. Neither alone is sufficient. Re-seeded per `train()` call offset by the call
index, so a second fit is reproducible without replaying the first. `run_grid` now passes the
cell's seed through.

`tests/test_model_determinism.py` asserts bit-identical `p_ct` across seeded fits, **and** that
unseeded fits differ — without that negative control the positive test would keep passing on a
model that had become deterministic for an unrelated reason.

---

## DE-20 — fuzziness mapping is `g(f) = f`, the note specifies `g(f) = √f` · FIXED (PR #61)

The note: *"We also apply a concave mapping g(f) … in the reported experiments, we use
g(f) = √f."* The generator applies the identity, so the benchmark's difficulty axis is not the
published one.

---

## Open questions

- **Q-A — is the `p_ct` detach intended?** DE-18. Blocks DE-3, DE-5, DE-6, DE-10, DE-12.
- **Q-B — minibatch weighting.** Does a minibatch estimate weight the N cell terms against the
  C+M global terms in eq 7's ratio? Ships as `xfail(strict=True)` until answered.
- **Q-C — which estimand does a posterior summary report?** `E[NMI(J)]` vs `NMI(E[J])`.
- **Q-D — weight decay as a prior.** It reaches `q_p_c_raw`/`q_p_ct_raw`, whose param-store
  leaves are `log θ`, so it pulls every clone row toward the uniform simplex point. That is a
  prior acting through an optimizer setting; declare it or remove it.

---

# PR plan — the stack

Seven PRs, submitted as one `gh stack` chain. Each is independently reviewable (one concern, one review audience) and independently revertible (contract text and the code it governs travel together, so a revert never leaves a conformance test red). The chain is linear, so the order below is also the merge order.

```
gh stack init  seed-and-record
gh stack add   training-mechanics
gh stack add   stopping-policy
gh stack add   metric-joint
gh stack add   generator-fidelity
gh stack add   benchmark-protocol
gh stack add   guide-concentration      # conditional — see Q-A
gh stack submit
```

Ordering principle: the two PRs that move no number at all land first, because PR 1 is the measuring instrument for everything after it and PR 2's only numeric effect is the one defect it closes. Every subsequent PR moves published numbers, and each lands against a baseline that the PR below it made reproducible.

Three defects are opened by this plan and are not yet in the register above: **DE-18** (`p_ct` has no data term), **DE-19** (the fit is unseeded), **DE-20** (fuzziness mapping is `g(f)=f` where the note specifies `g(f)=√f`). They are written into the register by PR 1.

## Stack at a glance

| # | Branch | Closes | Contract first? | Numbers move |
|---|---|---|---|---|
| 1 | `seed-and-record` | DE-19, DE-16 | no | none |
| 2 | `training-mechanics` | DE-1, DE-4, DE-17 | **yes** — new training contract + 2 model-contract amendments | MI/NMI **+6e-4** |
| 3 | `stopping-policy` | DE-2, DE-3 | no (flips xfails from PR 2) | MI/NMI **0 to +4e-3** |
| 4 | `metric-joint` | DE-6, DE-7, DE-15 | **yes** — metrics manifest + API contract | MI/NMI **+0.05 to +0.21** |
| 5 | `generator-fidelity` | DE-13, DE-14, DE-20 | no | benchmark difficulty axis only |
| 6 | `benchmark-protocol` | DE-8, DE-9, DE-10, DE-11, DE-12 | no | every benchmark cell |
| 7 | `guide-concentration` | DE-5 | **yes** — model manifest | posterior read only; **0** if Q-A keeps the detach |

---

## PR 1 — `seed-and-record`

**Title:** Seed the fit, and correct the deviation record

**Closes:** DE-19 (new), DE-16. Opens DE-18 and DE-20 with their confirmations.

**Files**

- `tcri/model/_model.py` — `seed: int | None = None` on `TCRIModel.__init__`; store as `self._seed`; call `lightning.seed_everything(seed, workers=True)` and `pyro.set_rng_seed(seed)` immediately before `TCRIModule(...)` is constructed, and again at the top of `train()` using `self._seed + self._n_train_calls` so a second fit is reproducible without being a replay of the first. Fall back to `scvi.settings.seed` when `None`.
- `benchmarks/run_grid.py` — pass the cell's `seed` into `TCRIModel(...)`; it currently reaches the simulator and the metric draw only.
- `docs/contract/REFACTOR_AGENDA.md` — `[SUPERSEDED <date>]` annotations at `:207` and `:218` (the second carries the same staleness for `[E]` and `[G]`); new rule 6 under "How to use this doc" making the DIARY and AUDIT LOG append-only history and naming `METHODS_CONFORMANCE.md` + `SANCTIONED_DEVIATIONS` as live status.
- `docs/contract/DEFECTS.md` — close DE-16; open DE-18/19/20; record Q-A…Q-D.
- New `tests/test_model_determinism.py`, `tests/test_docs_deviation_status.py`.

**Contract change:** none. `seed` goes on `__init__`, which is not declared in `tcri/_contract.pyi`. Putting it on `train()` would be an API-contract change to `_contract.pyi:35-38`, gated by `tests/test_contract_conformance.py::test_signature_matches_contract` and by CODEOWNERS. Do not do that to save a line.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```
177 passed / 3 skipped, plus the new files. `tests/test_docs_deviation_status.py` must be shown failing at `REFACTOR_AGENDA.md:207` and `:218` on the parent commit, or it is not testing what it claims.

**Metric outputs that move:** none. No expected value in the suite changes. What it removes is the ~1.8e-3 unseeded fit-to-fit spread, which is larger than the effect of PR 2 and comparable to the effect of PR 3 — without it, neither of those PRs is measurable from a single paired fit.

---



---

## PR 2 measured effect (DE-1 + DE-4)

Paired seeded fits, 1200 cells / 20 clones / 5 phenotypes, 60 epochs, `normalize_mode="average"`,
`n_samples=0`. Same-seed fits are bit-identical after PR 1, so these deltas are the change itself
rather than fit-to-fit noise.

| seed | parent | with PR 2 | delta |
|---|---|---|---|
| 0 | 0.123834 | 0.124480 | +6.5e-4 |
| 1 | 0.149407 | 0.149736 | +3.3e-4 |
| 2 | 0.194495 | 0.193689 | −8.1e-4 |

Magnitude matches the plan's ~6e-4 estimate. **The sign does not** — the plan predicted a
consistent upward move on the reasoning that removing the validation-time flattening leaves
`p_ct` rows sharper. Two of three seeds move up and one moves down, so the effect is not
monotone in the way the mechanism suggests at this configuration. Recorded as measured; not
worth chasing, since it is an order of magnitude below anything the stack is trying to resolve.


---

## DE-18 + DE-5 — SUPERSEDED: DE-18 WITHDRAWN, DE-5 STANDS

> **2026-08-07.** DE-18 is withdrawn as not-a-defect (see its entry above) and the observed
> phenotype likelihood has been reverted. DE-5 — the free guide concentration, eq 6 — is an
> independent fix and remains landed.
>
> One measurement below now reads the other way round. The **pre-DE-18 configuration is the
> most accurate**: surrogate-only reads NMI 0.170 against a true 0.2145 at 600 epochs (error
> 0.045), while likelihood + surrogate reads 0.122 (error 0.093). This was written up at the
> time as "the data term works, but NMI accuracy gets worse," and the accuracy loss was set
> aside as possible synthetic circularity. It was the signal.
>
> Two behaviours of the detached hierarchy are expected, not defects, and are recorded here
> **once** so they are not re-litigated from new symptoms:
> 1. `q(ϕ_m)`'s concentration does not track clone size. This follows structurally: with no
>    direct data term the only ϕ-bearing ELBO term is −KL(q(ϕ_m) ‖ Dir(β·ω)), whose optimum over
>    a free λ'_m is β·ω for every row, independent of cell count. DE-5 frees the parameter, as
>    eq 6 requires; it does not make it data-informed.
>
>    A measured r = −0.038 on Yost top-50 is consistent with this but is **not** evidence for
>    it, and should not be cited as such: that subset is the 50 largest clones, spanning 42–901
>    cells — 1.33 decades, with no singletons — against the 3–4 decades of a full repertoire.
>    Correlating concentration against clone size over a narrow band at the top of a heavy tail
>    is range-restricted by construction and would read ≈0 whether or not the dependence exists.
>    The structural argument is the content; testing it needs the untruncated repertoire.
> 2. `p_ct`'s L1 to the observed crosstab grows 0.242 → 0.321 over 600 epochs as the guide
>    relaxes toward the archetype prior.
>
> Whether the hierarchy should concentrate with data, and through what coupling, is a model
> question for the forthcoming supplemental note.

**Superseded decision (kept for the record): land DE-18 and DE-5 together.** Either alone is
insufficient — DE-18 supplies the evidence, DE-5 lets the posterior respond to it — and the
combination reverses the degradation that motivated the whole investigation.

### Two implementation attempts; the first was a no-op

Adding `pyro.sample("phenotype", Categorical(logits=ell), obs=...)` alone did nothing, because
`ell` was built from `log_phi` derived from `phi = p_ct[ct_idx].detach()`. The detach still
severed the gradient, so the new likelihood reached the classifier only. Measured: `p_ct`'s L1
to the observed crosstab still grew 0.286 → 0.510 over 900 epochs, exactly as before.

The fix keeps two views. `phi_live` carries gradient and builds `ell` for the likelihood;
`phi_det` stays detached as the surrogate's alignment target, which it must be or
`−γ·KL(probs‖φ)` is self-referential.

This is only visible because the change was measured rather than assumed.

### With the gradient path correct

1200 cells, 20 clones, 5 phenotypes, seed 0, `normalize_mode="average"`, `n_samples=0`.
True NMI = 0.2145.

| configuration | ep60 L1 / NMI | ep600 L1 / NMI |
|---|---|---|
| likelihood + surrogate (as implemented) | 0.273 / 0.138 | 0.309 / 0.122 |
| likelihood only, γ=0 | 0.405 / 0.138 | 0.396 / 0.103 |
| surrogate only (pre-DE-18) | 0.242 / 0.136 | 0.321 / **0.170** |

**The drift is reduced** — likelihood + surrogate grows least (+0.036 over the run, versus
+0.079 for the pre-fix behaviour), and the earlier single-config run showed 0.510 → 0.325 at
900 epochs. The data term works.

**But NMI accuracy gets worse.** The pre-fix configuration is closest to truth at 600 epochs
(0.170, error 0.045); with the fix it reads 0.122 (error 0.093). Dropping the surrogate is worse
still on both counts.

### Why this is a decision and not a bug to chase

On `simulate_tcri` the generator draws ω from a Dirichlet and `q_p_ct_raw` is initialised at the
observed crosstab — which is already a good estimator of `P(φ|c)`. So a configuration that stays
near its initialisation scores well on NMI *without inferring anything*, and the comparison may
be rewarding least-drift-from-initialisation rather than best inference. This is the circularity
problem: the synthetic is drawn from the model's own family.

The residual drift is also plausibly **DE-5**, not DE-18: the guide's total concentration is
pinned to β regardless of how many cells a group has, so the posterior cannot concentrate in
proportion to the data term this PR just added. DE-18 supplies the evidence; DE-5 is what lets
the posterior respond to it. That would make the two a single change rather than PRs 3 and 8.

**Open:** ship DE-18 as-is and let PR 8 (DE-5) supply the missing half; or land DE-18 and DE-5
together; or hold both until a non-circular test bed exists. One seed, one configuration, one
generator — thin evidence for a model change of this size.


### Result of landing both

1200 cells, 20 clones, 5 phenotypes, seed 0, `normalize_mode="average"`, `n_samples=0`.
True NMI = 0.2145.

| epochs | L1 to crosstab | NMI | conc range | corr(conc, group size) |
|---|---|---|---|---|
| 60 | 0.234 | 0.150 | 9.25–11.71 | 0.270 |
| 200 | 0.160 | 0.170 | 7.96–17.69 | 0.402 |
| 600 | 0.210 | **0.220** | 5.71–36.00 | **0.558** |

Accuracy at 600 epochs, against a truth of 0.2145:

| configuration | NMI | error |
|---|---|---|
| pre-fix (surrogate only, β-pinned guide) | 0.170 | 0.045 |
| DE-18 only (data term, β-pinned guide) | 0.122 | 0.093 |
| **DE-18 + DE-5** | **0.220** | **0.0055** |

Three things changed at once, and they are the three symptoms this investigation started from:

- **The estimate converges to the truth** rather than away from it. Error is 8× smaller than the
  pre-fix code and 17× smaller than DE-18 alone.
- **Training no longer degrades the answer.** NMI rises monotonically with epochs. The optimum
  at 30–120 epochs followed by decay — the behaviour that produced the "bias floor" reading and
  cost a day — does not occur.
- **The posterior concentrates with data.** Totals spread from 9–12 to 6–36 and their
  correlation with group size climbs 0.27 → 0.56. This is the property eq 6 specifies and the
  β-pin removed, and it is why credible intervals were previously prior-set.

Caveat, stated because the evidence is thinner than the result sounds: one seed, one
configuration, one generator — and that generator is drawn from the model's own family. The
direction is unambiguous and the mechanism is visible in the concentration/size correlation, but
the magnitude should not be quoted until the benchmark grid re-runs.
