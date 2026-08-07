I have all the confirmations. Writing the specification.

## Source verification

`docs/contract/source/supplementary_note_1_SS_2026-08-03.pdf`, §"Inference Details", is one sentence on optimization: *"SVI (Stochastic Variational Inference) in Pyro - the probabilistic programming language used here - uses mini-batching, KL scaling for Dirichlet and discrete terms, and Adam optimization."* Nothing else in the note mentions epochs, learning rate, patience, warmup, stopping, initialization or seeding — the words do not appear anywhere in the 17 pages. §"Variational Inference" gives eq 7 and *"We maximize L via stochastic gradient ascent, using the reparameterization trick for continuous latents, and parallel enumeration for discrete latents."* The brief's reading is correct: **a training contract must be authored, not derived** — with one exception, flagged as I6/Q1 below, because that sentence does say something about KL scaling that the code does not do.

---

# Part 0 — Confirmations on current `main`

All measured with `MPLBACKEND=Agg .venv/bin/python`. Fixture sizes stated per item.

| # | Claim | Evidence |
|---|---|---|
| C1 | `validation_step` takes optimizer steps | `tcri/model/_training.py:141` calls `super().training_step`, which is `scvi/train/_trainingplans.py:1492-1502` → `self.svi.step(...)`. Instrumented `pyro.infer.SVI.step`: **36 calls in training, 2 in validation** over a 12-epoch/`check_val_every_n_epoch=5` run |
| C2 | Exactly which parameters move | Param-store ΔL1 per validation batch: `q_p_c_raw` **0.1726**, `q_p_ct_raw` **0.2140**, all 22 network tensors **0.0**. Mechanism: at validation entry, 22 params have `.grad is None` (torch's Adam skips them) and exactly 2 carry a zeroed grad, so the weight-decay term `wd·p` becomes the entire gradient for the two guide parameters |
| C3 | The decay acts in log space | `torch.distributions.transform_to(constraints.positive)` = `ComposeTransform(ExpTransform(), AffineTransform())`. Demo: constrained `[0.2, 5.0]` → leaf `[-1.609, +1.609]`. L2 on that leaf pulls every entry of a clone's row toward θ=1; the guide then row-normalizes (`_module.py:311-323`), so the pull is toward the uniform simplex point |
| C4 | Warmup unit is **STEPS** | `_my_global_step += 1` at `_training.py:135`, inside `training_step` only. Measured `(epoch, step, kl_weight)`: `(0,0,1e-6) (0,1,1e-6) (0,2,0.05)` — 3 steps in epoch 0 on a 200-cell/batch-64 fixture, i.e. `ceil(0.9·200/64)` |
| C5 | Shipped-default epoch equivalent | `DataSplitter` `n_train = ceil(0.9·N)` (`scvi/dataloaders/_data_splitting.py:52`), no `drop_last`. N=5000, batch=1024 → 4500 train cells → 5 steps/epoch → `n_steps_kl_warmup=2000` = **400 epochs**. At `train()`'s own defaults (batch=1000) also 5 steps/epoch = **400 epochs** |
| C6 | scvi's rival epoch schedule is inert — conditionally | `LowLevelPyroTrainingPlan.__init__` defaults `n_epochs_kl_warmup=400`, and `_compute_kl_weight` prefers epochs over steps (`_trainingplans.py:64-69`). It never fires only because `model()` has no `kl_weight` parameter, so `use_kl_weight=False` (`_trainingplans.py:1332-1336`). Adding that parameter would silently install a second, conflicting schedule |
| C7 | The monitor is non-stationary during warmup | Traced `elbo_validation` with `kl_weight` at each check: `0.061, 0.128, 0.194, … 0.994, 1.000`. The objective's definition changes at every check for the whole warmup |
| C8 | The monitor is noise-dominated | At **fixed** parameters, 500-cell validation batch, 30 repeats: 1-particle ELBO **sd 51.7**, range 219.7 on a mean of 5173. 8 particles: **sd 17.5**. `early_stopping_min_delta` defaults to 0.00 (`scvi/train/_trainer.py:111`) and tcri never sets it. On a 400-cell fixture the monitor's "best" was epoch 33 at 464.3 against a trendless local scatter of 535–658 |
| C9 | Train and validation ELBO are not on one scale | Same parameters, same batch: eval mode 5173.5, train mode 4999.8 — a **173.7 nat** offset (3.4× the MC sd) from dropout/BN mode alone, before the batch-size difference |
| C10 | Best weights are unreachable via `state_dict` | `"q_p_ct_raw" in module.state_dict()` → `False`. Non-`scvi$$$` param-store keys: `['q_p_c_raw', 'q_p_ct_raw']`. A `get_state`/`set_state` path already exists in-repo at `tcri/utils/_utils.py:167` |
| C11 | Patience is counted in checks | `lightning/pytorch/callbacks/early_stopping.py:62-64` states it, `:278` increments `wait_count` in `_run_early_stopping_check`, reached from `on_validation_end` (`:220`). `_model.py:300-301` sets `patience=300` and `check_val_every_n_epoch=5` → **1500 epochs** |
| C12 | Warmup restarts per `train()` | `_training.py:85` `self._my_global_step = 0`; `_model.py:282` constructs a fresh plan per call |
| C13 | The per-step objective is batch-size dependent | `pyro.plate("data", batch_size)` at `_module.py:216` and `:337` — no `size=`/`subsample=`, and `TCRIModule` has no `n_obs` attribute, so scvi's `n_obs_training` setter is a no-op. Traced log-prob sums on the 5000-cell benchmark fixture: **\|global\|/\|local\| = 0.0255 at batch=1024 vs 0.0064 at batch=4500** — a 4× change in the priors' relative weight from a batch-size choice |
| C14 | No seeding path exists | No `manual_seed`/`seed_everything`/`scvi.settings.seed` anywhere in `tcri/` or `benchmarks/`. `benchmarks/run_grid.py:86-137`: `seed` reaches `simulate_tcri` and the metric's `random_state`, never the fit. Measured: seed-immediately-before-each-fit → init identical, `p_ct` max\|diff\| **0.0**; seed-once-then-two-fits → init differs, `p_ct` max\|diff\| **0.0116** after 8 epochs |
| C15 | Wiring-only knobs | `_training.py:84` `self.reconstruction_loss_scale` is never read; the live value is `module.reconstruction_loss_scale` set at `_model.py:259` and consumed at `_module.py:269`. `tests/test_model_knobs.py:146` asserts the dead field. `num_particles=5` (`_training.py:42`) is inert on the default `Trace_ELBO()` path |
| C16 | Vectorized particles crash the model | `Trace_ELBO(num_particles=8, vectorize_particles=True)` → `IndexError: index 8 is out of bounds for dimension 0 with size 8` at `_module.py:212` (`p_c[self.ct_to_c]` indexes dim 0, which the particle dim displaces). Sequential particles work |
| C17 | Lightning cannot clip these gradients | `lightning/pytorch/trainer/configuration_validator.py:120-125` raises on `gradient_clip_val` under manual optimization, which `PyroTrainingPlan` sets (`_trainingplans.py:1477`). Clipping must come from `pyro.optim.ClippedAdam` (present) |

**Not reproducible / closed:** none of the above. Every item in the brief's premise held.

---

# Part (a) — Derivable invariants

These follow from eq 7 and are not matters of taste. **Home: `tcri/model/_model_contract.py`**, as a new `OBJECTIVE_INVARIANTS` dict beside `SEMANTIC_INVARIANTS`, asserted by `tests/test_model_contract_conformance.py`. Rationale: eq 7 and the surrogate already live there, and these say what "optimizing eq 7" operationally means. All CODEOWNER-restricted.

**I1 — `objective_is_eq7_plus_surrogate_only`.** The only quantity any optimizer descends is `−(L#(x;Λ,Θ) + γ·Σ KL(probs‖ϕ))`. No auxiliary loss, no second optimizer, no side objective. *Status: holds today* (the phantom torch-Adam was removed; `configure_optimizers` is scvi's one-dummy-param shim, guarded by `tests/test_model_guardrails.py:120-127`).

**I2 — `no_parameter_update_outside_training_step`.** `SVI.step` — and anything else that reaches `loss_and_grads` or a `PyroOptim.__call__` — may be invoked only from `training_step`, and only on training-split batches. **Violated:** C1, C2. This is the exact statement of DE-1, and it also forbids a future `predict`/`get_p_ct`/`tl.*` path from mutating the store.

**I3 — `monitor_is_a_fixed_function_of_parameters`.** Whatever quantity a stopping or selection rule compares across time must be the same function of Λ, Θ at every evaluation: same `kl_weight`, same module mode, same particle count, same data. Otherwise "improved" is undefined. **Violated:** C7 (`kl_weight` differs at every check for 400 epochs).

**I4 — `reported_model_is_the_selected_model`.** If a selection criterion is declared, the parameters left in the store when `train()` returns are the ones that criterion chose. **Violated:** C10 / DE-3.

**I5 — `kl_weight_is_schedule_only`.** `kl_weight` is a training-schedule multiplier with no standing in eq 7; any run whose ELBO or posterior is quoted must have reached `kl_weight_max = 1.0`. *Partly held* — `tests/test_model_knobs.py:181-206` already asserts monotone ramp to ceiling within one fit.

**I6 — `minibatch_estimate_is_unbiased_for_eq7`.** Eq 7 is an expectation over the full dataset. A per-minibatch estimate of it must weight the N cell-level terms against the C+M global Dirichlet terms in the N : 1 ratio the full objective has. **Violated:** C13 — the priors are over-weighted by `steps_per_epoch`, and their relative weight moves 4× with `batch_size`. **This one is not a training-plan fix; it changes what is optimized, so it is a model-contract change and it needs the author.** See Q1.

**I7 — `every_knob_is_behavioural`.** A declared knob reaches the object it configures *and* changes an observable. Wiring-only is this project's dominant defect class; the bound must be behavioural. **Violated:** C15.

---

# Part (b) — Authored bounds

**Home: a new fourth contract** — `tcri/model/_training_contract.py` + `docs/contract/TRAINING_CONTRACT.md` + `tests/test_training_contract_conformance.py`, added to the CLAUDE.md table and `.github/CODEOWNERS`. Rationale for a separate file rather than more entries in the model contract: (a) cannot be renegotiated without contradicting the note; (b) can be, by @nceglia/@salehis, and needs its own change log. These are properties, not values — no bound below names a learning rate.

**B1 — Monotone, terminating annealing.** `kl_weight` is non-decreasing within a fit, reaches `kl_weight_max` strictly before the last training step, and is non-decreasing **across** `train()` calls on the same model unless the caller passes `reset_schedule=True`. (DE-4: today it restarts, so staged training sees a sawtooth.)

**B2 — Warmup is declared in the unit it is counted in.** Definitive answer to DE-17/DUX-2: **STEPS**, incremented once per training minibatch at `_training.py:135`, and **2000 steps = 400 epochs** at the shipped defaults (5000 cells, `batch_size` 1000 or 1024 → 5 steps/epoch; C5). The bound: the API must accept the warmup in **epochs** and convert at fit time using the run's actual `steps_per_epoch`, recording both. A schedule whose meaning is `2000/ceil(0.9N/B)` epochs silently becomes 4 epochs at 500 000 cells. The plan must also pass `n_epochs_kl_warmup=None` to `super().__init__` so scvi's latent 400-epoch schedule (C6) can never wake up.

**B3 — Patience means what the docstring says.** `patience` is declared in **epochs** and translated explicitly: `early_stopping_patience = ceil(patience_epochs / check_val_every_n_epoch)`. Today the two are conflated and 300 means 1500 (C11).

**B4 — `min_delta` exceeds the monitor's own noise.** An improvement threshold of 0 against a monitor with sd 51.7 nats (C8) selects a lucky draw. The bound is relational, not a value: `min_delta ≥ 2·sd̂`, where `sd̂` is the monitor's Monte-Carlo standard deviation measured at fixed parameters — a quantity the plan can measure once at fit start and record. Equivalently, drive `sd̂` down with `val_num_particles` (8 particles → sd 17.5) and set `min_delta` from the result.

**B5 — Selection is gated on a converged schedule.** Neither early stopping nor the best-weight snapshot may act before `kl_weight` has reached its ceiling. scvi supports this directly (`early_stopping_warmup_epochs`, `_trainer.py:113,148`); tcri does not set it.

**B6 — Every advertised knob changes an output.** For each knob in `train()` and `UnifiedTrainingPlan.__init__` there is a test asserting a *result* moves, not only that an attribute was assigned. A knob with no behavioural test is removed, not documented.

**B7 — A run is a function of `(seed, data, knobs)`.** No dependence on ambient process RNG state, at any point including network construction. Measured today: bit-identical when seeded immediately before construct+fit, `p_ct` max|diff| 0.0116 across two fits from one process seed (C14).

**B8 — Optimizer settings that act as priors are declared as priors.** Weight decay on `q_p_c_raw`/`q_p_ct_raw` is one: their param-store leaves are `log θ` (C3), so L2 there is a pull toward uniform clonotype rows. The current `SANCTIONED_DEVIATIONS['optimizer_weight_decay']` text says decay applies "to the network parameters" — measurably it applies to the two variational Dirichlet parameters as well, and those are the only ones it reaches on a validation step. The contract text needs correcting whether or not the behaviour changes.

**B9 — The plan emits provenance.** Epochs actually run, stop reason, best epoch, best monitor value, warmup steps and their epoch equivalent, seed, effective steps/epoch, and the monitor's measured `sd̂`, written to `model.training_record_` and persisted by `to_anndata`. This closes DE-8 structurally instead of in the benchmark harness, where the same mistake will recur.

---

# Part (c) — Target design for `UnifiedTrainingPlan`

Whole-file rewrite of `tcri/model/_training.py` plus a new callback module and a ~40-line change in `TCRIModel.train`. Diff-shaped, in dependency order. **Measured** = I ran it; **inferred** = reasoned from source or literature.

### C-1. Read-only validation (fixes I2 / DE-1)

**Change.** Delete the `super().training_step(batch, batch_idx)` call at `_training.py:141`. Replace with a direct ELBO evaluation:

```python
# tcri/model/_training.py — validation_step
args, kwargs = self.module._get_fn_args_from_batch(batch)
was_training, saved_klw = self.module.training, self.module.kl_weight
self.module.eval()
self.module.kl_weight = self.module.kl_weight_max      # B5 / I3: stationary monitor
try:
    with torch.no_grad():
        total = self._val_loss_fn.loss(self.module.model, self.module.guide, *args, **kwargs)
finally:
    self.module.kl_weight = saved_klw
    if was_training:
        self.module.train()
```

**Why this and not the alternatives.** Pyro's own read-only path is `SVI.evaluate_loss`, which is literally `with torch.no_grad(): self.loss(model, guide, *args, **kwargs)` (`pyro/infer/svi.py:119-132`) — calling `Trace_ELBO.loss` directly is the same computation without needing an SVI object, and `Trace_ELBO.loss` (`pyro/infer/trace_elbo.py:64-80`) never calls `torch_backward`. Two rejected alternatives: wrapping the existing `svi.step` in `pyro.poutine.block` (blocks sites, not the optimizer — the store still moves), and passing `blocked=[...]` to `PyroTrainingPlan` (applies to training too). *Measured: `Trace_ELBO.loss` leaves the store bit-identical.*

**Reporting basis.** Accumulate the per-batch loss and the per-batch cell count, and in `on_validation_epoch_end` log `elbo_validation = Σ loss / Σ n_cells` — a **per-cell** figure. Lightning's default epoch reduction is an unweighted mean over batches, which is wrong the moment the validation loader yields a partial last batch. The C+M global Dirichlet terms still enter once per batch, so the per-cell number carries a constant `n_val_batches/N_val` offset; that offset is fixed across epochs for a fixed loader, which is all a comparison needs — state it in the docstring rather than pretending it away. Do the same for `elbo_train` by overriding `on_train_epoch_end` (scvi's version at `_trainingplans.py:1364-1374` is a per-batch mean of un-normalized losses). After this, the two curves are comparable **in shape but not in level**: *measured*, dropout/BN mode alone offsets them by 173.7 nats at identical parameters (C9), and train is evaluated at the running `kl_weight` while validation is pinned at the ceiling.

**Particles.** `self._val_loss_fn = Trace_ELBO(num_particles=val_num_particles, vectorize_particles=False)`, default 8. *Measured:* sd 51.7 → 17.5. `vectorize_particles` must stay False until `_module.py:212` is made particle-dim-safe (C16) — worth a separate defect entry, since it also blocks `num_particles` on the training path.

**Buys:** DE-1 closed exactly (not approximately); I3 satisfied; the monitor's variance drops 3×; the val curve becomes readable next to the train curve.

### C-2. Persistent, epoch-declared KL schedule (fixes B1, B2, DE-4)

**Change.** Move the counter off the plan onto the module (`module._kl_steps_taken`, initialized in `TCRIModule.__init__`, incremented in `training_step`). Add `n_epochs_kl_warmup: float | None` to `train()` as the front door; on `setup`, compute `steps_per_epoch = ceil(n_train / batch_size)` from the datamodule and set `self._warmup_steps = round(n_epochs_kl_warmup * steps_per_epoch)`. Keep `n_steps_kl_warmup` as an explicit override, and make passing both an error. Pass `n_epochs_kl_warmup=None` to `super().__init__` (C6). Add `reset_schedule: bool = False` to `train()`.

**Schedule shape:** keep linear. Cyclical annealing ([Fu et al., NAACL 2019](https://aclanthology.org/N19-1021/)) is the main published alternative and targets KL vanishing in text VAEs; there is no evidence of posterior collapse here — the note's VampPrior is itself the stated mitigation ("mitigates posterior collapse", §Generative Model item 3) — and a cyclical schedule would make the monitor non-stationary forever, contradicting I3. **Authored, not imported:** linear-to-ceiling with a declared epoch length.

**Buys:** DE-4 closed; the warmup stops changing meaning with dataset size; the scvi shadow schedule is disarmed.

### C-3. Early stopping that means what it says (fixes B3, B4, B5)

**Change, in `TCRIModel.train` (`_model.py:297-303`):**

```python
check_every = kwargs.setdefault("check_val_every_n_epoch", 5)
kwargs.setdefault("early_stopping_patience", max(1, ceil(self.patience / check_every)))
kwargs.setdefault("early_stopping_min_delta", min_delta)          # from B4
kwargs.setdefault("early_stopping_warmup_epochs", warmup_epochs)  # from C-2
```

and rename the constructor argument `patience` → `patience_epochs` (keeping `patience` as a deprecated alias for one release), with a docstring that states the translation.

**Is validation ELBO defensible as the monitor?** Partly, and the honest answer has two halves. It is the field's default — scvi-tools monitors `elbo_validation` with `patience=45` checks and `min_delta=0.0` (`scvi/train/_trainer.py:108-114`; [docs](https://docs.scvi-tools.org/en/stable/api/reference/scvi.train.Trainer.html)) — and there is very little published beyond that; the search for principled VAE early-stopping monitors returns the same generic "stop at minimum validation loss" advice Pyro's own tutorial gives ([SVI Part IV](https://pyro.ai/examples/svi_part_iv.html)). It is defensible as a **guard against divergence and overfitting of the density model**. It is *not* defensible as a proxy for metric accuracy, and the brief's own measurements are the proof: the gate-0 deterministic read matches the label oracle at 30–120 epochs (0.1931/0.1933/0.1916 vs 0.1927) and has fallen to 0.1353 by 4000 while validation ELBO is still improving. The recommendation is therefore to keep validation ELBO as the *stopping* rule and to add a second, explicitly-declared *selection* rule — a held-out clone-level agreement statistic between `get_p_ct()` and the observed clone×phenotype crosstab, which the brief already shows bottoms at ~120 epochs where the accuracy optimum is. That second monitor is authored, is not in the note, and must be labelled as such; the point is that the criterion be *stated and enforced* (I4), not that ELBO be replaced silently.

**Buys:** patience stops meaning 5× its label; the selected epoch stops being a draw; the gap between "best ELBO" and "best metric" becomes a declared choice instead of an accident.

### C-4. Best-weight restore across the param-store boundary (fixes I4 / DE-3)

**New file `tcri/model/_callbacks.py`:** a Lightning `Callback`, not plan code, so the selection rule is one auditable object.

```python
class ParamStoreCheckpoint(Callback):
    # on_validation_end: if monitor improved by > min_delta and warmup is done,
    #     self._best = copy.deepcopy(pyro.get_param_store().get_state())
    #     self._best_epoch, self._best_value = trainer.current_epoch, value
    # on_fit_end: pyro.get_param_store().set_state(self._best)
```

`get_state()` returns live references — the `deepcopy` is load-bearing. `set_state` is already used in-repo at `tcri/utils/_utils.py:167`, so this is an existing, tested seam. Restoring discards the Adam state, which is correct because restoration happens once, at the end. `enable_checkpointing` stays False: Lightning's checkpointer writes `state_dict()`, which provably does not contain `q_p_ct_raw` (C10), so it cannot serve this purpose no matter how it is configured — worth saying explicitly in `TRAINING_CONTRACT.md` so nobody proposes it again.

**Buys:** DE-3 closed; `best_epoch`/`best_value` become available for B9.

### C-5. Optimizer (fixes B8, adds clipping)

**Change.** Default to `pyro.optim.ClippedAdam` with per-parameter argument groups:

```python
def _optim_args(module_name, param_name):
    base = {"lr": lr, "betas": betas, "eps": eps, "clip_norm": clip_norm}
    # variational Dirichlet params live as log θ; L2 there is a prior toward uniform
    if param_name in ("q_p_c_raw", "q_p_ct_raw"):
        return {**base, "weight_decay": guide_weight_decay}   # default 0.0
    return {**base, "weight_decay": weight_decay}
```

Pyro's `PyroOptim` accepts a callable `optim_args(module_name, param_name)`, so this needs no new machinery.

**Is weight decay in the unconstrained log space of a positive-constrained parameter defensible?** No, not as an unlabelled optimizer setting. It is not a shrinkage toward zero of a magnitude — because the leaf is `log θ` (C3, measured), it is shrinkage of θ toward **1**, and since the guide row-normalizes `q_p_ct_raw**(1/T)` (`_module.py:319-323`), that is a pull of every clone's posterior toward the uniform simplex point, i.e. toward *lower* clone–phenotype mutual information. That is a prior on `q(ϕ_m)` with strength `1e-4` and no equation number. The field has nothing to say about this specific case — a search for weight decay on variational Dirichlet concentrations returns only the general AdamW literature ([Loshchilov & Hutter](https://arxiv.org/pdf/1711.05101)), whose entire point is that coupling decay into the gradient interacts badly with adaptive step sizes, which is exactly the failure mode C2 exhibits. **Recommendation: `guide_weight_decay=0.0` by default, keep `weight_decay=1e-4` on the networks, and amend `SANCTIONED_DEVIATIONS['optimizer_weight_decay']` to say what it actually touches.** Anyone wanting the uniform pull can set it deliberately.

**Gradient clipping:** must come from the optimizer, because Lightning refuses `gradient_clip_val` under manual optimization (C17). `ClippedAdam` is Pyro's own recommendation for SVI ("Use Adam or ClippedAdam by default when doing SVI... the smoothing they provide via per-parameter momentum is often essential when the optimization problem is very stochastic" — [SVI Part IV](https://pyro.ai/examples/svi_part_iv.html)). Default `clip_norm=10.0` — **authored**; the objective contains `log(x+1e-8)` terms and a `pyro.factor` KL, both of which can spike.

**`eps=1e-5`** (vs torch's 1e-8) is already an authored departure and should be declared rather than left as a literal at `_training.py:59`.

### C-6. Seeding (fixes B7)

**Change.** `seed: int | None = None` on **both** `TCRIModel.__init__` (network construction happens there, before `train()` is ever called) and `TCRIModel.train`. When set, `lightning.seed_everything(seed, workers=True)` — `workers=True` matters, it seeds dataloader worker RNGs — plus `pyro.set_rng_seed(seed)`, immediately before the object that consumes randomness is built. Honor `scvi.settings.seed` when the argument is None. Fix `benchmarks/run_grid.py:86-129` to pass its `seed` to the fit as well as the simulator.

**Buys:** the ~0.0018 fit-to-fit spread disappears; grid cells become comparable; the reproducibility bound becomes testable as bitwise equality.

### C-7. Remove the wiring-only knobs (fixes I7 / C15)

Drop `reconstruction_loss_scale` from `UnifiedTrainingPlan.__init__` and have the plan assign `module.reconstruction_loss_scale` in `setup()` (the plan owns the run's objective settings). Drop the `num_particles=5` default from the signature or honor it on `Trace_ELBO` too — do not leave a default that describes a path the code does not take.

---

# Part 5 — Dependencies affected, and which numbers move

| Change | Direction on reported MI/NMI | Magnitude | Basis |
|---|---|---|---|
| C-1 read-only validation | **up** | +0.0006 (0.5002 → 0.5008 at 5000 cells / 2000 epochs) | measured in DE-1; removes the per-check flattening of `q_p_ct_raw` |
| C-4 best-weight restore | **up**, small | +0.004 in the recorded instance (0.4979 at epoch 2964 → 0.5020 at epoch 1464) | measured in DE-3. Restores the *chosen* model; does **not** reach the accuracy optimum, which sits at 30–120 epochs |
| C-3 patience in epochs (300 checks → 300 epochs) | **up**, large, regime-dependent | The brief's curve: 0.1353 at 4000 epochs vs 0.1931 at 30–120, oracle 0.1927 | derived from the brief's measured facts. This is the single largest mover among the training fixes |
| C-5 `guide_weight_decay=0` | **up**, small | not yet measured — must be measured before adoption | mechanism confirmed (C3); magnitude is not |
| C-6 seeding | none | removes ~0.0018 fit-to-fit spread | measured (C14) |
| C-1 particle count / C-3 `min_delta` | none on a given fit | changes *which* epoch is selected | measured (C8) |
| I6 minibatch scaling (**if adopted**) | **up**, potentially large | unmeasured; would let `p_ct` concentrate with data instead of being held at the prior | C13. **Do not adopt without the author** |

**Published numbers.** Every NMI in the benchmark grid moves. The direction is uniformly toward the oracle in the flat-coupling regime and small in the sharp regime, but the grid must be re-run; treat the current table as superseded, not adjusted.

**Existing tests whose expected value changes.**
- `tests/test_model_knobs.py:146` — `assert plan.reconstruction_loss_scale == 5e-3` asserts a field nothing reads. Rewrite against `module.reconstruction_loss_scale`.
- `tests/test_model_knobs.py:155-163` — `assert es[0].patience == 4` becomes `ceil(4/check_val_every_n_epoch)` once B3 lands.
- `tests/test_model_knobs.py:181-206` and `tests/test_model_guardrails.py:110-116` construct `UnifiedTrainingPlan` directly; both break on the C-2/C-7 signature change.
- `tests/test_model_classifier.py:130` (`recovery >= 0.9`, chance 0.2) is the one accuracy oracle in CI and runs 200 epochs. It is the test most likely to move if C-3 shortens runs — check it, do not relax it.
- Downstream metric tests (`tests/test_tools/`, `tests/test_diag/`, `tests/test_plotting/`) assert identities and structure against the `trained_model` session fixture, not pinned values, so they are insensitive. Verified by inspection.

---

# Part 6 — Verification tests

Every tolerance below comes from an identity, an exact integer, or a quantity the test measures for itself. None is calibrated on current behaviour.

**T1 — store immutability outside `training_step` (I2).** Wrap `PyroOptim.__call__` with a counter; run 6 epochs with `check_val_every_n_epoch=1`; assert zero calls from any non-training frame, and assert every param-store tensor is **bitwise identical** across a validation epoch. Extend to `predict`, `get_p_ct`, `to_anndata` and one `tl.mutual_information` call. Tolerance: exact equality.

**T2 — objective equals eq 7 (I1, and I6 if adopted).** With `batch_size >= n_train` (one step per epoch), the per-step training loss must equal `Trace_ELBO().loss` evaluated on the full training split at the same parameters, to within the estimator's own MC sd — which the test measures in the same run by repeating the evaluation 30× at fixed parameters. An analytic identity, not a calibrated number.

**T3 — monitor stationarity (I3).** Record `(kl_weight, module.training)` at every validation check. Assert `kl_weight == kl_weight_max` and `module.training is False` at all of them.

**T4 — selected model is the reported model (I4).** Snapshot the store at the recorded `best_epoch` via the callback's own path; at `on_fit_end`, assert the live store is bitwise equal to that snapshot, and that the monitor recomputed at the restored parameters equals the recorded `best_value` within the sd the test measures.

**T5 — warmup arithmetic (B2).** Exact integers: for `(N, batch_size, n_epochs_kl_warmup)` assert `plan._warmup_steps == round(n_epochs_kl_warmup * ceil(ceil(0.9N)/batch_size))`, and assert `kl_weight` first equals `kl_weight_max` at exactly that step index. Plus a regression pinning the shipped default: 2000 steps at N=5000/batch=1024 **is 400 epochs**.

**T6 — schedule persists across `train()` (B1).** Call `train(max_epochs=3)` twice; assert the concatenated `kl_weight` trace is non-decreasing. Assert `reset_schedule=True` restarts it.

**T7 — patience units (B3).** Construct with `patience_epochs=10`, `check_val_every_n_epoch=5`; assert the installed callback has `patience == 2`; and with a deliberately flat monitor, assert the run stops within `10 ± check_val_every_n_epoch` epochs of the last improvement.

**T8 — `min_delta` exceeds monitor noise (B4).** The test measures `sd̂` itself: evaluate the validation monitor 30× at frozen parameters, then assert `min_delta >= 2·sd̂`. Self-calibrating and re-derived on every run.

**T9 — reproducibility (B7).** In one process: seed → construct → fit → snapshot store; repeat with the same seed; assert **bitwise** equality of every store tensor and of `get_p_ct()`. Then assert two *different* seeds differ, so the test cannot pass by the model being degenerate.

**T10 — behavioural knobs (I7, B6).** Parametrize over every `train()`/plan knob; for each, assert an *output* changes (`get_p_ct`, `predict`, `elbo_validation`, or the recorded epoch count), not an attribute. `reconstruction_loss_scale` and `num_particles` are the two that fail this today.

**T11 — provenance (B9).** Assert `model.training_record_` records epochs actually run, stop reason, best epoch, warmup steps *and* their epoch equivalent, seed, and `steps_per_epoch`; assert it survives `to_anndata` → `save_tcri_session` → `load_tcri_session`.

Run everything as `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q`. Baseline before any change: 177 passed, 3 skipped.

---

# Part 7 — The one question that must go to the author

**Q1 — "KL scaling for Dirichlet and discrete terms" (Note 1, §Inference Details).** This is the only optimization instruction the note gives, and the code does not do it under either reading:

- If it means **minibatch KL scaling** (Pyro's usual sense — `pyro.plate(..., size=N, subsample=...)` so local terms are upweighted to the full dataset), then `_module.py:216` and `:337` are missing it, and the measured consequence is that the Dirichlet priors are over-weighted by `steps_per_epoch` and their relative strength moves 4× with `batch_size` (C13). Fixing it changes what is optimized — a model-contract change under CLAUDE.md, and one that would plausibly move every published number upward by letting `p_ct` concentrate with data.
- If it means **KL annealing**, the note says it applies to the Dirichlet and discrete terms; the code anneals only the `latent` (z) term, and the model contract records that as `SANCTIONED_DEVIATIONS['kl_warmup_z_only']` — which under this reading is the inverse of what the note specifies.

CLAUDE.md says to ask rather than infer a definition from what makes the code come out right, and both readings imply a contract edit, so this does not belong in the training plan and I have not designed for it. It is the largest single unknown in the model's effective objective.

**Q2 — secondary, for the same conversation.** `SANCTIONED_DEVIATIONS['optimizer_weight_decay']` says the decay applies "to the network parameters." Measured, it applies to `q_p_c_raw` and `q_p_ct_raw` as well, in log space, and on a validation step those are the *only* parameters it reaches. The text needs correcting regardless of whether C-5 changes the default.

---

## Files touched by the plan

- `tcri/model/_training.py` — rewrite (C-1, C-2, C-5, C-7)
- `tcri/model/_callbacks.py` — new (C-4)
- `tcri/model/_model.py:244-314` — `train()` signature and early-stopping translation (C-2, C-3, C-6, B9)
- `tcri/model/_model.py:97-227` — `seed` on `__init__` (C-6)
- `tcri/model/_module.py` — `_kl_steps_taken` counter (C-2)
- `tcri/model/_model_contract.py` — new `OBJECTIVE_INVARIANTS`; amend `optimizer_weight_decay` (a), Q2
- `docs/contract/MODEL_CONTRACT.md` — mirror
- `tcri/model/_training_contract.py`, `docs/contract/TRAINING_CONTRACT.md`, `tests/test_training_contract_conformance.py` — new (b)
- `tests/test_model_contract_conformance.py` — assert `OBJECTIVE_INVARIANTS`
- `tests/test_model_knobs.py:146,155-163,181-206`, `tests/test_model_guardrails.py:110-116` — expected values change
- `benchmarks/run_grid.py:86-137` — seed the fit; record actual epochs from `training_record_`
- `CLAUDE.md`, `.github/CODEOWNERS`, `docs/contract/DEFECTS.md`, `docs/contract/REFACTOR_AGENDA.md`

Sources: [Pyro SVI API](https://docs.pyro.ai/en/dev/inference_algos.html) · [Pyro SVI Part IV: Tips and Tricks](https://pyro.ai/examples/svi_part_iv.html) · [scvi-tools Trainer](https://docs.scvi-tools.org/en/stable/api/reference/scvi.train.Trainer.html) · [scvi-tools TrainingPlan](https://docs.scvi-tools.org/en/stable/api/reference/scvi.train.TrainingPlan.html) · [Fu et al. 2019, Cyclical Annealing Schedule](https://aclanthology.org/N19-1021/) · [Loshchilov & Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101)