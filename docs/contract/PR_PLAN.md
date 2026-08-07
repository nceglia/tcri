# PR plan — the stack

Seven PRs, submitted as one `gh stack` chain. Each is independently reviewable (one concern, one review audience) and independently revertible (contract text and the code it governs travel together, so a revert never leaves a conformance test red). The chain is linear, so the order below is also the merge order.

```
gh stack init  seed-and-record
gh stack add   training-mechanics
gh stack add   phenotype-likelihood     # DE-18 — decided 2026-08-07
gh stack add   stopping-policy
gh stack add   metric-joint
gh stack add   generator-fidelity
gh stack add   benchmark-protocol
gh stack add   guide-concentration
gh stack submit
```

Ordering principle: the two PRs that move no number at all land first, because PR 1 is the measuring instrument for everything after it and PR 2's only numeric effect is the one defect it closes. Every subsequent PR moves published numbers, and each lands against a baseline that the PR below it made reproducible.

Three defects are opened by this plan and are not yet in the register above: **DE-18** (`p_ct` has no data term), **DE-19** (the fit is unseeded), **DE-20** (fuzziness mapping is `g(f)=f` where the note specifies `g(f)=√f`). They are written into the register by PR 1.

**Amended 2026-08-07.** Q-A is answered: condition on the observed phenotype. DE-18 becomes **PR 3**, ahead of everything it blocks (DE-3, DE-5, DE-6, DE-10, DE-12), and the stack is eight PRs. PR 7 `guide-concentration` is no longer conditional — it becomes meaningful only once PR 3 gives φ a data term.

## Stack at a glance

| # | Branch | Closes | Contract first? | Numbers move |
|---|---|---|---|---|
| 1 | `seed-and-record` | DE-19, DE-16 | no | none |
| 2 | `training-mechanics` | DE-1, DE-4, DE-17 | **yes** — new training contract + 2 model-contract amendments | MI/NMI **+6e-4** |
| 4 | `stopping-policy` | DE-2, DE-3 | no (flips xfails from PR 2) | MI/NMI **0 to +4e-3** |
| 5 | `metric-joint` | DE-6, DE-7, DE-15 | **yes** — metrics manifest + API contract | MI/NMI **+0.05 to +0.21** |
| 6 | `generator-fidelity` | DE-13, DE-14, DE-20 | no | benchmark difficulty axis only |
| 7 | `benchmark-protocol` | DE-8, DE-9, DE-10, DE-11, DE-12 | no | every benchmark cell |
| 3 | `phenotype-likelihood` | DE-18 | **yes** — model manifest (new observed site) | all of them; every prior baseline invalidated |
| 8 | `guide-concentration` | DE-5 | **yes** — model manifest | posterior width; meaningful only after PR 3 |

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

## PR 2 — `training-mechanics`

**Title:** Read-only validation, per-model KL schedule, and a training contract

**Closes:** DE-1, DE-4, DE-17.

**Commit order inside the PR** (contract first, per CLAUDE.md):

1. Contract: new `tcri/model/_training_contract.py`, `docs/contract/TRAINING_CONTRACT.md`, `tests/test_training_contract_conformance.py`; amendments to `tcri/model/_model_contract.py` + `docs/contract/MODEL_CONTRACT.md`; one row in CLAUDE.md's contract table; three lines in `.github/CODEOWNERS`.
2. `validation_step` becomes a read (DE-1).
3. KL counter moves to the module; scvi's shadow schedule is disarmed (DE-4, DE-17).
4. Wiring-only knobs removed.
5. Existing test expectations updated.

**Files**

- `tcri/model/_training.py` — replace `validation_step:138-169`. Delete the `super().training_step(batch, batch_idx)` call at `:141` and evaluate instead:

  ```python
  args, kwargs = self.module._get_fn_args_from_batch(batch)
  with torch.no_grad():
      total = self._val_loss_fn.loss(self.module.model, self.module.guide, *args, **kwargs)
  ```

  `Trace_ELBO.loss` is exactly what `SVI.evaluate_loss` wraps and never calls `torch_backward`. Drop the `self.module.eval()` / `self.module.train()` toggle at `:140`/`:142` — Lightning's evaluation loop already puts the module in eval, and the toggle is what currently causes the diagnostic block at `:144-166` to run in train mode with classifier dropout active. Keep the whole method in eval and under `no_grad`; the diagnostic block is presently outside `no_grad`, unlike its `training_step` twin at `:113`, and builds an autograd graph for values that are only logged.
  Also: read `self.module._kl_warmup_step` at `:97-98` and increment it at `:135`; pass `n_epochs_kl_warmup=None` to `super().__init__` at `:71-81`; add `on_train_start` emitting a `UserWarning` when `n_steps_kl_warmup > max_epochs * trainer.num_training_batches`, naming the fraction of `kl_weight_max` the run will reach; delete the never-read `self.reconstruction_loss_scale` field at `:84` and the inert `num_particles=5` default at `:42`.
- `tcri/model/_module.py` — `self._kl_warmup_step = 0` beside `self.kl_weight` at `:85`. A plain `int`, deliberately not a buffer: a buffer changes the checkpoint key set and breaks `load_state_dict(strict=True)` against every previously saved model.
- `tcri/model/_model.py` — `reset_schedule: bool = False` on `__init__`; assign `module.reconstruction_loss_scale` unchanged at `:259`; docstring on `train()` stating the warmup unit and its epoch equivalent; `training_record_` populated with epochs run, warmup steps and their epoch equivalent, seed, and `steps_per_epoch`.
- `tests/test_model_knobs.py:146` (asserts a field nothing reads), `:181-206`, `tests/test_model_guardrails.py:110-116` (both construct `UnifiedTrainingPlan` directly and break on the signature change).

**Contract change:** yes, and it is the substantive half of the PR.

*Derived invariants* — these follow from eq 7 and are not negotiable. Home: `tcri/model/_training_contract.py::DERIVED_INVARIANTS`.

| | Statement | After PR 2 |
|---|---|---|
| I1 | The only quantity any optimizer descends is `−(L# + γ·Σ KL(probs‖ϕ))` | holds |
| I2 | No parameter update outside `training_step`, and only on training-split batches | **fixed here** |
| I3 | The monitored quantity is a fixed function of Λ,Θ — same `kl_weight`, module mode, particle count, data | asserted `xfail(strict=True)`; PR 3 flips it |
| I4 | The parameters left in the store when `train()` returns are the ones the declared criterion selected | asserted `xfail(strict=True)`; PR 3 flips it |
| I5 | `kl_weight` is schedule-only; a quoted ELBO or posterior comes from a run that reached `kl_weight_max` | tightened here |
| I6 | A minibatch estimate weights the N cell terms against the C+M global terms in eq 7's ratio | **open — Q-B**; ships `xfail(strict=True)` |
| I7 | A declared knob reaches its object *and* changes an observable | **fixed here** |

*Authored bounds* — `AUTHORED_BOUNDS` in the same file, changeable by @nceglia/@salehis: B1 monotone terminating annealing across `train()` calls unless `reset_schedule=True`; B2 warmup declared in the unit it is counted in, with the epoch equivalent recorded; B3 patience declared in epochs and translated explicitly; B4 `min_delta` exceeds the monitor's measured noise; B5 no selection before the schedule converges; B6 every advertised knob has a behavioural test; B7 a run is a function of `(seed, data, knobs)`; B8 optimizer settings that act as priors are declared as priors; B9 the plan emits provenance. B3–B5 are `xfail(strict=True)` until PR 3.

Model-contract amendments: `SANCTIONED_DEVIATIONS['optimizer_weight_decay']` currently says decay applies "to the network parameters" — it also reaches `q_p_c_raw`/`q_p_ct_raw`, whose param-store leaves are `log θ`, making it a pull of every clone row toward the uniform simplex point (Q-D). `SANCTIONED_DEVIATIONS['kl_warmup_z_only']` gains the unit and the size-dependence: `n_steps_kl_warmup=2000` at the shipped defaults is 400 epochs at 5000 cells and 2000 epochs at 1000 cells. Manifest and prose move in the same commit or `tests/test_model_contract_conformance.py` goes red.

**Which parts of the recommended `UnifiedTrainingPlan` design land here, and which do not**

| Design element | Here | Why |
|---|---|---|
| Read-only validation via `Trace_ELBO.loss` | **yes** | the defect itself; effect is bounded and measured |
| Diagnostic block in eval mode, under `no_grad` | **yes** | changes a logged series nothing monitors |
| Per-cell reporting basis for `elbo_train` / `elbo_validation` | **yes** | a constant divisor per epoch, so monotone-equivalent to the current series; the stop epoch cannot move while `min_delta=0` |
| Per-model KL counter, `reset_schedule` | **yes** | provably zero on the single-`train()` path |
| `n_epochs_kl_warmup=None` to `super()` | **yes** | provably inert today (`use_kl_weight=False`), removes a trap |
| Warmup-unreachable warning | **yes** | no behaviour |
| Removal of wiring-only knobs | **yes** | the fields are unread |
| **Pinning `kl_weight = kl_weight_max` during validation (I3)** | **no — PR 3** | `early_stopping=True` is the default (`_model.py:296`), so changing the monitored series changes the stop epoch and therefore the fitted result |
| **`val_num_particles=8`** | **no — PR 3** | same reason: it changes the monitor's noise (sd 51.7 → 17.5) and therefore which epoch wins |
| **`min_delta`, `warmup_epochs`, patience translation, best-state restore** | **no — PR 3** | these are the stopping policy |
| **`ClippedAdam`, per-parameter `guide_weight_decay=0.0`, `clip_norm`** | **no — deferred, own PR after the grid re-run** | mechanism is confirmed but the magnitude is unmeasured; B8 records the requirement, Q-D corrects the text, the default change waits for a measurement |
| **Minibatch KL scaling (I6)** | **no — Q-B** | changes what is optimized; a model-contract change requiring the author |

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_training_contract_conformance.py \
  tests/test_model_contract_conformance.py tests/test_model_knobs.py \
  tests/test_model_guardrails.py -q && \
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

Two tests must be written with a `lightning.pytorch.Callback` driving a real `model.train()`, not a standalone `plan.validation_step(batch, 0)` — that call raises `RuntimeError: UnifiedTrainingPlan is not attached to a Trainer` at `_trainingplans.py:1503`, so the register's harness cannot be shown failing on the parent commit. The callback snapshots in `on_validation_start` and compares in `on_validation_end`; measured drift on `main` is 0.12–0.14 per check, and `len(plan.training_step_outputs)` goes 5 → 6.

Two further test corrections. The validation estimator's equivalence must not be checked against `evaluate_loss` (circular post-fix) — use (a) seed-paired equality `pyro.set_rng_seed(k); evaluate_loss(...)` vs `pyro.set_rng_seed(k); svi.step(...)` at `rel=1e-9`, and (b) site-set identity via `poutine.trace` under both paths, asserting `"phenotype_alignment"` is present. (b) is the only assertion that fails if the validation ELBO silently loses a term. And DE-4's "no movement on the single-call path" proof must be a permanent closed-form test, not a one-off review check: record `module.kl_weight` at every `training_step` of one seeded `train()` and assert element-wise equality to `[max(1e-6, kl_max*i/K) for i in range(n)]`.

**Metric outputs that move**

- `tl.mutual_information`, `tl.compare_groups`: **up**, +6e-4 (0.5002 → 0.5008 at 5000 cells / 2000 epochs).
- `tl.clonotypic_entropy`, `tl.phenotypic_entropy` (conditional forms): **down** correspondingly.
- `tl.phenotypic_flux` (D_KL): **up**.
- `model.predict`, `obsm['X_tcri_probabilities']`, `obs['tcri_phenotype']`: shift with `p_ct`.
- DE-4 and DE-17 contribute exactly zero on the single-`train()` path. DE-4 changes results only for callers who train the same model twice, where `kl_weight` no longer drops to 1e-6 at the start of call 2.
- `elbo_train` stops including validation batches and is reported per cell; `diag.loss` panel 1 changes scale. No test reads it.

+6e-4 is below the pre-PR-1 noise floor and above nothing; it is measurable here only because PR 1 landed.

---

## PR 3 — `phenotype-likelihood`

**Title:** Condition on the observed phenotype (eq 4 with `z^φ` observed)

**Closes:** DE-18. Unblocks DE-3, DE-5, DE-6, DE-10, DE-12.

**The change.** Inside the `data` plate in `TCRIModule.model`, after `ell` is formed:

```python
pyro.sample("phenotype", dist.Categorical(logits=ell),
            obs=self._target_phenotypes[indices])
```

`ell = π·cls_logits + (1−π)·log φ`, so the observed label gives gradient to **both** the
classifier and `p_ct`. This is eq 4 with `z^φ` observed rather than latent. `_target_phenotypes`
is already populated from `_model.py:235` and read nowhere — this is its intended use.

**Open sub-decision, to be settled by measurement during implementation:** with `z^φ` observed,
the surrogate's enumeration justification disappears and `−γ·KL(probs ‖ φ)` becomes a second
term acting on the same quantity the likelihood now constrains. Either drop it, or keep it as a
γ-weighted regulariser with γ configurable and default measured. Do not decide this from
first principles — fit both and compare recovery and `p_ct` drift.

**Files**

- `tcri/model/_module.py` — the observed site; `.detach()` on `phi` is retained or removed
  depending on the surrogate decision (if the surrogate is dropped, the detach goes with it).
- `tcri/model/_model_contract.py` — declare the new observed site. **This changes the joint
  distribution**, which `CLAUDE.md` names explicitly as a model-mathematics change.
- `docs/contract/MODEL_CONTRACT.md` — prose twin, same commit.
- `docs/contract/METHODS_CONFORMANCE.md` — the eq-4 row moves from `◐ via surrogate` to an
  observed site; the symbols table entry for `z^φ_i` changes from "not sampled" to observed.
- `tests/test_model_contract_conformance.py` — the traced site set gains `phenotype`.

**Contract change:** yes, and it is the substance of the PR. Contract commit first, per
`CLAUDE.md`. CODEOWNER-gated.

**Metric outputs that move: all of them, and this is the point.** `p_ct` becomes data-informed,
so it should stop drifting away from the observed crosstab with training. Expected, to be
confirmed rather than assumed: `p_ct` L1 to the crosstab stops growing; the 30–120 epoch optimum
extends rather than decaying; MI/NMI stop degrading with epochs; `gate=1` stops reading ~0
because `cls_logits` finally receives gradient from real labels.

**Every baseline recorded before this PR is invalidated.** Runs across it are not comparable.

**Verification tests**

1. The traced model contains an observed site `phenotype` with `Categorical` family in the
   `data` plate — contract conformance.
2. **`p_ct`'s L1 to the observed crosstab does not grow monotonically with training.** This is
   the direct behavioural assertion for DE-18 and it fails on the parent commit — measured
   0.284 → 0.515 over 120 → 4000 epochs there. Show it red before it is green.
3. Classifier recovery on separable synthetic stays at 1.000, and `gate_prob=1` on the fitted
   benchmark fixture reads materially above 0 (it reads 0.0022 / 0.0000 today).
4. Full suite green.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

---

## PR 4 — `stopping-policy`

**Title:** Patience in epochs, and restore the selected weights

**Closes:** DE-2, DE-3.

**Files**

- New `tcri/model/_callbacks.py` — `RestoreBestState(Callback)`. On `on_validation_end`, if the monitor improved by more than `min_delta` **and** warmup has completed, snapshot `pl_module.module.named_parameters()` (CPU clones) plus `q_p_c_raw` / `q_p_ct_raw` from the Pyro store. On `on_fit_end`, restore in place under `no_grad`. Snapshot `named_parameters()`, **not** `state_dict()` — the latter carries `c_array`, `ct_array`, `_target_phenotypes`, three int64 buffers of length `n_obs` that are constant during training, so the cost scales with cells (~12 MB per snapshot at 500k cells) and PR 3's cadence change multiplies the frequency by five. Restore must be in place; `ParamStoreDict.set_state` rebinds `self._params[name]`, which would leave the store holding clones while the `nn.Module` keeps the originals.
- `tcri/model/_training.py` — pin `self.module.kl_weight = self.module.kl_weight_max` for the duration of `validation_step` and restore it in a `finally`; `self._val_loss_fn = Trace_ELBO(num_particles=val_num_particles, vectorize_particles=False)`, default 8. Keep `vectorize_particles=False`: `Trace_ELBO(num_particles=8, vectorize_particles=True)` raises `IndexError` at `_module.py:212` because `p_c[self.ct_to_c]` indexes dim 0, which the particle dimension displaces. Log the annealed value separately as `elbo_validation_annealed` if the train/validation shape comparison is wanted; the two series are comparable in shape but not level (measured 173.7 nat offset from module mode alone at identical parameters).
- `tcri/model/_model.py:296-303` — `check_val_every_n_epoch` 5 → 1; `patience` → `patience_epochs` with `patience` kept as a deprecated alias for one release; `early_stopping_patience = max(1, ceil(patience_epochs / check_every))`; `early_stopping_min_delta` set from the monitor's measured sd; `early_stopping_warmup_epochs` set from the resolved warmup; `restore_best_weights: bool = True` on `__init__`; append the resolved stopping policy to the `logger.info` at `:238`. Complete `training_record_` with best epoch, best value, stop reason, and the measured `sd̂`.
- `tcri/model/_module.py` — three-line `device` property. `PyroBaseModuleClass` has none, so scvi's `SaveCheckpoint.on_train_end` raises `AttributeError` today for anyone who passes `enable_checkpointing=True`. `enable_checkpointing` stays False and `TRAINING_CONTRACT.md` should say why: Lightning's checkpointer writes `state_dict()`, which provably does not contain `q_p_ct_raw`.
- `tests/test_training_contract_conformance.py` — flip I3, I4, B3, B4, B5 from `xfail` to asserted.
- `tests/test_model_knobs.py:163` — `es[0].patience == 4` becomes `ceil(4 / check_val_every_n_epoch)`.

**Contract change:** none of its own. The bounds it satisfies were declared in PR 2; this PR only turns their strict xfails into assertions, which is why PR 2 must not declare them as already holding.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_training_contract_conformance.py \
  tests/test_model_knobs.py tests/test_model_smoke.py tests/test_model_classifier.py \
  tests/test_session_round_trip.py -q && \
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

`tests/test_model_classifier.py:130` (`recovery >= 0.9`, chance 0.2, 200 epochs) is the only accuracy oracle in CI and the one most likely to move if the stopping policy shortens runs. Check it; do not relax it. Replace the register's wall-clock test with a work count — spy on `validation_step` and assert `n_calls == max_epochs * trainer.num_val_batches[0]` — because timing in CI is the wrong instrument for a 5× cadence change. Two further test corrections: use `data_ptr()` equality, not `is`, to assert restore preserved parameter identity (`state_dict()` returns detached views); and script the monitor with an interior minimum (`[5,4,3,9,9,9,9]`), asserting bitwise equality to the minimum check and inequality to the last, having first asserted the two differ.

**Metric outputs that move**

- Where validation ELBO plateaus then drifts: **up**, +0.004 in the recorded instance (0.4979 at epoch 2964 → 0.5020 at epoch 1464). Conditional entropies **down** correspondingly.
- Where validation ELBO improves monotonically to the cap: **exactly zero** — the restore is a no-op, and there is a test for that regime so nobody reads the fix as a uniform improvement.
- DE-2's own contribution is ~0 because DE-3 selects the same epoch either way; that is the whole reason they ship together. Alone, DE-2 would move the kept weights from epoch 2964 to ~1764 for a reason nobody chose.
- Pinning `kl_weight` and raising `val_num_particles` move the stop epoch by an amount that has not been measured. That is the risk this PR carries and the reason it is separate from PR 2: if the paired single-cell check comes back outside `[0, +6e-3]`, the two are separable within this PR and can be re-measured one at a time.

**This PR does not recover the metric optimum, and TRAINING_CONTRACT.md should say so in those words.** The measured facts have validation ELBO still improving while the gate-0 deterministic read falls 0.1931 (epoch ~60) to 0.1353 (epoch 4000). Best-ELBO restore makes early stopping mean what it says. It does not make the ELBO a proxy for metric accuracy; that is Q-A's territory.

---

## PR 5 — `metric-joint`

**Title:** Freeze the table the metrics read

**Closes:** DE-6, DE-7, DE-15.

**The DE-6 candidate decision, revised from the register: take candidate 2, not candidate 1.** Flip the default on both `tcri/tools/_common.py::joint_draws` and `tcri.tl.joint_distribution` to `use_logits=False`. Candidate 1 (metric seam only) breaks a documented identity: `tests/test_tools/test_metrics.py:41-42` builds `jd = tcri.joint_distribution(adata, covariate=cov, n_samples=0)` at the default and asserts it matches the adata path, and that identity is a contract statement — `tcri_api_and_responsibilities.md:480` says the AnnData path computes J "via §7.1 (`use_logits=True`)" and §7.9 at `:617` scopes the equivalence to `n_samples=0`. Candidate 2 preserves it. Its cost is one line: `tests/test_tools/test_joint.py:124` asserts `df.attrs["params"]["use_logits"] is True` on a default call. `tests/test_contract_conformance.py` compares `(name, kind, has_default)` and never default values, so no signature conformance breaks. Keep `use_logits=True` reachable and tested as the explicit eq-4 prediction-table path — `test_gate_aware_combine_direct` stays unchanged as the guard that eq 4 still exists for `predict`.

**Files**

- `tcri/tools/_metrics_contract.py` — new exported `JOINT_CONSTRUCTION` with four keys: `rows` (P(φ|c) is ϕ_m, eq 2, read from `uns[P_CT]`; eq 4's gated combination is the per-cell prediction rule and belongs to `predict()`), `clone_mass` (P(c) is not specified by the METRICS document; `weighted=False` is the API-contract §0.8 default, Note 1's benchmark specifies abundance weighting, and the two reverse sign along the coupling axis), `temperature`, `gate`. New `IDENTITIES["metric_joint_is_independent_of_the_gate"]`. `OPEN_QUESTIONS["clone_mass_default"]` and `["metrics_source_is_an_excerpt"]`. All four keys are required, not just the two the register's plan adds — `test_every_metric_knob_is_declared` fails on `main` for `weighted`, `use_logits`, `temperature` **and** `gate`, and cannot ship green otherwise.
- `docs/contract/METRICS_CONTRACT.md` — a "What table" section placed upstream of the definitions.
- `tcri/tools/_common.py:23-24`, `tcri/tools/_joint.py` — the default flip.
- `tcri/diagnostics/_ppc.py:29` — drop the hardcoded `use_logits=True` so the package's goodness-of-fit check validates the table the metrics read rather than one nothing consumes. Note the change in the PPC docstring.
- `tcri/diagnostics/_ppc.py:127-167` — DE-15: rename `permutation_null` → `label_permutation_test`; drop the dead `groupby` (the token appears exactly once, in the signature) and the single-valued `metric`; expose `normalized` / `normalize_mode` (it hardcodes `mode="min"` and reports NMI under a column named `observed`); rename `observed` → `observed_nmi` and add a `statistic` column; floor the p-value at `(1 + #{null ≥ obs}) / (R + 1)`; first docstring line states it tests the empirical label table and is not a null for `tl.mutual_information`.
- `tcri/_contract.pyi:17-20` and `:142-145`; `tests/test_contract_conformance.py:35`; `docs/contract/tcri_api_and_responsibilities.md` §7.1, `:480`, §7.8 at `:613` (delete the false shared-draw-stack claim), §7.9 at `:617`, plus the name occurrences at `:168`, `:211`, `:721`, `:723`, `:732`.
- `dev/live_test_rnr.py:142-144`; `tests/test_diag/test_diag.py:36`; `tests/test_tools/test_joint.py:124`.
- Docstring lines naming the estimand on `tcri/tools/_mutual_information.py:39`, `_entropy.py:106,118`, `_flux.py:51` and the four `pl.*` twins.

**Contract change:** yes — metrics manifest, metrics prose, and the API contract (DE-15's rename is breaking). Contract commits first; `tests/test_contract_conformance.py` failing on the pre-manifest commit is the forcing function.

**DE-7 is document-only.** `weighted` stays `False`. Choosing the no-movement option is what lets DE-7 share a PR with DE-6 without confounding DE-6's measured shift. Its `test_ground_truth_comparisons_use_abundance_mass` must monkeypatch `tcri.tl.mutual_information` with a recorder and call `run_cell`, not grep the source — `run_grid.py:115-116` already contains prose near those call sites.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_metrics_contract_conformance.py \
  tests/test_contract_conformance.py tests/test_tools tests/test_diag tests/test_recovery.py -q && \
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

**Metric outputs that move.** All four families and every `pl.*` twin, measured on a 47×10 Zipf fixture at gate 0.5 → 0 with an inert classifier:

| Metric | Direction | Sharp coupling | Flat coupling |
|---|---|---|---|
| `tl.mutual_information` | **up** | 0.3239 → 0.5318 | 0.0224 → 0.0701 |
| `tl.phenotypic_entropy` | **down** | 0.5931 → 0.3543 | 0.9713 → 0.9133 |
| `tl.clonotypic_entropy` | **down** | 0.7655 → 0.6242 | 0.9830 → 0.9489 |
| `tl.phenotypic_flux` | **up** | 1.2574 → 3.3512 | — |

Accuracy at `n_samples=0` goes from +0.0761 to +0.0006 against the label oracle. At `n_samples>0` the fold's attenuation stops opposing the Jensen inflation, so posterior-mean values rise by roughly the measured gap (+0.10 weak coupling, +0.017 strong) — a single-signed, attributable residual replacing a two-signed tuned one, and exactly what PR 7 removes. Every entropy and MI in any figure produced from this code changes. DE-7 and DE-15 move nothing except DE-15's own p-column, upward, bounded by `1/(R+1)`.

State plainly in the PR body: with Q-A unresolved, `tl.*` after this change is the plug-in statistic of the slowly-decaying initialisation plus Dirichlet shrinkage. That is what the model currently knows, and this PR makes it visible rather than masked.

---

## PR 6 — `generator-fidelity`

**Title:** Generator label space, temperature scaling, and the fuzziness mapping

**Closes:** DE-13, DE-14, DE-20 (new).

**Files:** `tcri/datasets/_simulate.py` only, plus new `tests/test_datasets/test_simulate.py` and `tests/test_datasets/test_temperature_scale.py`.

- DE-13: `categories=` on `phenotype` and `true_phenotype` only. Leave `clone_id` / `covariate` data-derived — declaring an unrealized *clone* level propagates into the estimator as an all-uniform prior row that adds fictitious mass to the metric joint at `weighted=False`. Record `realized_clones` / `realized_phenotypes` in `uns['tcri_truth']`.
- DE-14: log-space, support-preserving rewrite of `temperature_scale` (`:245-247`), validation on `T`, and the finiteness guard on `mi_from_joint_oracle:55` — `NaN <= 0` is `False`, so it currently proceeds and returns a structurally plausible all-zero oracle. Docstring must retract the "verbatim behaviour of `sc_simulator.temperature_scale_conditional`" claim, note that the `eps` kwarg is gone, and name the one deliberate difference: the T→∞ limit becomes uniform-on-support rather than uniform-on-all-columns.
- DE-20: `_simulate.py:322` and the matching block in `simulate_tcri` — `g = np.sqrt(fuzziness)`, citing the note (p. 11).

**Contract change:** none. `tcri/datasets` appears in no manifest, no conformance test, and no CODEOWNERS path.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_datasets tests/test_recovery.py -q && \
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

Copy DE-14's test construction throughout: write the superseded power form **inline in the test** and assert exact agreement at T ∈ {0.1, 0.5, 2.0} on a strictly-positive matrix at `rtol=1e-12`. It is the only construction in the register that proves a rewrite is a reparameterisation rather than a re-tuning.

**Metric outputs that move:** none in the package. DE-13 moves nothing on the benchmark's synthetic path (all 5 phenotypes realise at every grid size and seed); on the fit-params path a run that was losing a level gains a near-zero column, moving `tl.mutual_information` by <1e-3, **down** at `normalize_mode="average"` and unchanged at the default `"min"`. DE-14 moves nothing at the published anchors T ∈ {0.1, 0.5, 1.0}. **DE-20 moves every benchmark difficulty cell**: the K-Means init NMI declines monotonically 0.2344 / 0.2223 / 0.2011 / 0.1551 / 0.1005 instead of bunching at the top end. `true_nmi` and `empirical_nmi` do not move under any of the three — the interpolation is expression-side, and `omega`, `pi` and the realised `(z, φ)` counts are untouched. That is the first regression check on this PR, at tolerance 0.

---

## PR 7 — `benchmark-protocol`

**Title:** Benchmark provenance, protocol, and per-cell noise floor

**Closes:** DE-8, DE-9, DE-10, DE-11, DE-12.

**Files:** `benchmarks/run_grid.py`, new `tests/test_bench_harness.py`.

**Internal commit order:** DE-8 + DE-9 (instrumentation, moves nothing) → DE-10 + DE-11 (protocol) → DE-12 (floor, measured in the final regime) → one full grid re-run.

Four corrections to the register's entries, all of which change what gets written:

- **DE-10's framing.** The note says "a maximum of 2,000 epochs" — a maximum presupposes a stopping rule, and DE-10's own measurement is `epochs_actual == 2000` in all four runs because effective patience was 1500 epochs. Sequenced after PR 3, patience is real and the protocol is reproducible. State separately that under the current model the 2000-epoch read is *further* from the truth than the 60-epoch read; the conformance gain and the accuracy question are two line items and the second is Q-A's.
- **DE-11's unlisted mechanism.** `P = k_infer` means that at k_infer ∈ {8, 12} the estimate lives on a different phenotype support from the oracle, which is computed over the generator's 10 phenotypes. `ae_vs_true` at K=8 carries a structural floor and both NMI denominators change. Add `P_supplied` and `P_true` columns and name the mechanism, or the K-axis separation reads as estimator sensitivity to K.
- **DE-12's flag and memo key.** `interpretable = true_nmi > null_mean + 2*null_sd` with `null_sd` from 3 seeds is a 2-degree-of-freedom variance estimate; use `n_null_seeds >= 5`. The fuzziness-invariance that licenses memoising the model floor was measured at 60 epochs with true-label init — the regime DE-10 and DE-11 remove. Re-measure invariance in the final regime, or keep `fuzziness` in the memo key until you have.
- **Three tests must be rewritten.** DE-8 T3 passes on the defect (`epochs_run == epochs_requested == 400` satisfies `0 < epochs_run <= 400`) — force a truncation and assert the columns diverge, plus `row["epochs_run"] == model.trainer.current_epoch` from a spy in the same run. DE-11 T4 is vacuous — the fix writes `km_*` categories against `phen_*`, so "differs on >1% of cells" is 100% by construction; replace with the real invariant: permute `obs['true_phenotype']` and `obs['phenotype']`, call `_kmeans_labels` again, assert the label vectors are element-wise identical. DE-10 T6 and DE-11 T7 encode live defects as expected values and go red when PRs 3 and 4 land — replace T6 with a pure-function test of `effective_patience_epochs(patience, cv)`, and T7 with "the gate=1 read is *equal* between `phenotype_init='true'` and `'kmeans'` at the same seed", which is the attribution claim DE-11 actually needs and stays true after the classifier becomes informative.

Also: `--baseline` default `kmeans` → `gmm`. The note specifies GMM, and after DE-11 a K-Means baseline is numerically identical to TCRi's own initialisation.

**Contract change:** none. `benchmarks/` appears in no manifest.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_bench_harness.py -q && \
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

**Metric outputs that move:** nothing in the package; every number in the benchmark. `tcri_nmi` **falls** by an amount that grows with fuzziness (−0.002 at f=0.1, −0.070 at f=0.9 from DE-11; a further −0.005 to −0.013 from DE-10). `ae_vs_true` falls at high fuzziness (0.117 → 0.045) — report that as a **cancellation, not accuracy**: the K-Means crosstab under-reads the truth (0.1294 vs 0.1816) while the estimator over-reads by ~+0.06. `ae_vs_empirical` moves more and stops being near-zero by construction. The MAE-vs-fuzziness curve goes from flat (range 0.0014–0.0026) to sloped (0.062–0.067). The K axis begins to exist. `true_nmi` and `empirical_nmi` do not move. Any benchmark figure already produced is superseded, not adjusted.

---

## PR 8 — `guide-concentration`

**Title:** Free the guide concentration magnitude

**Closes:** DE-5. **Held until Q-A is answered.** Placed last so a hold does not stall anything else; its only ordering constraint is that it lands after PR 4.

**Files:** `tcri/model/_model_contract.py` (GUIDE_SITES notes for `p_c`/`p_ct`, `GUIDE_PARAMS` gains `q_p_c_mag` / `q_p_ct_mag`, one new invariant), `docs/contract/MODEL_CONTRACT.md`, `docs/contract/METHODS_CONFORMANCE.md:47-48,115-150` (close deviation [I], delete the two false arrow lines), `tcri/model/_module.py` (guide, ~8 lines), `tcri/model/_model.py::to_anndata` (`uns["tcri_conc_ct"]`), `tcri/tools/_joint.py:83-89` and `tcri/_compute/_joint.py:100-106` (accept the concentration vector, scalar fallback so older AnnData objects still load), `tests/test_model_knobs.py:223-239`, `benchmarks/run_grid.py:115-116,207` (the prose "the TOTAL Dirichlet concentration on p_ct" becomes false).

**Contract change:** yes, manifest first, code second — `test_guide_registers_variational_params` failing on the pre-manifest commit is the forcing function.

**Two versions.** If Q-A removes the detach at `_module.py:241`, this is candidate A as written, and the engine plumbing is **not optional** — without it the metric keeps drawing from the prior-set width regardless of what the guide learned. If Q-A keeps the detach, this ships as a **conformance-only** change and the manifest must say so: the register's `SEMANTIC_INVARIANTS["guide_concentration_magnitude_is_free"]` and its `max/min sum(conc_ct) > 1.05` test would then pass only by non-convergence and would reject a correct fix, because with ϕ detached the only ϕ-bearing term's argmax over a free λ′ is β·ω — total β again. Replace with the structural invariant: `torch.autograd.grad(Trace_ELBO().differentiable_loss(...), store._params["q_p_ct_mag"])` is finite and non-zero. That fails on `main` (the parameter does not exist), fails if the parameter is registered but detached from the objective, and passes for a correct fix under either answer. Ship `test_posterior_concentrates_with_cells_per_group` as `xfail(strict=True)`.

**Test command**

```
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_model_contract_conformance.py \
  tests/test_model_knobs.py tests/test_recovery.py -q && \
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q
```

**Metric outputs that move:** at `n_samples=0`, **unchanged by construction** — `get_p_ct()` returns the Dirichlet mean under both parameterisations. At `n_samples>0` the posterior-mean NMI **falls** by roughly the Jensen gap (−0.10 weak coupling, −0.017 strong) and HDI widths narrow for large groups, stopping being identical across a 100× cell-count range. That last property is what to judge the fix by, not the NMI error. If the detach stays, essentially nothing moves and the entry must say so.

---

# Benchmark grid re-runs

The grid is expensive and PR 6 changes it. Three runs total.

**Run A — reference baseline, `published_quick`, immediately after PR 1 merges.** `--preset published_quick`, 60 epochs, current protocol, seeds fixed. **Its numbers are not a result and must never be quoted**: the MAE table pools across temperature, the difficulty axis is inert, the 60-epoch reads sit within 0.0005 of their own initialisation, and there is no interpretability flag. It is worth running for exactly one reason — it is the first grid output in the project's history that is bit-reproducible, so it is a diff target. Every subsequent PR is checked against it cell-by-cell rather than against a remembered number. Cost: 27 cells at ~3 s each.

**Is a full `published` run before PR 6 worth anything? No.** It would cost 4500 cells to produce a table whose headline aggregation is wrong, whose sweep axis does not move, and whose per-cell interpretability is unknown. Do not run one.

**Run B — full `published`, after PR 6 merges.** This is the first run that reproduces the note's design: 2000-epoch budget, K-Means initialisation, real K axis, per-axis summary, per-cell noise floor, recorded epochs. It is the run that supersedes every existing benchmark figure. Budget it as the single expensive event in the plan (~35× the old per-cell cost from DE-10, +3% from DE-12's amortised null fits).

**Run C — required, after PR 3.** Q-A is answered and DE-18 is PR 3, so this is no longer
conditional. PR 3 gives `p_ct` a data term and invalidates every baseline recorded before it;
Run B is not comparable across it. Re-baseline immediately after PR 3 merges, before PR 4.

Between runs, every number-moving PR (2, 3, 4) is checked with a **single fixed cell**, not a grid: `(seed=0, n_cells=1000, temperature=1.0, fuzziness=0.1)`, run pre and post on the same commit pair, recording `tcri_nmi`, actual epochs trained (`model.trainer.current_epoch`, not the requested `max_epochs`), and the best epoch. That is ~6 s per side and is the only comparison that can attribute a delta to one PR.

---

# Probe re-runs

Four probes, each with a stated trigger and a required rewrite.

**Gate probe** (shipped gate vs gate=0 vs gate=1, deterministic and posterior reads against the label oracle). Its existing readings were taken at 4000 epochs and at 60 epochs, both of which the register shows are the wrong place to read — 60 is the initialisation, 4000 is past the degradation. **Re-run at the head of PR 4, against a PR-3 model**, i.e. at the restored best-`elbo_validation` state with `training_record_['best_epoch']` recorded alongside. Re-run again immediately after PR 4 to record the post-fix error. Expected: the +0.076 metric-path error at the shipped gate collapses to +0.0006 at `n_samples=0`.

**Metric-scale probe** (NMI at `normalize_mode` min vs average, weighted vs unweighted, against the plug-in and label oracles). Same trigger and same reason: it was read at an epoch count where the estimate had not left its initialisation. **Re-run at the head of PR 4.** It is also the probe that produces DE-7's sign-reversal evidence in the post-PR-5 fuzziness regime, so re-read it once more after PR 5.

**Overtrain probe** (metric read vs epoch, against validation ELBO). **Rewrite before PR 2 lands, as a `lightning.pytorch.Callback` inside a single `train()` call.** The current construction — repeated `train(max_epochs=k)` calls with increasing `k` — is invalid on `main` for a reason the register documents (DE-4: each call restarts the KL ramp from 1e-6, so the model sees a sawtooth) and stays invalid after PR 2 for three further reasons: each call rebuilds the `Trainer`, rebuilds the dataloader, and discards Adam's momentum. The callback records, at every validation check: `epoch`, `module.kl_weight`, `elbo_validation`, `get_p_ct()`, the gate-0 deterministic read, the shipped read, and the label oracle. Re-run after PR 2 (to confirm the read-only validation changed the trajectory in the predicted direction and by the predicted amount) and after PR 3 (to see where the restore lands on that curve). This probe is also the evidence for Q-A, since it is what shows `p_ct` relaxing away from the crosstab while validation ELBO improves.

**KL-schedule probe** (`kl_weight` vs step vs epoch, at several `(n_cells, batch_size)`). Same rewrite, same reason. Re-run after PR 2 to confirm the counter is per-model and the ramp is continuous across a `reset_schedule=False` second call, and to record the epoch equivalents that go into `MODEL_CONTRACT.md`.

Park the rewritten probes in `dev/probes/` with the seeds baked in, so each re-run is a diff rather than a re-derivation.

---

# Regression checkpoints

Each checkpoint names what is compared against what. A silent change is caught between PRs, not at the end.

**After every PR, without exception**

- `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q` — count as well as status. The baseline is 177 passed / 3 skipped; a PR that adds tests must state the new count in its body, and a *drop* in collected tests is a finding regardless of green.
- All three (later four) conformance tests green. A conformance test that had to be edited to go green is a contract change and belongs in the PR's contract commit, not in its code commit.

**PR 1**

- Two `TCRIModel(seed=s)` constructions plus fits in one process: every Pyro-store tensor and `get_p_ct()` compare `torch.equal`. Then two *different* seeds must differ, so the check cannot pass on a degenerate model.
- `run_cell` called twice at identical arguments: `row["tcri_nmi"]` compares `==`. Until this holds, no benchmark delta in this plan is measurable.

**PR 2**

- Param-store tensors bitwise identical across a validation loop; `len(training_step_outputs)` unchanged across one. Both fail on the parent commit (drift 0.12–0.14, 5 → 6).
- Trajectory invariance: `max_epochs=6` with `check_val_every_n_epoch=1` versus `limit_val_batches=0`, `get_p_ct()` compares `torch.equal`.
- Single-cell paired delta against Run A: `ΔNMI` in `[0, +2e-3]`. Expected +6e-4. Negative or outside that band means the fix removed something other than the perturbation.
- `module.kl_weight` sequence for one seeded single `train()` matches `[max(1e-6, kl_max*i/K)]` element-wise — the proof DE-4 contributed nothing.
- Site-set identity between the training and validation traces, `"phenotype_alignment"` present. This is the one silent failure mode of the PR.

**PR 3**

- Post-`train()` state bitwise equal to the scripted-minimum check and unequal to the last.
- `kl_weight == kl_weight_max` and `module.training is False` at every logged check.
- `best_epoch < last_epoch` on at least one benchmark-shaped cell. If `best_epoch == last_epoch` everywhere, early stopping is inert and the restore is a no-op for the wrong reason — check the patience arithmetic before believing the delta.
- Single-cell paired delta against the PR-2 head: `ΔNMI` in `[0, +6e-3]`. A negative delta on a plateauing cell means the monitored objective and the metric are anti-correlated there — record it as a finding, do not loosen the check.
- `tests/test_model_classifier.py::recovery` still ≥ 0.9.

**PR 4**

- `tl.mutual_information` bitwise identical across `uns[GATE_PROB] ∈ {0.0, 0.5, 1.0, NaN}`, at **both** `n_samples=0` and `n_samples>0` with a fixed `random_state`. These four cannot coincide on the parent commit: `_compute/_joint.py` treats NaN as `use_gate=False` → `ell + log_b`, distinct from `gate=0.0` → `log_b`.
- `test_mutual_information_scalar_and_fastpath` green. If it is red, candidate 1 was taken without the §7.9 prose work.
- `diag.joint_distribution_ppc`'s `mean_distance` **changed**. If unchanged, `_ppc.py:29` was missed and the goodness-of-fit check is validating a table nothing consumes.
- `tests/test_recovery.py::test_posterior_hdi_covers_the_truth` still covers 8/8. This is what makes PR-4-before-PR-7 the safe order; confirm before merging.
- `test_every_metric_knob_is_declared` green — meaning `temperature` and `gate` were declared, not just `rows` and `clone_mass`.

**PR 5**

- `true_nmi` and `empirical_nmi` bitwise identical across the DE-20 edit, tolerance 0. If they move, the interpolation touched the label-generating path and the whole difficulty axis is compromised.
- `temperature_scale` agrees with the inline old power form at T ∈ {0.1, 0.5, 2.0} to `rtol=1e-12`, and the `nmi_min` at the published anchors matches to `rtol=1e-9`.
- No declared-but-unrealised *clone* level in `obs['clone_id'].cat.categories`.

**PR 6**

- `epochs_run < epochs_requested` on at least one cell. If they are equal everywhere, PR 3's patience arithmetic did not survive and "a maximum of 2,000 epochs" is still a fixed-length fit.
- `init_crosstab_nmi` declines along fuzziness under `phenotype_init='kmeans'`. Flat means either `_kmeans_labels` is not being used or DE-20 did not land.
- `module.P == k_infer` at k_infer ∈ {8, 10, 12}.
- `interpretable` neither all-True nor all-False across the grid; and the null read with the run's own `normalize_mode`, since `diag`'s version hardcodes `"min"` at `_ppc.py:152`.
- `null_model_mean > null_plugin_p95` on every cell. If not, either the estimator's inflation has vanished (a real result worth being told about) or the model floor is computed on the wrong object.
- The Pyro param store empty at entry to each real fit after a null fit — `_model.py:131-141` only *warns* on a dirty store.
- `true_nmi` / `empirical_nmi` bitwise identical to Run A on every shared cell, tolerance 0.

**PR 7**

- HDI widths no longer identical across a 100× cell-count range. If they are, the engine plumbing was skipped and the metric is still drawing from the prior width.
- Concentration at init equals `clamp(β · q_p_ct_sharp, 1e-3)` to 1e-12, so the change is a reparameterisation and the before/after is valid.
- `get_p_ct()` unmoved at fixed `q_p_ct_raw`.

---

# Open questions gating the stack

Recorded in `docs/contract/DEFECTS.md` by PR 1. Q-A gates PR 7 and re-scopes PR 6's expectations; Q-B and Q-D are recorded in the training contract by PR 2; Q-C is recorded in the metrics manifest by PR 4. None of them is inferred from what makes the code come out right.

**Q-A — ANSWERED 2026-08-07: condition on the observed phenotype; DE-18 is PR 3.** Original statement of the question: `tcri/model/_module.py:241` is `phi = p_ct[ct_idx].detach()`. `p_c` appears in `model()` only at `:208` and `:212`; `p_ct` only at `:214` and `:241`; `_target_phenotypes` is registered at `:132`/`:160` and read nowhere in `tcri/`. So the only forces on `q_p_ct_raw` are the two prior KLs, and the classifier's only training term targets a detached constant, minimised by any phenotype-constant logit vector. That single fact accounts, without further hypotheses, for the optimum at 30–120 epochs, the decay to 0.1353 by 4000, `p_ct`'s L1 to the crosstab growing to 0.515 while validation ELBO improves, and gate=1 reading 0.0000. Is the detach intended as a stop-gradient on the alignment target, and was `_target_phenotypes` meant to enter the objective?

**Q-B — "KL scaling for Dirichlet and discrete terms" (Note 1, §Inference Details).** The note's only sentence about optimization, and the code does not do it under either reading. `pyro.plate("data", batch_size)` at `_module.py:216` and `:337` carries no `size=`/`subsample=`, so the C+M global Dirichlet terms enter once per minibatch: measured `|global|/|local|` = 0.0255 at batch 1024 versus 0.0064 at batch 4500, a 4× change in the priors' relative weight from a batch-size choice. Under the annealing reading, the note anneals the Dirichlet and discrete terms while the code anneals z, which is the inverse of `SANCTIONED_DEVIATIONS['kl_warmup_z_only']`. Either reading is a model change; invariant I6 ships `xfail(strict=True)` until it is answered.

**Q-C — `clone_mass` default, and the archived metrics source.** The archived METRICS document begins mid-document at the Entropy section, so the section defining p(c,φ) is not in the repo while `SOURCES['METRICS']['owns']` claims "eqs 2–7". Note 1's benchmark specifies abundance weighting; the shipped default is `weighted=False` on a recorded §0.8 decision. PR 4 assumes they stay divergent.

**Q-D — the weight-decay deviation text.** `SANCTIONED_DEVIATIONS['optimizer_weight_decay']` says decay applies to the network parameters. It also reaches `q_p_c_raw` and `q_p_ct_raw`, whose param-store leaves are `log θ`, so it is a pull of every clone row toward the uniform simplex point — a prior on `q(ϕ_m)` with strength 1e-4 and no equation number. PR 2 corrects the text. Whether the default should become `guide_weight_decay=0.0`, and whether to move to `ClippedAdam`, is a separate measured change and is deliberately not in this stack.