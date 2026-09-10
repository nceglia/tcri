# Training contract

How the model is fit. Enforced by `tests/test_training_invariants.py`, behaviourally: every
statement below names a test that exercises the behaviour, not the wiring. Nothing else binds
the training plan.

Two kinds of statement. **Invariants** follow from the objective in `MODEL_CONTRACT.md`; a
violation is a defect. **Bounds** are ours, because the objective says nothing about
schedules or stopping; they change with a recorded reason.

## Invariants

- **I1 One objective.** The only quantity any optimizer descends is −L of `MODEL_CONTRACT.md`
  eq 7: the ELBO with the label readout and the alignment surrogate. Enforced by the model
  conformance test (site set, factor sign).
- **I2 No optimizer update outside `training_step`.** Validation evaluates and never steps.
  The best-weight restore at `on_fit_end` applies no gradient and consumes no validation batch.
  `test_validation_does_not_update_parameters`, `test_validation_step_does_not_call_training_step`.
- **I3 The monitored quantity is a fixed objective.** `objective_validation_percell` is the
  per-cell block only, `latent` + `phenotype_alignment` + `phenotype_label` + `obs`, over the
  validation split, divided by the plate size; evaluated at `kl_weight_max`, in eval mode,
  with a fixed particle count, the same split every check, and a forked fixed RNG seed. The
  global sites `p_c` and `p_ct` are excluded because their KL is identical whatever is held out.
  It is not an ELBO and is not called one. `test_monitor_is_invariant_to_ramp_position`,
  `test_monitor_excludes_the_global_block`, `test_validation_pin_restores_the_training_schedule`.
- **I4 The reported model is the selected one.** A best-by-monitor snapshot at each gated
  check, restored in place at `on_fit_end`, spanning `state_dict()` (BatchNorm running
  statistics included) and the Pyro param store through `named_parameters()` in unconstrained
  space. `test_restored_model_is_the_selected_one`.
- **I5 Annealing is schedule-only.** The warmup counter lives on the module, so a resumed
  `train()` continues the ramp rather than restarting it.
  `test_kl_ramp_is_monotone_across_resumed_training`, `test_warmup_counter_is_owned_by_the_module_not_the_plan`.
- **I7 A declared knob changes an observable.** `tests/test_model_knobs.py`; partial.
- **I8 A minibatch is an unbiased estimate of eq 7.** The data plate is declared at the size
  of the training split with the batch as its subsample, so the mean of the batch objectives
  over a partition of the cells equals the full-batch objective.
  `test_minibatch_objective_is_unbiased_for_the_full_batch`, `test_plate_size_tracks_the_training_split`,
  and `test_data_plate_is_scaled_to_the_dataset` in the model conformance test.

## Bounds

- **B1** The `kl_weight` schedule is non-decreasing within and across `train()` calls and
  reaches `kl_weight_max` in finite steps. `train()` has no reset knob; construct a new model
  for a fresh schedule. `test_no_reset_knob_on_train`.
- **B2** `n_steps_kl_warmup` counts optimizer steps; a run records its epoch equivalent.
- **B3** Patience is in epochs: `check_val_every_n_epoch=1` and the knob is `patience_epochs`
  (`patience` is a deprecated alias).
- **B3a** `max_epochs` defaults to 2000. A fit that reaches the cap warns and records
  `stopped_early: False`.
- **B4** `min_delta` must exceed the monitor's noise; the fixed evaluation seed removes the
  Monte-Carlo component.
- **B5** Selection begins only after the ramp completes, read from one counter by both the
  stopping and the snapshot callback. If the ramp never completes: warn, do not raise, and
  record `selection_criterion = "last epoch (ramp incomplete)"`.
  `test_selection_is_gated_until_the_ramp_completes`.
- **B6** Every advertised knob has a behavioural test, never a wiring check.
- **B7** A fit is a function of (seed, data, knobs) and of nothing that ran before `train()`.
  `train()` forces train mode, because an earlier `predict()`, `to_anndata()`, session load,
  or scvi's own end-of-fit `eval()` otherwise leaves dropout off and BatchNorm frozen for the
  whole fit, reproducibly. `test_train_resets_module_mode`, `tests/test_model_determinism.py`.
- **B8** Optimizer settings that act as priors are declared or removed. Weight decay reaches
  the network parameters only; the two guide concentrations receive `weight_decay=0`, since
  decay on their log-space leaves is a flat Dirichlet prior applied through the optimizer.
  `test_weight_decay_does_not_reach_the_guide_concentrations`.
- **B9** A fit records provenance in `training_record_`: epochs actually run, warmup steps and
  their epoch equivalent, `ramp_completes_at_epoch`, `ramp_completed`, `selection_criterion`,
  `selected_epoch`, `stopped_early`, `seed`; `kl_weight` is logged per epoch.

## The stopping policy in one paragraph

Annealing is a continuation method: training descends a family of surrogates whose endpoint
is the objective meant. A series of different functions has no argmin, so the selection
criterion must be a fixed function of the parameters and the held-out data (I3), selection may
only start once every check comes from the same endpoint (B5), and early stopping has two
outputs, a stop time and the argmin weights, so the selected weights are restored (I4).

## Two ways to get the restore silently wrong

Restore `state_dict()`, not `named_parameters()`: the encoder and VampPrior carry BatchNorm
running statistics, which are buffers that `predict()` reads in eval mode. Restore the Pyro
store through `named_parameters()` in unconstrained space, never `items()`: the two guide
concentrations are positive-constrained, so `items()` yields a non-leaf transform output and a
`.data.copy_()` on it does nothing, without error. Do not use `ParamStore.set_state()`; it
rebinds the store away from the module that registered the parameters.
