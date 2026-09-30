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
  conformance test (site set, factor sign). The training plan declares the estimator SVI
  descends: Pyro's `Trace_ELBO` averaged over `train(num_particles=...)` draws of the latents
  (default 1, drawn in sequence). The model has no discrete latent, since the phenotype in the
  label readout is summed out in closed form, so there is nothing to enumerate. The validation
  criterion uses one draw whatever the count (I3). `test_svi_steps_on_the_plans_loss`,
  `test_num_particles_averages_that_many_draws`.
- **I2 No optimizer update outside `training_step`.** Validation evaluates and never steps.
  The best-weight restore at `on_fit_end` applies no gradient and consumes no validation batch.
  `test_validation_does_not_update_parameters`, `test_validation_step_does_not_call_training_step`.
- **I3 The monitored quantity is a fixed objective.** `objective_validation_percell` is the
  per-cell block only, `latent` + `phenotype_alignment` + `phenotype_label` + `obs`, over the
  validation split, divided by the plate size; evaluated at `max_kl_weight`, in eval mode,
  with a fixed particle count, the same split every check, and a forked fixed RNG seed. The
  global sites `p_c` and `p_ct` are excluded because their KL is identical whatever is held out.
  It is not an ELBO and is not called one. `test_monitor_is_invariant_to_ramp_position`,
  `test_monitor_excludes_the_global_block`, `test_validation_pin_restores_the_training_schedule`.
- **I4 The reported model is the selected one.** A best-by-monitor snapshot at each gated
  check, restored in place at `on_fit_end`, spanning `state_dict()` (BatchNorm running
  statistics included) and the Pyro param store through `named_parameters()` in unconstrained
  space, restricted to the module's OWN namespace: with a namespace per model the store holds
  other fits' concentrations too, and restoring those would write another fit's parameters over
  this one's. `test_restored_model_is_the_selected_one`.
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
  reaches `max_kl_weight` in finite steps. `train()` has no reset knob; construct a new model
  for a fresh schedule. `test_no_reset_knob_on_train`.
- **B2** `n_steps_kl_warmup` counts optimizer steps; a run records its epoch equivalent.
- **B3** Patience is in epochs: `train()` validates once per epoch and the knob is
  `train(early_stopping_patience=...)`, default 150. `early_stopping_patience` checks in a row
  without an improvement stop the fit (B4 defines one). `early_stopping=False` installs no
  stopping rule and trains for `max_epochs`; the best check is restored either way (I4).
  `test_fit_stops_once_improvements_fall_below_min_delta`, `test_restored_model_is_the_selected_one`.
- **B3a** `max_epochs` defaults to 2000. With early stopping on, a fit that reaches the cap
  warns and records `stopped_early: False`.
- **B4** A check is an improvement only if it is below the best check so far by more than
  `train(early_stopping_min_delta=...)`, in the monitor's own units (per cell).
  `early_stopping_min_delta` must exceed the monitor's noise; the fixed evaluation seed removes
  the Monte-Carlo component, so what remains is the epoch-to-epoch movement of the fit. The
  defaults come from fits of `tcri.datasets.simulate_tcri(n_clones=1000, n_phenotypes=6,
  n_cells=13700, n_covariates=6, omega_concentration=0.4, fuzziness=0.1, seed=0)` at 150, 500
  and 2000 genes, a parent and a phenotype null at each and a second parent seed at 500, at
  `train()` defaults with stopping off for 3000 epochs. Noise is 1.4826 x MAD of the residuals
  from a 101-check centered rolling median on the post-ramp plateau. It did not grow with the
  monitor's magnitude, so the threshold is absolute, and `early_stopping_min_delta` defaults to
  0.05, twice the largest noise, rounded up. `early_stopping_patience` is the smallest of 25,
  50, 100, 150, 200, 300 under which every one of those fits selects a checkpoint within its
  noise of the one a threshold of 0 with patience 300 selects.
  `test_fit_stops_once_improvements_fall_below_min_delta`,
  `test_a_null_stops_under_its_parents_rule`.
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
  decay on their log-space leaves is a flat Dirichlet prior applied through the optimizer. The
  exemption matches the parameter's TAIL under any namespace (`x.q_p_ct_raw` as well as
  `q_p_ct_raw`); an exact match would silently reinstate that prior for every named model.
  `test_weight_decay_does_not_reach_the_guide_concentrations`.
- **B9** A fit records provenance in `training_record_`: epochs actually run, warmup steps and
  their epoch equivalent, `ramp_completes_at_epoch`, `ramp_completed`, `selection_criterion`,
  `selected_epoch`, `stopped_early`, `seed`; `kl_weight` is logged per epoch. `stopped_early`
  is true only when the stopping rule ended the fit; a fit ended by another limit, such as
  `max_steps`, records false and warns. `test_a_fit_ended_by_another_limit_is_not_an_early_stop`.
- **B10** A permutation null is fitted with the parent's knobs, the parent's seed and the
  parent's train/validation split; only the permutation draws from a stream of its own, keyed by
  the kind so that two kinds never share one. `train()` records the arguments it actually ran
  with, the stopping arguments included, and a null replays them, because a null fitted at
  `train()`'s defaults is not the parent's model on permuted labels. The permutation is stored beside the fit it produced. A null's
  parameter namespace is its own, so I4 and B8 apply per namespace and neither the parent nor
  any other null is touched by its fit. `tests/test_nulls.py`.
- **B11** No argument is accepted and then ignored. `train()` raises for a Trainer argument
  this model replaces (`early_stopping_warmup_epochs`, `early_stopping_monitor`,
  `early_stopping_mode`, `trainer_config`, `learning_rate_monitor`, `enable_checkpointing`,
  `checkpointing_monitor`) and for `check_val_every_n_epoch` other than 1 or
  `val_check_interval` other than 1.0, which would change what patience counts. Every argument
  scvi's Trainer adds to Lightning's is either forwarded with its plain meaning, set by
  `train()`, or raises. The constructor raises for an unknown name; a saved record's
  `kl_weight_max` maps to `max_kl_weight`, and its `patience_epochs`/`patience` are dropped,
  both with a warning; its `use_enumeration` is dropped, with a warning only when it is true.
  `test_train_rejects_arguments_it_replaces`,
  `test_every_scvi_trainer_argument_is_forwarded_or_replaced`,
  `test_an_unknown_constructor_argument_raises`, `test_a_saved_use_enumeration_is_dropped`.

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
