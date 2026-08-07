"""Frozen properties of the TRAINING PLAN (the *training contract*).

The model structure is specified by Supplementary Note 1 and machine-checked by the model
contract. The training plan was not specified at all — no contract, no bounds — and it has
large effect on every reported number. That absence is why the same defects kept being
rediscovered from new symptoms: DE-1 through DE-4 are four faces of one unspecified subsystem.

**The asymmetry is in the source document, not just in our contracts.** Note 1 gives the model
(eqs 1-5), the variational family (eq 6), the ELBO (eq 7) and the surrogate, but on optimization
says only: SVI in Pyro, mini-batching, KL scaling for Dirichlet and discrete terms, Adam. No
epochs, patience, warmup schedule or stopping criterion. So this contract has two halves that
carry different authority:

``DERIVED_INVARIANTS``
    Follow from eq 7. Not matters of taste, and not changeable without changing what the
    package claims to optimize. A violation is a defect.

``AUTHORED_BOUNDS``
    Properties any valid run must have, authored here because the note does not specify them.
    Deliberately NOT hyperparameter values — not "lr must be 1e-3" (that is tuning) but
    "a declared knob must change an observable". Changeable by @nceglia / @salehis with a
    recorded reason.

A bound must be BEHAVIOURAL. The knob test verified that ``patience`` *arrives* at
``EarlyStopping.patience`` and marked it ✅, while patience was silently counted in validation
checks (300 x check_val_every_n_epoch=5 = 1500 epochs). Wiring-only verification is this
project's dominant defect class; asserting that a value is connected is not asserting that the
behaviour is right.

Checked by ``tests/test_training_invariants.py`` (behaviour) and
``tests/test_training_contract_conformance.py`` (this manifest).
"""
from __future__ import annotations

__all__ = ["DERIVED_INVARIANTS", "AUTHORED_BOUNDS", "OPEN"]


#: Consequences of eq 7 + the surrogate. Each carries its enforcement status.
DERIVED_INVARIANTS = {
    "I1_single_objective": {
        "statement": (
            "The only quantity any optimizer descends is -(L# + gamma * sum_i "
            "KL(probs_i || phi_g(i))) -- eq 7 with the z^phi terms replaced by the surrogate."
        ),
        "status": "holds",
        "enforced_by": "tests/test_model_contract_conformance.py (factor sign, site set)",
    },
    "I2_no_optimizer_update_outside_training_step": {
        "statement": (
            "No OPTIMIZER update happens outside training_step, and only on training-split "
            "batches. Validation evaluates; it never steps.\n"
            "\n"
            "The word 'optimizer' is load-bearing and was added when I4 was resolved. The "
            "best-weight restore writes parameters at on_fit_end, which the earlier wording "
            "('no parameter is updated outside training_step') forbade. The thing that must "
            "never happen is a gradient step on non-training data; deterministically writing "
            "back an already-selected snapshot is not that. Any future restore, swap or "
            "re-initialisation must still satisfy: it applies no gradient, it consumes no "
            "validation batch, and it is a pure function of state already computed."
        ),
        "status": "holds (DE-1 fixed)",
        "enforced_by": "tests/test_training_invariants.py::test_validation_does_not_update_parameters",
        "history": (
            "validation_step called super().training_step() -> SVI.step() -> the Pyro "
            "optimizer. Lightning zeroes .grad on the LightningModule's parameters before "
            "validation so the networks were spared, but q_p_c_raw/q_p_ct_raw are not "
            "LightningModule parameters, kept Pyro's zeroed grad, and were stepped on "
            "weight_decay*theta in the unconstrained log space of a positive-constrained "
            "parameter -- pulling every clone row toward uniform. Measured 0.54 L1 per check."
        ),
    },
    "I3_monitored_quantity_is_a_fixed_objective": {
        "statement": (
            "The quantity early stopping monitors is a FIXED function of (Lambda, Theta) and of "
            "held-out data. Fixed means all five of:\n"
            "  1. kl_weight pinned to module.kl_weight_max, not inherited from the last "
            "     training step;\n"
            "  2. module in eval mode;\n"
            "  3. a fixed particle count;\n"
            "  4. the same validation data every check;\n"
            "  5. a FIXED evaluation RNG seed -- the same draws every check.\n"
            "\n"
            "Clause 5 is not pedantry. A Monte-Carlo estimator that redraws each check is not a "
            "function of the parameters at all, so an argmin over it is an argmin over noise. "
            "Without this clause the invariant would certify a lottery.\n"
            "\n"
            "WHY THIS IS DERIVED AND NOT AUTHORED. Annealing is a continuation method: training "
            "descends a family of surrogates L_beta whose endpoint L_(kl_weight_max) is the "
            "objective actually meant. The function descended and the criterion selected on are "
            "therefore different objects. A series of DIFFERENT functions has no argmin to take. "
            "This follows from what 'argmin' means, not from taste."
        ),
        "status": "SPECIFIED -- the criterion is decided and written below; the code does not "
                  "implement it yet. Monitor will be `objective_validation_percell`, evaluated "
                  "at kl_weight_max under a forked, fixed RNG, in eval mode, on the validation "
                  "split. Moves to 'holds' only when the enforcing test exists and passes.",
        "enforced_by": "pending -- tests/test_training_invariants.py::"
                       "test_monitor_is_invariant_to_ramp_position",
        "history": (
            "validation_step deliberately did not set kl_weight, so elbo_validation inherited "
            "whatever the last training batch left. The stated reason was that this keeps "
            "elbo_validation on the same scale as elbo_train. That property was never real: the "
            "two are computed on different splits and, per SCOPE below, on different site sets."
        ),
        "scope": (
            "The monitor covers the PER-CELL block only -- `latent` + `phenotype_alignment` + "
            "`obs` -- summed over the validation split and divided by N_val.\n"
            "\n"
            "The global sites p_c and p_ct are EXCLUDED, and this is a deliberate departure: it "
            "means the monitored number is NOT the ELBO and must never be called one. The "
            "reason is that both global plates are declared at full size with no subsampling "
            "(_module.py `pyro.plate('clonotypes', c_count)` and `pyro.plate('ct_plate', "
            "ct_count)`), so their KL contributes the SAME value regardless of which cells are "
            "held out. Including them means selecting partly on how well the guide matches its "
            "prior on the TRAINING data -- a quantity a validation criterion must not contain. "
            "Their share of the total is small at benchmark scale (tens of clones) and large at "
            "repertoire scale, because real repertoires are singleton-dominated; making that "
            "number stationary would have fixed the well-posedness of the wrong quantity.\n"
            "\n"
            "This is NOT the reconstruction-only shortcut. `phenotype_alignment` scores held-out "
            "cells against phi = p_ct[ct_idx], so the Dirichlet branch is still covered by the "
            "criterion; what is dropped is only the prior-matching term that held-out data "
            "cannot speak to.\n"
            "\n"
            "The global block is logged as its own diagnostic series and never monitored."
        ),
    },
    "I4_reported_model_is_the_selected_one": {
        "statement": (
            "The parameters left in the store when train() returns are the ones the declared "
            "selection criterion chose.\n"
            "\n"
            "Early stopping has TWO outputs: a stop time and the argmin weights (Prechelt, "
            "'Early Stopping -- But When?'). A run that stops at the argmin and keeps the last "
            "weights has implemented half of it and is not doing early stopping.\n"
            "\n"
            "A snapshot must carry BOTH sources or it restores a model no check ever evaluated:\n"
            "  - module.state_dict() -- and state_dict(), NOT named_parameters(): the encoder "
            "    and VampPrior carry BatchNorm running statistics (FCLayers defaults "
            "    use_batch_norm=True), which are buffers, absent from named_parameters(), and "
            "    read by predict()/get_latent_representation() in eval mode. Measured on a "
            "    minimal fixture: 6 running buffers, 21 state_dict keys absent from "
            "    named_parameters().\n"
            "  - the Pyro param store -- q_p_c_raw and q_p_ct_raw are NOT in state_dict() at "
            "    all, and they are what every metric reads."
        ),
        "status": "SPECIFIED -- the mechanism is decided and written below; the code does not "
                  "implement it yet. Best-by-monitor snapshot at each gated improving check, "
                  "restored in place at on_fit_end across BOTH state_dict and the param store. "
                  "Moves to 'holds' only when the enforcing test exists and passes.",
        "enforced_by": "pending -- tests/test_training_invariants.py::"
                       "test_restored_model_is_the_selected_one",
        "param_store_hazard": (
            "The param store MUST be snapshotted and restored through "
            "store.named_parameters() in UNCONSTRAINED space, never through store.items().\n"
            "\n"
            "q_p_c_raw and q_p_ct_raw carry constraints.positive, so store.items() yields a "
            "NON-LEAF ExpTransform output. Verified by direct experiment: writing 5.0 via "
            "`.data.copy_()` on that tensor leaves the store still reading 2.0, with no error "
            "and no warning. A restore written the obvious way therefore silently does nothing "
            "-- the same class of failure as DE-1, where a real defect hid behind an operation "
            "that looked like it worked.\n"
            "\n"
            "Snapshotting constrained values and restoring them as unconstrained (or the "
            "reverse) inflates every clone row by exp(.). Pick one space -- unconstrained -- and "
            "assert the round trip in a test.\n"
            "\n"
            "Do not use ParamStore.set_state(): it rebinds _params[name], desyncing the store "
            "from the nn.Module that registered it. Copy into the existing leaves instead."
        ),
    },
    "I5_annealing_is_schedule_only": {
        "statement": (
            "kl_weight is a schedule, not a model parameter. A quoted ELBO or posterior comes "
            "from a run that reached kl_weight_max."
        ),
        "status": "holds (DE-4 fixed: the counter is on the module, so a resumed fit continues "
                  "the ramp rather than restarting it)",
        "enforced_by": "tests/test_training_invariants.py::test_kl_ramp_is_monotone_across_resumed_training",
    },
    "I7_declared_knobs_change_an_observable": {
        "statement": (
            "A knob the API advertises reaches its object AND changes something measurable. "
            "Arriving is not sufficient."
        ),
        "status": "partial -- tests/test_model_knobs.py covers most knobs behaviourally; those "
                  "with a '--' in its behavioural column are wiring-only and are the gap.",
        "enforced_by": "tests/test_model_knobs.py, tests/test_shared_defaults.py",
    },
}


#: Authored, because the note does not specify them. Bounds, not values.
AUTHORED_BOUNDS = {
    "B1_monotone_terminating_annealing": (
        "The TRAINING schedule for kl_weight is non-decreasing within a fit and across resumed "
        "train() calls, and reaches kl_weight_max in finite steps. There is deliberately no "
        "reset knob on train(): restarting the ramp is the behaviour DE-4 removes, and adding "
        "one would be an API-contract change. Construct a new model for a fresh schedule.\n"
        "\n"
        "'TRAINING schedule' is load-bearing and was added with I3. Validation pins kl_weight to "
        "kl_weight_max for the duration of the check and restores the schedule value in a "
        "try/finally at batch scope. Read as a bound on the instantaneous attribute, that pin "
        "would violate B1 on the way back down; the bound is on the schedule the ramp advances, "
        "not on every transient value the attribute holds."
    ),
    "B2_warmup_declared_in_its_own_unit": (
        "n_steps_kl_warmup counts OPTIMIZER STEPS, not epochs (DE-17/DUX-2, open since July "
        "with no answer recorded). At batch_size=1024 on 5000 cells that is ~5 steps/epoch, so "
        "2000 steps is ~400 epochs -- and ~2000 epochs at 1000 cells. Any run that quotes a "
        "warmup must record the epoch equivalent for its data size."
    ),
    "B3_patience_declared_in_epochs": (
        "Patience is stated in the unit a reader assumes. Lightning counts it in VALIDATION "
        "CHECKS, so patience=300 with check_val_every_n_epoch=5 is 1500 epochs of "
        "non-improvement -- measured exactly: best at epoch 1464, stop at 2964.\n"
        "\n"
        "Resolved by setting check_val_every_n_epoch=1 so the two units coincide by "
        "construction, and renaming the knob patience -> patience_epochs. Deliberately NOT "
        "resolved by dividing at the call site: two units with a silent conversion between them "
        "is the same trap in a new place. Cost of cv=1 is a measured +7.6% wall clock on the "
        "worst-case fixture (2 training batches per epoch); less on benchmark shapes.\n"
        "\n"
        "Note for the implementation: `patience` appears in init_params_, so the rename needs a "
        "real deprecated alias or load_tcri_session() breaks on previously saved models.\n"
        "\n"
        "Consequence worth stating: at the OLD defaults early stopping could never fire at all "
        "-- 300 checks x 5 epochs = 1500 epochs of patience against a max_epochs of 1000. Every "
        "default run trained to the budget. DE-2 and DE-3 were therefore latent rather than "
        "active, which is why neither had been noticed from a wrong number."
    ),
    "B4_min_delta_exceeds_monitor_noise": (
        "A stopping threshold below the monitored series' own noise is not a criterion. "
        "min_delta=0 with a noisy ELBO stops on sampling variation. Note that I3's fixed "
        "evaluation seed removes the Monte-Carlo component of that noise, so min_delta now has "
        "to clear only genuine parameter movement."
    ),
    "B5_selection_begins_only_after_the_ramp_completes": (
        "No check is RECORDED, compared or snapshotted before "
        "module._kl_warmup_step >= n_steps_kl_warmup. Checks before that point are logged and "
        "nothing else.\n"
        "\n"
        "I3 makes each check well-posed; this bound makes the SERIES comparable, by ensuring "
        "every entry in it comes from the same objective L_(kl_weight_max). Together they are "
        "the whole of 'the argmin means something'.\n"
        "\n"
        "One gate, one predicate, one object: both the stopping callback and the snapshot "
        "callback read the same counter. Do NOT also set scvi's early_stopping_warmup_epochs -- "
        "two counters in two different units can disagree by a check at the boundary.\n"
        "\n"
        "If the ramp does not complete inside the run: WARN LOUDLY, do not raise, and record "
        "selection_criterion='last epoch (ramp incomplete)' in the training record. Raising "
        "would break every short fit in tests/ and the entire benchmark grid, whose 60-epoch "
        "default reaches roughly 15% of a 2000-step ramp."
    ),
    "B6_every_advertised_knob_has_a_behavioural_test": (
        "The check is that behaviour changed, never that a value arrived. This is the rule the "
        "knob test's '--' column violates and the reason DE-2 survived a green tick."
    ),
    "B7_a_run_is_a_function_of_seed_data_knobs": (
        "Given (seed, data, knobs) a fit is reproducible. Before DE-19 the network init and "
        "minibatch order were unseeded and the same nominal seed gave ~1.8e-3 NMI spread -- "
        "larger than several of the effects this stack measures."
    ),
    "B8_optimizer_settings_that_act_as_priors_are_declared": (
        "weight_decay reaches q_p_c_raw/q_p_ct_raw, whose param-store leaves are log theta, so "
        "it pulls every clone row toward the uniform simplex point. That is a prior acting "
        "through an optimizer setting. Declare it as one or remove it (Q-D, open)."
    ),
    "B9_the_plan_records_provenance": (
        "A fit records what actually happened -- epochs ACTUALLY run (not requested), warmup "
        "steps and their epoch equivalent, seed, steps per epoch. Silent truncation of "
        "max_epochs=4000 to 2964 epochs went unnoticed for a day because nothing recorded it.\n"
        "\n"
        "Extended with I3/I5: kl_weight must be LOGGED per epoch. It is absent from history "
        "entirely today, which is exactly what makes I3 and I5 unfalsifiable after a run has "
        "finished -- there is no way to tell from the record whether the ramp completed.\n"
        "\n"
        "Also record, per fit: selection_criterion (the monitor name, or 'last epoch (ramp "
        "incomplete)'), the epoch the snapshot came from, and ramp_completed_at_epoch = "
        "n_steps_kl_warmup / steps_per_epoch. That last number is one line and it tells a "
        "reader immediately which regime a run was in: a ramp that finishes early leaves most "
        "of the run at a stationary objective, while one that never finishes means the prior "
        "was substantially switched off for the whole fit."
    ),
}


#: Questions that must be answered before the corresponding invariant can be enforced.
OPEN = {
    "Q-B_minibatch_weighting": (
        "Does a minibatch estimate weight the N cell terms against the C+M global terms in "
        "eq 7's ratio? Unanswered; affects whether the ELBO is an unbiased estimate of eq 7 "
        "at any batch size other than the full data."
    ),
    "Q-D_weight_decay_as_prior": "See AUTHORED_BOUNDS['B8_optimizer_settings_that_act_as_priors_are_declared'].",
}
