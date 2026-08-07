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
    "I2_no_update_outside_training_step": {
        "statement": (
            "No parameter is updated outside training_step, and only on training-split "
            "batches. Validation evaluates; it never steps."
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
            "The quantity early stopping monitors is a fixed function of (Lambda, Theta): same "
            "kl_weight, module mode, particle count and data across checks. Selecting a minimum "
            "over a series whose objective is still changing does not mean what it appears to."
        ),
        "status": "OPEN -- elbo_validation currently inherits the last training step's annealed "
                  "kl_weight, so it is comparable to elbo_train but is not stationary while the "
                  "ramp climbs. Resolved by PR 4 (stopping-policy).",
        "enforced_by": "pending",
    },
    "I4_reported_model_is_the_selected_one": {
        "statement": (
            "The parameters left in the store when train() returns are the ones the declared "
            "selection criterion chose."
        ),
        "status": "OPEN -- early stopping truncates and keeps the final weights; nothing "
                  "restores the best. Compounded because q_p_ct_raw is not in state_dict(), so "
                  "checkpointing alone cannot provide it. Resolved by PR 4 (DE-3).",
        "enforced_by": "pending",
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
        "kl_weight is non-decreasing within a fit and across resumed train() calls, and reaches "
        "kl_weight_max in finite steps. There is deliberately no reset knob on train(): "
        "restarting the ramp is the behaviour DE-4 removes, and adding one would be an "
        "API-contract change. Construct a new model for a fresh schedule."
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
        "non-improvement -- measured exactly: best at epoch 1464, stop at 2964. Resolved by "
        "PR 4."
    ),
    "B4_min_delta_exceeds_monitor_noise": (
        "A stopping threshold below the monitored series' own noise is not a criterion. "
        "min_delta=0 with a noisy ELBO stops on sampling variation."
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
        "max_epochs=4000 to 2964 epochs went unnoticed for a day because nothing recorded it."
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
