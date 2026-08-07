# Training contract

**Manifest:** `tcri/model/_training_contract.py` · **Tests:** `tests/test_training_contract_conformance.py` (manifest + structure), `tests/test_training_invariants.py` (behaviour)

## Why this exists

The model *structure* is specified by Supplementary Note 1 and machine-checked by the model
contract. The *training plan* was specified nowhere — no contract, no bounds — and it has large
effect on every reported number.

That absence is the reason the same defects kept being rediscovered. DE-1 through DE-4 are not
four bugs; they are four faces of one unspecified subsystem, each found from a different symptom
(slow convergence, then a plateau, then degradation) and each diagnosed locally.

## The asymmetry, and why this contract has two halves

Note 1 gives the model (eqs 1–5), the variational family (eq 6), the ELBO (eq 7) and the
surrogate. On optimization it says one sentence: SVI in Pyro, mini-batching, KL scaling for
Dirichlet and discrete terms, Adam. No epochs, patience, warmup schedule or stopping criterion.

So unlike the model contract, this one cannot be wholly derived. It is split accordingly:

| | authority | changing it |
|---|---|---|
| **`DERIVED_INVARIANTS`** | follow from eq 7 | a violation is a **defect** |
| **`AUTHORED_BOUNDS`** | ours, because the note is silent | changeable by @nceglia / @salehis **with a recorded reason** |

Keeping them apart matters: collapsing the two would let a preference acquire the authority of
the manuscript. A test asserts the key sets are disjoint.

## A bound must be behavioural

The knob test verified that `patience` *arrives* at `EarlyStopping.patience` and marked it ✅.
Its behavioural column held a dash. Meanwhile patience was counted in validation checks
(300 × `check_val_every_n_epoch=5` = **1500 epochs**), `validation_step` was taking optimizer
steps on the parameters every metric reads, and the KL ramp restarted on every `train()` call.

Asserting that a value is connected is not asserting that the behaviour is right. That gap is
this project's dominant defect class, and it is why the behavioural assertions live in a
separate file from the manifest checks.

## Current status

| | statement | status |
|---|---|---|
| I1 | one objective: `−(L# + γ·Σ KL(probs‖φ))` | holds |
| I2 | no parameter update outside `training_step` | **holds** — DE-1 fixed |
| I3 | the monitored quantity is a fixed objective | **open** — `elbo_validation` inherits the annealed `kl_weight`, so it is comparable to `elbo_train` but not stationary while the ramp climbs. PR 4. |
| I4 | the reported model is the one the criterion selected | **open** — early stopping truncates and keeps the final weights; `q_p_ct_raw` is not in `state_dict()`, so checkpointing alone cannot provide a restore. PR 4. |
| I5 | annealing is schedule-only and terminates | **holds** — DE-4 fixed |
| I7 | a declared knob changes an observable | partial |

I3 and I4 are labelled open deliberately. A contract that overstates its coverage is worse than
one that admits a gap — that is what the ✅ on `patience` did.

## Open questions

**Q-B — minibatch weighting.** Does a minibatch estimate weight the N cell terms against the
C+M global terms in eq 7's ratio? Affects whether the ELBO is unbiased for eq 7 at any batch
size below the full data.

**Q-D — weight decay as a prior.** It reaches `q_p_c_raw`/`q_p_ct_raw`, whose param-store leaves
are `log θ`, so it pulls every clone row toward the uniform simplex point. That is a prior
acting through an optimizer setting. Declare it as one or remove it.

## Notes on specific bounds

**B1 — no reset knob.** `train()` deliberately has no `reset_schedule`. Restarting the ramp is
the behaviour DE-4 removes, and adding the parameter would also be an API-contract change to
`_contract.pyi`. Construct a new model for a fresh schedule.

**B2 — warmup units.** `n_steps_kl_warmup` counts **optimizer steps**. At `batch_size=1024` on
5000 cells that is ~5 steps/epoch, so 2000 steps ≈ 400 epochs — and ≈ 2000 epochs at 1000 cells.
The unit was open as DUX-2 since July with no answer recorded; this is the answer.

**B9 — provenance.** A fit records what actually happened, including epochs **actually run**
rather than requested. `max_epochs=4000` silently trained 2964 epochs and `8000` trained 2774,
and the resulting near-identical numbers were read as an estimator property for a full day
because nothing recorded the real count.
