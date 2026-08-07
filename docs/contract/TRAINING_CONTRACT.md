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
| I2 | no **optimizer** update outside `training_step` | **holds** — DE-1 fixed |
| I3 | the monitored quantity is a fixed objective | **specified, not implemented** — criterion decided below; code pending |
| I4 | the reported model is the one the criterion selected | **specified, not implemented** — mechanism decided below; code pending |
| I5 | annealing is schedule-only and terminates | **holds** — DE-4 fixed |
| I7 | a declared knob changes an observable | partial |

`SPECIFIED` is a third status, added with this revision. It means the design is settled and
written into the manifest but the code does not do it yet. A contract that overstates its
coverage is worse than one that admits a gap — that is what the ✅ on `patience` did — so
`holds` is now machine-checked: an invariant may only claim it if `enforced_by` names a test
file that exists, and a function inside it that exists
(`test_a_holds_claim_names_a_test_that_exists`).

## The stopping policy (I3 + I4)

### The principle

Annealing is a **continuation method**. Training descends a family of surrogates `L_β` whose
endpoint `L_β_max` is the objective actually meant. So the function you *descend* and the
criterion you *select on* are different objects, and must be allowed to differ.

Two consequences, and neither is a matter of taste:

1. **A series of different functions has no argmin.** So the selection criterion must be a
   fixed function of (parameters, held-out data). This follows from what "argmin" means.
2. **Early stopping has two outputs**, a stop time *and* the argmin weights (Prechelt,
   *Early Stopping — But When?*). A run that stops at the argmin and keeps the last weights has
   implemented half of it, and is not doing early stopping.

### What is monitored

`objective_validation_percell` — the **per-cell block only**: `latent` + `phenotype_alignment`
+ `obs`, over the validation split, divided by `N_val`.

It is deliberately **not** the ELBO, and must never be called one: it excludes the global sites
and contains an unnormalized `pyro.factor` and a ρ-tempered `obs`.

Excluding `p_c`/`p_ct` is the one substantive judgement here. Both global plates are declared at
full size with no subsampling, so their KL contributes the **same value regardless of which
cells are held out**. Including them means selecting partly on how well the guide matches its
prior on the *training* data — which a validation criterion must not contain. Their share is
small at benchmark scale (tens of clones) and large at repertoire scale, since real repertoires
are singleton-dominated.

This is *not* a reconstruction-only criterion. `phenotype_alignment` scores held-out cells
against `φ = p_ct[ct_idx]`, so the Dirichlet branch is still covered; what is dropped is only
the prior-matching term that held-out data cannot speak to. The global block is logged as its
own series and never monitored.

### How it is evaluated

Pinned `kl_weight = kl_weight_max` (try/finally, batch scope), eval mode, fixed particle count,
same validation split, and a **forked, fixed RNG seed** — the same draws every check. The seed
clause is load-bearing: a Monte-Carlo estimator redrawn each check is not a function of the
parameters at all, so an argmin over it is an argmin over noise.

### When selection begins

Not until `module._kl_warmup_step >= n_steps_kl_warmup`. I3 makes each check well-posed; **B5**
makes the *series* comparable, by ensuring every entry came from the same `L_β_max`. One gate,
one predicate, one object, read by both the stopping and snapshot callbacks — do not also set
scvi's `early_stopping_warmup_epochs`, as two counters in two units can disagree at the
boundary. If the ramp never completes: warn loudly, do **not** raise, and record
`selection_criterion = "last epoch (ramp incomplete)"`.

### What is restored, and two ways to get it silently wrong

Snapshot at each gated improving check; restore in place at `on_fit_end`. The snapshot must
carry **both** sources or it restores a model no check ever evaluated:

**`state_dict()`, not `named_parameters()`.** The encoder and VampPrior carry BatchNorm running
statistics (`FCLayers` defaults `use_batch_norm=True`), which are buffers — absent from
`named_parameters()`, absent from the param store, and read by `predict()` in eval mode.
Measured on a minimal fixture: 6 running buffers, 21 `state_dict` keys absent from
`named_parameters()`.

**The param store, through `named_parameters()` in unconstrained space — never `items()`.**
`q_p_c_raw`/`q_p_ct_raw` carry `constraints.positive`, so `store.items()` yields a *non-leaf*
`ExpTransform` output. Verified directly: writing `5.0` via `.data.copy_()` on that tensor
leaves the store reading **2.0**, with no error and no warning. A restore written the obvious
way silently does nothing — the same failure class as DE-1. Snapshotting constrained and
restoring unconstrained (or the reverse) inflates every clone row by `exp(·)`. Do not use
`ParamStore.set_state()`: it rebinds `_params[name]` and desyncs the store from the `nn.Module`
that registered it.

### Why this was invisible

At the old defaults early stopping **could never fire**: patience of 300 checks ×
`check_val_every_n_epoch=5` = 1500 epochs, against `max_epochs=1000`. Every default run trained
to the budget and stopped there. DE-2 and DE-3 were latent rather than active, which is why
neither had ever produced a wrong number to notice.

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

**B3 — patience units.** Resolved by setting `check_val_every_n_epoch=1` so the two units
coincide by construction, and renaming `patience` → `patience_epochs`. Deliberately *not*
resolved by dividing at the call site: two units with a silent conversion between them is the
same trap in a new place. Costs a measured +7.6% wall clock on the worst-case fixture. Since
`patience` is in `init_params_`, the rename needs a real deprecated alias or
`load_tcri_session()` breaks on previously saved models.

**B5 — selection begins after the ramp.** See "When selection begins" above.

**B9 — provenance.** A fit records what actually happened, including epochs **actually run**
rather than requested. `max_epochs=4000` silently trained 2964 epochs and `8000` trained 2774,
and the resulting near-identical numbers were read as an estimator property for a full day
because nothing recorded the real count.

Extended with I3/I5: `kl_weight` must be **logged per epoch**. It is absent from `history`
entirely today, which is exactly what makes I3 and I5 unfalsifiable once a run has finished.
Also record `ramp_completes_at_epoch = n_steps_kl_warmup / steps_per_epoch` — one line, and it
tells a reader which regime a run was in. A ramp that finishes early leaves most of the fit at a
stationary objective; one that never finishes means the prior was substantially switched off
throughout. The benchmark grid's 60-epoch default reaches roughly 15% of a 2000-step ramp.
