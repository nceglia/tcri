# tcri — contributor rules

Single-cell TCR+RNA information-theory metrics on scvi-tools / pyro / scanpy.

## Source of truth, and who may change it

**The manuscript is upstream of the contracts.** Sohrab Salehi's Supplementary Note is
ground truth for the model; the metrics document is ground truth for the metrics. Where a
contract disagrees with them, **the contract is wrong**. The documents are the first line:
if one is ambiguous, **ASK** — never infer a definition from what makes the code, a test,
or a benchmark come out right.

Both are archived in `governance/source/` with their hashes recorded in the metrics
manifest and checked by a test, so a revision is detectable rather than something someone
has to notice.

**Their equation numbers COLLIDE.** Note 1 numbers eqs 1–12 (generative model, variational
family, ELBO, perturbation). The metrics document independently numbers eqs 2–7
(entropies, MI, NMI, KL). "eq 3" means the VampPrior in one and the clonotypic entropy in
the other — every reference must name its document. Note 1 contains **no** entropy or MI
definitions at all; a metric citing it is citing the wrong source.

**Only @nceglia and @salehis may change a contract, a conformance test, or a source
document.** Enforced by `.github/CODEOWNERS`, which requires "Require review from Code
Owners" on the `main` branch protection rule to actually block a merge.

## The four contracts

This repo is governed by four frozen contracts. All are machine-checked; a failing
conformance test means **stop and decide**, not "adjust the contract until it passes."

| | freezes | manifest | prose | test |
|---|---|---|---|---|
| **API contract** | the public *interface* | `tests/contracts/api.pyi` | `governance/API_CONTRACT.md` | `tests/test_contract_conformance.py` |
| **Model contract** | the generative *mathematics* | `tests/contracts/model.py` | `governance/MODEL_CONTRACT.md` | `tests/test_model_contract_conformance.py` |
| **Metrics contract** | what the *metrics compute* | `tests/contracts/metrics.py` | `governance/METRICS_CONTRACT.md` | `tests/test_metrics_contract_conformance.py` |
| **Training contract** | how the model is *fit* | `tests/contracts/training.py` | `governance/TRAINING_CONTRACT.md` | `tests/test_training_contract_conformance.py` + `tests/test_training_invariants.py` |

### Model integrity (read before touching `tcri/model/`)

The model implements **Supplementary Note 1** (`governance/source/supplementary_note_1_SS_2026-08-03.pdf`)
— the source of truth. Changing its mathematics means: adding/removing a stochastic
site, changing a distribution family or plate, altering the ELBO or the phenotype
surrogate, or changing what a prior is scaled by (α on eq 1, β on eq 2).

**Update the model contract FIRST, then the code.** Cite the note equation and state
what changes in the joint distribution. Then make the code agree.

**Never loosen the manifest to make a conformance failure go away.** That silently
rewrites the model the package claims to implement. A failure is either an intended
model change (update the contract deliberately, as a reviewed model change) or a
regression (fix the code).

If you are an AI agent and a model change appears necessary, **surface the contract
implication to the user** rather than editing the manifest to fit your change.

Accepted departures from the note live in `SANCTIONED_DEVIATIONS`
(`tests/contracts/model.py`) with a rationale, mirrored in `MODEL_CONTRACT.md`. Anything
not listed there that departs from the note is a defect.

`governance/METHODS_CONFORMANCE.md` is the eq-by-eq code map + deviation history.

### Metric integrity (read before touching `tcri/tools/`)

The entropies and mutual information are frozen by the **metrics contract**. Changing
a definition means changing what every published number means. Same rule: update
`tests/contracts/metrics.py` + `METRICS_CONTRACT.md` first, then the code.

The conformance test pins numeric identities, the keystone being
`I(c;φ) = H(c) − Σ_φ P(φ)·H[P(c|φ)]` — it ties the entropy and MI families together so
neither can be redefined alone.

**`normalize_mode` deviates from the note deliberately.** Eq 6 defines
`NMI = I/(½(H(c)+H(φ)))` (the mean denominator, `normalize_mode="average"`), but the
default is `"min"` — the mean denominator scales with `log2(C)` and so is not
comparable across groups with different clone counts. Anything reproducing the note's
benchmark must pass `normalize_mode="average"` explicitly. Recorded in
`SANCTIONED_EXTENSIONS['normalize_mode_default']` and pinned by a test that asserts the
default does *not* equal eq 6, so the divergence cannot go silent.

### Training integrity (read before touching `tcri/model/_training.py`)

The model *structure* is specified by the note; the *training plan* is not. Note 1 says only
"SVI in Pyro, mini-batching, KL scaling, Adam" — no epochs, patience, warmup schedule or
stopping rule. So the training contract has two halves and they carry different authority:
**`DERIVED_INVARIANTS`** follow from eq 7 and a violation is a defect;
**`AUTHORED_BOUNDS`** are ours, changeable with a recorded reason.

**A bound must be behavioural.** The knob test verified that `patience` *arrives* at
`EarlyStopping.patience` and ticked it ✅ — while patience was counted in validation checks
(300 × 5 = 1500 epochs), `validation_step` was taking optimizer steps, and the KL ramp
restarted on every `train()` call. Asserting a value is connected is not asserting the
behaviour is right, and that gap is why the same defects kept resurfacing from new symptoms.

I3 and I4 are **open** and say so. Do not mark an invariant satisfied without a test.

## Working agreement

- **`dev/REFACTOR_AGENDA.md` is the living tracker** — read it before starting, write a diary
  entry after each PR, and run the Standing Audit in it. **`dev/` is gitignored**, so it is
  local-only and a fresh clone will not have it: it records what is *pending*, which is a
  different kind of claim from what a contract freezes, and mixing the two meant a reader could
  not tell which sentences bind. Anything in there that others must act on belongs in a GitHub
  issue, not in the tracker.
- **`governance/` holds the four contracts, their prose, and the source documents.** It sits
  outside `docs/` deliberately — it is the frozen record, not documentation, and at its old
  `docs/contract/` home it was one letter from `docs/contracts/`, the published reader page.
- **Removal is a hard bar.** Delete dead code rather than keeping it "just in case";
  git has it. Tick the Removal Ledger.
- **Never read the `example/` notebooks.** They are disposable *outputs* of the
  refactor, never an input — no caller census, no "is-it-used" checks. They are untracked and
  purged from history; `example/` is gitignored.
- **After moving or renaming anything, grep the WHOLE repo** — `grep -rIn "<old-path>" .` with
  no `--include` filters and no directory scoping. Extensionless files (`CODEOWNERS`), docs
  trees, and build-time-resolved references (Sphinx `automodule`) are exactly what a filtered
  grep hides, and each has already caused a silent break. Then run the acceptance check that
  *consumes* the path — build the wheel, build the docs — because the unit suite cannot see any
  of them.
- Run tests with the pinned venv: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q`.

### Branching (this has gone wrong more than once)

- **Always branch from a fresh `main`.** `git checkout main && git pull` first. Never
  branch off a branch that has an open PR: the second PR then *contains* the first, and it
  goes stale the moment the first merges.
- **After a PR merges, return to `main` and pull** before starting the next piece of work.
- **Before pushing to a branch with an open PR**, check it is not behind:
  `git fetch && git rev-list --count <branch>..origin/main` must be `0`. If it is not,
  rebase — and read what landed on `main` first, because someone else's work may already
  cover what you were about to write.
- Duplicated work is the symptom to watch for. The eq 3–4 erratum was addressed in three
  separate PRs (#41 closed, #42 and #44 both merged with identical titles) because each was
  started from a stale base.
