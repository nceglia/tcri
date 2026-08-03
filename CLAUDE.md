# tcri — contributor rules

Single-cell TCR+RNA information-theory metrics on scvi-tools / pyro / scanpy.

## The three contracts

This repo is governed by three frozen contracts. All are machine-checked; a failing
conformance test means **stop and decide**, not "adjust the contract until it passes."

| | freezes | manifest | prose | test |
|---|---|---|---|---|
| **API contract** | the public *interface* | `tcri/_contract.pyi` | `docs/contract/tcri_api_and_responsibilities.md` | `tests/test_contract_conformance.py` |
| **Model contract** | the generative *mathematics* | `tcri/model/_model_contract.py` | `docs/contract/MODEL_CONTRACT.md` | `tests/test_model_contract_conformance.py` |
| **Metrics contract** | what the *metrics compute* | `tcri/tools/_metrics_contract.py` | `docs/contract/METRICS_CONTRACT.md` | `tests/test_metrics_contract_conformance.py` |

### Model integrity (read before touching `tcri/model/`)

The model implements **Supplementary Note 1** (`tcri_supplementary_methods_04_30_26.pdf`)
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
(`_model_contract.py`) with a rationale, mirrored in `MODEL_CONTRACT.md`. Anything
not listed there that departs from the note is a defect.

`docs/contract/METHODS_CONFORMANCE.md` is the eq-by-eq code map + deviation history.

### Metric integrity (read before touching `tcri/tools/`)

The entropies and mutual information are frozen by the **metrics contract**. Changing
a definition means changing what every published number means. Same rule: update
`_metrics_contract.py` + `METRICS_CONTRACT.md` first, then the code.

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

## Working agreement

- **`docs/contract/REFACTOR_AGENDA.md` is the living tracker** — read it before
  starting, write a diary entry after each PR, and run the Standing Audit in it.
- **Removal is a hard bar.** Delete dead code rather than keeping it "just in case";
  git has it. Tick the Removal Ledger.
- **Never read the `example/` notebooks.** They are disposable *outputs* of the
  refactor, never an input — no caller census, no "is-it-used" checks.
- Run tests with the pinned venv: `MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q`.
