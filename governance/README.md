# `governance/` — the policy, the four contracts, and the source documents

Internal governance docs for `tcri`. It lives outside `docs/` on purpose: this is not
documentation, it is the record the project is held to, and while it sat at `docs/contract/` it
was one letter away from `docs/contracts/`, the published reader-facing page.

**Start with `RULES.md`.** It is the policy, stated once: what the contracts are for, the three
tiers of code and their tags, the lock, how to change things, and what each test enforces. No
other file restates the policy; where one appears to, it is stale.

**The manuscript is upstream of the contracts.** `source/` holds Supplementary Note 1 and the
metrics document, hash-pinned and checked by a test. Where a contract disagrees with them, the
contract is wrong. Their equation numbers **collide**, so every reference names its document.

## The four contracts

Each pairs a machine-checked manifest under `tests/contracts/` with prose here. The manifests sit
under `tests/` because nothing in `tcri` imports them; they are enforcement, not package code.

| Contract | Freezes | Manifest | Prose here |
|---|---|---|---|
| **API** | the public interface | `tests/contracts/api.pyi` | `API_CONTRACT.md` |
| **Model** | the generative mathematics | `tests/contracts/model.py` | `MODEL_CONTRACT.md` |
| **Metrics** | what the metrics compute | `tests/contracts/metrics.py` | `METRICS_CONTRACT.md` |
| **Training** | how the model is fit | `tests/contracts/training.py` | `TRAINING_CONTRACT.md` |

`METHODS_CONFORMANCE.md` is the equation-by-equation code map and deviation history.

The source documents and the published core are owner-reviewed via `.github/CODEOWNERS`.
Additions to the surface are ordinary pull requests; see `RULES.md` for the status of that
transition.

## What used to be here

The pre-refactor planning corpus and the refactor trackers are gone. The planning corpus, a
131-function inventory, an early target-API contract, a dependency map in three formats, their
generators, and the superseded implementation plans, was removed because it had begun to read
as current while describing the package as it was before the refactor. Recoverable from git at
`dfbb4cd`. `REFACTOR_HISTORY.md` followed once the refactor closed; git has the chronology. The
running trackers moved to the gitignored `dev/` and are local scratch, not part of the record.
