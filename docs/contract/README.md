# `docs/contract/` — the four contracts and their history

Internal governance docs for `tcri`. **Not** part of the published Sphinx site (excluded via
`exclude_patterns` in `docs/conf.py`); the reader-facing overview is `docs/contracts/index.md`.

**The manuscript is upstream of the contracts.** `source/` holds Supplementary Note 1 and the
metrics document, hash-pinned and checked by a test. Where a contract disagrees with them, the
contract is wrong. Their equation numbers **collide** — every reference must name its document.

## The four contracts

Each pairs a machine-checked manifest with prose. A failing conformance test means **stop and
decide**, never "loosen the manifest until it passes."

| Contract | Freezes | Manifest | Prose here |
|---|---|---|---|
| **API** | the public interface | `tcri/_contract.pyi` | `API_CONTRACT.md` |
| **Model** | the generative mathematics | `tcri/model/_model_contract.py` | `MODEL_CONTRACT.md` |
| **Metrics** | what the metrics compute | `tcri/tools/_metrics_contract.py` | `METRICS_CONTRACT.md` |
| **Training** | how the model is fit | `tcri/model/_training_contract.py` | `TRAINING_CONTRACT.md` |

`METHODS_CONFORMANCE.md` is the equation-by-equation code map and deviation history.

Only @nceglia and @salehis may change a contract, a conformance test, or a source document
(enforced by `.github/CODEOWNERS`).

## Working documents

| File | What it is |
|---|---|
| `REFACTOR_AGENDA.md` | The living tracker — read before starting, diary entry after each PR, Standing Audit. |
| `REFACTOR_HISTORY.md` | Chronology and settled decisions, including what was removed and where it lives. |
| `DEFECTS.md` | The defect log. |

**Hard rule:** the `example/` notebooks are **disposable outputs**, never an input — never read
them, and never use them for a caller census or an "is it used" check.

## What used to be here

The pre-refactor planning corpus — a 131-function inventory, an early target-API contract, a
dependency map in three formats, their three generator scripts, the superseded implementation
and PR plans, the training spec draft, and the contaminated `_quarantine/` directory — was
removed in the cleanup pass (22 files, ~1.5 MB). They described the package as it was *before*
the refactor and had begun to read as current: `tcri_dependency_map.md` announced itself as
"the post-refactor API" while its call graph still contained four functions deleted in PR6/PR7.
The generators hardcoded their tables rather than introspecting the package, so they could not
be brought up to date by re-running them. Recoverable from git at `dfbb4cd`.
