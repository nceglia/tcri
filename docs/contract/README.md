# TCRI refactor contract

Internal planning/design docs for the `tcri` API refactor (target: a standalone scverse-ecosystem
package, "Door A"). **Not** part of the published Sphinx site (excluded via `exclude_patterns` in
`docs/conf.py`). Structural reference is the `../grafiti` package.

## ⚠ Read first — governance

| File | What it is |
|---|---|
| **`REFACTOR_HISTORY.md`** | Single source of truth: the **Hard Rules**, the chronology, and all settled decisions (§2). Read before touching anything. |
| **`REDO_LIST.md`** | The corrected **disposition map** (keep / drop) + the clean re-implementation plan. |

**Hard Rules (never violate):** the `example/` notebooks are **disposable** — never read them or use them
for any decision. **Non-core = DROP (delete); nothing moves to `examples/`.** Disposition is decided by one
test: *is it core?*

## The authoritative pair

| File | What it is |
|---|---|
| **`tcri_api_and_responsibilities.md`** | Final API surface + per-function math/stats spec + prior-vs-mean resolution. **Clean** (post-recovery). |
| **`tcri_implementation_plan.md`** | Ordered PR sequence + model→AnnData streamline + GPU architecture + testing/CI. **Clean** (post-recovery). |
| `tcri_consistency_sufficiency.md` | Argument consistency deltas + sufficiency-from-responsibility (clean re-derivation). |

## Supporting artifacts

| File | What it is |
|---|---|
| `tcri_function_inventory.md` (+ data, `build_tcri_inventory.py`) | 131-function labeled inventory |
| `tcri_api_contract.*`, `tcri_dependency_map.*` (+ build scripts) | early target API + dependency graph |

## `_quarantine/` — do not use

Earlier versions of the two authoritative docs + their audit data, contaminated by a **notebook-caller-census
leak** (a workflow prompt treated the disposable notebooks as an authority on what to keep, resurrecting
`gene_entropy`/`probability_ternary`). Their **disposition sections are wrong**; their math/design was clean and
was salvaged into the current authoritative docs. Kept only as a record. See `REFACTOR_HISTORY.md`.
