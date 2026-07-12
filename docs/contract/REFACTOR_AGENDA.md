# TCRI Refactor — Agenda, Tasklist & Diary (LIVING DOC)

**Update this every working session. Refer back to it before starting anything.** It is the operational
tracker + running diary for the whole refactor. The detailed spec lives in `tcri_api_and_responsibilities.md`
+ `tcri_implementation_plan.md`; the rules/history in `REFACTOR_HISTORY.md`; the scratch/deferred pile in
`REFACTOR_NOTES.md` (not checked in).

## How to use this doc
1. **Removal is a hard bar.** The default failure mode is keeping old code around and over-engineering to
   preserve it. **Do the opposite.** Every function in the Removal Ledger must actually be deleted, its
   `__all__`/import sites cleaned, and the checkbox ticked. If keeping something "just in case" feels tempting —
   don't. Delete it; git has it.
2. **Never read the `example/` notebooks.** They are disposable and are an *output* of the refactor, never an
   input. No caller census, no "is-it-used," no "sufficiency."
3. **After every PR, run the Standing Audit** (below) and write a diary entry.
4. **Frequent audits**: at minimum after each PR; ideally mid-PR when a component is touched. Log them in the
   Audit Log.
5. **Usability is a first-class check** — every session ask "is this easier to use than before?"

## Standing Audit (run after each PR — copy into the diary entry)
- [ ] **Removed everything slated?** (cross-check the Removal Ledger; `__all__` + import-sites clean; `import tcri` green)
- [ ] **Added everything wanted?** (the PR's deliverables all present)
- [ ] **Tests:** what components can now be unit-tested that weren't? Added?
- [ ] **Streamline:** any duplication / dead branch / needless complexity spotted? Removed or logged?
- [ ] **Usability:** simpler signatures / clearer errors / fewer steps than before?
- [ ] **Contract conformance green?** (`test_contract_conformance`)
- [ ] Diary entry written; ledger + statuses updated.

## Status legend: ☐ todo · ◐ in progress · ✅ done · ⚠ blocked

## PR Agenda
| # | PR | Status | Risk | Depends | Gate |
|---|---|---|---|---|---|
| 0 | Contract freeze + CI scaffolding | ✅ | none | — | conformance green |
| 1 | Shared helpers + `_keys` | ✅ | low | 0 | existing tests green |
| 2 | Safe deletions | ✅ | very low | 1 | import-graph clean |
| 3 | Model module split | ✅ | low | 1 | model/pyro tests green |
| 4 | Model→AnnData streamline | ☐ | HIGH | 1,3 | session round-trip |
| 5 | Engine consolidation | ☐ | HIGH | 4 | joint identities |
| 6 | Metric-API consolidation | ☐ | HIGH | 5 | metric tests |
| 7 | Plotting split + pl twins | ☐ | medium | 6,1 | twins render |
| 8 | `diag/` seeding | ☐ | low-med | 4,5 | PPC columns |
| 9 | PGM→docs; utils finalize | ☐ | low | 1,8 | import green sans daft |
| 10 | Notebook rewrite (fresh) | ☐ | low | 4–8 | nbmake tutorial |
| 11 | Public API + scverse CI | ☐ | low-med | all | ecosystem checklist |

## Removal Ledger (the hard bar — every one MUST end deleted)
Tick only when the symbol is gone from source AND `__all__`/imports AND `import tcri` is green.

**Phase 2 (dead / out-of-scope):** ✅ ALL DELETED (PR2) — 14 symbols, 384 lines, `import tcri` green.
- [x] `pp.get_latent_embedding` · [x] `pp.group_small_clones` · [x] `pp.register_probability_columns`
- [x] `pp.remove_meaningless_genes` · [x] `pp.gene_entropy` · [x] `pp.classify_phenotypes`
- [x] `pl.polar_plot` · [x] `pl.probability_distribution` · [x] `pl.bayesian_mutual_information`
- [x] `metrics._ent` · [x] `tl.clone_fraction` · [x] `metrics.dkl` (→ `_distance.kl_divergence`)
- [x] `ut.probabilities` (+ its `_plotting.py` import, same PR) · [x] `SankeyNode.hex_to_rgb`

**Phase 4 (folded into `to_anndata` / session):**
- [ ] `pp.register_model` (→ `model.to_anndata`) · [ ] `pp.register_phenotype_key` · [ ] `pp.register_clonotype_key`
- [ ] `pp._compute_logits_and_prior` · [ ] `ut.write_adata_safely` · [ ] `ut._pop_nonserializables`
- [ ] uns keys `tcri_manager`, `tcri_clone_key`, `tcri_phenotype_key`, obsm `X_tcri_phenotypes`

**Phase 5/6 (consolidated away — delete WITH replacement, never before):**
- [ ] `pp.joint_distribution_posterior` (→ unified `joint_distribution`) · [ ] `metrics._mi_from_joint` (→ `_mutual_information`)
- [ ] `tl.mi_compare` (→ `compare_groups`) · [ ] `tl.delta_clonotypic_entropy` · [ ] `tl.delta_entropy_table` · [ ] `tl.flux_table`
- [ ] `tl.clonotypic_entropy_base` · [ ] `tl.clonality` · [ ] `tl.dkl` local `dkl_func`
- [ ] plural `*_entropies` shims · [ ] `metrics/` package (after migration to `tools/`)

**Phase 7 (non-core plots — DROP, not to examples):**
- [ ] `pl.probability_ternary` · [ ] `pl.top_clone_umap` · [ ] `pl.clone_size_umap` · [ ] `pl.plot_phenotype_probabilities`
- [ ] `pl.compare_phenotypes` · [ ] `pl.ridge_delta_entropy` · [ ] `pl.flux` boxplot · [ ] `pl.clonality` plot
- [ ] `pl.tcri_boxplot` (→ private `_metric_boxplot`) · [ ] `pl.set_color_palette` (→ `resolve_palette`)
- [ ] `pl.plot_pheno_sankey` (→ private `_sankey`) · [ ] leaked aliases `centropy`/`pentropy`/`*_tl`

**Phase 9 (out of the package):**
- [ ] `ut.build_nested_tcri_pgm` (→ `docs/`) · [ ] `ut.draw_tcri_pgm_nested` (→ `docs/`) · [ ] `daft` runtime dep

**Phase 3/9 (model/utils cleanup):**
- [ ] `_ascii_hist` (+ all `graph=`/ASCII paths) · [ ] `ml.plot_loss` (→ `diag.loss`) · [ ] `ml.plot_archetypes` (→ `diag.archetypes`)

---

# DIARY

Template per PR: **Goal · Status · What happened · Issues & fixes · Added ✓ · Removed ✓ (hard bar) · Test opportunities · Streamline opportunities · Usability.**

## PR 0 — Contract freeze + CI scaffolding  ·  ✅ done (branch `refactor/pr0-contract-freeze`)
- **Goal:** freeze the target public API as a `.pyi` contract; add signature-drift + import-smoke CI. Zero package behavior change.
- **What happened:** hand-authored `tcri/_contract.pyi` (27 target functions, namespaced `tl/pp/pl/diag/ut` + `TCRIModel`; locked decisions baked in — `n_samples=250`, `weighted=False`, `use_logits`, `normalize_mode`). Ported grafiti's AST signature logic into `tests/test_contract_conformance.py` (contract-parses · live-vs-contract for `IMPLEMENTED` · unimplemented worklist · import-smoke). `IMPLEMENTED={}` (nothing migrated yet). **Full suite: 26 passed, 1 skipped** (23 existing + PR0; additive, zero regressions).
- **Issues & fixes:** `tl`/`pl` share function names → used **namespace container classes** in the `.pyi` so both twins declare cleanly; parser keys as `Namespace.func`; `_strip_receiver` drops `self`/`cls` for the future `TCRIModel` method checks.
- **Added:** ✅ `tcri/_contract.pyi` ✅ `tests/test_contract_conformance.py` ✅ import-smoke
- **Removed (hard bar):** n/a (additive). The contract lists ONLY the kept surface → the 27 declared + the Removal Ledger are the two halves of "done."
- **Test opportunities:** conformance is now a live guardrail; each future PR onboards its functions into `IMPLEMENTED` and drift fails CI. TODO: add `--nbmake` (Phase 10) and multi-py import-smoke in CI yaml.
- **Streamline:** none this PR.
- **Usability:** `_contract.pyi` doubles as the one-screen human-readable target signature reference.
- **Standing Audit:** removed✅(n/a) · added✅ · tests✅(guardrail live) · streamline✅(none) · usability✅ · conformance✅ green · diary✅.
- **Committed** on branch `refactor/pr0-contract-freeze`.

## PR 1 — Shared helpers + `_keys`  ·  ✅ done
- **Goal:** create `_keys`/`_console`/`_stats`/`_distance`; then adopt `_keys` at every read/write site, dedup console, move stats out of utils. API unchanged.
- **What happened (foundation, done):** created the 4 helper modules — `_keys.py` (all uns/obsm/obs constants incl. NEW `GATE_PROB`/`CLASSIFIER_TEMPERATURE`; legacy keys listed for the removal step); `_console.py` (single `_ok/_info/_warn/_fin` + all ANSI aliases both spellings); `_stats.py` (`stars`/`auc_and_label_permutation`/`bootstrap_auc` + **NEW** true `hdi`, `eti`, `prob_direction`, `mann_whitney`); `_distance.py` (`kl_divergence`/`l1_distance`/`jensen_shannon` + `phenotype_distance` dispatch, single eps, bits). Added `tests/test_helpers.py` (8 tests). Suite green.
- **Issues & fixes:** **`hdi` off-by-one** — first version spanned `ceil(prob·n)+1` points, so on a right-skewed sample it returned the full range (HDI==ETI). Caught by a sanity assertion; fixed to the arviz `floor(prob·n)` window → now correctly hugs the low-mass region (`HDI=(0,0.2)` vs `ETI=(0,2.12)`). A concrete "validate the math up front" catch.
- **Added:** ✅ `_keys.py` ✅ `_console.py` ✅ `_stats.py` ✅ `_distance.py` ✅ `tests/test_helpers.py`
- **Removed / deduped (hard bar):** ✅ console dedup — deleted the **12** copied `_ok/_info/_warn/_fin` defs (metrics+preprocessing+plotting) → one `_console`. ✅ moved `stars`/`auc_and_label_permutation`/`bootstrap_auc` out of `utils` → `_stats` (plotting repointed; dead `stars` import dropped). Suite green (34 passed).
- **Test opportunities:** ✅ done — 8 unit tests for pure helpers that were previously embedded/untestable.
- **Streamline:** foundation enables deleting ~3 console dupes + 2 `dkl` copies + the `utils` stats block on adoption.
- **Usability:** internal only this step.
- **`K.*` migration (done):** replaced **85** canonical key literals with `K.*` across preprocessing/metrics/plotting/utils via a verified script (model=0; only legacy keys there); added `test_no_canonical_key_literals` guard — none remain. `dkl` reassigned (dead `metrics.dkl`→Phase 2; `flux` inner→Phase 6). Legacy keys left as literals until their removal phases. **PR1 COMPLETE — 35 passed / 1 skipped.**

## PR 2 — Safe deletions  ·  ✅ done (branch `refactor/pr2-safe-deletions`)
- **Goal:** delete the Phase-2 dead / out-of-scope symbols outright (they go to the trash, not to examples). Zero behavior change to the kept surface; `import tcri` stays green.
- **What happened:** verified **every** target has zero in-package call-sites (grep for calls/imports, plus a precise `NAME(` call-site check for the three ambiguous ones — `_ent`, module-level `dkl`, `probabilities` — all 0), then deleted via AST-span (top-level `FunctionDef` line ranges; the `SankeyNode.hex_to_rgb` method inside its `ClassDef`; the dead `from ..utils._utils import probabilities` line). **14 symbols, 384 lines removed** across metrics/plotting/sankey/preprocessing/utils. `import tcri` green; **full suite 35 passed / 1 skipped** (unchanged — nothing referenced them).
- **Why safe (not deferred):** the "dkl" refs that remain are the **string** distance-metric name in `flux` + the `_distance` registry, not the deleted module-level `dkl` function (`flux` keeps its own inner `dkl_func` → Phase 6). The `polar_plot`/`probability_distribution` "callers" were self-referential (own docstring examples / the self-recursion bug). `probabilities` was imported into plotting but never called.
- **Added:** n/a (pure removal PR).
- **Removed (hard bar):** ✅ all 14 Phase-2 ledger items ticked — `pp.get_latent_embedding`, `pp.group_small_clones`, `pp.register_probability_columns`, `pp.remove_meaningless_genes`, `pp.gene_entropy`, `pp.classify_phenotypes`, `pl.polar_plot`, `pl.probability_distribution`, `pl.bayesian_mutual_information`, `metrics._ent`, `tl.clone_fraction`, `metrics.dkl`, `ut.probabilities` (+ import), `SankeyNode.hex_to_rgb`. Confirmed gone: no `def` remains, no explicit re-export/`__all__` names them.
- **Test opportunities:** none new (removal); the existing suite is the regression gate and stayed green.
- **Streamline:** shrinks preprocessing (−127) and plotting (−225) meaningfully ahead of the Phase 3/4/7 splits. Removed the one import this PR orphaned (`cosine_similarity` — sole user was the deleted `classify_phenotypes`); other module-top imports left for the file-split phases (conservative; the PR1 audit already handled the utils ones).
- **Usability:** removes broken/dead public entry points (`probability_distribution` self-recursion, `bayesian_mutual_information` bad kwarg, `polar_plot` undefined-name) from the surface so nobody trips on them.

## PR 3 — Model module split  ·  ✅ done (branch `refactor/pr3-model-split`)
- **Goal:** split the 1074-line `model/_model.py` into scvi-style sibling files (`_model` + `_module` + `_priors` + `_classifier` + `_training`), rename `c2p_mat → clone_phenotype_prior`. Mechanical; **no behavior change**.
- **What happened:** verified up front that **no code outside `model/` references any moved internal** (only `TCRIModel` is imported externally). Extracted the 7 top-level defs via `ast.get_source_segment` (formatting-preserving) into the target files along the clean dependency DAG `_classifier`/`_priors` (leaf) → `_module` → `_training` → `_model`; `_model` re-imports all six moved symbols so the `tcri.model.*` surface is byte-for-byte unchanged (private helpers carry `# noqa: F401`). Applied the `c2p_mat → clone_phenotype_prior` rename with a word-boundary regex (13 sites; left the unrelated `c2p_torch` local and the module buffer `clone_phen_prior` untouched). Dropped 3 provably-dead top-level imports surfaced by the per-file import rebuild (`setup_anndata_dsp`, `cosine_similarity`, the `torch.distributions` `Categorical/Dirichlet/MixtureSameFamily` trio — all uses were `dist.`-prefixed pyro). File sizes: `_model` 462, `_module` 326, `_training` 154, `_priors` 147, `_classifier` 21.
- **Issues & fixes:** (1) my first smoke silently imported a **stale `site-packages/tcri`** copy (a script's dir, not the repo, leads `sys.path`) and failed in `setup_anndata` on an old-copy/scvi mismatch — a red herring; forcing the repo copy onto the path, the smoke passes. (The stale install is an env-hygiene note, not a code issue — pytest already uses the repo copy, which is why the suite validates the split.) (2) The train path was **entirely uncovered** by the suite (`trained_model` fixture defined but unused) — so the split was only import-verified. Fixed by adding a real end-to-end smoke test.
- **Added:** ✅ `tests/test_model_smoke.py` — construct → train (2 epochs) → `get_latent_representation` / `get_p_ct` / `get_cell_phenotype_probs` (asserts shapes + prob normalization), plus asserts the `clone_phenotype_prior` rename landed and `build_archetypes` returns centers **and** labels. Runs in ~1s inside the suite.
- **Removed (hard bar):** n/a — PR3 is a structural split, not a removal PR. `ml.plot_loss`/`ml.plot_archetypes` stay on `TCRIModel` until `diag/` exists (Phase 8); no Phase-2 style deletions here.
- **Test opportunities:** ✅ closed the biggest gap (model construct/train/query now covered). The rewritten `test_session_round_trip` (Phase 4) will extend this to save/load.
- **Streamline:** the split makes Phase 4 (model→AnnData) and Phase 5 (engine) tractable — the pyro module, priors, classifier, and training plan are now editable in isolation.
- **Usability:** each file now has a docstring stating its role; the model file reads as a clean `BaseModelClass` API surface.
- **Deferred (logged):** **M5** (`build_archetypes` default `K=4` vs `TCRIModel` `K=10`) — behavior-neutral today (the model always passes `K=10` explicitly), reconciled with persisted `labels` when `diag.archetypes` lands (Phase 8). Not touched here to keep the split purely mechanical.
## PR 4 — Model→AnnData streamline  ·  ☐ todo
## PR 5 — Engine consolidation  ·  ☐ todo
## PR 6 — Metric-API consolidation  ·  ☐ todo
## PR 7 — Plotting split + pl twins  ·  ☐ todo
## PR 8 — diag/ seeding  ·  ☐ todo
## PR 9 — PGM→docs; utils finalize  ·  ☐ todo
## PR 10 — Notebook rewrite  ·  ☐ todo
## PR 11 — Public API + scverse CI  ·  ☐ todo
- **Logged test (from grafiti parity):** once `pl.__all__` exists, add a conformance assertion `set(pl.__all__) == {pl entries in _contract.pyi}` — catches *extra/missing* plot functions (whole-surface), not just signature drift on onboarded ones. (tcri's namespaced `.pyi` checks drift incrementally via `IMPLEMENTED`; this closes the whole-surface gap grafiti gets from its markdown+`__all__` channel.)

---

# AUDIT LOG
_(dated entries; what was audited, findings, actions)_

- **(PR0 ✅):** agenda + removal ledger established; standing-audit checklist defined. Contract frozen (27 fns) + conformance guardrail live. Full suite 26 passed / 1 skipped, zero regressions.
- **(PR1 ◐):** shared-helper foundation created (`_keys`/`_console`/`_stats`/`_distance`) + 8 unit tests. Caught & fixed an `hdi` off-by-one before it shipped. **Adoption pending** (dedup, stats-move, `K.*` migration) — no ledger items ticked yet; foundation is additive, suite green. Logged: key-literal test (PR1), `pl.__all__` whole-surface test (PR11).
- **(PR0+PR1 multi-agent audit — 3 lenses):** verdict FIX. Caught a real regression — the `K.*` find/replace over-reached into **10** display/warning/docstring strings (`register_model`/`load_tcri_session` printed `"K.X_LOGITS"` etc.). **Fixed:** restored readable key text in all 10 (AST-span, delimiter-safe); made the key-literal guard **AST-based** (checks real subscripts/`.get`, ignores prose); removed 3 dead `utils` imports the audit flagged. Suite 35 passed. Two non-blocking items deferred to `REFACTOR_NOTES` (contract↔api-doc reconciliation; helper-name canonicalization) — noted in the PR body.
