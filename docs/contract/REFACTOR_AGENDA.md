# TCRI Refactor — Agenda, Tasklist & Diary (LIVING DOC)

**Update this every working session. Refer back to it before starting anything.** It is the operational
tracker + running diary for the whole refactor. The detailed spec lives in `API_CONTRACT.md`; the
rules/history in `REFACTOR_HISTORY.md`; the scratch/deferred pile in `REFACTOR_NOTES.md` (not checked
in).

## How to use this doc
1. **Removal is a hard bar.** The default failure mode is keeping old code around and over-engineering to
   preserve it. **Do the opposite.** Every function in the Removal Ledger must actually be deleted, its
   `__all__`/import sites cleaned, and the checkbox ticked. If keeping something "just in case" feels tempting —
   don't. Delete it; git has it.
2. **Never read the `example/` notebooks.** They are disposable and are an *output* of the refactor, never an
   input. No caller census, no "is-it-used," no "sufficiency."
3. **The DIARY and AUDIT LOG are append-only history.** An entry records what was true when it was written; do not rewrite one to reflect a later state. When it is overtaken, annotate it `**[SUPERSEDED <date> — see ...]**` and leave the original text. **Live status lives in `METHODS_CONFORMANCE.md` and `SANCTIONED_DEVIATIONS`, never here** — this file is read before starting work, so a stale 'deferred' sends someone to redo closed work. Enforced by `tests/test_docs_deviation_status.py`.
4. **After every PR, run the Standing Audit** (below) and write a diary entry.
5. **Frequent audits**: at minimum after each PR; ideally mid-PR when a component is touched. Log them in the
   Audit Log.
6. **Usability is a first-class check** — every session ask "is this easier to use than before?"

## What a green suite does not catch

Two bugs shipped through a green suite and were found only by **rendering the figures and
looking at them** (#80, #81):

- `resolve_colors` assigned colour by POSITION, not level. Within one figure,
  `phenotypic_entropy` drew NR purple while `phenotypic_flux` beside it drew R purple. The
  existing test called it twice with the same order, so the property it was written to protect
  was never actually exercised.
- the delta endpoints size legend read "clones matched: 4" on a 4-phenotype fixture. For a
  phenotype-item metric the matched clone count is not in `result` at all.

Both were in code written specifically to prevent those failures. That is the point worth
keeping: **the suite verified the structure, and the structure was right.** The values flowing
through it were wrong in a way no shape assertion can see. Where a function's output is a
figure, the figure is the artifact under test — render it and look before calling the work
done, and add the discriminating assertion once you know what to assert.

## Standing Audit (run after each PR — copy into the diary entry)
- [ ] **Removed everything slated?** (cross-check the Removal Ledger; `__all__` + import-sites clean; `import tcri` green)
- [ ] **Added everything wanted?** (the PR's deliverables all present)
- [ ] **Tests:** what components can now be unit-tested that weren't? Added?
- [ ] **Streamline:** any duplication / dead branch / needless complexity spotted? Removed or logged?
- [ ] **Usability:** simpler signatures / clearer errors / fewer steps than before?
- [ ] **Contract conformance green?** (`test_contract_conformance`)
- [ ] **If the change affects a figure, RENDER IT and look.** See "What a green suite does not
      catch" above — twice now, a green suite passed a plot that was visibly wrong.
- [ ] Diary entry written; ledger + statuses updated.

## Status legend: ☐ todo · ◐ in progress · ✅ done · ⚠ blocked

## PR Agenda
| # | PR | Status | Risk | Depends | Gate |
|---|---|---|---|---|---|
| 0 | Contract freeze + CI scaffolding | ✅ | none | — | conformance green |
| 1 | Shared helpers + `_keys` | ✅ | low | 0 | existing tests green |
| 2 | Safe deletions | ✅ | very low | 1 | import-graph clean |
| 3 | Model module split | ✅ | low | 1 | model/pyro tests green |
| 4 | Model→AnnData streamline | ✅ | HIGH | 1,3 | session round-trip |
| 5 | Engine consolidation | ✅ | HIGH | 4 | joint identities |
| 6 | Metric-API consolidation | ✅ | HIGH | 5 | metric tests |
| 7 | Plotting split + pl twins | ✅ | medium | 6,1 | twins render |
| 8 | `diag/` seeding | ✅ | low-med | 4,5 | PPC columns |
| 9 | PGM→docs; utils finalize | ✅ | low | 1,8 | import green sans daft |
| 10 | Public API + scverse CI | ☐ | low-med | all | ecosystem checklist |

## Repo cleanup  ·  ✅ done (own pass, no PR — 2026-08-18)

All four items closed. `docs/contract/` went from 33 files to 11. Kept below in full rather than
compressed to ticks, because three of the four turned up a **generalisable failure mode** and those
are the point: a rename that silently drops a governance rule, a generated doc outliving its
generator's input, and a future-tense contract clause that never expired.

- ✅ **Re-audited the Phase 5/6 removal ticks.** The rule was right, the tick was wrong: nobody
  checked that the named replacement covered the same *axis*, only that something was named. A tick
  must now say **what** it replaces, not just **with what**. Result of re-reading all nine:

  | Removed | Named replacement | Verdict |
  |---|---|---|
  | `pp.joint_distribution_posterior` | `tl.joint_distribution(n_samples>0)` | ✅ same axis, strictly wider |
  | `metrics._mi_from_joint` | the `_compute` engine | ✅ internal, same axis |
  | `tl.mi_compare` | `compare_groups` | ✅ **verified in code this pass** |
  | `tl.flux_table` | `tl.phenotypic_flux` | ⚠️ axis covered, one column dropped |
  | `tl.clonotypic_entropy_base` | `tl.clonotypic_entropy` | ✅ strictly wider |
  | `tl.clonality` | `tl.clonotypic_entropy` | ✅ **deleted** (verified gone); the tick's stated reason was wrong |
  | `tl.delta_clonotypic_entropy`, `tl.delta_entropy_table` | `compare_groups` | ❌ known bad — rebuilt in #81 |
  | plural `*_entropies` shims, `metrics/` dir | renames / deletion | ✅ |

  **`mi_compare` clears.** The worry was that it enumerated all group pairs and the replacement only
  handled a two-level split. It does not: `compare_groups` builds `itertools.combinations` over the
  `splitby` levels exactly as `mi_compare` did over `groupby`, and adds a `reference=` mode it never
  had. Same axis, wider.

  **`flux_table` drops `clone_size`.** `tl.phenotypic_flux` covers the metric axis (per-clone
  distance `cov_from`→`cov_to`, one row per clone per draw, `splitby` via the shared machinery), but
  `flux_table` also carried a `clone_size` column and **computed it within each `splitby` group**
  (`adata_g.obs[clone_col].value_counts()`). `pp.clone_size` writes a **global** count over the whole
  object, so joining it back is *not* equivalent — a user who wants the old column must subset first.
  It is denormalised convenience data rather than a metric, so the scope principle leaves it to the
  user; recorded here so it is not rediscovered as a defect.

  **`tl.clonality` — deleted, and the deletion is not in question.** Re-verified this pass: no `def`
  anywhere in `tcri/`, no mention in any module, not on `tl.__all__`, not importable. The ledger tick
  is accurate.

  What was wrong is only the **justification written beside the tick**, which named
  `tl.clonotypic_entropy` as the replacement. That is not what replaced it. `clonality` used the
  **observed (hard)** clone and phenotype labels — `1 − H/log₂K` on raw clone counts per phenotype,
  needing no model at all — while `tl.clonotypic_entropy` has no hard-label path (`n_samples=0` is
  the deterministic plug-in of the *posterior*). Naming it as the replacement asserts an equivalence
  that does not hold: it changes the **estimator**, which is the same class of substitution that made
  the delta tick wrong. The correct justification is the scope principle — computing clonality needs
  no metric applied at a level the public surface hides; it is a `value_counts` and a Shannon entropy
  on `obs`, so it belongs to the user. **The tick should have said that**, and a replacement that
  changes the estimator needs its own justification rather than inheriting one.

- ✅ **Renamed `tcri_api_and_responsibilities.md` → `API_CONTRACT.md`**, so all four contracts are
  named for what they are. Referrers updated: `CLAUDE.md`, `README.md`, `tcri/_contract.pyi`,
  `tcri/tools/_joint.py` — **and `.github/CODEOWNERS`, which the first sweep missed** because it
  matched on file extension and CODEOWNERS has none. That rule is what makes the API contract
  maintainer-approval-only; for the length of that sweep it pointed at a path that no longer
  existed and therefore protected nothing. A rename that silently drops a governance rule is the
  failure mode to watch: **grep for the old name with `-I` across the whole tree, not by
  extension.**
- ✅ **Issues do not live in the contract — swept, and the problem was smaller than assumed.**
  Across all five contract documents there were **six** tracker-flavoured passages, not the pile the
  item implied. Four were correct as written and stay: the `paired=True` estimand (§7.6) and the
  training contract's two open questions are *genuinely pending decisions*, which the rule permits;
  the Sankey enhancement (§8.4) already points at its own issue, which is the pattern we want; and
  the eqs 8–12 perturbation row in `METHODS_CONFORMANCE.md` is deviation history, not a to-do.

  **§7.7 of the API contract was the real defect, and it was worse than stale prose.** It was titled
  "forward-compat with the **deferred** `@tl_result` uns-cache" long after `@tl_result` shipped, and
  it specified three carriers that do not exist in the code: a `_provenance` JSON-string column, a
  `df.attrs["params"]` durable copy, and a cache key hashed over the argument list. The `attrs`
  mechanism had not merely failed to ship — it was **removed during implementation** because attrs
  does not survive most pandas operations (`tcri/tools/_joint.py:220` says so), so the contract was
  mandating the exact carrier the code had already rejected on evidence. Anyone conforming new work
  to §7.7 would have been conformed to the wrong design by the document whose job is to prevent that.
  Rewritten to describe what shipped (`{table, result, stats, params, version}` siblings, a `draw`
  column, `_encode_index` for MultiIndex), with the correction recorded inline.

  **The lesson generalises past this section.** A "deferred" or "forward-compat" clause is a claim
  about the future, and nothing makes it expire when the future arrives by a different route. The
  same disease produced `tcri_dependency_map.md` announcing itself as "the post-refactor API".
  **Every contract clause written in the future tense needs an owner and a re-read at the moment the
  thing lands** — the landing PR is the trigger, not a later audit.

- ✅ **Stale and duplicate artifacts in `docs/contract/` — 22 files, ~1.5 MB, removed** (33 files → 11).
  Deleted: the three generated pre-refactor snapshots and *their generators*
  (`tcri_function_inventory.md` + `tcri_inventory_data.json` + `build_tcri_inventory.py`;
  `tcri_api_contract.md`/`.html` + `build_tcri_contract.py`;
  `tcri_dependency_map.md`/`.dot`/`.html` + `build_tcri_depgraph.py`), the superseded plans
  (`tcri_implementation_plan.md`, `tcri_consistency_sufficiency.md`, `TRAINING_SPEC_DRAFT.md`,
  `REDO_LIST.md`, `PR_PLAN.md`), the `_quarantine/` directory, and a 0-byte `.Rhistory`.
  `docs/contract/README.md` was rewritten around the four contracts — it had been describing two of
  the deleted files as "the authoritative pair".

  **The generators were the finding.** The three `build_*.py` scripts read as live tooling, so the
  obvious move was "regenerate rather than delete". They do not introspect the package — none of them
  imports `tcri` — they print hardcoded tables, which is why `build_tcri_inventory.py` still reports
  **131 functions** for a package that now has 39 public ones. Re-running them would have re-emitted
  the same stale documents with a fresh timestamp, which is worse than deleting them: it launders a
  pre-refactor snapshot as current. That is what `tcri_dependency_map.md` had already become —
  it announced itself as "the post-refactor API" while its call graph carried 49 references to
  `tl.delta_clonotypic_entropy`, `tl.delta_entropy_table`, `pl.ridge_delta_entropy` and
  `pp.joint_distribution`, all deleted in PR6/PR7. **A generated doc is only as live as its
  generator's input; check what the generator reads before trusting "just regenerate it."**

## Removal Ledger (the hard bar — every one MUST end deleted)
Tick only when the symbol is gone from source AND `__all__`/imports AND `import tcri` is green.

**Machine-checked since the cleanup pass — `tests/test_removal_ledger.py`.** The ticks below were
hand-maintained and nothing verified them, which is how the `delta_clonotypic_entropy` tick stayed
wrong for four PRs and why "is it still gone?" could only ever be answered by someone re-checking by
hand. 43 removed symbols are now asserted absent from their public namespace, one test per symbol.
Two entries are deliberately narrower: `compare_groups` is asserted *internal* (removed from `tl`,
still importable from `tcri.tools._compare` — the ledger's own wording is "it is not dead, it is no
longer a step the user performs"), and the reinstated delta pair is asserted **present**, so that a
failure cannot be "fixed" by quietly re-adding them to the removed list. **When you tick a row here,
add the symbol there.**

**Phase 2 (dead / out-of-scope):** ✅ ALL DELETED (PR2) — 14 symbols, 384 lines, `import tcri` green.
- [x] `pp.get_latent_embedding` · [x] `pp.group_small_clones` · [x] `pp.register_probability_columns`
- [x] `pp.remove_meaningless_genes` · [x] `pp.gene_entropy` · [x] `pp.classify_phenotypes`
- [x] `pl.polar_plot` · [x] `pl.probability_distribution` · [x] `pl.bayesian_mutual_information`
- [x] `metrics._ent` · [x] `tl.clone_fraction` · [x] `metrics.dkl` (→ `_distance.kl_divergence`)
- [x] `ut.probabilities` (+ its `_plotting.py` import, same PR) · [x] `SankeyNode.hex_to_rgb`

**Phase 4 (folded into `to_anndata` / session):** ✅ functions folded + manager stash retired (PR4).
- [x] `pp.register_model` (→ `model.to_anndata`) · [x] `pp.register_phenotype_key` · [x] `pp.register_clonotype_key`
- [x] `pp._compute_logits_and_prior` · [x] `ut.write_adata_safely` · [x] `ut._pop_nonserializables`
- [x] uns key `tcri_manager` (retired at `setup_anndata`; `test_model_setup` asserts it's gone)
- [x] `tcri_clone_key` / `tcri_phenotype_key` / obsm `X_tcri_phenotypes` — **DONE**. The deferral said "still read by not-yet-refactored `metrics`/`plotting`", but those were rewritten in PR6/PR7; the last reader was a raw literal in `pp.clone_size`, now migrated to `uns[METADATA][CLONE_COL]` with a clear error. `to_anndata` no longer writes the shims and the three `LEGACY_*` constants are deleted (`LEGACY_MANAGER` stays — `save_tcri_session` still pops it defensively).

**Phase 5/6 (consolidated away — delete WITH replacement, never before):**
- [x] `pp.joint_distribution_posterior` · [x] `metrics._mi_from_joint` (verified: no `def` remains in `tcri/`)
- [x] `tl.mi_compare` · [x] `tl.delta_clonotypic_entropy` · [x] `tl.delta_entropy_table` · [x] `tl.flux_table`
- [x] `tl.clonotypic_entropy_base` · [x] `tl.clonality` · [x] `tl.dkl` local `dkl_func`
- [x] plural `*_entropies` shims · [x] `metrics/` package — the `.py` files went in PR6 but an **empty dir with a `.DS_Store` survived**; removed now

**Phase 7 (non-core plots — DROP, not to examples):**
- [x] `pl.probability_ternary` · [x] `pl.top_clone_umap` · [x] `pl.clone_size_umap` · [x] `pl.plot_phenotype_probabilities`
- [x] `pl.compare_phenotypes` · [x] `pl.ridge_delta_entropy` · [x] `pl.flux` boxplot · [x] `pl.clonality` plot
- [x] `pl.tcri_boxplot` (→ `_metric_boxplot` → `_boxstrip`) · [x] `pl.set_color_palette` (→ `resolve_palette` → `resolve_colors`)
- [x] `pl.plot_pheno_sankey` (→ `_sankey`) · [x] leaked aliases `centropy`/`pentropy`/`*_tl`

**Phase 9 (out of the package):**
- [x] `ut.build_nested_tcri_pgm` (→ `docs/`) · [x] `ut.draw_tcri_pgm_nested` (→ `docs/`) · [x] `daft` runtime dep

**Phase 3/9 (model/utils cleanup):**
- [x] `_ascii_hist` (dead: zero callers) · [x] `ml.plot_loss` (→ `diag.loss`) · [x] `ml.plot_archetypes` (→ `diag.archetypes`)

**Metrics store-once (passes 1–2):**
- [x] the precomputed-joint path — `adata_or_jd` on three metrics, `is_precomputed_joint`,
  `reject_stacked_covariate_joint`, API doc §7.9 · [x] `joint_distribution(groupby=)`
  (declared at the first freeze, never implemented) · [x] `df.attrs["params"]` on the joint
- [x] `tl.compare_groups` off the public surface (code kept as an internal helper with one
  caller; it is not dead, it is no longer a step the user performs)
- [x] `pl.resolve_palette` (→ `resolve_colors`) · [x] the duplicate 30-entry `tcri_colors` in
  `utils/_utils.py` · [x] `_stats.eti` (no caller outside its own test)
- [x] every metric argument on every `pl` signature — `covariate`, `groupby`, `splitby`,
  `n_samples`, `weighted`, `normalize_mode`, `distance_metric`, `clones`, `temperature`
- [x] the four per-metric copies of the `groupby` loop (→ `_common.metric_table`)

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
- **What happened:** verified up front that **no code outside `model/` references any moved internal** (only `TCRIModel` is imported externally). Extracted the 7 top-level defs via `ast.get_source_segment` (formatting-preserving) into the target files along the clean dependency DAG `_classifier`/`_priors` (leaf) → `_module` → `_training` → `_model`. Each module declares an explicit `__all__` (a Phase-3 deliverable — plan §Phase 3), so `tcri.model.*` is now pinned to exactly the public API the frozen contract promises — `{TCRIModel}` — and the incidental third-party re-export leaks the old `import *` exposed (`pyro`/`dist`/`Encoder`/`KMeans`/… 17 names, none tcri-defined, none referenced anywhere) are no longer surfaced. Applied the `c2p_mat → clone_phenotype_prior` rename with a word-boundary regex (13 sites; left the unrelated `c2p_torch` local and the module buffer `clone_phen_prior` untouched). Dropped 3 provably-dead top-level imports surfaced by the per-file import rebuild (`setup_anndata_dsp`, `cosine_similarity`, the `torch.distributions` `Categorical/Dirichlet/MixtureSameFamily` trio — all uses were `dist.`-prefixed pyro). File sizes: `_model` 462, `_module` 326, `_training` 154, `_priors` 147, `_classifier` 21.
- **Issues & fixes:** (1) my first smoke silently imported a **stale `site-packages/tcri`** copy (a script's dir, not the repo, leads `sys.path`) and failed in `setup_anndata` on an old-copy/scvi mismatch — a red herring; forcing the repo copy onto the path, the smoke passes. (The stale install is an env-hygiene note, not a code issue — pytest already uses the repo copy, which is why the suite validates the split.) (2) The train path was **entirely uncovered** by the suite (`trained_model` fixture defined but unused) — so the split was only import-verified. Fixed by adding a real end-to-end smoke test.
- **Added:** ✅ explicit `__all__` in all 5 model modules (`_model`=`{TCRIModel}`; siblings export their own class(es)). ✅ `tests/test_model_smoke.py` — construct → train (2 epochs) → `get_latent_representation` / `get_p_ct` / `get_cell_phenotype_probs` (asserts shapes + prob normalization), plus asserts the `clone_phenotype_prior` rename landed and `build_archetypes` returns centers **and** labels. Runs in ~1s inside the suite.
- **Removed (hard bar):** n/a — PR3 is a structural split, not a removal PR. `ml.plot_loss`/`ml.plot_archetypes` stay on `TCRIModel` until `diag/` exists (Phase 8); no Phase-2 style deletions here.
- **Test opportunities:** ✅ closed the biggest gap (model construct/train/query now covered). The rewritten `test_session_round_trip` (Phase 4) will extend this to save/load.
- **Streamline:** the split makes Phase 4 (model→AnnData) and Phase 5 (engine) tractable — the pyro module, priors, classifier, and training plan are now editable in isolation.
- **Usability:** each file now has a docstring stating its role; the model file reads as a clean `BaseModelClass` API surface.
- **Deferred (logged):** **M5** (`build_archetypes` default `K=4` vs `TCRIModel` `K=10`) — behavior-neutral today (the model always passes `K=10` explicitly), reconciled with persisted `labels` when `diag.archetypes` lands (Phase 8). Not touched here to keep the split purely mechanical. Also deferred (auditor's own recommendation): the stale `c2p_mat` descriptors in the contract **generator** (`build_tcri_contract.py:81,267`) + regenerating the contract HTML — bundled with the Phase-8 `diag.archetypes`/M5 pass (they describe that future function). The inventory rename-table row (`c2p_mat → clone_phenotype_prior`) is correct and stays.
- **Audit (workflow — 3 lenses × adversarial verify, 8 agents):** 2 lenses PASS, plan-contract FIX. 5 findings, **all confirmed, all LOW/MED** — no behavior/correctness defect (behavior lens confirmed byte-identical class bodies + zero F821 undefined-names + suite green). Fixed here: the **MED** — explicit `__all__` per module (plan §Phase 3) was omitted — now added to all 5 files, which also resolves the two LOW "surface not byte-for-byte" findings (surface is now the explicit `{TCRIModel}`; diary wording corrected; the 3 `# noqa: F401` re-exports removed as no longer needed). Suite 36 passed / 1 skipped.
## PR 4 — Model→AnnData streamline  ·  ✅ done (branch `refactor/pr4-model-anndata`)
- **Goal:** kill the `uns['tcri_manager']` hack; fold `register_model → model.to_anndata` (writing the full canonical set incl. new `GATE_PROB`/`CLASSIFIER_TEMPERATURE`); rename `get_cell_phenotype_probs → predict` (labelled DataFrame); rewrite the round-trip gate. Behavior change.
- **Env (prereq):** built a fresh py3.12 venv on the latest scverse stack (anndata 0.13.1, scanpy 1.12.2, **scvi-tools 1.5.0**, torch 2.13, numpy 2.4, **pandas 3.0.3**), pinned in `requirements.txt`; all runs use `.venv`. One pandas-3.0 compat fix (legacy `tcri_boxplot` `groupby.median()` positional `numeric_only`). Suite green in the new env.
- **What happened:** `setup_anndata` → keyword-only, returns `None`, no manager stash (registration only). `predict` → labelled `DataFrame` (obs_names × phenotypes), `eval()` → deterministic. `to_anndata` → metadata + categories, `P_CT`/`CT_TO_COV`/`CT_TO_C`/per-cell `CT_ARRAY`/`COV_ARRAY`, `LOCAL_SCALE`, **`GATE_PROB`**, **`CLASSIFIER_TEMPERATURE`**, `X_TCRI`/`X_LOGITS`/`X_LOGPOSTERIOR`, `X_PROBABILITIES` (from `predict`) + argmax labels. Deleted the register cluster + `write_adata_safely`/`_pop_nonserializables` (inlined h5ad write). Onboarded the 6 `TCRIModel` methods into the contract (`IMPLEMENTED`) — conformance now enforces the model surface.
- **Issues & fixes:** the rewritten round-trip test passed in isolation but failed in the full suite — the **process-global pyro param store** (§5.2) is clobbered by other model-training tests, so the session-scoped `trained_model`'s store isn't its own at save-time. Fixed with a function-scoped `fresh_trained_model` fixture that trains inside the test and owns the store.
- **Correction to the PR3 note:** my "train path was uncovered" claim was wrong — the fixture is `trained_model` (I grepped the wrong name `fitted_model`), and `test_session_round_trip` consumes it, so train WAS covered. The PR3 smoke is still a faster/targeted guard; the framing was off.
- **Added:** ✅ rewritten `test_session_round_trip.py` (canonical write-set · setup-obs invariant · reloaded model reproduces p_ct/latent/predict) ✅ `dev/real_data_to_anndata.py` — **LOCAL-only, gitignored, not CI** — builds the 50-largest patient-specific clones of yost (`trb_unique = trb+patient`; 7682 cells / 10 patients) and runs setup→train→to_anndata (canonical keys OK; prior-driven recovery 0.90).
- **Removed (hard bar):** ✅ `register_model`, `register_phenotype_key`, `register_clonotype_key`, `_compute_logits_and_prior`, `write_adata_safely`, `_pop_nonserializables`; `uns['tcri_manager']` stash. Legacy `tcri_clone_key`/`tcri_phenotype_key`/`X_tcri_phenotypes` deferred to Phase 6/7 (live readers).
- **Test opportunities:** the **Model knob-test matrix** below (new deliverable). **Two dead knobs surfaced** — see "Model correctness debt."
- **Streamline / Usability:** `predict` returns a labelled DataFrame; `to_anndata` is one call replacing the heavy `register_model`; save/load is plain h5ad.

---

# MODEL KNOB-TEST MATRIX  ·  ✅ COMPLETE (`tests/test_model_knobs.py`, 32 tests)

**Two layers, and the split is the point.**

- **WIRING** — does the value actually reach the object it configures? This layer was
  added after `lr` sat marked "hooked up" for months while never reaching Pyro's
  optimizer. The model still converged, so *every behavioral test passed*. A silently
  ignored knob is invisible to convergence testing; assert the plumbing directly.
- **BEHAVIOR** — the mathematically-correct input→output assertion.

| Knob | Wiring test | Behavior test | Status |
|---|---|---|---|
| `n_latent` | `module.n_latent` | latent width == n_latent | ✅ |
| `n_pseudo_obs` | `vamp_prior.pseudo_inputs.shape[0]` | — | ✅ |
| `K` | `mixture_concentration.shape[0]` | `centers.shape[0] == K` | ✅ |
| `n_hidden`,`n_layers` | encoder param count scales | — | ✅ |
| `global_scale` (α) | `module.global_scale` | scales the eq-1 prior concentration ([G] fix) | ✅ |
| `local_scale` (β) | `module.local_scale` | p_ct draw variance == p(1−p)/(β+1) | ✅ |
| `prior_temperature` | `module.prior_temperature` | T>1 raises `clone_phen_prior` row-entropy | ✅ |
| `guide_temperature` | `module.guide_temperature` | T<1 lowers `get_p_ct()` row-entropy | ✅ |
| `gate_prob` (π) | `module.gate_prob` | π=0 ⇒ softmax(log φ); π=1 ⇒ softmax(f_cls) | ✅ |
| `classifier_temperature` | `module.classifier_temperature` | logits(T=2) == logits(T=1)/2 | ✅ |
| `classifier_dropout` | `classifier.mlp[2].p` | — | ✅ (was ⛔ not-plumbed) |
| `classifier_hidden`,`classifier_n_layers` | layer widths / Linear count | — | ✅ (was ⛔ untrained) |
| `kl_weight_max` | `module.kl_weight_max` | ramp ceiling | ✅ |
| `guide_init_scale` | `module.guide_init_scale` | — | ✅ |
| `phenotype_kl_weight` (γ) | `module.phenotype_kl_weight` | classifier recovery (model contract) | ✅ |
| `lr`,`weight_decay`,`betas`,`eps` | **`plan.optim.pt_optim_args`** | recovery/ELBO move with lr | ✅ **(was DEAD)** |
| `n_steps_kl_warmup` | `plan.n_steps_kl_warmup` | `kl_weight` ramps 0→max, monotonic | ✅ |
| `reconstruction_loss_scale` | `module.reconstruction_loss_scale` | — (deviation [E]) | ✅ wiring |
| `max_epochs`,`patience` | `trainer.max_epochs`, EarlyStopping `.patience` | — | ✅ |
| `batch_size` | dataloader batch shape | `predict` invariant (float32 tol) | ✅ |
| `use_enumeration` | selects `TraceEnum_ELBO` vs `Trace_ELBO` | — | ✅ |
| `phenotype_weights` | — | — | **REMOVED** (was dead; deleted in Phase 1a) |

**Findings from the run:** no new dead knobs. The two initial failures were both
*test* bugs, not code bugs — scvi installs `LoudEarlyStopping` (not `EarlyStopping`),
and `predict` differs by ~1.2e-07 across batch sizes (float32 kernel paths, not a
logic error). Previously-⛔ classifier knobs are unblocked now that the classifier
trains. `n_steps_kl_warmup`'s step-vs-epoch semantics remain open as **DUX-2**.

## PR 5 — Engine consolidation  ·  ✅ done (branch `refactor/pr5-engine`)
- **Goal:** build the unified `joint_distribution` engine (`tools/` + `_compute/`) that every metric will consume — the substrate. **Additive** this PR: the old `joint_distribution`/`joint_distribution_posterior` stay until Phase 6 migrates the metrics onto the new engine. HIGH risk (math-heavy).
- **What happened:** created `_compute/_xp.py` (torch-first device seam — CPU / torch-CUDA, lazy GPU import, `asnumpy` boundary; grafiti parity but torch-first since the draws are torch), `_compute/_joint.py::_joint_draws` (the `[S, n_clones, P]` core), `_compute/_reduce.py` (batched entropy/MI, **bits/log2 default** per the user, float64 accumulators), `tools/_joint.py` (the DataFrame wrapper), re-exported `tcri.joint_distribution`. Onboarded `tl.joint_distribution` into the contract (added `device`).
- **Engine invariants implemented (§7.1):** temper the base **once** (`T==1` is the *exact* identity — no eps round-trip, so both closed-form identities are exact/near-exact); **draw over all ct rows once then slice per covariate** (shared-draw invariant → draw-count == `n_samples` regardless of #covariates); `n_samples=0` deterministic; `use_logits=True` folds per-cell logits with `log(base)` **gate-aware** and scatter-adds per clone (matching `predict`); `weighted` scales clone rows by the **ct-keyed** cell count (fixes the old clone-indexed `Counter` bug); torch-seeded Dirichlet (`random_state`).
- **Verified (the gate — `tests/test_tools/test_joint.py`, all green):** `use_logits=False,n=0,T=1 == uns[P_CT]` restricted (**dev 0.0**); `use_logits=True,n=0,T=1 == predict` per-clone aggregation (**dev 4e-8**; compared to the *frozen* `X_PROBABILITIES` so it's store-independent); `n=0` bit-identical; `n>0` seeded-reproducible + Dirichlet mean→base; weighting == clone cell counts; `covariate=None` slice == per-covariate call (shared draw); provenance JSON-serializable. Plus `test_reduce` (entropy of uniform == `log2(P)` bits; MI independent==0, coupled==1 bit), gate-aware combine (direct), T≠1 temper, and the subset/`local_scale` guards. Suite **66 passed**.
- **Added:** ✅ `_compute/{_xp,_joint,_reduce}.py` ✅ `tools/{__init__,_joint}.py` ✅ `tcri.joint_distribution` re-export ✅ `tests/test_tools/{test_joint,test_reduce}.py`.
- **Removed (hard bar):** n/a — additive PR; the old engines are deleted in Phase 6 (with the metric migration), per the ledger.
- **Deferred (deliberate scope boundary, logged):** **`groupby`** — raises `NotImplementedError` for now. Its correct semantics (full-space cell/clone restriction, the clone-determined guard, and sharing the draw across groups) are substantial and land with the Phase-6 metric consumers + `_metric_boxplot` (Phase 7) that actually exercise it. `_compute/_reduce` is created but not yet consumed (Phase-6 metrics use it). The `+1e-8`-vs-clamp draw bug is **fixed** in the new engine (`clamp(local_scale·base, 1e-3)`); the old engines keep the bug until they're deleted.
- **Decision baked in (user):** all entropy/MI default to **bits (log2)**, consistent with `_distance`.
## PR 6–9 — Metrics · Plotting · diag · PGM  ·  ✅ done (branch `refactor/pr6-9`, one PR at the end)
_(committed together on a single branch per the `/goal` directive — one multi-commit PR at the end.)_
- **PR 6 (metric-API consolidation):** the 4 metrics + `compare_groups` rewritten onto the PR5 engine in `tools/` (bits/log2; `normalize_mode='min'` default; support-only/NaN fixes). `groupby` done at the metric level via the engine's `clones=` restriction (**with a clone-disjointness guard** — added after the audit — that raises if a clone spans groups, instead of silently contaminating). `tl` repointed `metrics`→`tools`; 5 metrics onboarded into the contract. **Removed:** `mi_compare`, `delta_*`, `flux_table`, `clonotypic_entropy_base`, `clonality`, `_mi_from_joint` (old), the old `metrics/` package, and the old `preprocessing` `joint_distribution`/`joint_distribution_posterior` engines.
- **PR 7 (plotting):** split the 1437-line `_plotting.py` into `_base`/`_colors`/`_entropy`/`_mutual_information`/`_flux` (explicit `__all__`); the 4 tl↔pl twins are **cache renderers** over the tidy `tl` results (no slice-and-call). `resolve_palette` mutates in place. Dropped the non-core plots + `tcri_boxplot`/`set_color_palette`.
- **PR 8 (diag):** `diagnostics/_ppc.py` (`joint_distribution_ppc`, `phenotype_calibration`, `reconstruction_ppc`, `permutation_null` — all → DataFrame) + `_training.py` (`loss`, `archetypes`, relocated off `TCRIModel`; `plot_loss`/`plot_archetypes` deleted). 6 onboarded into the contract.
- **PR 9 (PGM→docs):** moved `build_nested_tcri_pgm`/`draw_tcri_pgm_nested` out to `docs/model_pgm.py`; dropped `daft` from runtime deps (`import tcri` no longer imports daft).
- **Gate:** full suite **82 passed** (default pytest config) in the pinned venv; contract conformance green (all metrics + diag enforced).
- **Deferred (logged):** engine `groupby` param (metrics use the metric-level path); `_provenance` sidecar + GPU guardrails (PR5); the §7.2–§7.6 api-doc code blocks still lag the frozen `.pyi` (reconcile in the docs pass); the classifier-training fix (its own PR); flux Sankey (twin renders a box for now).
## Model PR — classifier training + methods conformance  ·  ✅ done (branch `model/classifier-fix`)
_(the deferred "classifier-training fix, its own PR" above — plus a full audit of the model vs the Supplementary Methods note.)_
- **The classifier now trains.** Two coupled bugs: (1) `cls_logits` never entered the ELBO → added `pyro.factor("phenotype_alignment", −γ·KL(probs‖φ))` in `model()` (the note's "Inference Details" surrogate, `γ=phenotype_kl_weight`); (2) the alignment target `φ=p_ct[ct_idx]` was indexed by the **local** pyro plate index → scrambled labels across shuffled minibatches → `f_cls` collapsed to a constant. Fixed by threading **global** cell `indices` through `_get_fn_args_from_batch → model()/guide()`; the `indices=None` path now `assert`s rather than silently falling back. Pure-classifier (gate=1.0) recovery on the perfect dataset: **0.200 (chance) → 1.000**.
- **Note conformance (new `docs/contract/METHODS_CONFORMANCE.md`).** Eq-by-eq code↔note map + deviation table. Fixed alongside: `gate_prob` default `None→0.5` (π), `classifier_dropout` plumbed into `PhenotypeClassifier`, dead `class_weights`/`phenotype_weights` removed (3 signatures), dead `encoder(x)` forward in `model()` removed.
- **Tests:** new `tests/test_model_classifier.py` (perfect-recovery guard at gate 1.0 + 0.5, asserts f_cls weights actually move — with a module-local param-store isolation fixture); round-trip now guards `phenotype_kl_weight`/`gate_prob`/`classifier_dropout`. Full suite **89 passed**.
- **[G] fixed (author-approved):** α (`global_scale`) now applied to the eq-1 clonotype prior (`expanded_conc = global_scale * centroids`), removing the prior/guide scale mismatch; classifier recovery unchanged (1.000), suite green.
- **Deferred:** **[E]** `reconstruction_loss_scale=1e-3` vs eq-7 full weight (over-generation symptom) — author deferred; may be an intentional β-VAE reweighting, and raising it needs a retrain + R/NR revalidation. Tracked as a follow-up investigation. **[F]** in-silico perturbation (eqs 8–12) not implemented (additive). **[SUPERSEDED 2026-08-07 — [D], [E] and [G] are resolved; see METHODS_CONFORMANCE.md and SANCTIONED_DEVIATIONS for live status.]**
## Metrics store-once, pass 1 — the `tl` layer  ·  ✅ done (branch `feat/metrics-migration`)
_(pass 2 — `pl` reading `tcri.get`, `compare_groups` internal, colours, `eti`/AUC — is the next branch.)_
- **All five `tl` store once.** Each returns `{table, result, stats}` and writes the same object
  to `uns[key_added or "tcri_<metric>"]` with a `params` provenance block, via the `@tl_result`
  decorator that landed in #73. `joint_distribution` is one of the five: it is probabilistic
  (`n_samples`, `temperature`, `weighted`, `random_state`) and was the only `tl` that recorded
  nothing about how it was derived.
- **The precomputed-joint path is deleted** — `adata_or_jd`, `is_precomputed_joint`,
  `reject_stacked_covariate_joint`, and the §7.9 API-doc section. It was declared in the first
  contract freeze (`7599959`), implemented because it was declared, and had exactly one caller
  in the repo. Its root cause was `joint_distribution` returning a naked DataFrame; once every
  `tl` stores its result there is nothing to hand back in.
- **Shared machinery in `tools/_common.py`** — `resolve_groupby`, `validate_splitby`,
  `metric_table`, `build_result`, `build_stats`, `across_groups`. The group loop lives in one
  place so the four metrics cannot diverge on what `groupby` means (the divergence #64 keeps
  producing).
- **`stats` is its own slot, not inline columns.** The plan said follow grafiti and put
  statistics in columns of the tidy frame. Implementing it showed the pattern does not
  transfer: grafiti's tests are per-motif, so "one row per thing tested" and "one row per
  result row" coincide. tcri's contrast is BETWEEN SPLITS — a different cardinality from
  `result` — so columns would either broadcast one contrast across every group row or pick a
  row to hang it on. Recorded at `tools/_common.py`.
- **Tests:** 289 with `--runslow` (baseline on `main`: 256), 276 fast. `test_recovery`'s four
  oracle/invariance tests now call `_mi_from_joint` directly — they score a count table built
  from `obs`, so routing them through a public tool added a store step to a test with no
  AnnData and hid which function the oracle was compared against.
- **Bugs the migration surfaced and fixed:** `build_result` grouped by a named list of label
  columns, which silently averaged away `cov_from`/`cov_to` and lost the flux's endpoints — it
  now groups by every non-`{draw,value}` column; `metric_table` carried an all-NaN `covariate`
  column on flux.
- **Standing Audit:** removed everything slated ✅ (precomputed path + `df.attrs["params"]` +
  `joint_distribution(groupby=)`); deliverables present ✅; new unit tests for the axes, the
  cache, and the no-clobber guard ✅ (each mutation-checked); duplication removed ✅ (four
  copies of the group loop → one); usability ✅ (one return shape instead of
  float/Series/DataFrame/dict by axis); contract conformance green ✅ (59 passed);
  `METRICS_CONTRACT.md` + `_metrics_contract.py` updated in the same PR ✅.
- **Deferred to pass 2 (not a regression, but load-bearing):** `pl` and `diagnostics` still
  recompute their metric. They now pass `inplace=False` so a plot cannot overwrite the user's
  cached result under the plot's own arguments — including the `groupby`
  `pl.mutual_information` manufactures from `batch_col`. Guarded by a test.

## Metrics store-once, pass 2 — `pl`, colours, surface  ·  ✅ done (branch `feat/metrics-pass2`)
- **`pl` no longer computes.** Every twin is `(adata, key=, display args)`. It reads the result
  `tl` stored, so the covariate, groupby, splitby, n_samples and distance it renders are the
  ones `tl` actually used. Three defects die structurally rather than being fixed: the
  `tl`/`pl` `distance_metric` disagreement (`"kl"` vs `"l1"`, so the flux axis label and the
  numbers under it could describe different quantities); the `groupby` manufactured from
  `batch_col` whenever the caller passed none, which grouped the figure by a column nobody
  named and made an ungrouped MI unreachable; and a plot disagreeing with the frame in hand.
- **`stats` reaches the figure.** The contrast is bracketed with its stars, and only where
  `stats` has a row for that exact pair of x levels — so an R-vs-NR contrast cannot appear
  over the phenotype axis.
- **One contrast implementation.** `build_stats` collapses items to groups (the
  pseudoreplication step) then delegates to `compare_groups`, which leaves the public surface.
  A test pins the delegation so the two cannot drift. `across_groups` is now wired: `stats`
  carries `ci_*`/`sd_*`/`n_*` per arm beside `result`'s within-group `hdi_*`.
- **One palette.** `resolve_colors` replaces `resolve_palette`, which had no way to READ an
  existing assignment and so reassigned on every call — the same patient changed colour
  between two figures in one notebook. Persists under scanpy's `uns["<key>_colors"]`, so a
  palette set here also colours `sc.pl.umap`.
- **`_stats`:** `eti` deleted; `auc_and_label_permutation` + `bootstrap_auc` exported as
  `tcri.ut.*`. The surface conformance test failed the moment they went public — the contract
  working as intended, not an obstacle.
- **Tests:** 304 fast / **317 with `--runslow`** (289 after pass 1; 256 on `main` before).
  New `cohort` fixture — 6 patients, 3 per arm — because a contrast needs replicates: with one
  patient per arm a Mann-Whitney returns p=1.0 whatever the data says, so nothing downstream
  of `splitby` was testable at all.
- **Mutation testing found two real gaps**, which is the point of running it:
  1. the pseudoreplication collapse was only tested on `mutual_information` — the one metric
     with *no* item axis, where the collapse is a no-op. `per_group = result` passed. It is
     now parametrized over the item-bearing metrics, where the defect actually bites.
  2. `pl` reading `batch_col` instead of the cached `params` was invisible because `batch_col`
     IS `"patient"` in every fixture. The test now breaks the registry and asserts the plot
     does not notice.
  A third (`x == splitby` before annotating) is redundant with the level match inside
  `_annotate_contrasts` — mutating either alone leaves the figure correct. Recorded in the
  docstring rather than left implying it is independently verified.
- **Standing Audit:** removals ticked ✅ (ledger updated); deliverables present ✅; new tests
  for the renderer contract, the palette, the delegation and the across-group tier ✅;
  duplication removed ✅ (two contrast implementations → one, two palettes → one);
  usability ✅ (`pl` signatures went from ~16 arguments to 9, none of them computable);
  contract conformance green ✅ (60 passed); API doc §7.9 + §7.10 and the nine stale code
  blocks regenerated from live signatures ✅.
- **Deferred, with issues:** `compare_groups`'s paired branch has no producer — it wants a
  frame whose cells are draw vectors. Kept, not deleted, because `table` makes a paired
  posterior contrast genuinely reachable now and which estimand it should use is a question
  for the authors. Sankey rebuild; `tcri.ut` star-import with no `__all__`.

## Mark rule + the paired entropies  ·  ✅ done (branches `feat/pl-mark-rule`, `feat/metrics-delta`)

**The mark rule.** A mark shows ONE variance component: within an x position the sample is the
coarsest unit that varies there, replicate > item > draw. Measured defect it fixes, on
`phenotypic_entropy(groupby="patient", splitby="response")` — the box and strip described
**47 clones** while the p-value bracketed above them described **6 patients**. That is the
pseudoreplication `build_stats` collapses away, surviving in the marks.
`collapse_to_replicates` is extracted and called by both, so they cannot disagree by
construction. `_bars` becomes `_points` (a bar's area encodes a magnitude from zero these
metrics do not have), and draws are drawn as violins rather than summarised into a whisker.

**The paired entropies.** `delta_clonotypic_entropy` and `delta_phenotypic_entropy`:
`value(cov_to) − value(cov_from)` per item, within a draw. Two functions, not three — MI has
no item axis, so a "ΔMI" is a subtraction of two cached scalars and belongs to the caller.
This restores a capability deleted in PR6 whose nominal replacement, `compare_groups`, was on
a different axis entirely. Support is the intersection within each replicate, which for the
clonotypic metric also makes `log2(C)` cancel; seeding follows flux so a self-delta is exactly
0. No `p_gt`. `pl.delta_*(kind="delta"|"endpoints")`, with the endpoints view rendered from the
delta result so the unmatched version of that figure is unreachable.

- **Tests:** 340 fast, 352 `--runslow` (from 256 on `main` before this run of work). Nine
  mutations checked across the two branches, each failing exactly its own test.
- **Rendering the figures found two bugs the tests could not.** `resolve_colors` assigned
  colour by POSITION, so within one figure `phenotypic_entropy` drew NR purple while
  `phenotypic_flux` beside it drew R purple — the exact bug that function exists to prevent,
  missed because the test called it twice with the same order. And the delta endpoints legend
  read "clones matched: 4" on a 4-phenotype fixture: for a phenotype-item metric the matched
  clone count is not in `result` at all. Both fixed, both now covered.
- **Two of my own test bugs, worth recording:** an HDI need NOT contain the mean (it is the
  narrowest interval holding 94% of the mass, so an outlying draw is excluded from the
  interval while still pulling the mean — the median is the invariant); and a test asserting a
  cached key is absent must clear it, since `cohort` is session-scoped and the assertion
  otherwise passes or fails on file ordering.
- **Standing Audit:** removals ✅ (`_SELF`/dead `subkey`); deliverables ✅; new tests for the
  mark rule, the strip unit, the connector prohibition, colour stability and the whole delta
  family ✅; duplication removed ✅ (one collapse, one contrast, one palette); usability ✅;
  contract conformance green ✅ (64); contract written BEFORE the code ✅.

## PR 10 — Public API + scverse CI  ·  ☐ todo
- **Logged test (from grafiti parity):** once `pl.__all__` exists, add a conformance assertion `set(pl.__all__) == {pl entries in _contract.pyi}` — catches *extra/missing* plot functions (whole-surface), not just signature drift on onboarded ones. (tcri's namespaced `.pyi` checks drift incrementally via `IMPLEMENTED`; this closes the whole-surface gap grafiti gets from its markdown+`__all__` channel.)

---

# AUDIT LOG
_(dated entries; what was audited, findings, actions)_

- **(GOAL RUN — metrics contract · knob test · [E] · legacy-key removal; self-audited):** five ordered items, all landed on `model/contract`. **1) Entropy verification → HALTED as instructed.** The code differs from Supplementary Note 1 eqs 3–4, and the *note* is wrong: both equations weight by the **marginal** while taking the log of the **conditional** (a cross-entropy), and eq 4's left-hand side is mislabelled `H(p(c))` while its right side sums over φ. Proof the code is right: MI must satisfy `I(c;φ)=H(c)−E_φ[H(c|φ)]`; on a test joint with true MI **0.288703** the implemented conditional entropy reproduces it exactly while the literal formula gives **−0.345883** — a negative MI. Author confirmed: keep the code, record the erratum. **2) Metrics contract (separate, by my call).** `_metrics_contract.py` + `METRICS_CONTRACT.md` + 12 identity tests; separate from the model contract because that one is verified by *tracing* `model()`/`guide()` while metrics are pure functions pinned by *numeric identities*. Keystone identity is the decomposition above. Fixed a real wart found en route: `np.where(p>0, p*log2(p), 0)` still evaluates `log2(0)` (numpy computes both branches) — masked before the log. **3) Knob test completed with a new WIRING layer** — 32 tests. The layer exists because `lr` sat marked "hooked up" for months while dead; convergence tests cannot see a silently-ignored knob. **No new dead knobs found**; both initial failures were *test* bugs (scvi installs `LoudEarlyStopping`, not `EarlyStopping`; `predict` differs ~1.2e-07 across batch sizes = float32 kernel paths, not logic). **4) [E] re-measured** now that the phantom optimizer is gone: real-yost library ratio **1.40 at 1e-3 → 0.99 at 1e-2**; recovery (1.000) and latent separation (7.19) unchanged, so the default was raised. The synthetic reads 1.00 everywhere and **could not** have detected this — only the real 1000-gene/87%-dropout data discriminates. Three inconsistent defaults (1e-3/1e-3/1e-2) unified. **5) Deferred Phase-4 legacy keys removed** — the deferral ("still read by metrics/plotting") was stale after PR6/PR7; the last reader was a raw literal in `pp.clone_size`, now on `uns[METADATA][CLONE_COL]` with a real error message. `_ascii_hist` deleted (zero callers). Ledger ticked. **SELF-AUDIT (mutation test of the new contract):** 5 mutations — note's-literal weighting, bits→nats, zero-mass→0, epsilon-clip, MI min→average — **all caught**. Note the first attempt at the note's-literal mutation was a **no-op** (`col[supp].sum() == col.sum()`, so it changed nothing) and appeared to "escape"; re-done with the true marginal it fails correctly. Mutations must be verified to actually change behavior before their result is trusted. Suite **152 passed**; all three conformance tests green (27 / 12 / 12). **[SUPERSEDED 2026-08-07 — [D], [E] and [G] are resolved; see METHODS_CONFORMANCE.md and SANCTIONED_DEVIATIONS for live status.]**

- **(MODEL PR — methods-conformance audit vs Supplementary Note 1 — WORKFLOW, 6 lenses × 2 adversarial verifiers, 53 agents):** **22 findings survived adversarial verification, 1 refuted.** Verdict: the classifier fix (ELBO factor + global-index alignment target) is **correct and faithful to the note's surrogate**. Caught a real **HIGH the local run missed**: the new `test_model_classifier.py` leaked the process-global Pyro param store → the *full* suite was RED (1 failed / 88 passed) though the file passed in isolation — fixed with a **module-local** autouse `clear_param_store` fixture (a conftest autouse would wipe the session-scoped `trained_model`). **Fixed now:** dead `class_weights`/`phenotype_weights` removed ([D]); `indices=None` silent fallback → `assert` (re-hardens A2); dead `encoder(x)` forward in `model()` removed; `gate_prob: Optional[float]`; surrogate-KL **sign** clarified in the doc (code's `−γ·KL` realizes the note's `+γ` *penalty* intent under SVI-maximization); KL-warmup z-only scope + `num_particles` enumeration-only scope documented; round-trip now guards the new scalars. Suite **89 passed**. **Confirmed CONFORMANT:** ZINB (eq 5), β (eq 2), VampPrior (eq 3), gated ℓ rule in `predict`. **Deferred (author sign-off, change fitted results):** [E] `reconstruction_loss_scale=1e-3` vs eq-7 full weight; [G] α not applied to the eq-1 clonotype prior (prior/guide scale mismatch). [F] perturbation (eqs 8–12) additive/not implemented. **[SUPERSEDED 2026-08-07 — [D], [E] and [G] are resolved; see METHODS_CONFORMANCE.md and SANCTIONED_DEVIATIONS for live status.]**
- **(PR0 ✅):** agenda + removal ledger established; standing-audit checklist defined. Contract frozen (27 fns) + conformance guardrail live. Full suite 26 passed / 1 skipped, zero regressions.
- **(PR1 ◐):** shared-helper foundation created (`_keys`/`_console`/`_stats`/`_distance`) + 8 unit tests. Caught & fixed an `hdi` off-by-one before it shipped. **Adoption pending** (dedup, stats-move, `K.*` migration) — no ledger items ticked yet; foundation is additive, suite green. Logged: key-literal test (PR1), `pl.__all__` whole-surface test (PR11).
- **(PR0+PR1 multi-agent audit — 3 lenses):** verdict FIX. Caught a real regression — the `K.*` find/replace over-reached into **10** display/warning/docstring strings (`register_model`/`load_tcri_session` printed `"K.X_LOGITS"` etc.). **Fixed:** restored readable key text in all 10 (AST-span, delimiter-safe); made the key-literal guard **AST-based** (checks real subscripts/`.get`, ignores prose); removed 3 dead `utils` imports the audit flagged. Suite 35 passed. Two non-blocking items deferred to `REFACTOR_NOTES` (contract↔api-doc reconciliation; helper-name canonicalization) — noted in the PR body.
- **(PR2 multi-agent audit — 3 lenses):** PASS on all three (doc↔code · deletion safety · plan/contract). Independently re-derived: all 14 deletions have zero call-sites, all on-plan Phase-2/DROP, none in `_contract.pyi`. 3 LOW items fixed before push (orphaned `cosine_similarity` import; a −129→−127 count; a stale plan line calling `classify_phenotypes` a Phase-4 fold).
- **(PR7–9 + live-plan audit — WORKFLOW, 2 lenses × adversarial verify, 10 agents):** **8 confirmed (all MED/LOW).** Fixed: **the `pl.*` surface is now contract-enforced** (reconciled `_contract.pyi` `resolve_palette` + onboarded all 5 pl.* into `IMPLEMENTED`); **`reconstruction_ppc` now mirrors the module's generative distribution exactly** (`module.eps` + `nb_logits` clamp `[-10,10]`) so the PPC samples from what the model defines; api-doc §9.1 param names reconciled; the live plan now states its **underpowered-subset caveat** and uses an explicit `pre→post` flux direction. **Deferred (LOW):** `joint_distribution_ppc` per-covariate aggregate lives in `df.attrs`; a few surface pieces (`save/load` round-trip, `group_singletons`) not hit by the live plan (covered by CI tests). Suite **87 passed**.
- **(LIVE R/NR TEST — real yost data, `dev/live_test_rnr.py`):** **16/16 steps OK** on a 2259-cell / 76-clone / 10-patient subset (R+NR, pre+post, 6 clusters). Train → `to_anndata` → all 6 `diag` → all 4 metrics + `compare_groups` (R vs NR) → all 4 `pl` twins (figures render). Only failure was a **test-harness** pandas bug (`reset_index`), fixed. Diagnostics surfaced two genuine **model-quality** findings (ZINB decoder over-generates counts ~6×; classifier untrained → calibration reflects the prior) — both already tracked as model debt, **not refactor defects**. Detail in `REFACTOR_NOTES`.
- **(PR6 audit — WORKFLOW, 2 lenses × adversarial verify, 12 agents):** **7 confirmed (2 MED correctness/infra, 5 doc/schema).** Fixed: the metric `groupby` now **validates clone-disjointness** and raises on a clone spanning groups (was silent cross-group contamination) + `synthetic_adata` clones made patient-disjoint; test `__init__.py` added (prepend-mode basename collision); `groupby` on a precomputed joint raises the §7.9 `ValueError`; `compare_groups` unified column schema (keeps `p_lt`); added `n_clones_ref` to `clonotypic_entropy` (+ contract). Remaining doc-only: the §7.2–§7.6 code blocks lag the frozen `.pyi` (reconcile in the docs pass). PR7–9 audit runs next (combined).
- **(PR5 audit — WORKFLOW, 3 lenses × adversarial verify, 14 agents; correctness lens math-focused):** **11 findings confirmed, 0 refuted — 2 MED, 9 LOW.** Both MED were real safety gaps, **fixed here:** (C4) restored the subset/filtered-AnnData length guard the old engine had — a sliced AnnData now errors instead of silently misaligning full-space `uns` vs subset `obsm`; (C8) `n_samples>0` now **raises** on a missing `uns[LOCAL_SCALE]` instead of silently defaulting to 1.0. Also fixed: made `clones=` ordering consistent across the single- and MultiIndex paths, and added the missing tests — the **gate-aware combine** (direct `_joint_draws` unit test; the fixture is gate=None), the T≠1 temper, and both new guards (66 passed). Reconciled stale api-doc §4.1/§4.2 (`get_xp`→`torch_device`; the real `_joint_draws(decomposed args)->(blocks,n_draws)`). **Deferred (LOW, logged):** the `_provenance` sidecar column (§7.7, with the Phase-6 cache) and GPU guardrails 5/7/8 (§7.4, with the GPU path). The correctness lens confirmed the core math (exact `P_CT` identity, `predict` parity dev 4e-8, clamped-Dirichlet draw, ct-keyed weighting, shared-draw invariant, bits/log2).
- **(PR4 audit — WORKFLOW, 4 lenses × adversarial verify, 23 agents):** the standard three (doc↔code · correctness · plan/contract) **plus a dedicated knob-test-plan lens**. **19 findings confirmed, 0 refuted — all LOW/MED, zero HIGH.** The correctness lens independently **confirmed the streamline is behavior-preserving** (44 passed) and reproduced both dead-knob findings (classifier untrained; `class_weights` dead). **Fixed here:** requirements count (36→44), a value-equality assertion for `LOCAL_SCALE`/`GATE_PROB`/`CLASSIFIER_TEMPERATURE` in the round-trip, `pyro.clear_param_store()` in the `trained_model` fixture, api-doc `to_anndata` signature reconciled to the contract, and **7 knob-matrix corrections** — the Dirichlet variance is `∝ 1/(scale+1)` (not `1/scale`); `classifier_dropout` split out as a *separate* not-plumbed knob; gate=1 *formula* is testable now (only recovery blocked); `guide_temperature` needs a post-train test; scale-variance tests co-located; `kl_weight_max`/`reconstruction_loss_scale` and `use_enumeration` justifications sharpened. **Deferred (LOW/MED, logged in `REFACTOR_NOTES`):** the `group_singletons`-ordering guard, `predict` order hardening, round-trip exclusivity (Phase 6/7), and the stale `build_tcri_depgraph.py`.
- **(PR3 audit — WORKFLOW, 3 lenses × adversarial verify, 8 agents):** behavior + doc-code lenses PASS, plan-contract FIX. **5 findings, all confirmed, all LOW/MED — zero behavior/correctness defect.** Behavior lens verified class bodies are byte-identical to the pre-split monolith (modulo the sanctioned rename), zero F821 undefined-names across all 5 files (every import header complete), and suite/smoke green. **Fixed:** the MED — explicit `__all__` per module (plan §Phase 3, line 279) was omitted → added to all 5 files (also resolves the two LOW surface-wording findings; surface now pinned to `{TCRIModel}`). **Deferred (auditor-recommended):** stale `c2p_mat` in the contract *generator* → Phase 8 with `diag.archetypes`. Suite 36 passed / 1 skipped.
