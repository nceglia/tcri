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
| 4 | Model→AnnData streamline | ✅ | HIGH | 1,3 | session round-trip |
| 5 | Engine consolidation | ✅ | HIGH | 4 | joint identities |
| 6 | Metric-API consolidation | ✅ | HIGH | 5 | metric tests |
| 7 | Plotting split + pl twins | ✅ | medium | 6,1 | twins render |
| 8 | `diag/` seeding | ✅ | low-med | 4,5 | PPC columns |
| 9 | PGM→docs; utils finalize | ✅ | low | 1,8 | import green sans daft |
| 10 | Public API + scverse CI | ☐ | low-med | all | ecosystem checklist |

## Removal Ledger (the hard bar — every one MUST end deleted)
Tick only when the symbol is gone from source AND `__all__`/imports AND `import tcri` is green.

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
- [x] `_ascii_hist` (dead: zero callers) · [x] `ml.plot_loss` (→ `diag.loss`) · [x] `ml.plot_archetypes` (→ `diag.archetypes`)

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
- **Deferred:** **[E]** `reconstruction_loss_scale=1e-3` vs eq-7 full weight (over-generation symptom) — author deferred; may be an intentional β-VAE reweighting, and raising it needs a retrain + R/NR revalidation. Tracked as a follow-up investigation. **[F]** in-silico perturbation (eqs 8–12) not implemented (additive).
## PR 10 — Public API + scverse CI  ·  ☐ todo
- **Logged test (from grafiti parity):** once `pl.__all__` exists, add a conformance assertion `set(pl.__all__) == {pl entries in _contract.pyi}` — catches *extra/missing* plot functions (whole-surface), not just signature drift on onboarded ones. (tcri's namespaced `.pyi` checks drift incrementally via `IMPLEMENTED`; this closes the whole-surface gap grafiti gets from its markdown+`__all__` channel.)

---

# AUDIT LOG
_(dated entries; what was audited, findings, actions)_

- **(MODEL PR — methods-conformance audit vs Supplementary Note 1 — WORKFLOW, 6 lenses × 2 adversarial verifiers, 53 agents):** **22 findings survived adversarial verification, 1 refuted.** Verdict: the classifier fix (ELBO factor + global-index alignment target) is **correct and faithful to the note's surrogate**. Caught a real **HIGH the local run missed**: the new `test_model_classifier.py` leaked the process-global Pyro param store → the *full* suite was RED (1 failed / 88 passed) though the file passed in isolation — fixed with a **module-local** autouse `clear_param_store` fixture (a conftest autouse would wipe the session-scoped `trained_model`). **Fixed now:** dead `class_weights`/`phenotype_weights` removed ([D]); `indices=None` silent fallback → `assert` (re-hardens A2); dead `encoder(x)` forward in `model()` removed; `gate_prob: Optional[float]`; surrogate-KL **sign** clarified in the doc (code's `−γ·KL` realizes the note's `+γ` *penalty* intent under SVI-maximization); KL-warmup z-only scope + `num_particles` enumeration-only scope documented; round-trip now guards the new scalars. Suite **89 passed**. **Confirmed CONFORMANT:** ZINB (eq 5), β (eq 2), VampPrior (eq 3), gated ℓ rule in `predict`. **Deferred (author sign-off, change fitted results):** [E] `reconstruction_loss_scale=1e-3` vs eq-7 full weight; [G] α not applied to the eq-1 clonotype prior (prior/guide scale mismatch). [F] perturbation (eqs 8–12) additive/not implemented.
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
