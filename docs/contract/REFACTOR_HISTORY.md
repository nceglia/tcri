# TCRI Refactor — History, Decisions, and Hard Rules

This is the single source of truth for **what we decided**, **the order we did things**, **what went
wrong**, and **the rules that must never be violated again**. Read this before touching any refactor doc.

---

## 0. HARD RULES (never violate)

1. **The `example/` notebooks are DISPOSABLE.** They call an old/divergent API and will be rewritten.
   **Never read them, never rely on them, never use them as evidence for ANY decision** — not disposition,
   not "sufficiency," not "is it used," not deletion-safety. A notebook using a function is **NOT** a reason
   to keep it.
2. **Non-core = DROP (delete).** **Nothing moves to `examples/`.** `examples/` is not a destination and is
   not consulted for anything.
3. **Disposition is decided by ONE question: "is it core?"** per the settled design (§2). Core = the model,
   the joint-distribution engine, the four metrics + their plots, session I/O, PPC diagnostics, shared helpers.
   Everything else is dropped.
4. **Sufficiency means:** does a *kept* function's arguments cover *its own responsibility*? Never "did a
   notebook pass this arg."
5. **Never write a workflow prompt that runs a "caller census over notebooks."** That instruction is what
   caused the leak.
6. **Settled decisions are not re-litigated.** They are in §2.

---

## 1. Chronology of what we did (in order)

1. **Full codebase read** → catalogued every function (metrics/preprocessing/plotting/model/utils). Found:
   no `__all__` anywhere; ANSI/console helpers triplicated; `tl`/`pl` twin names; phenotype-prob computed 3×;
   several broken/dead functions; the joint-distribution engine as the substrate.
2. **Target API design** (chat). Created `docs/contract/` with `build_tcri_contract.py` → `tcri_api_contract.*`
   and `build_tcri_depgraph.py` → `tcri_dependency_map.*`. **(CLEAN — code-based only.)**
3. **scvi-tools / scverse investigation.** Decided **Door A** = standalone scverse-ecosystem package;
   Door C (scvi-hub) later; Door B (in-tree `scvi.external`) a possible stretch. Structural reference = the
   `../grafiti` package (one file per topic, private `_state`/`_compute`, explicit `__all__` NOT `import *`,
   `diagnostics` returns DataFrames).
4. **Package validation.** `import tcri` OK; 23 tests pass; the notebooks call a **drifted old API**
   (`phenotypic_entropies`/`clonotypic_entropies` plural, `probability_ternary`, `phenotypic_entropy_delta`,
   `ml.JointProbabilityDistribution`). **Decision: notebooks are a mess to be REWRITTEN clean — never a
   proving ground.** (This is exactly why Rule 1 exists.)
5. **131-function inventory** — 9-agent workflow → `tcri_function_inventory.md` (+ `tcri_inventory_data.json`,
   `build_tcri_inventory.py`), labeled from CODE, critic-verified 0 missing. **(CLEAN input.)**
   *Caveat: its plotting-triage used a "move to examples" disposition — now BANNED by Rule 2; those become DROP.*
6. **Design decisions locked** (see §2).
7. **← LEAK POINT. 13-agent draft→audit→synthesize workflow** → `tcri_api_and_responsibilities.md` +
   `tcri_implementation_plan.md`. The math/stats resolution, prior-vs-mean answer, and GPU architecture it
   produced are CLEAN. **But my audit prompts instructed a "caller census over `example/` + `docs/` notebooks,"
   which resurrected `gene_entropy` (→examples) and `probability_ternary` (→kept) *because notebooks use them*,
   and baked a "deletion-safety gated on notebook callers" rule into §11.**
8. **Consistency/sufficiency** — 6-agent workflow → `tcri_arg_consistency_sufficiency.md`. **Amplified the
   leak** (sufficiency measured against notebooks); the consistency findings themselves are clean.

## 1b. What is CLEAN vs CONTAMINATED

- **CLEAN, keep:** `tcri_api_contract.*`, `tcri_dependency_map.*`, `tcri_function_inventory.md` (+ data)
  *(apply examples→DROP)*. And, *as salvageable content*: the math/stats resolution, prior-vs-mean answer, GPU
  architecture, metric design, layout, rename map — all inside the quarantined docs but not themselves contaminated.
- **CONTAMINATED, quarantined:** `tcri_api_and_responsibilities.md`, `tcri_implementation_plan.md`,
  `tcri_arg_consistency_sufficiency.md`, `tcri_refactor_audit_data.json`, `tcri_arg_audit_data.json`.
  The contamination is confined to the **deletion/disposition** sections (API §11; Plan §3 + Phase 0 + risk rows)
  and the **sufficiency** findings.

---

## 2. Settled design decisions (canonical — do not re-litigate)

**Target:** standalone scverse package (Door A). Layout mirrors grafiti; explicit `__all__` re-export, **no `import *`**;
`_keys.py` now, `@tl_result` uns-cache deferred (build toward it).

**Engine (first-class FUNCTION, not an object):** `tcri.tl.joint_distribution(adata, *, covariate=None, groupby=None,
n_samples=0, use_logits=True, clones=None, temperature=1.0, random_state=None) -> DataFrame`. Re-exported top-level
as `tcri.joint_distribution`. `covariate=None` → all covariate values in one shared-draw pass. Unifies the two old engines.

**Four metrics (tl↔pl twins; `groupby` = aggregation unit, `splitby` = comparison cohort — BOTH kept, distinct):**
`clonotypic_entropy`, `phenotypic_entropy`, `mutual_information`, `phenotypic_flux` (renamed from `flux`).
`compare_groups` is the single public stats helper (replaces `mi_compare`/`*_table`/`*_delta`). **No `*_delta`/`*_table`.**

**Sampling:** `n_samples=0` = deterministic **posterior-MEAN** point estimate (`E_q[p_ct] = get_p_ct() = uns[tcri_p_ct]`);
`n_samples>0` = draws from the exact guide `Dirichlet(clamp(local_scale·m̃, 1e-3))`. Drop `point_estimate=`. `n_samples=0`
and `n_samples>0`-mean are **different estimators** (Jensen gap) — documented, never asserted equal.

**Prior-vs-mean RESOLVED:** the point estimate is the closed-form posterior mean; the **prior is rejected** (argmax-hard-label
init → leakage). The old `posterior=` flag was never prior-vs-posterior — it's a logits-mixing switch → **renamed `use_logits`**.

**Model (ml):** `TCRIModel.setup_anndata / train / get_latent_representation / predict (was get_cell_phenotype_probs) /
get_p_ct`. Streamline: `setup_anndata → TCRIModel → train → model.to_anndata` (thin; writes minimal canonical state;
no manager-in-uns; kills `write_adata_safely`).

**pp** shrinks to `register/to_anndata`, `group_singletons` (separate), `clone_size`.

**diagnostics (diag):** PPCs + model validation, returns DataFrames. Seeds: `joint_distribution_ppc` (fixed
`compare_joint_distribution`), calibration, reconstruction PPC; relocate `loss` (was plot_loss), `archetypes`.

**Shared helpers:** `_keys.py`, `_console.py` (leveled colored logging via scanpy verbosity; drop raw ANSI prints;
**drop `_ascii_hist` and all `graph=`/ASCII paths**), `_stats.py`, `_distance.py`. Sankey primitives → private `plotting/_sankey.py`.

**American spelling everywhere** (`normalized`, `normalize_mode`, `color`). Reproducibility via a seeded `torch.Generator`
named `random_state` (the old `np.random.seed` was a no-op).

**PGM** (`build_nested_tcri_pgm`/`draw_tcri_pgm_nested`) → `docs/`, out of the package.

**DROP (delete) — NOT to examples:** `clonality`, `probability_distribution`, `bayesian_mutual_information`, `polar_plot`,
`compare_phenotypes`, `probability_ternary`, `gene_entropy`, `top_clone_umap`, `clone_size_umap`,
`plot_phenotype_probabilities`, all `*_table`/`*_delta`, `mi_compare`, `clonotypic_entropy_base`, `ridge_delta_entropy`,
`classify_phenotypes`, `get_latent_embedding`, `group_small_clones`, `register_probability_columns`,
`remove_meaningless_genes`, `clone_fraction`, `dkl` (→ `_distance.kl_divergence`), `probabilities`, `_ent`,
`SankeyNode.hex_to_rgb`. (Deletion decided by "not core" — never by notebook callers.)

---

## 3. Open (still parked — not decided by this recovery)

- Whether to keep a `posterior=`/prior axis at all beyond `use_logits` (audit leaned: no).
- Adopt `@tl_result` uns-cache now vs later (recommend: `_keys.py` now, decorator later).
