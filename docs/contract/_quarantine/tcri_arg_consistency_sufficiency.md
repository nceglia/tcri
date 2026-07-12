# TCRI — Argument Consistency & Sufficiency (Final)

**Scope.** The analysis/plot/diagnostic/preprocessing/stats/engine surface defined in `docs/contract/tcri_api_and_responsibilities.md` (§0–§12). Pure internal helpers with no shared-vocabulary arguments (`_console`, priors/module/classifier/training internals, `_register._write_*`, session utils, `_compute` reducers, `resolve_device`/`get_xp`/`asnumpy`) are omitted as all-`—` rows.

**Method.** A full parameter matrix was extracted, every shared concept was checked against the four RULES (same concept → same name/order/default/type; kw-only except the primary positional; American spelling; `groupby`≠`splitby`; `covariate=None`=all; `n_samples=0`=point estimate), then an argument-sufficiency pass compared the surface to the six example notebooks. A verifier then adjudicated every finding against both contract docs and the notebooks.

**This document folds in the verifier's verdicts:** refuted findings are dropped (listed once, for the record); revised findings are restated in their corrected form; the verifier's missed items are added. Each item carries its audit id and status: **[confirmed]**, **[revised]**, or **[missed]**.

**Dropped as refuted (not actionable — recorded so they are not re-raised):**
- **S3 `weighted=`** — deliberately removed and documented (§0.8, appendix row-and-changelog). Behavior change, not an oversight.
- **S8 `pp.gene_entropy(batch_key=)`** — relocated to `examples/` with its notebook (§11), so `batch_key`/`agg_function` travel with it. Not lost.
- **S12 `base=`** — single-base (`log2`/bits) is an intentional unit-consistency fix (§3.4, §7.2/7.3, appendix row 12). Re-adding `base=` would undo it. `decimals=` is cosmetic; drop.

---

## 1. PARAMETER MATRIX

Cells show `name=default` (keyword-only unless noted), `name(req)` = required keyword-only (no default), `POS` = positional, `—` = absent, **DIVERGENT** = the concept exists under a different name/default, **— (missing)** = the concept is applicable but absent.

### Table A — engine / tl / pl / diag / pp core vocabulary

| Function (ns/module) | primary(pos) | covariate | cov_from/cov_to | order | groupby | splitby | n_samples | temperature | clones | use_logits | normalized | normalize_mode | distance_metric | n_clones_ref | random_state | device | palette | ax | figsize | save | return |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tl.joint_distribution | adata | covariate=None | — | — | groupby=None | — | n_samples=0 | temperature=1.0 | clones=None | use_logits=True | — | — | — | — | random_state=None | device=None | — | — | — | — | — |
| tl.clonotypic_entropy | adata_or_jd | covariate=None | — | — | groupby=None | — | n_samples=0 | temperature=1.0 | clones=None | — | normalized=True | — | — | n_clones_ref=None | random_state=None | device=None | — | — | — | — | — |
| tl.phenotypic_entropy | adata_or_jd | covariate=None | — | — | groupby=None | — | n_samples=0 | temperature=1.0 | clones=None | — | normalized=True | — | — | — | random_state=None | device=None | — | — | — | — | — |
| tl.mutual_information | adata_or_jd | covariate=None | — | — | groupby=None | — | n_samples=0 | temperature=1.0 | clones=None | — | normalized=True | normalize_mode="min" | — | — | random_state=None | device=None | — | — | — | — | — |
| tl.phenotypic_flux | adata | — | cov_from(req), cov_to(req) | — | groupby=None | — | n_samples=0 | temperature=1.0 | clones=None | — | — | — | distance_metric="l1" | — | random_state=None | device=None | — | — | — | — | — |
| tl.compare_groups | df | — | — | — | **DIVERGENT: by(req)** | **DIVERGENT: by(req)** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| pl.clonotypic_entropy | adata | covariate=None | — | **— (missing)** | groupby=None | splitby=None | n_samples=0 | temperature=1.0 | clones=None | — | normalized=True | — | — | n_clones_ref=None | random_state=None | **— (missing)** | palette=None | ax=None | figsize=(6,3) | save=None | return_df=False |
| pl.phenotypic_entropy | adata | covariate=None | — | **— (missing)** | groupby=None | splitby=None | n_samples=0 | temperature=1.0 | clones=None | — | normalized=True | — | — | — | random_state=None | **— (missing)** | palette=None | ax=None | figsize=(8,4) | save=None | return_df=False |
| pl.mutual_information | adata | covariate=None | — | **— (missing)** | groupby=None | splitby=None | n_samples=0 | temperature=1.0 | clones=None | — | normalized=True | normalize_mode="min" | — | — | random_state=None | **— (missing)** | palette=None | ax=None | figsize=(8,4) | save=None | return_df=False |
| pl.phenotypic_flux | adata | — | — | order(req) | groupby=None | **— (missing)** | **— (missing n_samples)** | temperature=1.0 | clones=None | — | **DIVERGENT: normalize=True** | — | distance_metric="l1" | — | random_state=None | **— (missing)** | **DIVERGENT: phenotype_colors=None** | ax=None | figsize=(6,3) | save=None | return_axes=False |
| pl.probability_ternary | adata | **— (missing)** | — | — | groupby=None | **— (missing)** | **— (missing)** | **— (missing)** | clones=None | — | — | — | — | — | **— (missing)** | — | palette=None | ax=None | figsize=(5,5) | save=None | return_axes=False |
| pl._metric_boxplot (priv) | adata, function POS | — | — | order=None | groupby=None | splitby=None | — | — | — | — | — | — | — | — | — | — | palette=None | ax=None | figsize=(8,4) | — | — |
| pl._finish (priv) | fig, ax POS | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ax POS | — | save=None | return_axes=False; **show=None (only here)** |
| pl.resolve_palette | adata, columns POS | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | palette=None | — | — | — | — |
| diag.joint_distribution_ppc | adata | covariate=None | — | — | **— (missing)** | — | **— (missing)** | temperature=1.0 | — | — | — | — | distance_metric="l1" | — | **— (missing)** | — | — | — | — | — | — |
| diag.phenotype_calibration | adata | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | (n_bins=10) |
| diag.reconstruction_ppc | model, adata=None POS | — | — | — | — | — | **DIVERGENT: n_samples=100** | — | — | — | — | — | — | — | **DIVERGENT: seed=0** | — | — | — | — | — | — |
| diag.permutation_null | adata | covariate=None | — | — | groupby=None | **— (missing)** | **— (metric="mutual_information"; no metric passthrough)** | — | — | — | — | — | — | — | **DIVERGENT: seed=0** | — | — | — | — | — | (n_permutations=1000) |
| diag.loss | model | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ax=None | — | save=None | (log_scale=False) |
| diag.archetypes | model | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ax=None | — | save=None | — |
| pp.group_singletons | adata | — | — | — | **DIVERGENT default: groupby="patient"** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | (clonotype_key="trb", target_col="trb_unique", min_clone_size=10) |
| pp.clone_size | adata | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | (key_added="clone_size", return_counts=False) |

### Table B — distance / stats / contrast / RNG vocabulary

| Function | primary(pos) | metric-selector | base / eps | by | value | reference | paired | hdi_prob | alternative | seed / random_state | resample count | pos_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared.phenotype_distance (priv) | p, q POS | **DIVERGENT: metric="l1"** | — | — | — | — | — | — | — | — | — | — |
| shared.kl_divergence (priv) | p, q POS | — | base=2.0, eps=1e-12 | — | — | — | — | — | — | — | — | — |
| shared.js_divergence (priv) | p, q POS | — | base=2.0, eps=1e-12 | — | — | — | — | — | — | — | — | — |
| shared.hdi (priv) | samples POS | — | — | — | — | — | — | hdi_prob=0.94 | — | — | — | — |
| shared.summarize (priv) | samples POS | — | — | — | — | — | — | hdi_prob=0.94 | — | — | — | — |
| shared.mann_whitney (priv) | a, b POS | — | — | — | — | — | — | — | alternative="two-sided" | — | — | — |
| tl.compare_groups | df POS | — | — | **by(req)** | value(req) | reference=None | paired=False | hdi_prob=0.94 | alternative="two-sided" | — | — | — |
| shared.auc_and_label_permutation (priv) | scores, labels POS | — | — | — | — | — | — | — | — | **seed=42** | n_perm=200_000 (max_exact=200_000) | pos_label=None |
| shared.bootstrap_auc (priv) | scores, labels POS | — | — | — | — | — | — | — | — | **seed=42** | n_boot=5000 | pos_label=None |
| diag.reconstruction_ppc | model POS | — | — | — | — | — | — | — | — | **seed=0** | n_samples=100 | — |
| diag.permutation_null | adata POS | — | — | — | — | — | — | — | — | **seed=0** | n_permutations=1000 | — |
| _compute._joint_draws (priv) | adata POS | — | — | — | — | — | — | — | — | random_state(kw) | n_samples(kw) | (gate_prob kw; **order/set differs from joint_distribution**) |

### Consistency legend
- **RNG:** `random_state=None` (engine+tl+pl) vs `seed=42` (stats) vs `seed=0` (diag) — 3 spellings/defaults for one concept.
- **distance selector:** `distance_metric="l1"` (tl/pl flux, diag ppc) vs `metric="l1"` (`_distance.phenotype_distance`, private).
- **clones↔temperature order:** engine + `pl.phenotypic_flux` emit `clones` before `temperature`; every other metric emits `temperature` before `clones`.
- **normalize vs normalized:** `normalize=` (distribution bool) only on `pl.phenotypic_flux`; `normalized=` (metric [0,1] scaling bool) everywhere else — a near-collision.
- **device:** present on engine + all tl metrics; absent on every `pl.*` metric plot.
- **splitby:** present on the tl→pl box plots; absent on `pl.phenotypic_flux` and `pl.probability_ternary`.
- **figsize defaults:** (6,3) clonotypic_entropy/flux · (8,4) phenotypic_entropy/MI/_metric_boxplot · (5,5) ternary.
- **overloaded `splitby` (legacy):** on box plots the legacy `splitby` was the cohort-hue; on ternary/flux the legacy `splitby` was the covariate **column** paired with `conditions=`/`order=` **values** — two different concepts under one name.

### Clean — verified consistent (no findings)
`temperature=1.0`; the `n_samples=0` point-estimate convention (single exception: `reconstruction_ppc`); `hdi_prob=0.94`; `alternative="two-sided"`; `use_logits` correctly engine-only; **no `point_estimate=` / `posterior=` survivors**; **no British-spelling residue** anywhere in the target surface.

---

## 2. CONSISTENCY — confirmed inconsistencies + canonical decision

Each item: the divergence → the single canonical name/order/default/type to adopt → the functions that change. Duplicate audit ids are merged. Private-symbol renames are marked *(private → optional/lower stakes)*.

### High

**H1 — RNG fragmentation → `random_state`** *(C1 ≡ S10, confirmed).*
`random_state=None` (engine/tl/pl) vs `seed=42` (`_stats`) vs `seed=0` (diag). One concept, three names/defaults/types.
**CANONICAL:** `random_state=None`, keyword-only, typed `int | numpy.Generator | torch.Generator | None`, placed as the **penultimate** compute arg (immediately before `device`) everywhere it appears. Drop the literal `42`/`0` defaults for `None` (seed internally from the passed generator).
**Load-bearing (public):** `diag.reconstruction_ppc`, `diag.permutation_null` (rename `seed`→`random_state`, default `None`). *(Private → optional):* `_stats.auc_and_label_permutation`, `_stats.bootstrap_auc` may keep `seed` internally but should be documented as the split.

**H2 — `clones`/`temperature` order → temperature-before-clones** *(C2, confirmed; resolve with H8/C12).*
Engine emits `(n_samples, use_logits, clones, temperature)` and `pl.phenotypic_flux` emits `(clones, normalize, temperature)` — `clones` before `temperature` — while every other metric emits `(n_samples, temperature, clones)`.
**CANONICAL** shared block, left→right: `covariate`/`cov_*`, `groupby`, `splitby`, `n_samples`, `temperature`, `clones`, `use_logits` (engine-only, after `clones`), `normalized`/`normalize_mode`/`distance_metric`, `random_state`, `device`. Reorder `tl.joint_distribution` and `pl.phenotypic_flux` to `temperature`→`clones`.

**H3 — `device` absent from every `pl.*` metric plot → add `device=None`** *(C3, revised).*
Confirmed absent on `pl.clonotypic_entropy`, `pl.phenotypic_entropy`, `pl.mutual_information`, `pl.phenotypic_flux`. Revision: on the three box plots the dangling knob is that they expose posterior draws + `random_state` but cannot steer the draw device; on `pl.phenotypic_flux` the finding's "draws-without-device" premise is wrong (flux exposes **no** `n_samples`), so its dangling knob is `random_state` — fix flux by adding **both** `device` and `n_samples` (see S-flux below).
**CANONICAL:** add `device=None` (`str | torch.device | None`, kw-only, penultimate before nothing/after `random_state`) to all four `pl.*` metric plots and thread it into the underlying `tl` call.

**H4 — overloaded `splitby` on ternary/flux → `covariate_key` + `conditions`/`order`, NOT a cohort `splitby`** *(C4 revised ≡ M2 missed).*
`splitby` is genuinely absent from `pl.probability_ternary` and `pl.phenotypic_flux`, but the legacy notebooks passed `splitby=<column>` there to name the covariate **column** (paired with `conditions=`/`order=` selecting the **values**), which is a *different* concept from the box-plot cohort-hue `splitby`. Adding a cohort-hue `splitby` here would re-introduce the exact name-collision the RULES forbid.
**CANONICAL:** reserve `splitby` strictly for the cohort/faceting hue. For ternary/flux add `covariate_key=None` (the covariate **column** override) plus `conditions=None` (ternary: 1–2 covariate levels → start/end simplices) / reuse `order` (Sankey: the ordered value series). Document that legacy `splitby=<column>` maps to `covariate_key`, not `splitby`, in the Phase-10 notebook rewrite.
**Feasibility caveat (must be resolved in the contract):** the engine's joint is built from `p_ct` indexed by the trained covariate (`ct_to_cov`), so a `covariate_key` override is only implementable on an **empirical per-cell-probability aggregation path**, not the `p_ct` engine path. The contract must state which path multi-column flux/ternary use before `covariate_key` is added.

### Medium

**M1 — `n_samples=100` overloads the `n_samples=0` convention → rename to `n_sims`** *(C5, confirmed).*
`diag.reconstruction_ppc.n_samples=100` counts simulated PPC datasets, colliding with the surface-wide `n_samples=0`=point / `>0`=draws convention.
**CANONICAL:** rename to `n_sims=100` (`int`). Reserve `n_samples` (default `0`) exclusively for the posterior-draw convention.

**M2 — distance selector `metric=` vs `distance_metric=`** *(C6, confirmed).*
`_distance.phenotype_distance(metric="l1")` (private dispatcher) vs `distance_metric="l1"` on `tl.phenotypic_flux`, `pl.phenotypic_flux`, `diag.joint_distribution_ppc`.
**CANONICAL:** `distance_metric` is the public name everywhere. Since `phenotype_distance` is private, keep its `metric` **only** as an internal dispatcher detail (documented) — no public churn required.

**M3 — `normalize` vs `normalized` + `phenotype_colors` vs `palette` on `pl.phenotypic_flux`** *(C7, confirmed).*
`normalize=True` (distribution-normalization bool feeding `_phenotype_mass_per_clone`) look-alikes `normalized=True` (metric [0,1] scaling) used everywhere else; `phenotype_colors=None` duplicates `palette`.
**CANONICAL:** rename `phenotype_colors`→`palette`. Rename the flux bool to `normalize_distributions=True` (or drop it if the engine always feeds normalized distributions). Never ship both `normalize` and `normalized` as look-alike names.

**M4 — `order`/`hue_order` inconsistent across the box-plot family → expose both on all three** *(C9 ≡ S2 ≡ S13, confirmed).*
`pl.clonotypic_entropy` has `hue_order` but no `order`; `pl.phenotypic_entropy` and `pl.mutual_information` have neither; the shared `_metric_boxplot` supports `order`; notebooks constantly pass `order=`.
**CANONICAL:** expose `order=None` (x-axis category order) **and** `hue_order=None` (`splitby` level order), both `list[str] | None`, on all three metric box plots, wired to `_metric_boxplot.order` and the `splitby` hue.

**M5 — `compare_groups.by` names the cohort → rename `by`→`splitby`** *(C10, confirmed; defensible-either-way).*
`by` is the column whose levels are contrasted (with `reference`/`paired`) — semantically the comparison **cohort**, i.e. the plotting surface's `splitby`.
**CANONICAL:** rename `by`→`splitby` so the aggregation-unit(`groupby`)/comparison-cohort(`splitby`) vocabulary is uniform metric→plot→contrast. (`by` is pandas-idiomatic; if the team prefers `by`, that is a documented, deliberate exception rather than drift.)

**M6 — `pp.group_singletons` defaults `clonotype_key="trb"`, `groupby="patient"` → keep, document as pre-registration defaults** *(C11, revised).*
Values confirmed, but the original "force `None`/match `unique_clone_id`" fix is wrong: `group_singletons` runs **before** `setup_anndata`, on the raw pre-registration column (`trb` → writes `target_col="trb_unique"`, which only later becomes the registered `clonotype_key`), so `unique_clone_id` doesn't exist yet; its `groupby` is the collapse **unit** (consistent with `groupby`=aggregation-unit), only the *default* differs.
**CANONICAL:** keep `clonotype_key="trb"` and `groupby="patient"` as intentional pre-registration defaults and **document** them as such; do not force `None`.

**M7 — `_joint_draws` keyword order/set diverges from `joint_distribution`** *(C12, confirmed; private → lower stakes).*
`_joint_draws(covariate, clones, n_samples, use_logits, temperature, gate_prob, …)` vs `joint_distribution(covariate, groupby, n_samples, use_logits, clones, temperature, …)`.
**CANONICAL:** make `_joint_draws`' keyword order a strict subset-in-order of the public engine: `(covariate, n_samples, temperature, clones, use_logits, gate_prob, random_state, device)` — i.e. temperature-before-clones (dovetails H2). Resolve `gate_prob` at the public layer or document it as an internal-only extra.

**M8 — `joint_distribution_ppc` arg-order + missing knobs** *(C13 ~ S15, confirmed).*
Orders `(covariate, distance_metric, temperature)` (temperature after the selector) and omits `groupby`/`clones` (and `n_samples`/`random_state`) that its metric siblings expose.
**CANONICAL:** reorder to `(covariate, groupby, clones, temperature, distance_metric)`; add `groupby=None`, `clones=None`. It is model-free/deterministic, so add `n_samples=0`/`random_state=None` **only if** a draw-based comparison is intended (flag as a contract decision, not a default addition).

**M9 — `covariate=None` dual meaning in the `adata_or_jd` fast path** *(C19 ≡ S14, confirmed).*
All signatures default `covariate=None`, but §7.2(c)/§7.9 let the precomputed-`jd` path read `covariate` as "required/the one baked into this jd," contradicting the RULE that `covariate=None`=all covariate values.
**CANONICAL:** `covariate=None` **always** means all covariate values on the adata path. On the precomputed-`jd` fast path, `covariate`/`n_samples`/`temperature`/`clones`/`random_state`/`device` are inert — **raise** a clear `ValueError` if any is set to a non-default — so `covariate=None`=all is preserved only where it is computed. Fix the entropy/MI docstrings and dispatch so `covariate` is never treated as required.

### Low

**L1 — figsize drift across box-plot siblings** *(C8, confirmed).* `pl.clonotypic_entropy=(6,3)` vs `(8,4)` for `pl.phenotypic_entropy`/`pl.mutual_information`/`_metric_boxplot`.
**CANONICAL:** `figsize=(8,4)` for the metric box-plot family; set `pl.clonotypic_entropy` to `(8,4)`. Aspect-driven plots keep purpose-specific defaults, documented as intentional: ternary `(5,5)`, Sankey flux `(6,3)`.

**L2 — 2nd-positional violations → keyword-only** *(C14, confirmed).* `pl.resolve_palette(adata, columns)`, `pl._metric_boxplot(adata, function)`, `diag.reconstruction_ppc(model, adata=None)` each carry a second positional.
**CANONICAL:** insert `*` after the single primary positional so `columns`, `function`, and `adata` are keyword-only. *Census note:* `_phenotype_mass_per_clone(adata, covariate, clones, normalize)` and `TCRIModule.prepare_two_level_params` also carry extra positionals (private, minor) — align in the same pass.

**L3 — resample-count `n_permutations` vs `n_perm`** *(C15, confirmed).* `diag.permutation_null.n_permutations=1000` vs `_stats.auc_and_label_permutation.n_perm=200_000`.
**CANONICAL:** `n_perm` for permutation counts everywhere; rename `permutation_null.n_permutations`→`n_perm` (keep its `1000` default). `n_boot=5000` (bootstrap) is a distinct concept and keeps its name.

**L4 — `show=` only on private `_finish`** *(C16, confirmed).* `_finish(show=None)` is unreachable from any public plot, though all expose `save=`/`return_*`.
**CANONICAL:** adopt the scanpy triad uniformly — thread `show=None` (`bool | None`, kw-only) through **every** public `pl.*` entry point (with `save`/`return_*`), keeping it on `_finish`.

**L5 — return-control name split** *(C17, confirmed; near-self-resolving).* `return_df` on DataFrame plots vs `return_axes` on figure plots.
**CANONICAL:** keep the split (payloads genuinely differ) but **standardize which every plot exposes** — DataFrame/metric plots → `return_df`; figure-only plots (ternary, Sankey) → `return_axes` — and document it. No plot may omit both (all currently comply).

**L6 — `eps` clip-floor drift + internal contradiction** *(C18 confirmed ≡ M3-missed strengthened).* Four floors: `1e-8` (`TCRIModel.predict`, `_compute_logits_and_prior`), `1e-6` (`TCRIModule.prepare_two_level_params`), `1e-12` (`kl_divergence`, `js_divergence`), `1e-15` (hardcoded in `_mi_from_joint`). Sharper than drift: §3.4 and appendix row 12 both assert "one ε=10⁻¹² **library-wide**, matching entropy/MI," yet the MI kernel — the exact metric that claim name-checks — uses `1e-15`. The document contradicts its own stated invariant.
**CANONICAL:** set `_mi_from_joint` `eps=1e-12` to satisfy the stated invariant (single probability-clip floor `1e-12` for the distance/MI/normalization paths). Keep context-specific floors (`predict`/`prepare_two_level_params`) only where numerically justified, and document why. Do not leave prose and kernel contradicting.

### Documentation / cross-document consistency (verifier's missed items)

**D1 — `§11` and the implementation plan disagree on live callers** *(M1, missed).* API §11 line 770 buckets `clonality` (tl+pl), `probability_distribution`, and `clone_fraction` under "Deleted … 0 live callers after census," but the plan (and the notebooks) show live callers: `tcri.pl.clonality` is called with full args (`groupby`/`splitby`/`order`/`palette`/`figsize`) across smith/renal/comparison/yost/zhang; `tcri.metrics.probability_distribution` is imported and called 4× via the alias `pdistribution(psubset, method=…)`. `compare_phenotypes` (1 live call in synthetic1) is dispositioned in the plan (DROP) but is **absent from §11 entirely**.
**CANONICAL:** reconcile §11 with plan §3 — move `clonality`, `probability_distribution`, `clone_fraction` out of the "0 live callers" bucket into a "removed **with** replacement + in-PR notebook rewrite (Phase 6/10)" category; add `compare_phenotypes` to §11; and correct the false "0 live callers after census" label (the census missed alias-imported and package-qualified call-sites).

**D2 — undispositioned legacy renames `method=` and `phenotype_names=`** *(M4, missed).* `method=` (e.g. `method="probabilistic"`) is passed to `pl.clonotypic_entropy`, `pl.mutual_information`, `pl.flux`, `polar_plot`, and `pdistribution`, but only `posterior=`/`point_estimate=` are mapped in §11 — `method=` is never named. `phenotype_names=` is the legacy positional/keyword on `probability_ternary`, and the kept §8.7 signature renames it to `phenotypes` with no rename entry.
**CANONICAL:** add disposition-map rows to §11: `method=` → removed, expressed via `n_samples`/`use_logits`; `phenotype_names` → `phenotypes` on `pl.probability_ternary`.

**D3 — caller-census counts are unreliable** *(M5, missed).* `polar_plot` is recorded as "1 notebook use," but the notebooks contain 4 local `def polar_plot` redefinitions plus multiple call-sites; the same def-plus-call pattern holds for `clonality`. Neither audit disambiguated def-vs-call.
**CANONICAL:** re-run the census distinguishing (a) package-qualified calls, (b) alias-imported calls, (c) notebook-local redefinitions; record accurate per-symbol counts; the Phase-10 rewrite must also strip shadow `def`s, not just swap call-sites.

**D4 — `pl.flux` box-plot disposition unstated** *(S9, revised).* The old per-clone flux-**distance** box-plot `pl.flux(..., paint=, distance_metric='dkl', ...)` is distinct from the Sankey and is used in renal/yost/zhang, but §11 only renames `flux`→`tl.phenotypic_flux` and never states this box-plot's fate. The plan settles it as DROP (the Sankey is the flux plot); re-adding a box variant would contradict that decision.
**CANONICAL:** state explicitly in §11 that the `pl.flux` distance box-plot is **DROP** (Sankey is the flux plot) — a doc-completeness fix, not a new function.

---

## 3. SUFFICIENCY — confirmed missing knobs (exact argument to add)

Every confirmed gap, with the exact `name=default` (type) and the functions that receive it. Refuted gaps (S3/S8/S12) are excluded per §Dropped.

| # | Argument to add | Default | Type | Functions receiving it | Audit id / status |
|---|---|---|---|---|---|
| U1 | `phenotypes` (rename from `phenotype_names`) | (req) | `list[str]` (3 axes) | `pl.probability_ternary` | S1/M4 confirmed |
| U2 | `conditions` | `None` | `list[str] \| None` (1–2 covariate levels → start/end) | `pl.probability_ternary` | S1/M2 confirmed |
| U3 | `scale_function` | `None` | `Callable[[float], float] \| None` (freq → marker size) | `pl.probability_ternary` | S1 confirmed (exercised in every zhang/yost ternary call) |
| U4 | `color` | `None` | `str \| None` | `pl.probability_ternary` | S1 confirmed |
| U5 | `covariate` | `None` | `str \| None` (value; `None`=all) | `pl.probability_ternary` | S1 confirmed (lower priority — historical default, not passed in notebooks) |
| U6 | `top_n` | `None` | `int \| None` | `pl.probability_ternary` | S1 confirmed (lower priority) |
| U7 | `n_samples` | `0` | `int` | `pl.probability_ternary`, **`pl.phenotypic_flux`** | S1, S5 confirmed |
| U8 | `temperature` | `1.0` | `float` | `pl.probability_ternary` | S1 confirmed (lower priority) |
| U9 | `order` | `None` | `list[str] \| None` (x-axis category order) | `pl.clonotypic_entropy`, `pl.phenotypic_entropy`, `pl.mutual_information` | S2 ≡ M4/C9 confirmed |
| U10 | `hue_order` | `None` | `list[str] \| None` (`splitby` level order) | `pl.phenotypic_entropy`, `pl.mutual_information` (already on `pl.clonotypic_entropy`) | S13 ≡ C9 confirmed |
| U11 | `device` | `None` | `str \| torch.device \| None` | `pl.clonotypic_entropy`, `pl.phenotypic_entropy`, `pl.mutual_information`, `pl.phenotypic_flux` | C3 confirmed/revised |
| U12 | `minimum_clone_size` | `None` | `int \| None` (filter `clone_size` before aggregation) | `pl.clonotypic_entropy`, `pl.phenotypic_entropy`, `pl.mutual_information` (optionally the `tl` metrics) | S6 confirmed |
| U13 | `phenotype_subset` | `None` | `list[str] \| None` (restrict rendered phenotype nodes) | `pl.phenotypic_flux` | S5 confirmed |
| U14 | `pair_on` | `None` | `str \| None` (unit column aligning the two levels under `paired=True`) | `tl.compare_groups` | S7 confirmed (closes the incomplete `delta_entropy_table` subsumption) |
| U15 | `covariate_key` | `None` | `str \| None` (covariate **column** override) | `tl.joint_distribution`, `tl.phenotypic_flux`, `pl.phenotypic_flux`, `pl.probability_ternary` | S4/M2 revised — **conditional** on the empirical-vs-`p_ct` path decision (H4 caveat) |
| U16 | metric pass-throughs: `temperature=1.0` (`float`), `normalized=True` (`bool`), `normalize_mode="min"` (`str`), `clones=None` (`list[str] \| None`), `splitby=None` (`str \| None`) | as noted | as noted | `diag.permutation_null` (so the null matches the reported statistic) | S11 confirmed |
| U17 | `clones` | `None` | `list[str] \| None` | `diag.joint_distribution_ppc`, `diag.permutation_null` | S15 ≡ C13 confirmed |
| U18 | `groupby` | `None` | `str \| None` | `diag.joint_distribution_ppc` | S15 ≡ C13 confirmed |
| U19 | `show` | `None` | `bool \| None` | all public `pl.*` plots (scanpy triad) | C16 confirmed |

**Not a new knob, but tied to sufficiency:** `tl.compare_groups` currently reproduces one-axis contrasts but not the paired-**delta-then-cohort-contrast** of `delta_entropy_table` in one call — `paired=True` aligns by `sample_id` (posterior-draw pairing), not pre/post-within-unit pairing, and there is no second cohort axis. `pair_on` (U14) supplies the within-unit pairing key; the §11 claim that `delta_entropy_table` is "expressed via `groupby` + `compare_groups`" must be softened to "with `pair_on` + an explicit cohort split" or documented as a two-pass recipe.

---

## 4. CORRECTED SIGNATURES

Final signatures for every function that changes. Deltas from the current contract are noted after each block. `# †` marks the conditional `covariate_key` (gated on the H4 empirical-path decision).

**`tl.joint_distribution`** — reorder to temperature-before-clones (`use_logits` engine-only, after `clones`).
```python
joint_distribution(
    adata, *,
    covariate=None,          # None → ALL covariate values (one shared draw)
    groupby=None,
    n_samples=0,
    temperature=1.0,
    clones=None,
    use_logits=True,         # engine-only; was posterior=; alias cell_informed=
    random_state=None,       # int | numpy.Generator | torch.Generator | None
    device=None,
) -> pandas.DataFrame
```

**`tl.compare_groups`** — `by`→`splitby`; add `pair_on`.
```python
compare_groups(
    df, *,
    value,
    splitby,                 # was by=
    reference=None,
    paired=False,
    pair_on=None,            # NEW: within-unit pairing key when paired=True
    hdi_prob=0.94,
    alternative="two-sided",
) -> pandas.DataFrame
```

**`tl.phenotypic_flux`** — add optional `covariate_key`.
```python
phenotypic_flux(
    adata, *,
    covariate_key=None,      # † covariate COLUMN override (H4 caveat)
    cov_from, cov_to,
    groupby=None,
    n_samples=0,
    temperature=1.0,
    clones=None,
    distance_metric="l1",
    random_state=None,
    device=None,
) -> pandas.DataFrame
```

**`pl.clonotypic_entropy`** — figsize (6,3)→(8,4); add `order`, `minimum_clone_size`, `device`, `show` (`hue_order` already present).
```python
clonotypic_entropy(
    adata, *,
    covariate=None, groupby=None, splitby=None,
    n_samples=0, temperature=1.0, clones=None,
    normalized=True, n_clones_ref=None, minimum_clone_size=None,
    order=None, hue_order=None,
    palette=None, ax=None, figsize=(8, 4),
    rotation=90, legend_fontsize=6, bbox_to_anchor=(1.15, 1.0),
    random_state=None, device=None,
    show=None, save=None, return_df=False,
)
```

**`pl.phenotypic_entropy`** — add `order`, `hue_order`, `minimum_clone_size`, `device`, `show`.
```python
phenotypic_entropy(
    adata, *,
    covariate=None, groupby=None, splitby=None,
    n_samples=0, temperature=1.0, clones=None,
    normalized=True, minimum_clone_size=None,
    order=None, hue_order=None,
    palette=None, ax=None, figsize=(8, 4),
    rotation=90, legend_fontsize=6, bbox_to_anchor=(1.15, 1.0),
    random_state=None, device=None,
    show=None, save=None, return_df=False,
)
```

**`pl.mutual_information`** — add `order`, `hue_order`, `minimum_clone_size`, `device`, `show`.
```python
mutual_information(
    adata, *,
    covariate=None, groupby=None, splitby=None,
    n_samples=0, temperature=1.0, clones=None,
    normalized=True, normalize_mode="min", minimum_clone_size=None,
    order=None, hue_order=None,
    palette=None, ax=None, figsize=(8, 4), rotation=90,
    legend_fontsize=6, bbox_to_anchor=(1.15, 1.0),
    random_state=None, device=None,
    show=None, save=None, return_df=False,
)
```

**`pl.phenotypic_flux`** — reorder temperature-before-clones; `normalize`→`normalize_distributions`; `phenotype_colors`→`palette`; add `covariate_key`, `n_samples`, `phenotype_subset`, `device`, `show`.
```python
phenotypic_flux(
    adata, *,
    covariate_key=None,               # † covariate COLUMN override (H4 caveat)
    order,                            # ordered covariate value series
    groupby=None,
    n_samples=0,                      # NEW: drives ribbon uncertainty (feeds random_state)
    temperature=1.0,
    clones=None,
    normalize_distributions=True,     # renamed from normalize=
    distance_metric="l1",
    phenotype_subset=None,            # NEW: restrict rendered phenotype nodes
    palette=None,                     # renamed from phenotype_colors=
    ax=None, figsize=(6, 3),
    show_legend=True, title=None,
    random_state=None, device=None,
    show=None, save=None, return_axes=False,
)
```

**`pl.probability_ternary`** — `phenotype_names`→`phenotypes`; add `covariate_key`, `covariate`, `conditions`, `n_samples`, `temperature`, `scale_function`, `top_n`, `color`, `random_state`, `show`. No cohort-hue `splitby` (H4).
```python
probability_ternary(
    adata, *,
    phenotypes,                       # renamed from phenotype_names; the 3 simplex axes
    covariate_key=None,               # † covariate COLUMN override (H4 caveat)
    covariate=None,                   # covariate VALUE (None → all)
    conditions=None,                  # 1–2 covariate levels → start/end simplices
    groupby=None, clones=None,
    n_samples=0, temperature=1.0,
    scale_function=None,              # freq → marker size
    top_n=None, color=None,
    palette=None, ax=None, figsize=(5, 5),
    random_state=None,
    show=None, save=None, return_axes=False,
)
```

**`diag.joint_distribution_ppc`** — reorder temperature-before-`distance_metric`; add `groupby`, `clones`.
```python
joint_distribution_ppc(
    adata, *,
    covariate=None,
    groupby=None,
    clones=None,
    temperature=1.0,
    distance_metric="l1",
    # add n_samples=0 / random_state=None ONLY if draw-based comparison is adopted
) -> pandas.DataFrame
```

**`diag.reconstruction_ppc`** — `adata` keyword-only; `n_samples`→`n_sims`; `seed`→`random_state` (default `None`).
```python
reconstruction_ppc(
    model, *,
    adata=None,
    n_sims=100,              # was n_samples=100
    random_state=None,       # was seed=0
) -> pandas.DataFrame
```

**`diag.permutation_null`** — `n_permutations`→`n_perm`; `seed`→`random_state`; add metric pass-throughs + `clones`.
```python
permutation_null(
    adata, *,
    metric="mutual_information",
    covariate=None, groupby=None, splitby=None, clones=None,
    temperature=1.0, normalized=True, normalize_mode="min",
    n_perm=1000,             # was n_permutations=1000
    random_state=None,       # was seed=0
) -> pandas.DataFrame
```

**`pl.resolve_palette`** — `columns` keyword-only.
```python
resolve_palette(adata, *, columns, palette=None) -> dict
```

**Private helpers (align in the same pass, lower stakes):**
```python
_metric_boxplot(adata, *, function, groupby=None, splitby=None,
                ylabel="", order=None, hue_order=None,
                palette=None, s=20, ax=None, figsize=(8, 4)) -> (fig, ax)   # function kw-only; add hue_order
_joint_draws(adata, *, covariate=None, n_samples=0, temperature=1.0,
             clones=None, use_logits=True, gate_prob=None,
             random_state=None, device=None)                                # subset-in-order of joint_distribution
# _stats.auc_and_label_permutation / bootstrap_auc: optionally seed→random_state (private; document the split)
# _distance.phenotype_distance: keep metric= as an internal dispatcher detail (no public change)
```
Also: `_mi_from_joint` — change hardcoded `eps=1e-15`→`1e-12` (L6). `pp.group_singletons` / `pp.clone_size` — **no signature change** (M6: defaults kept and documented as intentional pre-registration values).

---

## 5. READY-TO-APPLY PATCH LIST — edits to fold into `docs/contract/tcri_api_and_responsibilities.md`

1. **§7.1 engine** — reorder the `joint_distribution` block to `covariate, groupby, n_samples, temperature, clones, use_logits, random_state, device` (temperature before clones; `use_logits` after `clones`). Update the §7.1(c) arguments table row order to match.
2. **§7.6 `compare_groups`** — rename `by`→`splitby` (signature + the "grouping column" comment + Math/Return prose); add `pair_on=None` between `paired` and `hdi_prob`; soften the "subsumes `delta_entropy_table`" claim to require `pair_on` (+ a cohort split) or document the two-pass recipe.
3. **§7.5 `tl.phenotypic_flux`** — add optional `covariate_key=None` (gated on the H4 empirical-path decision; add a one-line note stating the chosen path).
4. **§7.2 / §7.9** — pin `covariate=None`=all covariate values on the adata path; state that the precomputed-`jd` fast path **ignores** `covariate`/`n_samples`/`temperature`/`clones`/`random_state`/`device` and **raises** if any is non-default (remove any "covariate required" reading).
5. **§8.1 `pl.clonotypic_entropy`** — `figsize=(6,3)`→`(8,4)`; add `order=None`, `minimum_clone_size=None`, `device=None`, `show=None`.
6. **§8.1 `pl.phenotypic_entropy`** — add `order=None`, `hue_order=None`, `minimum_clone_size=None`, `device=None`, `show=None`.
7. **§8.2 `pl.mutual_information`** — add `order=None`, `hue_order=None`, `minimum_clone_size=None`, `device=None`, `show=None`.
8. **§8.3 `pl.phenotypic_flux`** — reorder temperature-before-clones; rename `normalize`→`normalize_distributions` and `phenotype_colors`→`palette`; add `covariate_key=None` (†), `n_samples=0`, `phenotype_subset=None`, `device=None`, `show=None`; update the Sankey prose to reference `n_samples`-driven ribbons.
9. **§8.7 `pl.probability_ternary`** — rename `phenotype_names`→`phenotypes`; add `covariate_key=None` (†), `covariate=None`, `conditions=None`, `n_samples=0`, `temperature=1.0`, `scale_function=None`, `top_n=None`, `color=None`, `random_state=None`, `show=None`; do **not** add a cohort `splitby`.
10. **§8.5** — make `_metric_boxplot`'s `function` keyword-only; add `hue_order=None`. **§8.6** — make `resolve_palette`'s `columns` keyword-only.
11. **§8.5 `_finish` / all §8 public plots** — thread `show=None` through every public `pl.*` entry point (scanpy `show`/`save`/`return_*` triad); document the `return_df` (DataFrame) vs `return_axes` (figure) split so no plot omits both.
12. **§9.1 `joint_distribution_ppc`** — reorder to `(covariate, groupby, clones, temperature, distance_metric)`; add `groupby=None`, `clones=None`; note `n_samples`/`random_state` deferred unless draw-based comparison is adopted.
13. **§9.1 `reconstruction_ppc`** — `adata` keyword-only; `n_samples`→`n_sims`; `seed`→`random_state` (default `None`).
14. **§9.1 `permutation_null`** — `n_permutations`→`n_perm`; `seed`→`random_state`; add `splitby=None`, `clones=None`, `temperature=1.0`, `normalized=True`, `normalize_mode="min"` pass-throughs.
15. **§3.3** — optionally rename `_stats.auc_and_label_permutation`/`bootstrap_auc` `seed`→`random_state`, or document that the private primitives keep `seed` while the public diagnostics standardize on `random_state`.
16. **§3.4 / §7.4 / appendix row 12** — set `_mi_from_joint` `eps` to `1e-12`, OR amend the "one ε=10⁻¹² library-wide, matching entropy/MI" claim to carve out the MI-specific `1e-15` with a numerical justification. Resolve the prose↔kernel contradiction either way.
17. **§0.10** — state the canonical RNG type once: `random_state: int | numpy.Generator | torch.Generator | None`, keyword-only, penultimate before `device`; note the `_stats` `seed` exception if kept.
18. **§6.1 `pp.group_singletons`** — add a note that `clonotype_key="trb"` and `groupby="patient"` are intentional pre-registration defaults (operate on the raw column before `setup_anndata`); do not change them.
19. **§11 census (correctness fix)** — move `clonality` (tl+pl), `probability_distribution`, `clone_fraction` **out** of "Deleted … 0 live callers" into a "removed **with** replacement + in-PR notebook rewrite (Phase 6/10)" category; **add `compare_phenotypes`** (currently absent) as DROP-with-rewrite; correct the false "0 live callers" label. Add an explicit disposition line for the **`pl.flux` distance box-plot: DROP** (Sankey is the flux plot).
20. **§11 renamed/removed map** — add rows: legacy `method=` → removed (expressed via `n_samples`/`use_logits`); `phenotype_names` → `phenotypes` on `pl.probability_ternary`. Note that the caller census must distinguish package-qualified calls, alias-imported calls, and notebook-local `def` redefinitions (`polar_plot`, `clonality`), and that Phase-10 must strip shadow definitions.
21. **§0.11 (new, cross-cutting note)** — record the H4 open decision: because `p_ct` is covariate-locked (`ct_to_cov`), any `covariate_key` column override on flux/ternary must be computed on the **empirical per-cell-probability path**, not the `p_ct` engine path; the contract must state which path these plots use before `covariate_key` (†) ships.