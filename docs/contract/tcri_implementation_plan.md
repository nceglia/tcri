# TCRI Refactor — Implementation Plan (Final)

**From** the current five-monolith package **to** the settled Door‑A, scverse‑ecosystem target (grafiti‑mirrored, one‑file‑per‑topic), with every audit fix folded in.

**Source of truth.** The *SETTLED DESIGN* block is authoritative; the *prior‑vs‑mean RESOLUTION* settles the one parked statistical question and is adopted verbatim. Where the settled design diverges from `docs/contract/tcri_function_inventory.md` §3/§9 (engine lives in `tl` not `pp`; `groupby` subsumes all `*_table`/`*_compare`/`*_delta`; `flux → phenotypic_flux`; `register_model → model.to_anndata`; `tl` mirrors grafiti's `tools/`), this plan follows the settled design and flags the divergence inline. This document supersedes the earlier draft wherever the audit corrected it; the corrections are integrated at the point they bite, and every finding is cross‑referenced in **Appendix A** so nothing is lost.

**How to read.** §1 fixes the invariants and records the resolved decisions that unblock the engine/metric phases. §2 is the target tree. §3–4 are the rename/disposition map and the shared‑helper extraction. §5 is the model→AnnData streamline. §6 is the engine + metric **numeric contract** (all math/stats fixes live here). §7 is the GPU/optimization architecture, grounded in grafiti. §8 is the ordered, independently‑landable PR sequence. §9 is testing + scverse‑CI. §10 is the ordering‑hazard graph. §11 is risks & open items. §12 is the per‑PR checklist.

---

## 1. Invariants and resolved decisions

### 1.1 Invariants held at every PR boundary
- **`import tcri` stays green** and the public handles `tl / pp / pl / ml / diag / ut` remain importable. No PR leaves `main` with a broken import or red CI. **Corollary:** "0‑caller" means *no call‑sites **and** no import‑sites in the package* — a symbol imported at module top (e.g. `utils.probabilities` at `_plotting.py:18`) is a live dependency even with zero calls, so it and its import are removed in the same PR. (The `example/` notebooks are disposable and never counted.)
- **One behavior change per PR.** Mechanical moves (splits, helper extraction) never change numerics; numeric changes (engine, metrics) never also move files. Every diff stays reviewable and every regression bisectable.
- **`_keys.py` is the only place a key string is written**, from Phase 1 on. No new `uns/obsm/obs` key literal may appear outside it.
- **No `import *`.** Each split lands with an explicit `__all__` and named re‑exports; the top‑level `__init__` flip is the last PR.
- **`setup_anndata` performs no *analysis/label* obs mutation.** *(Corrected from the draft's "no obs mutation.")* It must still write and register the `obs['indices']` field the training step consumes (`batch['indices']`, `_model.py:603/641/678/684`). The invariant forbids writing *results* (probabilities, hard labels, latent) into `obs`, not the registration glue column.
- **GPU is never on the import path and never in `install_requires`.** Every accelerated path is pure opt‑in behind a device seam with a fully‑functional CPU fallback (§7).

### 1.2 Resolved decisions (adopted from the prior‑vs‑mean RESOLUTION + math/stats audit)

**(R1) Point estimate = closed‑form variational posterior mean.** `n_samples=0` returns `E_q[p_ct] = normalize(q_p_ct_raw) = TCRIModel.get_p_ct() = adata.uns[K.P_CT]`, read directly, **never sampled**. Because the guide is `Dirichlet(local_scale · m)` with `m` on the simplex, `Σα = local_scale` and the mean is exactly `m` — `local_scale` cancels. The **prior/archetype path is dropped** (it is guide‑init/generative anchor built from leaked hard labels, not what training learned); the **MAP/mode is rejected** (`α_k = local_scale·m_k` is routinely `< 1`, so the mode sits on the simplex boundary / is undefined); **mean‑of‑draws is rejected** (Rao‑Blackwell‑dominated, adds only MC noise to a closed‑form quantity).

**(R2) Rename the mis‑named `posterior=` axis to `use_logits` — a classifier‑mixing switch, not a prior/posterior switch.** There was never a live prior branch; what actually differs is *with logits vs without*. The **engine** `joint_distribution` keeps exactly one such flag, `use_logits` (alias `cell_informed`), replacing both `posterior=` and `combine_with_logits=`:
  - `use_logits=True` (default) folds per‑cell classifier logits into `log(base)` exactly like `model.predict` (gate‑aware, §5).
  - `use_logits=False` returns the ct‑level table directly.
  Both branches use the posterior **mean** (`n_samples=0`) or **draws** (`n_samples>0`) of `p_ct`; neither ever touches the generative prior. **The four metrics do NOT expose `use_logits`, `posterior=`, or `point_estimate=`** — given an `adata` they always compute the cell‑informed joint. `point_estimate=` is deleted.

**(R3) `n_samples=0` is a *plug‑in* estimator, not the posterior mean of the metric.** Entropy, MI, and l1/KL flux are **nonlinear** functionals, so `metric(E_q[p]) ≠ E_q[metric(p)]` by a **Jensen gap that is not Monte‑Carlo noise**: entropy (concave) plug‑in **over‑estimates**, flux (convex) plug‑in **under‑estimates**, MI is sign‑indeterminate. Therefore:
  - Document `n_samples=0` as **`metric‑at‑posterior‑mean`** and the `n_samples>0` `mean` column as **`E_q[metric]`** — two different, clearly‑labeled estimators.
  - **No test may assert `n_samples=0 == mean(n_samples>0)`.** The determinism/reproducibility guarantee (bit‑identical repeated `n_samples=0` calls) still holds and is tested; the *equality across estimators* claim is dropped.

**(R4) `n_samples>0` draws must reproduce the guide's clamped concentration.** Draw from `Dirichlet(clamp(local_scale · m̃, min=1e-3))`, reusing the guide's floor (`_model.py:490`). The three inconsistent current variants (`local_scale·m` unclamped; `local_scale·m + 1e-8`; the clamped guide form) are unified onto the guide form so reported HDIs summarize the distribution the model actually learned. Note that for committed clones (`m_k < 1e-3/local_scale`) the clamp makes the draw‑mean differ slightly from `m`; this is documented, and `n_samples=0` remains the closed‑form `m` by definition.

**(R5) `p_gt` (P(>0)) is only meaningful on a signed contrast.** Entropy, MI, and flux are all `≥ 0`, so `P(draw > 0) ≈ 1` and is vacuous. **Remove `p_gt` from the single‑metric `n_samples>0` summary** (emit `mean`, `sd`, `hdi_low`, `hdi_high` only). `p_gt` is computed **only** on a paired between‑group difference vector by the comparison helper (§4, §6).

**(R6) Temperature is a single analysis knob applied one consistent way.** `m̃ = softmax(log(m + 1e-8) / T)` (identity at `T=1`), applied identically in the mean and draw branches. To avoid double‑tempering, **`to_anndata` persists the *raw* posterior mean** `m = normalize(q_p_ct_raw)` (not the guide‑temperature‑adjusted vector), so analysis `temperature` is the sole tempering knob. At `T=1`, `use_logits=True` reproduces `model.predict` exactly. `temperature≠1` re‑centers the sampled distribution away from the learned posterior — documented in the metric docstrings.

**(R7) Reproducibility is via a seeded torch generator.** All draws move to a seeded `torch.Generator` (device‑aware; also seed cupy/torch.cuda RNG on GPU). Add `random_state: int | torch.Generator | None` to `joint_distribution` and every sampling metric/diag function; thread it into the Dirichlet draw. `np.random.seed` alone was a **no‑op** for the torch draws and is retired. `n_samples=0` is deterministic regardless.

**(R8) `use_logits=True` parity with `predict` requires three model facts persisted to `adata`.** The gate‑aware combination `gate_prob·logits + (1-gate_prob)·log(base)` and the classifier temperature are **model attributes**. `to_anndata` must persist `obsm[K.X_LOGITS]`, `uns[K.GATE_PROB]`, and `uns[K.CLASSIFIER_TEMPERATURE]` (§5). Without them the engine can only reproduce the additive rule — the exact disagreement the refactor set out to remove.

---

## 2. Target layout (grafiti‑mirrored, authoritative)

```
tcri/
  __init__.py             # explicit re-export; sys.modules aliases tl/pp/pl/ml/diag/ut;
                          #   top-level tcri.joint_distribution; NO import *
  _keys.py                # single source of every uns/obsm/obs key string (adopt in Phase 1)
                          #   NOW INCLUDES: X_LOGITS, GATE_PROB, CLASSIFIER_TEMPERATURE,
                          #   LOCAL_SCALE, P_CT, X_TCRI, X_PROBABILITIES, CLONE_COL/PHENO_COL
  _console.py             # leveled, silenceable logging over scanpy logging; NO ANSI, NO _ascii_hist
  _stats.py               # stars, auc_and_label_permutation, bootstrap_auc,
                          #   + posterior-comparison primitives: mann_whitney, prob_gt_zero, hdi (TRUE HDI)
  _distance.py            # kl_divergence (log2/bits), l1_distance, jensen_shannon, phenotype_distance dispatcher
  _compute/               # NEW — device seam + batched numeric core (grafiti/_compute parity)
    _xp.py                #   resolve_device / get_xp / asnumpy (torch-first, cupy optional, cpu default)
    _joint.py             #   _joint_draws(adata, covariate, n_samples, *, use_logits, temperature, device,
                          #     random_state) -> ndarray[n_samples, n_clones, P]  (scatter-add reduction)
    _reduce.py            #   batched entropy / mutual_information / flux over the [S, n_clones, P] stack
    _embedding.py         #   umap() behind _use_gpu gate (cuML on GPU, umap-learn CPU); lazy import
  model/                  # ml
    _model.py             #   TCRIModel: setup_anndata, train, get_latent_representation,
                          #     predict (was get_cell_phenotype_probs), get_p_ct, to_anndata
    _module.py            #   TCRIModule (pyro model/guide)
    _priors.py            #   MixtureDirichlet, VampPrior
    _classifier.py        #   PhenotypeClassifier
    _training.py          #   UnifiedTrainingPlan, build_archetypes (returns centers AND labels)
  preprocessing/          # pp  (shrinks to clone utilities; engine moved to tools/)
    _clones.py            #   group_singletons (must precede setup_anndata), clone_size
  tools/                  # tl  (metrics + engine; mirrors grafiti tools/)
    _joint.py             #   joint_distribution  (thin DataFrame wrapper over _compute._joint;
                          #     re-exported as tcri.joint_distribution)
    _entropy.py           #   clonotypic_entropy, phenotypic_entropy
    _mutual_information.py #  mutual_information  (+ private _mi_from_joint)
    _flux.py              #   phenotypic_flux     (was flux; cov_from / cov_to)
    _compare.py           #   compare_groups  (PUBLIC mid-level stats helper; replaces *_compare/*_delta)
  plotting/               # pl  (twins mirror tl by filename)
    _base.py              #   _metric_boxplot (was tcri_boxplot; groupby + splitby), _finish
    _colors.py            #   tcri_colors, resolve_palette (was set_color_palette)
    _entropy.py           #   clonotypic_entropy [FIX], phenotypic_entropy [FIX]
    _mutual_information.py #  mutual_information [FIX]
    _flux.py              #   phenotypic_flux (sankey)
    _sankey.py            #   SankeyNode, _phenotype_mass_per_clone
  diagnostics/            # diag  (NEW — PPCs + model validation; returns DataFrames)
    _ppc.py               #   joint-distribution PPC (fixed compare_joint_distribution),
                          #     calibration, reconstruction PPC (model-required), permutation-null
    _training.py          #   loss curves (was plot_loss), archetypes (was plot_archetypes)
  utils/                  # ut
    _session.py           #   save/load_tcri_session, _to_jsonable  (plain h5ad; no manager hack)
docs/                     # model PGM (build_nested_tcri_pgm) lives here, out of the package;
                          #   the rewritten tutorial notebook lives here too
```

Divergences from inventory §3, called out: **(a)** `tl` package is `tools/` (grafiti parity), not `metrics/`; **(b)** the engine's *numeric core* is `_compute/_joint.py` with a thin `tools/_joint.py` DataFrame wrapper — **not** `preprocessing/_engine.py`; **(c)** `preprocessing/` loses `_register.py`/`_engine.py` (registration collapses into `model.to_anndata`), leaving `pp` = `_clones.py`; **(d)** no `_tables.py` — `groupby` + `tl.compare_groups` subsume it; **(e)** a **new `_compute/` package** (not in the original inventory) is the device seam that makes the GPU wins additive.

---

## 3. Rename & disposition map

Freeze the map **before** Phase 5 (the first breaking PR). Renames are breaking; pre‑1.0 we pay once. The deltas this plan enforces — **including the four settled‑design corrections the draft had wrong** (marked ⚠):

| current | → target | lands in |
|---|---|---|
| `metrics/` package | `tools/` package (aliased `tl`) | Phase 5 |
| `joint_distribution` + `joint_distribution_posterior` | `tl.joint_distribution` (engine) → top‑level `tcri.joint_distribution` | Phase 5 |
| `posterior=` / `combine_with_logits=` (engine) | **`use_logits=`** (alias `cell_informed`), default `True` (R2) | Phase 5 |
| `point_estimate=`; public `posterior=` on metrics | **removed** — `n_samples` is the only point/draws knob (R1–R2) | Phase 5–6 |
| `flux` / `from_this` / `to_that` | `phenotypic_flux` / `cov_from` / `cov_to` | Phase 6 |
| ⚠ `clonality` | **DROP entirely — do NOT merge into `clonotypic_entropy`** (generic repertoire stat) | Phase 6 |
| ⚠ `clonotypic_entropy_base` | **DROP** (not merged); log base standardized via `_distance` (bits) | Phase 6 |
| ⚠ `ridge_delta_entropy` | **DROP** (not keep‑and‑fix) | Phase 7 |
| ⚠ `compare_phenotypes` | **DROP** | Phase 6 |
| `mi_compare`, `delta_entropy_table`, `flux_table`, `delta_clonotypic_entropy`, `phenotypic_entropy_delta` | **deleted** — expressed via `groupby=` + `tl.compare_groups` | Phase 6 |
| `tl.phenotypic_entropies` / `tl.clonotypic_entropies` (plural batch forms) | **deleted** — subsumed by `groupby=` on the singular metric | Phase 6 |
| `get_cell_phenotype_probs` | `predict` (scvi/CellAssign idiom; order‑preserving loader, indexed by `obs_names`) | Phase 4 |
| `register_model` (+ `register_*_key`) | `model.to_anndata` (thin) | Phase 4 |
| `classify_phenotypes` | — (DROP; superseded by `REDO_LIST.md`) | Phase 2 ✅ deleted |
| `register_clonotype_key` / `register_phenotype_key` | folded (private) into `to_anndata` | Phase 4 |
| `pl.clonotypic_entropy_by_phenotype` | `pl.clonotypic_entropy` | Phase 7 |
| `plot_pheno_sankey` | `pl.phenotypic_flux` (sankey) | Phase 7 |
| `plot_phenotype_probabilities` | **DROP** (not core) | Phase 6 |
| `probability_ternary` | **DROP** (not core — a bespoke simplex plot of no core metric) | Phase 6 |
| `gene_entropy` | **DROP** (out of scope — generic gene QC) | Phase 2 |
| `polar_plot` | **DROP** | Phase 2 |
| `pl.flux` boxplot | **DROP** (sankey is the flux plot) | Phase 7 |
| `tcri_boxplot` | `_metric_boxplot` (private; keeps a **`splitby=` axis**, §Phase 7) | Phase 7 |
| `set_color_palette` | `resolve_palette` | Phase 7 |
| `dkl`, `flux.dkl_func` | `_distance.kl_divergence` (log2/bits, single eps) + `_distance.jensen_shannon` | Phase 1 |
| `Δ` (unicode) | `delta` (ASCII, greppable) | Phase 6 |
| `c2p_mat` | `clone_phenotype_prior` | Phase 3 |
| `centropy` / `pentropy` / `*_tl` leaked aliases | removed via `__all__` | Phase 11 |
| `uns["tcri_clone_key"]`/`["tcri_phenotype_key"]` **and** `uns["tcri_metadata"][...]` (two schemes) | one scheme via `_keys.py` (single `tcri_metadata`) | Phase 1 |

---

## 4. Shared‑helper extraction (`_keys` / `_console` / `_stats` / `_distance`)

Lands as **Phase 1**, before any file move, because every later phase imports these. Pure internal dedup; public API unchanged except for the new public `tl.compare_groups` (Phase 6, built on `_stats`).

| new module | absorbs | notes / fixes folded in |
|---|---|---|
| `tcri/_keys.py` | every `uns/obsm/obs` key literal, both current schemes | constants only; unifies `tcri_clone_key`/`tcri_metadata['clone_col']` to one `tcri_metadata` scheme; **adds `X_LOGITS`, `GATE_PROB`, `CLASSIFIER_TEMPERATURE`, `LOCAL_SCALE`** (R4/R6/R8); `clone_size` and any other reader flips to `K.CLONE_COL` in the **same** change that retires the writer (no orphaned reader) |
| `tcri/_console.py` | triplicated `_ok/_info/_warn/_fin` (metrics/pp/pl copies) | reimplement over **scanpy's `logging`/verbosity**; drop raw ANSI; **drop `_ascii_hist` and every `graph=`/ASCII‑histogram code path** |
| `tcri/_stats.py` | `stars`, `auc_and_label_permutation`, `bootstrap_auc` (from `utils/_utils.py`) | **add** `mann_whitney`, `prob_gt_zero` (Bayesian P(>0), for signed contrasts only — R5), **`hdi` implemented as a TRUE highest‑density interval** (not the mislabeled equal‑tailed `percentile[2.5,97.5]`); document boundary instability for bounded skewed posteriors |
| `tcri/_distance.py` | module‑level dead `dkl` + `flux.dkl_func` | one `kl_divergence` + `l1_distance` + **`jensen_shannon`** + `phenotype_distance(metric=...)`; **fix the mixed‑units bug — standardize on `log2` (bits) across entropy/MI/KL with one eps**; document `dkl` as directional/unbounded and recommend JSD (bounded ≤1 bit) for symmetric shift; `l1` stays the safe bounded default |

`_mi_from_joint` stays module‑private in `tools/_mutual_information.py`. The **public comparison surface** is `tl.compare_groups` (§6) — not the private `_stats` primitives — so "comparisons via `groupby` + stats" is programmatically reproducible.

---

## 5. Model → AnnData streamline

The single highest‑risk behavior change, and the one that makes the metric↔model agreement guarantee (R8) achievable. It kills the `AnnDataManager`‑in‑`uns` hack and fixes the write‑set the draft under‑specified.

**5.1 `setup_anndata` — registration only (no *analysis* obs mutation).**
- Registers fields via scvi `REGISTRY_KEYS`; **keeps writing/registering `obs['indices']`** (training glue, `_model.py:678/684`) — this is not an analysis mutation and must not be removed.
- **Removes the manager stash `adata.uns['tcri_manager'] = adata_manager` (`_model.py:697`)** — *this is where the stash actually lives, not in `register_model`.* Removing it here is what lets `write_adata_safely`/`_pop_nonserializables` be deleted.
- `group_singletons` stays a **separate `pp` step that must run BEFORE `setup_anndata`** (it relabels clones; running it after desyncs `ct_to_c`/`p_ct` from `obs`). Enforced: `setup_anndata` errors if a later relabel is detected.

**5.2 `model.to_anndata(adata)` — thin, canonical write‑set.** Writes **only** the canonical minimum via `_keys`, and the canonical minimum now **explicitly includes the three items the engine needs for `predict` parity** (correcting the draft's "nothing else"):

| slot | key | why it is canonical |
|---|---|---|
| metadata/categories | `K.META`, covariate/phenotype/ct category maps | registry provenance |
| latent | `obsm[K.X_TCRI]` | embedding |
| phenotype probs + hard labels | `obsm[K.X_PROBABILITIES]`, `obs[...]` | `predict()` output; standard slot (retire `X_tcri_phenotypes`) |
| ct‑level prior mean | `uns[K.P_CT]` = **raw** `normalize(q_p_ct_raw)` (R6) | `n_samples=0` closed‑form mean |
| **per‑cell logits** | **`obsm[K.X_LOGITS]`** | `use_logits=True` engine path (R8) — hard‑required |
| **gate probability** | **`uns[K.GATE_PROB]`** (scalar or `None`) | gate‑aware combine parity with `predict` (R8) |
| **classifier temperature** | **`uns[K.CLASSIFIER_TEMPERATURE]`** | matches `predict`'s logit scaling (R8) |
| **local scale** | **`uns[K.LOCAL_SCALE]`** | draw variance for `n_samples>0`; engine **raises** (never defaults to 1.0) if missing when `n_samples>0` (R4/R8) |

- **Stops writing `uns['tcri_manager']`** (already removed in 5.1) → deletes the `write_adata_safely`/`_pop_nonserializables` hack.
- `predict` (renamed from `get_cell_phenotype_probs`): returns a per‑cell phenotype‑prob `DataFrame`; **asserts the inference `DataLoader` is order‑preserving (`shuffle=False`) and indexes by `adata.obs_names`** (or carries the registered `indices` field and reindexes) so ct assignment and barcode labels cannot drift.

**5.3 Session IO.** `write_adata_safely → save_tcri_session` writes a **plain h5ad** (nothing non‑picklable in `uns` anymore); `load_tcri_session` rebuilds the registry by re‑running `setup_anndata`. `get_p_ct` reads the **process‑global** pyro param store (`q_p_ct_raw`); load must set the store immediately before any `get_p_ct`/`to_anndata` call, and multi‑model/round‑trip diag workflows are documented as single‑model‑per‑process unless params are namespaced (§11).

---

## 6. Engine + metric numeric contract

This section is the substrate all metrics read; it folds in every math/stats and missing‑link correction.

**6.1 Engine `tools/_joint.py::joint_distribution`** (thin DataFrame wrapper over `_compute/_joint.py::_joint_draws`, §7):

```
joint_distribution(adata, *, covariate=None, groupby=None, n_samples=0,
                   use_logits=True, clones=None, temperature=1.0,
                   random_state=None) -> pandas.DataFrame
```
- Unifies `joint_distribution` + `joint_distribution_posterior`; re‑exported as `tcri.joint_distribution`. Provenance in a **serializable form** (a `params` column or a companion `uns` sidecar), **not** only `df.attrs` (R‑forward: h5ad round‑trips must not silently drop it, §11).
- `n_samples=0`: closed‑form path. `use_logits=False` → tempered `m̃` rows (`== uns[K.P_CT]` at `T=1`); `use_logits=True` (default) → per‑cell `softmax((logits + gate‑combine(log m̃))/T)` aggregated per clone, **identical to `model.predict`** (R6/R8).
- `n_samples>0`: draw `p_ct ~ Dirichlet(clamp(local_scale·m̃, 1e-3))` **once per sample via a seeded torch generator** (R4/R7), feed each draw through the same temperature + combine + scatter‑sum; stack a `sample_id` axis. **All clones within one draw share the SAME `p_ct` draw** (one coherent joint per `sample_id`) — never independent per‑clone draws.
- **`covariate=None` computes the joint across ALL covariate values in one pass from a single `p_ct` draw** (the all‑timepoints path the sankey and multi‑covariate metrics need).
- **Draw‑once invariant:** for `n_samples>0`, the number of Dirichlet draws is exactly `n_samples`, **independent of `#groups` and `#covariates`** — draws are reused across groups/covariates by cell‑masking, not re‑drawn. Enforced by a draw‑counter test (§9).
- **`groupby` is implemented by cell/clone RESTRICTION over the FULL `adata`** (positional masks into full‑space `uns` arrays + `clones=`), **never by passing a sliced `AnnData` to the engine** — this avoids the hard full‑space‑vs‑subset alignment guard that today's `tcri_boxplot` slicing would trip. `_metric_boxplot` is rewritten off the slice‑and‑call pattern.
- **`groupby` requires the cell‑informed path** (or group keys that are clone‑nested / constant within `clone × covariate`); it is **ill‑defined on the ct‑level table** for non‑clone‑determined columns. The design assumes **clones are disjoint across `groupby` groups** (a TCR clone does not span two patients) — now stated explicitly, with a validation that errors when a group split would bisect a clone's cells.
- **Engine bug fixes folded in:** weighting keyed on the **`ct` index** (not the clone index); consistent normalization (row‑ vs whole‑table) across the two old functions; all‑zero‑clone reindex yields **NaN, not inflated uniform entropy**; **torch‑seeded** determinism.

**6.2 Four metrics** (`tools/_entropy.py`, `_mutual_information.py`, `_flux.py`), uniform signatures; **none expose `use_logits`/`posterior=`/`point_estimate=`**:
```
tl.clonotypic_entropy(adata_or_jd, *, covariate=None, groupby=None, n_samples=0, temperature=1.0,
                      clones=None, normalized=True, random_state=None)
tl.phenotypic_entropy (adata_or_jd, *, covariate=None, groupby=None, n_samples=0, temperature=1.0,
                      clones=None, normalized=True, random_state=None)
tl.mutual_information (adata_or_jd, *, covariate=None, groupby=None, n_samples=0, temperature=1.0,
                      clones=None, normalized=True, normalize_mode='min', random_state=None)
tl.phenotypic_flux    (adata, *, cov_from, cov_to, groupby=None, n_samples=0, temperature=1.0,
                      clones=None, distance_metric='l1', random_state=None)
```
- **`mutual_information` default `normalize_mode='min'` (was `'average'`).** Under the settled uniform‑clone prior, `P(c)=1/C` pins the clone marginal entropy `H_c` to `log2(C)` (structural, uninformative), which throttles `'average'` normalization by `~1/log2(C)` and makes it non‑comparable across groups with different clone counts — breaking the whole `groupby` workflow. `'min'` (`I/H_p`, coefficient of constraint) is in `[0,1]`, reaches 1 when clone determines phenotype, and is `C`‑independent. `'average'` is documented as not‑recommended (or dropped).
- **`clonotypic_entropy` normalizer comparability:** divide by `log2` of the number of clones with **genuine support**, dropping requested‑but‑absent (all‑zero, reindexed) clones from `C`; expose a **common‑denominator option** (`log2` of a fixed reference clone count) for cross‑group plots. Document that normalized clonotypic entropy uses group‑specific denominators by default.
- **Dual input.** Each accepts `(adata + covariate)` [compute joint internally] **or** a precomputed joint `DataFrame` [fast path]. **A bare precomputed joint forces `n_samples=0` and `groupby=None`** (it carries no `p_ct`/`local_scale`/logits/cells to resample or re‑partition); `clones=` just re‑filters rows; **raise a clear error** for `n_samples>0`/`groupby` on a bare jd. `phenotypic_flux` correctly takes `adata` only (it needs two joints). Metrics **propagate the input joint's provenance** into their output.
- **Return‑shape rule (uniform):** no `groupby` & `n_samples=0` → scalar/Series; `groupby` set → tidy `DataFrame` (row per group [× phenotype/clone]); `n_samples>0` → adds a `sample_id` axis and, on reduction, summary columns **`mean`, `sd`, `hdi_low`, `hdi_high`** (**no `p_gt`** — R5). Metrics reduce the stack by **iterating the `sample_id` level** (per‑draw full‑joint metric, then summarize).
- **`n_samples>0` intervals are partial posterior** (only `p_ct` uncertainty; classifier logits held at their posterior‑mean encoding) — documented so users don't read them as full posterior‑predictive uncertainty.
- **h5ad‑serializable returns (build‑toward `@tl_result`):** flat columns, **no object‑array columns and no `df.attrs`‑only provenance**; per‑draw vectors go in a separate long frame or a `uns` sidecar. Define the cache key as a hash of `(covariate, groupby, n_samples, temperature, clones, normalized, normalize_mode, distance_metric, random_state)`.

**6.3 Comparisons via `groupby` + `tl.compare_groups` (public).** The deleted `*_compare`/`*_delta` functions are replaced by a **public mid‑level helper**, not private primitives:
```
tl.compare_groups(df, *, value, by, reference=None, paired=False, hdi_prob=0.94)
    -> tidy DataFrame with per-pair: mean_a, mean_b, delta, U, p (Mann-Whitney),
       p_gt (Bayesian P(delta>0)), hdi_low, hdi_high
```
This is where `p_gt`/HDI live (on the **signed** `delta`, R5). The docstring shows the recipe that recreates `mi_compare`'s per‑pair output, so "groupby + stats subsumes the tables/deltas" is actually reproducible.

---

## 7. GPU / optimization architecture

Grounded 1:1 in grafiti's `_compute/` wins. The **one architecturally load‑bearing decision**: write the engine's numeric core as a **batched, device‑routable function returning a `[n_samples, n_clones, P]` array**, with pandas only at the boundary — so every win below is designed *in*, not retrofitted later.

**7.1 Device seam — `tcri/_compute/_xp.py` (copy grafiti's 58‑line reference).** `resolve_device` / `get_xp` / `asnumpy`. Because **torch≥2.4.1 is already a hard dep**, a **torch‑tensor core is the first backend** (torch.cuda when present — zero new deps), with **cupy optional** as a numpy‑style second backend. Every accelerated function returns a plain numpy array via `asnumpy`. Device ladder: `None/'cpu'→cpu`; `'mps'→cpu`; `'auto'/'gpu'/'cuda'→GPU only if the lib imports **and** `getDeviceCount()>0`, else CPU (explicit `'cuda'` warns on fallback, `'auto'/'gpu'` silent).

**7.2 Batched engine core — `_compute/_joint.py::_joint_draws(...) -> ndarray[n_samples, n_clones, P]`.** Precompute clone integer codes **once**; batch the Dirichlet draw and the softmax on the leading sample axis; scatter‑sum by clone. `tools/joint_distribution` is a thin single‑draw/summary DataFrame wrapper over this stack. Metrics consume the stack and reduce vectorized (`_compute/_reduce.py`).

**7.3 The wins, priority‑ordered, with expected gains:**

| # | operation | current hot path | fix | expected gain | prio |
|---|---|---|---|---|---|
| P0 | joint‑by‑clone reduction | `pd.DataFrame(...).groupby(level=0).sum()` rebuilt every draw (`_preprocessing.py:320-322`) over 1e5–1e6 cells | precompute clone codes once; **scatter‑add** (`np.add.at`/`np.bincount` CPU, `torch.index_add_`/`cupy.bincount` GPU) batched across all `n_samples` — grafiti `contingency.py`/`edge_tensor` verbatim | **10–50× on CPU** (pandas groupby → integer‑keyed bincount), multiplied again on GPU; the single biggest win | P0 |
| P0/P1 | Dirichlet sampling loop + softmax | `for i in range(n_samples): joint_distribution_posterior(...)` (`_metrics.py:296/363/548/741/987`), redrawing the **full** `p_ct` each call | draw all `n_samples` at once, **restricted to the covariate's ct rows first**; batch `softmax((logits+log b)/T)` over the leading axis; route via torch.cuda | removes the `n_samples`‑fold Python loop + per‑iteration `.uns`/DataFrame setup; collapses 200–1000 iterations into a few kernels | P0/P1 |
| P1 | entropy / MI reductions | `scipy.stats.entropy` per draw + `jd.loc[cl]` per clone (`_metrics.py:315/562-569/744`) | batched `xlogx` reduction over `[S, n_clones, P]`; MI as joint‑vs‑outer‑product; `nanmean`/HDI over the sample axis — grafiti `joint.py::_entropy/_mi`, float64 accumulators | removes two nested Python loops + slow pandas `.loc`; medium‑high | P1 |
| P1 | share the joint across metrics | each of the 4 metrics rebuilds the joint for the same `(covariate, n_samples)` | compute the `[S, n_clones, P]` stack **once per covariate** and reduce it in all four metrics; **build `groupby` groups in one batched pass** | divides the dominant cost by `#metrics`; realizes the draw‑once invariant (§6.1) | P1 |
| — | deterministic point estimate | `n_samples=0` still enters the sampler / returns one draw today | zero‑draw read of `uns[K.P_CT]` (R1) | correctness **and** cheapest path (no Monte‑Carlo at all) | — |
| P2 | latent UMAP | `umap.UMAP` with `import umap` at module top (`_preprocessing.py:20`) | `cuml.manifold.UMAP` behind `_use_gpu`, umap‑learn CPU fallback, **lazy import moved inside the function**; return float64 ndarray | cuML UMAP commonly **10–50×** umap‑learn, but runs once per analysis → lower total impact; also fixes the module‑top heavy import | P2 |

**7.4 Grafiti guardrails (replicate ALL):** (1) every GPU lib imported **lazily inside** the function — `import tcri` never touches cupy/cuml/torch.cuda (the current `import umap` at module top already violates this and is fixed); (2) GPU deps never in `install_requires`; (3) permissive device ladder with `getDeviceCount()>0` verification; (4) `asnumpy` at every return boundary; (5) GPU body wrapped in `try/except` that degrades to CPU and reports which backend ran; (6) **float64 accumulators** so GPU matches CPU; (7) **validate the joint** (finiteness, nonnegativity, per‑row sum≈1 — per‑row, to catch cancelling defects) **on‑device before compute**; (8) **chunk the batched reduction** over cells/draws (grafiti Moran's‑I `chunk_size=256`, KDE `blk=8192`) to bound device memory on the large `[n_samples, n_cells, P]` tensor and avoid OOM.

**7.5 Seeding (R7).** Draws move to a seeded `torch.Generator` (and cupy/torch.cuda RNG on GPU); `random_state` is threaded from every public sampling function. The old `np.random.seed` was a no‑op for the torch draws.

---

## 8. Ordered PR sequence

Each PR is independently landable with green CI, lowest‑risk first.

### Phase 0 — Contract freeze + CI scaffolding *(docs/tests only; zero code change)*
- Frozen contract: `tcri/_contract.pyi` + `tests/test_contract_conformance.py`, ported from grafiti's `_pyi_gen.py`/`test_contract_conformance.py`. Seed `IMPLEMENTED = {}`; each target function flips to implemented as its phase lands. Markdown→`.pyi`→live‑signature drift fails CI.
- Land the corrected disposition map (§3) into `docs/contract/`. **Disposition is decided by one test — *is it core?* The disposable `example/` notebooks are never consulted for what to keep or drop.**
- **Risk:** none. **Depends on:** nothing.

### Phase 1 — Shared helpers + `_keys` adoption *(internal dedup; API unchanged)*
- Create `_keys.py` (incl. `X_LOGITS/GATE_PROB/CLASSIFIER_TEMPERATURE/LOCAL_SCALE`), `_console.py`, `_stats.py` (true HDI, `prob_gt_zero`), `_distance.py` (bits/log2, JSD) (§4).
- **Adopt `_keys` at every read/write site.** Migrate `clone_size` to `K.CLONE_COL` in the **same** change that retires `tcri_clone_key` (no orphaned reader).
- **Risk:** low (mechanical, no numerics). **Depends on:** Phase 0. **Hazard:** must precede Phases 4/5.

### Phase 2 — Safe deletions *(non-core symbols, unreferenced in the package)*
- Delete the non-core / dead: `get_latent_embedding`, `group_small_clones`, `register_probability_columns`, `remove_meaningless_genes`, `gene_entropy`, `polar_plot`, `metrics._ent`, `clone_fraction`, module‑level `dkl`, **`utils.probabilities` (and the `_plotting.py:18` import in the SAME PR)**, `pl.probability_distribution`, `pl.bayesian_mutual_information`, `SankeyNode.hex_to_rgb`.
- **Not deleted here** (each dropped in its own phase, *with* its replacement so no in‑package caller is orphaned): `compare_joint_distribution` (→ diag, Phase 8); `pl.mutual_information`/`pl.phenotypic_entropy` (keep+fix, Phase 7); the consolidated‑away `*_table`/`*_delta`/`clonality`/`clonotypic_entropy_base`/`ridge_delta_entropy` (Phase 6); `probability_ternary`/`plot_phenotype_probabilities` (Phase 6/7).
- **Risk:** very low. **Depends on:** Phase 1.

### Phase 3 — Model module split *(mechanical; no behavior change)*
- Split `model/_model.py` (1074 ln) → `_model.py` + `_module.py` + `_priors.py` + `_classifier.py` + `_training.py`. Rename `c2p_mat → clone_phenotype_prior`.
- **`build_archetypes` keeps returning `(centers, labels)`** (labels drive `diag.archetypes`' cluster ordering); persist labels on the model/`uns`. Reconcile the default‑`K` mismatch (`build_archetypes` default `K=4` vs model `K=10`).
- Explicit `__all__` per module. **Risk:** low. **Depends on:** Phase 1. **Verify:** `test_model_setup`, `test_pyro_params` unchanged‑green.

### Phase 4 — Model→AnnData streamline *(behavior change; kills the manager hack)*
- Implement §5: `setup_anndata` registration‑only (keeps `obs['indices']`; **removes the `_model.py:697` manager stash**); `group_singletons` enforced to precede `setup_anndata`.
- `register_model → model.to_anndata` writing the canonical set **including `X_LOGITS`, `GATE_PROB`, `CLASSIFIER_TEMPERATURE`, `LOCAL_SCALE`, and the raw `P_CT`** (R6/R8). `get_cell_phenotype_probs → predict` (order‑preserving loader, `obs_names` index).
- `write_adata_safely → save_tcri_session` (plain h5ad); load rebuilds the registry via `setup_anndata` and sets the pyro param store before any `get_p_ct`.
- **Risk:** HIGH. **Depends on:** Phase 1, Phase 3. **Gate:** `test_session_round_trip` rewritten — proves save/load reproduces `p_ct` + latent + `predict` probs with **no `tcri_manager` in `uns`**, `setup_anndata` leaves analysis `obs` untouched, and `to_anndata` writes **exactly** the canonical key set (asserts logits/gate/cls‑temp/local‑scale present).

### Phase 5 — Engine consolidation *(the substrate; §6.1, §7.2)*
- Create `tools/` (aliased `tl`) and `_compute/` (`_xp.py`, `_joint.py`, `_reduce.py`). Implement `_joint_draws` (batched, device‑routable, scatter‑add) and the thin `tools/joint_distribution` wrapper; re‑export as `tcri.joint_distribution`.
- Signature per §6.1 with **`use_logits`** (renamed from `posterior=`), `random_state`, `covariate=None` → all‑covariates one‑pass, draw‑once invariant, clamped‑Dirichlet draws, single‑knob temperature, groupby by full‑space restriction. Fold in the weighting/normalization/zero‑clone/seed bug fixes.
- **Risk:** HIGH — every metric reads this. **Depends on:** Phase 4. **Verify:** `test_tools/test_joint`: `use_logits=False & n_samples=0 == tempered uns[K.P_CT]` **exactly**; `use_logits=True & n_samples=0 & T=1 == model.predict` aggregation; repeated `n_samples=0` **bit‑identical**; `n_samples>0` **torch‑seeded reproducible** and drawn from `Dirichlet(clamp(local_scale·m̃,1e-3))`; **draw‑counter == n_samples independent of #groups/#covariates**; weighting keyed on `ct`; serializable provenance.

### Phase 6 — Metric‑API consolidation *(four metrics + `compare_groups`)*
- Populate `tools/_entropy.py`, `_mutual_information.py`, `_flux.py` per §6.2 (dual input, `normalize_mode='min'` default, support‑only clonotypic denominator + common‑denominator option, coherent per‑`sample_id` draws, `mean/sd/hdi` summary with **no `p_gt`**, serializable returns). Add public **`tl.compare_groups`** (§6.3).
- **Delete** `mi_compare`, `delta_entropy_table`, `flux_table`, `delta_clonotypic_entropy`, `phenotypic_entropy_delta`, `clonotypic_entropy_base`, `clonality`, `ridge_delta_entropy`, `compare_phenotypes`, the plural `*_entropies`, and `metrics.dkl`. Delete `metrics/` after migration.
- **Risk:** HIGH. **Depends on:** Phase 5. **Verify:** ranges (`[0,1]` normalized), `n_samples=0` determinism, `groupby` tidy shape, **dual‑input equivalence at `n_samples=0` only**, all‑zero‑clone → NaN, `phenotypic_flux` `cov_from/cov_to` + seeded draws, `compare_groups` recreates `mi_compare`'s per‑pair output, **no test asserts `n_samples=0 == mean(n_samples>0)`** (R3).

### Phase 7 — Plotting split + pl twins *(fix the broken core plots)*
- Split `plotting/_plotting.py` (1437 ln) → `_base.py` (`_metric_boxplot` **with a `splitby=` axis**, `_finish`), `_colors.py` (`resolve_palette`), `_entropy.py`, `_mutual_information.py`, `_flux.py`, `_sankey.py`.
- Ship the four tl↔pl twins: `pl.clonotypic_entropy` (was `_by_phenotype`), `pl.phenotypic_entropy` **[FIX]**, `pl.mutual_information` **[FIX]**, `pl.phenotypic_flux` (sankey). pl functions are **cache renderers** (no metric math). **Retain `splitby`** as a distinct box‑hue axis (design decision — `groupby`=aggregation unit, `splitby`=comparison cohort) so two‑axis figures (dots=patient, boxes=response, x=phenotype) work; document per‑figure the `groupby`+`splitby` recipe.
- **Changelog behavior notes:** the default flips to `weighted=False` → `pl.mutual_information` displayed MI changes (cell‑weighted → per‑clonotype); `weighted=True` restores the old behavior. `pl.flux` boxplot and `clonality` plot dropped.
- **Drop (non-core):** `top_clone_umap`, `clone_size_umap`, `plot_phenotype_probabilities`, `probability_ternary`. **Risk:** medium. **Depends on:** Phase 6, Phase 1. **Verify:** each twin returns a `Figure`/`Axes` from a tidy tl result; sankey renders.

### Phase 8 — `diag/` seeding *(new; additive)*
- `diag/_ppc.py`: the **fixed** `compare_joint_distribution` (no undefined‑global `NameError`; model `p(clone,phenotype)` vs empirical counts), phenotype‑probability calibration, reconstruction PPC, entropy/MI vs permutation null. **All return DataFrames.** Make the **model requirement explicit per function**: `joint_distribution_ppc`/calibration/permutation‑null run **adata‑only**; `reconstruction_ppc` **requires the live model** (ZINB decoder lives on the module). `diag/_training.py`: `plot_loss → loss curves`, `plot_archetypes → archetypes` (consumes `build_archetypes` labels).
- **Risk:** low‑medium. **Depends on:** Phase 4 (finalized model), Phase 5 (engine). **Verify:** each PPC returns the expected columns on the `trained_model` fixture; permutation‑null seeded; single‑model‑per‑process param‑store scoping honored.

### Phase 9 — PGM → docs; utils finalize
- Move `build_nested_tcri_pgm`/`draw_tcri_pgm_nested` **out of the package into `docs/`**; drop `daft` from runtime deps → docs extras only. `utils/_utils.py → utils/_session.py` (session‑io + `_to_jsonable` only).
- **Risk:** low. **Depends on:** Phase 1, Phase 8.

### Phase 10 — Notebook rewrite *(fresh tutorials against the new API)*
- Rewrite the `example/` notebooks **fresh** against the new API: `setup_anndata → TCRIModel → train → model.to_anndata`; `tcri.joint_distribution`; four metrics with `groupby`/`n_samples`; `tl.compare_groups`; four pl twins; `diag` checks. Dropped functions (`gene_entropy`, `polar_plot`, `probability_ternary`, `pl.flux` boxplot, `plot_phenotype_probabilities`, the `register_*_key` writers) have **no successor** — the rewrite simply does not call them. One canonical end‑to‑end tutorial notebook under `docs/`. **The notebooks are an OUTPUT of the refactor, never an input to it.**
- **Risk:** low (docs), high value. **Depends on:** Phases 4–8.

### Phase 11 — Public API finalize + scverse ecosystem CI
- `tcri/__init__.py`: **explicit** named re‑exports, `sys.modules` aliases for `tl/pp/pl/ml/diag/ut`, top‑level `tcri.joint_distribution`, **remove `import *`**, kill leaked aliases (`centropy`/`pentropy`/`*_tl`) via `__all__`. Flip **all** target functions to `IMPLEMENTED`; drift now hard‑fails CI. Turn on the full scverse‑ecosystem gate (§9.2).
- **Risk:** low‑medium. **Depends on:** all prior phases.

---

## 9. Testing + scverse‑CI strategy

### 9.1 Tests per phase

| phase | required tests |
|---|---|
| 0 | `test_contract_conformance` (markdown→`.pyi`→signature drift); import‑smoke py3.10/3.11 |
| 1 | `_stats` (`stars`, AUC/permutation, **true HDI vs equal‑tailed**, `prob_gt_zero` on a signed vector), `_distance` (`kl_divergence` bits/symmetry, JSD bound), `_console` silence flag, `_keys` "no stray literal" grep |
| 2 | every deleted symbol absent from `__all__` **and** unreferenced in the package (import‑graph test); `import tcri` green after `utils.probabilities` + its `_plotting.py:18` import go together |
| 3 | `test_model_setup`, `test_pyro_params` green through the split; `build_archetypes` returns `(centers, labels)`; submodule import smoke |
| 4 | **`test_session_round_trip` rewritten** — save/load reproduces `p_ct`+latent+`predict` probs with **no `tcri_manager`**; `setup_anndata` leaves analysis `obs` untouched but keeps `obs['indices']`; `to_anndata` writes **exactly** the canonical set incl. `X_LOGITS/GATE_PROB/CLASSIFIER_TEMPERATURE/LOCAL_SCALE`; `predict` order‑preserving |
| 5 | `test_tools/test_joint`: `use_logits=False,n=0 == tempered uns[P_CT]`; `use_logits=True,n=0,T=1 == predict` aggregation (**engine==predict** agreement test); repeat `n=0` bit‑identical; `n>0` torch‑seeded + clamped‑Dirichlet; **draw‑count == n_samples ⟂ #groups/#covariates**; weighting on `ct`; `groupby` via full‑space restriction does not trip the alignment guard |
| 6 | ranges, `n=0` determinism, `groupby` tidy shape, **dual‑input equivalence (n=0 only)**, all‑zero‑clone→NaN, `normalize_mode='min'` C‑independence, `compare_groups` recreates `mi_compare`; **golden regression** computing MI/entropy from a fixed `uns[P_CT]` by hand; **no `n=0==mean(n>0)` assertion** |
| 7 | each pl twin returns `Figure`/`Axes` from a tidy tl result without metric math; `splitby` two‑axis render; sankey renders |
| 8 | each PPC returns expected columns on `trained_model`; `reconstruction_ppc` requires model, others adata‑only; permutation‑null seeded |
| 9 | no `daft`/PGM import in the installed package (import‑graph test) |
| 10 | **tutorial execution** (`pytest --nbmake`) on the rewritten synthetic tutorial, end‑to‑end |
| 11 | contract test with **all** functions `IMPLEMENTED`; `__all__` completeness (public names ↔ contract); "no `import *`" AST test |

Reuse `conftest.py` fixtures (`synthetic_adata`, `trained_model`, `mock_adata`); extend `mock_adata` to the unified `_keys` scheme in Phase 1 and to `to_anndata`'s canonical key set (incl. logits/gate/cls‑temp/local‑scale) in Phase 4.

### 9.2 scverse‑ecosystem CI

Bring `.github/workflows/tests.yml` to the cookiecutter‑scverse bar, layered so each phase stays green:
- **Matrix** py3.10 + py3.11 (extend to 3.12 before 1.0), `pip install -e ".[test]"`, `pytest tests/ -v --cov=tcri`.
- **Import‑smoke job** (grafiti pattern): `python -c "import tcri; from tcri.model._model import TCRIModel; from tcri.tools._joint import joint_distribution"` — **and assert no GPU lib (cupy/cuml/torch.cuda) was imported** (guardrail #1).
- **Lint/format gate:** `pre-commit` (ruff + ruff‑format) — add in Phase 1, enforce from Phase 3.
- **Contract‑conformance job** (Phase 0 on) — the markdown→`.pyi`→signature guardrail that lets views land independently.
- **CPU‑only correctness job** — the full suite must pass with no GPU present; a separate optional GPU job (if a runner is available) asserts float64 GPU≈CPU parity.
- **Notebook‑execution job** (`nbmake`) from Phase 10.
- **Docs build** (sphinx + numpydoc; `.readthedocs.yaml` present) must pass; public API fully docstringed/typed.
- **Ecosystem checklist** (final gate, Phase 11): AnnData‑native `setup_anndata`, no `import *`, typed public surface, tutorial notebook, `diag` returns data not plots, GPU strictly opt‑in.

---

## 10. Dependencies & ordering hazards

```
0 ─▶ 1 ─┬─▶ 2
        ├─▶ 3 ─▶ 4 ─▶ 5 ─▶ 6 ─▶ 7 ─▶ 8 ─▶ 9 ─▶ 10 ─▶ 11
        └─────────────────────(1 gates 4,5,6)
```
Hard, load‑bearing constraints:
1. **`_keys` (1) before `to_anndata` (4) and the engine (5).** Both rewrite key IO; centralizing strings first keeps diffs local and prevents two‑scheme drift. `clone_size`'s reader flip and `tcri_clone_key`'s writer retirement must land **together**.
2. **Model split (3) before `to_anndata` (4).** `to_anndata` is a `TCRIModel` method; split first so the behavior‑change diff is readable.
3. **`to_anndata` (4) before the engine (5).** The engine reads exactly the canonical keys/**logits/gate/cls‑temp/local‑scale** `to_anndata` writes; if the write‑set changes after the engine exists, `use_logits=True` breaks silently. The **manager‑in‑`uns` removal (at `setup_anndata:697`) is the single highest‑risk hazard** — it changes the session round‑trip; `test_session_round_trip` is rewritten in the same PR.
4. **Engine (5) before metrics (6).** Metrics are thin readers; building them against a pre‑fix engine would bake in the weighting/normalization/seed/Jensen bugs.
5. **Metrics (6) before pl twins (7).** pl are cache renderers.
6. **Delete `*_table`/`*_delta`/`clonality`/`_base`/`ridge_delta_entropy`/`compare_phenotypes` WITH their replacement (Phase 6), never before** — landing the `groupby`/`compare_groups` replacement in the same PR keeps CI green with no half‑migrated in‑package state.
7. **Rename freeze before Phase 5.** `from_this/to_that → cov_from/cov_to`, `flux → phenotypic_flux`, `metrics/ → tools/`, `posterior= → use_logits`, dropping `point_estimate=` are all breaking; batch them into one breaking window.
8. **`group_singletons` before `setup_anndata`** (clone‑relabel ordering) — enforced by `setup_anndata`.
9. **Deferred, designed‑for:** `@tl_result` uns‑cache + `get.py`. Every tl function returns an **h5ad‑serializable** tidy result (flat columns, no object arrays, serializable provenance) so the decorator is a one‑line wrap; until then pl recomputes via the engine. Do **not** block the refactor on the cache.

**Independent / parallelizable:** Phase 2 (safe deletions) and Phase 8 (`diag`, once 4–5 land) proceed alongside neighbors without contending for the same files.

---

## 11. Risks & open items

- **Highest‑risk hazard: the manager‑stash removal + write‑set change (Phase 4).** It reshapes the train→`to_anndata`→session round‑trip and adds four persisted keys. Mitigation: rewrite `test_session_round_trip` in‑PR; assert the exact canonical set; keep the CPU‑only correctness job as the tripwire.
- **`use_logits=True` parity depends on persisted model facts (R8).** If any of `X_LOGITS`/`GATE_PROB`/`CLASSIFIER_TEMPERATURE`/`LOCAL_SCALE` is dropped, parity degrades silently to the additive rule. Mitigation: the engine==predict agreement test (Phase 5) and the canonical‑set assertion (Phase 4). Note `gate_prob` defaults to `None` (models ungated by default), so the bug is latent until someone trains with a gate — the guarantee still must hold.
- **Estimator semantics (R3) are a documentation risk, not a code risk.** `n_samples=0` (plug‑in) and the `n_samples>0` `mean` are *different estimators*; users will expect them to match. Mitigation: docstrings state the Jensen gap explicitly; no conformance test asserts equality.
- **Global pyro param store.** `get_p_ct`/`to_anndata`/diag PPCs read the process‑global `q_p_ct_raw`; loading two models in one process clobbers it. **Open item:** namespace params per model or ship a documented single‑model‑per‑process contract with load setting the store immediately before use. Pre‑existing, not introduced by the refactor, but the multi‑model diag/round‑trip workflows expose it.
- **`temperature≠1` re‑centers the sampled distribution** away from the learned posterior (R6); intervals under non‑default `T` are not the model posterior. Documented; `T=1` is the parity point.
- **Partial posterior.** `n_samples>0` intervals capture only `p_ct` uncertainty (classifier logits fixed). Documented so they are not read as full posterior‑predictive uncertainty.
- **GPU memory on large `[n_samples, n_cells, P]`.** Without chunked reductions (guardrail #8) the batching win becomes an OOM risk on realistic `n_samples × n_cells`. Mitigation: chunk over cells/draws; try/except degrade to CPU.
- **`splitby`** is retained as a distinct box‑hue axis by design (≠ `groupby`, not a notebook artifact); its `pl` signatures are fixed in Phase 7.
- **`@tl_result` cache is deferred.** Return shapes are constrained now to be h5ad‑serializable so the later wrap is a one‑liner; the cache‑key scheme is specified but not implemented.
- **cuML/rapids GPU parity** for UMAP is layout‑different (both valid, as grafiti documents); not bit‑identical to CPU. No test asserts embedding bit‑identity.
- **Open (deliberately deferred):** py3.12 matrix, the `get.py` accessor surface, and any KL/JSD default change for `phenotypic_flux` beyond offering JSD (l1 remains the bounded default).

---

## 12. Rollout checklist (per PR)

- [ ] One behavior change (or zero, if a mechanical move).
- [ ] `import tcri` green; `tl/pp/pl/ml/diag/ut` handles intact; **no GPU lib imported at import time**.
- [ ] No new key‑string literal outside `_keys.py`; no new `import *`.
- [ ] Deleted symbols removed from `__all__` and unreferenced **in the package** (import‑site + call‑site grep).
- [ ] Contract test updated (`IMPLEMENTED` flipped for functions this PR lands).
- [ ] Phase‑specific tests from §9.1 present and green on py3.10/3.11 (CPU‑only job passes).
- [ ] Rename/disposition rows for this phase applied everywhere (code + fixtures).
- [ ] Any persisted‑key change reflected in `_keys`, `to_anndata`'s canonical set, and the round‑trip test.

---

## Appendix A — Audit findings by severity (traceability)

Every finding folded into the body above, cross‑referenced so nothing is lost. **Src:** PC = plan‑correctness, MS = math/stats, ML = missing‑links, GPU = GPU/optimization, PVM = prior‑vs‑mean resolution.

### A.1 Blocking / High

| # | src | finding | resolved in |
|---|---|---|---|
| B1 | PC/ML | `to_anndata` "canonical minimum" drops `obsm['X_tcri_logits']` that the default engine path hard‑requires | §5.2, R8, Phase 4/5 |
| B2 | PC/ML | `gate_prob` (and `classifier_temperature`) are model attributes never persisted → gate‑aware parity infeasible from `adata` | §5.2, R8, Phase 4 |
| B3 | PC | `setup_anndata` cannot be "no obs mutation" — must keep registered `obs['indices']` | §1.1, §5.1, Phase 4 |
| B4 | PC | Deleting `utils.probabilities` (Phase 2) breaks `import tcri` via `_plotting.py:18` import | §1.1, Phase 2 |
| B5 | PC | `groupby` by slicing `adata` trips the full‑space‑vs‑subset alignment guard | §6.1, Phase 5 |
| B6 | MS | Jensen gap: `metric(E[p]) ≠ E[metric(p)]`; `n=0` (plug‑in) vs `mean(n>0)` disagree — drop the equality test | R3, §1.2, Phase 6 |
| B7 | MS | `p_gt`/P(>0) vacuous per‑metric (all metrics ≥0) — only on signed contrasts | R5, §6.2/6.3 |
| B8 | MS | `n>0` draws must use the guide's **clamped** `Dirichlet(clamp(local_scale·m,1e-3))` | R4, §6.1, Phase 5 |
| B9 | MS | MI `normalize_mode='average'` breaks under uniform‑clone prior (`H_c=log2 C`) — default `'min'` | §6.2, Phase 6 |
| B10 | ML | `_stats` is private — no public comparison entry point; add `tl.compare_groups` | §4, §6.3, Phase 6 |
| B12 | ML | `covariate × groupby` semantics unspecified; requires cell‑informed path / clone‑disjoint groups | §6.1, Phase 5/6 |
| B13 | ML | No mechanism shares one `p_ct` draw across groups/covariates → O(groups×cov×n_samples) | §6.1 draw‑once, §7, Phase 5 |
| B14 | GPU | Engine must be a batched device‑routable core returning `[n_samples,n_clones,P]`, DataFrame at boundary | §7.2, Phase 5 |

### A.2 Medium

| # | src | finding | resolved in |
|---|---|---|---|
| M1 | PC | `n=0 == uns[P_CT]` holds only for `use_logits=False`; default folds logits — scope the test | R2, §6.1, Phase 5 |
| M2 | PC | `groupby` unrepresentable in ct‑level table for non‑clone‑determined columns | §6.1 |
| M3 | PC/MS | `adata_or_jd` fast path incompatible with `n>0`/`groupby` — restrict to `n=0`,`groupby=None` | §6.2 |
| M4 | PC | `clone_size` reads retired `tcri_clone_key` — migrate reader with writer | §4, Phase 1 |
| M5 | PC | `build_archetypes` must keep returning `labels`; default‑K mismatch (4 vs 10) | §Phase 3, §Phase 8 |
| M6 | MS | `clonotypic_entropy` `log2(C)` denominator group‑dependent, inflated by absent clones | §6.2, Phase 6 |
| M7 | MS | "HDI" is actually equal‑tailed — implement true HDI or rename | §4 (`_stats`), Phase 1/6 |
| M8 | MS/ML | Dual‑input + `n>0` ill‑defined — force `n=0` on bare jd | §6.2 |
| M9 | MS/ML | Temperature re‑centers draws / double‑tempering with guide_temperature — single knob, store raw mean | R6, §5.2, §6.1 |
| M10 | ML | `splitby` is a distinct box‑hue axis (≠ `groupby`) — retain it on the pl twins | §Phase 7 |
| M12 | ML | No `seed`/`random_state` on engine/metrics — add and thread to torch | R7, §6, Phase 5/6 |
| M13 | ML | Normalized entropy/MI non‑comparable across groups (group‑specific denominators) | §6.2 common‑denominator option |
| M14 | ML | `@tl_result` h5ad‑hostile returns (MultiIndex, object arrays, `.attrs`) | §6.2, §10(9) |
| M15 | ML | `n>0` reduction must keep draws coherent (same `p_ct` per `sample_id`) | §6.1/6.2, Phase 5/6 |
| M16 | GPU | Seed torch (not `np.random`) for reproducible draws | R7, §7.5 |

### A.3 Low

| # | src | finding | resolved in |
|---|---|---|---|
| L1 | PC | `predict()` DataFrame indexing assumes order‑preserving loader | §5.2, Phase 4 |
| L2 | PC | `pl.mutual_information` default flip to `weighted=False` changes displayed numbers (`weighted=True` restores) | §Phase 7 changelog |
| L3 | PC | `tcri_manager` stash is in `setup_anndata:697`, not `register_model` | §5.1, Phase 4 |
| L4 | PC/ML | moved `plot_phenotype_probabilities` reads `X_tcri_phenotypes`, not `X_tcri_probabilities` | §3, Phase 10 |
| L5 | PC | `get_p_ct` reads the global pyro param store — multi‑model clobber | §5.3, §11, Phase 8 |
| L6 | MS | KL flux asymmetric/unbounded, base mismatch — bits/log2, offer JSD, l1 default | §4 (`_distance`) |
| L7 | MS | `n>0` intervals are partial posterior (p_ct only) — document | §6.2, §11 |
| L8 | MS | `local_scale` uns fallback `1.0` corrupts draw variance — `to_anndata` always writes it; engine raises if missing at `n>0` | §5.2, §6.1, Phase 4/5 |
| L9 | ML | `group_singletons` ordering vs `setup_anndata` unspecified | §5.1, Phase 4 |
| L10 | ML | `diag` model‑required vs adata‑only inconsistent; missing golden + engine==predict agreement tests | §Phase 8, §9.1 |
| L11 | PVM | current `softmax(log p_ct/T)` path is already the posterior mean at `T=1`; "prior/non‑posterior" label is a misnomer — collapse the axis to `use_logits` | R1/R2, §6.1 |
| L12 | GPU | module‑top `import umap` violates lazy‑import guardrail; cuML UMAP behind `_use_gpu` | §7.3/7.4, Phase 5+ |