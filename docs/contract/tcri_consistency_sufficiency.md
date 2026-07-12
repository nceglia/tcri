# TCRI — Argument Consistency & Sufficiency (clean re-derivation)

Replaces the quarantined `tcri_arg_consistency_sufficiency.md`. **Consistency** is checked over the API surface
only. **Sufficiency** is derived from each kept function's **responsibility** — the disposable notebooks are never
consulted (Hard Rules, `REFACTOR_HISTORY.md`). Findings here are deltas to fold into
`tcri_api_and_responsibilities.md` signatures.

## A. Consistency — canonical decisions (fold into the signatures)

| # | concept | drift found | CANONICAL |
|---|---|---|---|
| C1 | RNG arg | `seed=0` (diags) vs `random_state` (tl/pl) vs `n_perm` | **`random_state=None`** everywhere, typed `int \| numpy.Generator \| torch.Generator \| None`, keyword-only |
| C2 | arg order | `clones`/`temperature` order varies | canonical: `covariate`/`cov_*` · `groupby` · `splitby` · `n_samples` · `temperature` · `clones` · `use_logits` · `normalized` · `normalize_mode` · `distance_metric` · `random_state` · `device` |
| C3 | distance selector | `metric=` (`phenotype_distance`) vs `distance_metric=` (flux) | **`distance_metric`** everywhere |
| C4 | palette arg | `phenotype_colors=` (flux) vs `palette=` (rest) | **`palette`** |
| C5 | normalize flag | `normalize=` (flux) vs `normalized=` (metrics) | **`normalized`** (adjective), American spelling |
| C6 | category ordering | `order=`/`hue_order=` present on some pl plots, absent on siblings | expose **both** `order=None` and `hue_order=None` on all metric box plots |
| C7 | figsize | diverges across sibling box plots | unify `figsize=(8,4)` for the metric-plot family |
| C8 | keyword-only | `columns`/`function`/`adata` positional in places | only the primary data object is positional; everything else after `*` |
| C9 | scanpy `show=` | only on private `_finish` | thread `show=None` through every public `pl` function |
| C10 | PPC sample knob | `reconstruction_ppc(n_samples=100)` collides with the `n_samples=0` convention | rename to **`n_sims=100`**; reserve `n_samples` (default 0) for the point/draws convention |
| C11 | `compare_groups` axis | `by=` for the contrast column | rename **`by=`→`splitby=`** to match the two-axis vocab |
| C12 | `group_singletons` defaults | `clonotype_key="trb"`, `groupby="patient"` hardcoded | require or read the registered defaults (no dataset-specific literals) |
| C13 | eps floor | four different probability-clip epsilons | one `eps=1e-12` for the distance/MI/normalization paths (incl. the MI kernel) |
| C14 | spelling | British `normalise*` in the draft | **American** (`normalized`, `normalize_mode`, `color`) — already applied |
| C15 | `covariate=None` | one line said "required" on the `adata_or_jd` fast path | `covariate=None` = **all covariate values** everywhere; the jd fast path ignores it — already reconciled |

## B. Sufficiency — from responsibility (kept functions only)

| function | knob | why (its own responsibility) |
|---|---|---|
| `joint_distribution`, 4 metrics, sampling `diag` | `random_state=` | reproducible sampling is intrinsic to a posterior-draw function |
| per-clone metrics + `joint_distribution_ppc` | `clones=` | the function computes per-clone quantities → subsetting is intrinsic |
| the metric box plots | `order=` / `hue_order=` | a categorical plot must control axis / hue order |
| `compare_groups` | `pair_on=` | a **paired** covariate contrast (Pre→Post per unit) needs the unit-alignment column — a real responsibility, not notebook parity |

## C. REJECTED — notebook-driven "sufficiency", NOT re-added

These were flagged only because a disposable notebook used them. They correspond to **dropped** functions or
**deliberately removed** behavior, and are not re-added:

- `gene_entropy`, `probability_ternary`, `top_clone_umap`, `clone_size_umap`, `plot_phenotype_probabilities` — **dropped** (not core). No args to "restore."
- `weighted=` — deliberately removed (uniform-clonotype prior; `REFACTOR_HISTORY` §2).
- `minimum_clone_size=`, `base=`, `decimals=` — notebook conveniences on dropped/reworked functions; not core responsibilities.
- `covariate_key=` — **invalid**: the covariate column is fixed by `setup_anndata` and baked into `p_ct` via `ct_to_cov`; a call-time override would silently mismatch the model. Correctly absent.
