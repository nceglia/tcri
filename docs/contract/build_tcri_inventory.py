#!/usr/bin/env python3
"""Render the full TCRI function inventory + consolidation plan from the
inventory-workflow output (tcri_inventory_data.json).

Data-driven sections (inventory table, consolidation groups, deletions, helper
extraction, plotting triage) come straight from the workflow result. The
authored overlay sections (grafiti target shape, target tree, rename map, metric
conventions, diagnostics scoping) encode the five design decisions layered on
top. Regenerate: python3 docs/contract/build_tcri_inventory.py
"""
import json, os, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "tcri_inventory_data.json")
OUT_MD = os.path.join(HERE, "tcri_function_inventory.md")

res = json.load(open(DATA))
merged = res["merged"]
crit = res["critic"]
syn = res["synthesis"]

# ---- critic corrections (label overrides) --------------------------------
CRITIC_FIX = {
    ("pl", "clonality"): "plotting-beyond-core",
    ("pl", "tcri_boxplot"): "helper",
    ("ml", "TCRIModule.get_latent"): "model-construction",
    ("ml", "TCRIModule.get_p_ct"): "model-construction",
}
for r in merged:
    key = (r["namespace"], r["name"])
    if key in CRITIC_FIX:
        r["label"] = CRITIC_FIX[key]

# ---- disposition per row -------------------------------------------------
DISPO_DEFAULT = {
    "core": "keep",
    "redundant": "merge",
    "helper": "extract → shared",
    "model-construction": "keep (internal)",
    "session-io": "keep",
    "plotting-beyond-core": "move→examples / drop",
    "dead-broken": "delete",
}
# name-based overrides (apply our 5 overlays + the KEEP+FIX pl twins)
DISPO_OVERRIDE = {
    "mutual_information": None,   # ambiguous (tl vs pl) — handled by (ns,name) below
}
NAME_OVERRIDE = {
    ("pl", "mutual_information"): "keep + FIX",
    ("pl", "phenotypic_entropy"): "keep + FIX",
    ("pl", "ridge_delta_entropy"): "keep + FIX",
    ("pl", "compare_joint_distribution"): "→ diagnostics (PPC)",   # OVERLAY: was 'delete'
    ("ml", "TCRIModel.plot_loss"): "→ diagnostics",                # OVERLAY
    ("ml", "TCRIModel.plot_archetypes"): "→ diagnostics",          # OVERLAY
    ("ut", "build_nested_tcri_pgm"): "→ diagnostics",              # OVERLAY
    ("ut", "draw_tcri_pgm_nested"): "→ diagnostics",               # OVERLAY
}

def disposition(r):
    key = (r["namespace"], r["name"])
    if key in NAME_OVERRIDE:
        return NAME_OVERRIDE[key]
    d = DISPO_DEFAULT.get(r["label"], "?")
    if r["label"] == "redundant" and r.get("consolidate_into"):
        return "merge → " + r["consolidate_into"]
    return d

def mc(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()

def short(s, n=150):
    return textwrap.shorten(str(s), width=n, placeholder=" …")

# =====================================================================
CORE_DEF = """\
**Core = five things; everything else must justify itself against them.**
1. **Model (`ml`)** — `TCRIModel`: build / train / evaluate / register outputs onto an AnnData.
2. **Engine (`pp`)** — `joint_distribution`: Bayesian posterior sampling of the clone×phenotype distribution (the substrate every metric reads).
3. **Metrics (`tl`)** — `clonotypic_entropy`, `phenotypic_entropy`, `mutual_information` (+ `flux`, `delta_clonotypic_entropy`, and the tidy-table builders).
4. **Plotting (`pl`)** — plots that *directly* visualize those metrics and the joint distribution / flux.
5. **Utils (`ut`) + shared helpers** — session save/load; deduplicated console / stats / distance / color helpers.
"""

GRAFITI = """\
The reference layout (`../grafiti`) is a flat package with scanpy-style sub-packages aliased to short handles
(`model→ml`, `tools→tl`, `plotting→pl`, `preprocessing→pp`, `diagnostics→diag`, `datasets→ds`, `get.py→get`).
Six patterns tcri should copy:

1. **One file per topic, never a monolith.** `tools/_joint.py`, `tools/_motif.py`, … and `plotting/` mirrors them 1:1 by filename. (tcri's 1008-line `_metrics.py` and 1437-line `_plotting.py` are the anti-pattern.)
2. **Private cross-cutting sub-packages.** `_state/` (keys, resolve, storage, schemas) + `_compute/` (device-routed math) hold everything shared, so impl files stay thin. tcri analog: `_keys.py`, `_console.py`, `_stats.py`, `_distance.py` (and later a `_compute`).
3. **`__all__` at BOTH levels; NO `import *`.** Each impl module declares `__all__`; each `__init__` names every symbol explicitly and re-declares an aggregate `__all__` grouped by view. **This corrects the earlier `import *` instinct** — the mature pattern is explicit re-export, which keeps numpy/pandas/helpers *unexported*.
4. **`get.py` + `@tl_result` cache convention.** tl writes a versioned uns blob and returns a tidy result; `pl` functions are pure *cache renderers* (`load_result(adata, key)` → draw, never compute). *(tcri: adopt `_keys.py` now; defer the cache decorator — see open questions.)*
5. **`diagnostics/` returns DATA, not plots.** `gf.diag` runs read-only concordance/quality checks on the *finalized* model and returns a DataFrame ("did the model fit?"), deliberately outside the tl-writes / pl-reads loop. **This is exactly where PPCs + model-validation live.**
6. **Naming.** public package + private `_topic.py` impl modules; helper *packages* underscore-prefixed dirs, helper *files* underscore-prefixed, helper *functions* underscore-prefixed; tl↔pl twins share filename + function name.
"""

TARGET_TREE = """\
```
tcri/
  __init__.py             # explicit re-export + sys.modules aliases (tl/pp/pl/ml/ut/diag); NO import *
  _keys.py                # single source of every uns/obsm/obs key string
  _console.py             # _ok/_info/_warn/_fin/_ascii_hist          (was triplicated across 3 files)
  _stats.py               # stars, auc_and_label_permutation, bootstrap_auc
  _distance.py            # kl_divergence, l1_distance, phenotype_distance dispatch   (was dkl + flux.dkl_func)
  model/                  # ml
    _model.py             #   TCRIModel: setup_anndata, train, get_latent_representation, get_cell_phenotype_probs, get_p_ct
    _module.py            #   TCRIModule (pyro model/guide, get_latent, get_p_ct)
    _priors.py            #   MixtureDirichlet, VampPrior
    _classifier.py        #   PhenotypeClassifier
    _training.py          #   UnifiedTrainingPlan, build_archetypes
  preprocessing/          # pp
    _register.py          #   register_model  (+ folded register_*_key, _compute_logits_and_prior)
    _engine.py            #   joint_distribution(posterior=, n_samples=)   (unifies the two current fns)
    _clones.py            #   group_singletons, clone_size
  metrics/                # tl
    _entropy.py           #   clonotypic_entropy, phenotypic_entropy, delta_clonotypic_entropy
    _mutual_information.py #  mutual_information (+ private _mi_from_joint)
    _flux.py              #   flux
    _tables.py            #   mi_compare, delta_entropy_table, flux_table
  plotting/               # pl   (twins mirror tl by filename)
    _base.py              #   _metric_boxplot (was tcri_boxplot), _finish
    _colors.py            #   tcri_colors, resolve_palette   (was set_color_palette)
    _entropy.py           #   clonotypic_entropy (was _by_phenotype), phenotypic_entropy [FIX], ridge_delta_entropy [FIX]
    _mutual_information.py #  mutual_information [FIX], mi_compare
    _flux.py              #   phenotypic_flux (sankey)
    _sankey.py            #   SankeyNode, _phenotype_mass_per_clone
  diagnostics/            # diag   (NEW — PPCs + model validation, returns DataFrames)
    _ppc.py               #   joint-distribution PPC (was compare_joint_distribution, fixed) + calibration + reconstruction PPC
    _training.py          #   loss curves (was plot_loss), archetypes (was plot_archetypes)
    _pgm.py               #   model PGM (was build_nested_tcri_pgm)
  utils/                  # ut
    _session.py           #   save/load_tcri_session, write_adata_safely, _to_jsonable
examples/                 # bespoke one-offs move here: top_clone_umap, clone_size_umap,
                          #   phenotype_probabilities UMAP, compare_phenotypes  + the rewritten notebooks
```
"""

RENAME_MAP = """\
Freeze this **before** touching code or notebooks — renames are breaking and we only pay once (pre-1.0, Alpha).

### Modules / files
| current | → target |
|---|---|
| `metrics/_metrics.py` (1008 ln) | `tl/_entropy.py` + `_mutual_information.py` + `_flux.py` + `_tables.py` |
| `preprocessing/_preprocessing.py` (559) | `pp/_register.py` + `_engine.py` + `_clones.py` |
| `plotting/_plotting.py` (1437) | `pl/_entropy.py` + `_mutual_information.py` + `_flux.py` + `_base.py` + `_colors.py` |
| `model/_model.py` (1074) | `model/_model.py` + `_module.py` + `_priors.py` + `_classifier.py` + `_training.py` |
| `utils/_utils.py` (665) | `utils/_session.py` + new `_console.py` / `_stats.py` / `_distance.py` / `_keys.py` + `diagnostics/` |

### Functions
| current | → target | why |
|---|---|---|
| `joint_distribution` + `joint_distribution_posterior` | `joint_distribution(posterior=, n_samples=)` | one engine, one point/draws knob |
| `clonotypic_entropy_base` | merged into `clonotypic_entropy` | `_base` means nothing; it's the single-phenotype case |
| `pl.clonotypic_entropy_by_phenotype` | `pl.clonotypic_entropy` | tl↔pl twin name; matches notebook expectation |
| `plot_phenotype_probabilities` / `plot_pheno_sankey` | drop `plot_` prefix (`phenotype_probabilities`, internal `_sankey`) | scanpy/scvi pl fns are unprefixed |
| `tcri_boxplot` | `_metric_boxplot` (private helper) | generic engine, not public API |
| `dkl` / `flux.dkl_func` | `_distance.kl_divergence` | dedupe the KL kernel |
| `centropy` / `pentropy` / `*_tl` aliases | **removed** (via `__all__`) | leaked import aliases |
| `classify_phenotypes` | **removed** → `register_model` | duplicate phenotype-assignment path |
| `remove_meaningless_genes` | `filter_genes` (if kept) *or* delete | broken flag logic + 0 callers |

### Parameters
| current | → target |
|---|---|
| `from_this` / `to_that` (flux) | `cov_from` / `cov_to` |
| `point_estimate=True` (entropies) | **removed** — use `n_samples=0` |
| ad-hoc arg orders | standardize `covariate` / `splitby` / `groupby` / `clones` / `n_samples` / `temperature` / `posterior` everywhere |

### State keys
| current | → target |
|---|---|
| `uns["tcri_clone_key"]` / `["tcri_phenotype_key"]` **and** `uns["tcri_metadata"][...]` (two conventions) | one scheme via `_keys.py` constants (single `tcri_metadata`) |
| ad-hoc registry keys (`"clonotype_col_in_registry"`) | standard scvi `REGISTRY_KEYS` |

### Internal variables
| current | → target |
|---|---|
| `Δ` (unicode) in `delta_clonotypic_entropy` | `delta` (ASCII, greppable) |
| `c2p_mat` | `clone_phenotype_prior` |
| `p_ct` / `ct_to_c` / `ct_to_cov` (terse) | keep, but document (`ct` = (clone, covariate) index) |
"""

CONVENTIONS = """\
### Uniform point-estimate / sampling convention (applies to every metric)
- **`n_samples=0` → deterministic point estimate.** Posterior-*mean* `p_ct` (× logits), softmax, **no draw**, reproducible.
- **`n_samples=N>0` → N posterior draws** (adds a sampling axis; mean ± CI fall out).
- **Delete the `point_estimate=` argument** — `n_samples` is the only knob. This also fixes a latent bug: today `mutual_information(n_samples=0)` / `flux(n_samples=0)` return **one random draw**, not a deterministic estimate.

### Prior-vs-posterior — PARKED (open, do not collapse in this pass)
The `{prior, posterior} × {point, draws}` 2×2 is deferred. Until resolved the plan assumes `posterior=True` with
`n_samples` as the point/draws knob, and keeps `posterior=` as a documented-but-unfinalized argument
(the current prior-only branch raises `NotImplementedError`).

### diagnostics/ = PPCs + model validation
`gf.diag`-style, returns DataFrames, read-only on the finalized model. Seeded by:
- **joint-distribution PPC** — model p(clone,phenotype) vs empirical counts (the *fixed* `compare_joint_distribution`)
- phenotype-probability **calibration**; **reconstruction PPC** (ZINB simulate → compare library/dropout/mean-var); entropy/MI vs permutation null
- relocated: training curves (`plot_loss`), archetypes (`plot_archetypes`), model PGM (`build_nested_tcri_pgm`)
"""

# =====================================================================
def md():
    L = []
    A = L.append
    A("# TCRI — Full Function Inventory & Consolidation Plan")
    A("")
    A("_Generated from the inventory workflow (9 agents, 131 functions, completeness-verified) via "
      "`build_tcri_inventory.py`. Data: `tcri_inventory_data.json`. This is the working list we reduce "
      "the repo against._")
    A("")
    A("## 0. Core definition")
    A("")
    A(CORE_DEF)
    # counts
    from collections import Counter
    lab = Counter(r["label"] for r in merged)
    A("## 1. Label counts (critic-corrected)")
    A("")
    A("| label | count | disposition |")
    A("|---|---|---|")
    order = ["core","redundant","plotting-beyond-core","model-construction","session-io","helper","dead-broken"]
    disp = {
        "core":"keep (22 survive)","redundant":"merge into core (5 groups)",
        "plotting-beyond-core":"move to examples / drop","model-construction":"keep, split across model/_module,_priors,_classifier,_training",
        "session-io":"keep as utils/_session","helper":"dedupe → _console/_stats/_distance/_base/_colors",
        "dead-broken":"12 delete · 2 merge · 3 keep+fix",
    }
    for k in order:
        A(f"| {k} | {lab.get(k,0)} | {disp.get(k,'')} |")
    A(f"| **total** | **{len(merged)}** | |")
    A("")
    A("## 2. Grafiti reference layout (the target shape)")
    A("")
    A(GRAFITI)
    A("## 3. Target tcri layout (grafiti-mirrored)")
    A("")
    A(TARGET_TREE)
    # full inventory grouped by namespace
    A("## 4. Full inventory — every function")
    A("")
    A("Label is critic-corrected. Disposition folds in the five overlays "
      "(diagnostics reclass, keep+fix pl twins, n_samples convention).")
    NS_TITLE = {"ml":"ml — model","pp":"pp — preprocessing","tl":"tl — metrics","pl":"pl — plotting","ut":"ut — utils"}
    labrank = {l:i for i,l in enumerate(order)}
    for ns in ["ml","pp","tl","pl","ut"]:
        rows = [r for r in merged if r["namespace"]==ns]
        rows.sort(key=lambda r:(labrank.get(r["label"],9), r["name"]))
        A(f"### `tcri.{ns}` — {NS_TITLE[ns].split(' — ')[1]}  ({len(rows)} records)")
        A("")
        A("| name | kind | label | disposition | purpose |")
        A("|---|---|---|---|---|")
        for r in rows:
            A(f"| `{mc(r['name'])}` | {r['kind']} | {r['label']} | {mc(disposition(r))} | {mc(short(r['purpose'],140))} |")
        A("")
    # synthesis sections
    A("## 5. Consolidation groups (redundant → core)")
    A("")
    A("| into | members | rationale |")
    A("|---|---|---|")
    for g in syn.get("consolidation_groups",[]):
        A(f"| `{mc(g.get('into'))}` | {mc(', '.join(g.get('members',[])))} | {mc(short(g.get('rationale'),160))} |")
    A("")
    A("## 6. Deletions")
    A("")
    A("| function | reason |")
    A("|---|---|")
    for d in syn.get("deletions",[]):
        nm = d.get("name")
        note = mc(short(d.get("reason"),150))
        if "compare_joint_distribution" in nm:
            note = "**OVERLAY OVERRIDE → keep as diagnostics PPC** (was: " + note + ")"
        A(f"| `{mc(nm)}` | {note} |")
    A("")
    A("## 7. Helper extraction (dedupe → shared modules)")
    A("")
    A("| helper | → module | current copies |")
    A("|---|---|---|")
    for h in syn.get("helper_extraction",[]):
        A(f"| {mc(h.get('helper'))} | `{mc(h.get('into_module'))}` | {mc(short(h.get('current_copies'),90))} |")
    A("")
    A("## 8. Plotting triage")
    A("")
    pt = syn.get("plotting_triage",{})
    A("**Core (keep):** " + ", ".join(f"`{mc(x.split(' ')[0])}`" for x in pt.get("core_keep",[])))
    A("")
    A("**Beyond core (move→examples / drop):**")
    A("")
    for x in pt.get("beyond_core_drop_or_move",[]):
        A(f"- {mc(x)}")
    A("")
    A("## 9. Rename / readability map")
    A("")
    A(RENAME_MAP)
    A("## 10. Metric conventions & scoping")
    A("")
    A(CONVENTIONS)
    A("## 11. Open questions (decide before executing)")
    A("")
    oq = syn.get("open_questions") or ""
    if isinstance(oq, str) and oq.strip():
        A(oq)
    else:
        A("_(see synthesis output)_")
    A("")
    A("---")
    A("_Overlays applied on top of the workflow synthesis: (a) `diagnostics/` = PPCs + model validation "
      "(`compare_joint_distribution` reclassified from delete → diagnostics PPC seed; `plot_loss`/`plot_archetypes`/PGM relocated); "
      "(b) uniform `n_samples=0` point-estimate convention, drop `point_estimate=`; (c) prior-vs-posterior parked; "
      "(d) explicit `__all__` re-export (NOT `import *`) per grafiti; (e) full rename map._")
    return "\n".join(L)

open(OUT_MD,"w").write(md())
print("wrote", OUT_MD, f"({os.path.getsize(OUT_MD)} bytes)")
print("functions:", len(merged), "| consolidation:", len(syn.get('consolidation_groups',[])),
      "| deletions:", len(syn.get('deletions',[])), "| helpers:", len(syn.get('helper_extraction',[])))
