#!/usr/bin/env python3
"""Build the TCRI target API contract — emits .md (source) + color HTML.

Single source of truth for the *ideal* post-refactor API. Renders a compact
summary table, per-function detail cards (color-coded by namespace), the
adata-state schema (the data contract everything couples through), the shared
primitives index, and the dropped/merged list. Work the refactor toward this;
do not drift.

Companion docs:
  - tcri_api_refactor.md     (prose design spec + sequencing)   [optional]
  - tcri_dependency_map.*    (call + producer/consumer graph)   [build_tcri_depgraph.py]
"""
import html as _html
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(_HERE, "tcri_api_contract.md")
OUT_HTML = os.path.join(_HERE, "tcri_api_contract.html")

# ---- namespaces (color-coded) ----------------------------------------------
VIEWS = {
    "ml": ("#dc2626", "Model (ml)"),
    "pp": ("#0f766e", "Preprocess (pp)"),
    "tl": ("#7c3aed", "Metrics (tl)"),
    "pl": ("#2563eb", "Plotting (pl)"),
    "ut": ("#475569", "Session / IO (ut)"),
}

# ---- function inventory -----------------------------------------------------
# fields: ns, group, name, status, signature, returns, writes, reads, calls,
#         upstream (run-first), invariants, edges, mirror, source (provenance), notes
F = []
def fn(**k): F.append(k)

# ============================ ml — model ====================================
fn(ns="ml", group="model", name="TCRIModel.setup_anndata", status="keep",
   signature="(adata, *, layer=None, clonotype_key='unique_clone_id', "
             "phenotype_key='phenotype_col', covariate_key='timepoint', batch_key='patient')",
   returns="adata (registers scvi AnnDataManager)",
   writes="uns[tcri_manager]; registry{clonotype/phenotype/covariate/batch_col}",
   reads="obs[clonotype/phenotype/covariate/batch_key]",
   calls="—", upstream="—",
   invariants="all four obs columns must exist → ValueError otherwise",
   edges="layer=None → counts from X", mirror="—",
   source="keep as-is", notes="scvi classmethod entry point")
fn(ns="ml", group="model", name="TCRIModel.train", status="keep",
   signature="(max_epochs=1000, batch_size=1000, lr=1e-3, "
             "reconstruction_loss_scale=1e-3, n_steps_kl_warmup=2000, **kw)",
   returns="None (fits in place; populates history_)",
   writes="pyro param store (q_p_c_raw, q_p_ct_raw); model weights",
   reads="adata_manager", calls="UnifiedTrainingPlan, TrainRunner, DataSplitter",
   upstream="setup_anndata",
   invariants="early-stops on elbo_validation", edges="—", mirror="pl.model_loss",
   source="keep as-is", notes="")
fn(ns="ml", group="model", name="TCRIModel.get_latent_representation", status="keep",
   signature="(adata=None, indices=None, batch_size=None)",
   returns="ndarray[N, n_latent] — posterior-mean z",
   writes="—", reads="encoder", calls="module.get_latent", upstream="train",
   invariants="deterministic (mean, no sampling)", edges="—", mirror="—",
   source="keep as-is", notes="")
fn(ns="ml", group="model", name="TCRIModel.get_cell_phenotype_probs", status="keep — now the ONLY prob path",
   signature="(adata=None, batch_size=256, eps=1e-8)",
   returns="ndarray[N, P] — per-cell phenotype probabilities",
   writes="—", reads="encoder, classifier, get_p_ct(), module.ct_array",
   calls="module.get_p_ct", upstream="train",
   invariants="softmax(gate·logits + (1-gate)·log prior) OR additive when no gate",
   edges="use_gate toggles the two combination rules",
   mirror="—",
   source="canonical; absorbs pp._compute_logits_and_prior + pp.classify_phenotypes",
   notes="register_model now calls THIS instead of reimplementing the softmax")
fn(ns="ml", group="model", name="TCRIModel.get_p_ct", status="keep",
   signature="()", returns="ndarray[CT, P] — learned clone×covariate phenotype prior",
   writes="—", reads="pyro param store q_p_ct_raw", calls="module.get_p_ct",
   upstream="train", invariants="rows sum to 1; NaN→uniform guard", edges="—",
   mirror="—", source="keep as-is", notes="")
fn(ns="ml", group="model", name="TCRIModel.boost_phenotype_prior", status="keep — advanced",
   signature="(phenotype_name, boost_factor=5.0, *, affect_mixture=True)",
   returns="None (mutates clone_phen_prior / mixture in place)",
   writes="module.clone_phen_prior, module.mixture_concentration",
   reads="c2p_mat, centers", calls="—", upstream="(before) train",
   invariants="rows renormalized to 1 after boost", edges="unknown phenotype → ValueError",
   mirror="—", source="keep; drop its inline _ok (use _console)",
   notes="niche manual-prior knob; kept but flagged advanced")

# ============================ pp — preprocess ===============================
fn(ns="pp", group="register", name="register_model", status="keep",
   signature="(adata, model, *, latent_slot='X_tcri', batch_size=256, "
             "store_logits=True, store_logposterior=True, compute_umap=False, "
             "clonotype_key='trb_unique', ...)",
   returns="adata (all tcri_* state written)",
   writes="uns[tcri_metadata, tcri_p_ct, tcri_ct_to_cov, tcri_ct_to_c, tcri_local_scale, "
          "tcri_{covariate,clonotype,phenotype}_categories, tcri_{ct,cov}_array_for_cells]; "
          "obsm[X_tcri, X_tcri_logits, X_tcri_logposterior, X_tcri_probabilities]; obs[tcri_phenotype]",
   reads="model.module.*, adata_manager.registry",
   calls="model.get_latent_representation, model.get_cell_phenotype_probs, model.get_p_ct",
   upstream="model.train",
   invariants="per-cell arrays length == n_obs (guarded downstream)",
   edges="compute_umap optional; prob slot only written if absent",
   mirror="—",
   source="keep; folds in register_phenotype_key + register_clonotype_key (metadata only)",
   notes="THE bridge: model outputs → canonical adata state")
fn(ns="pp", group="engine", name="joint_distribution", status="merge — unified engine",
   signature="(adata, covariate, *, posterior=True, n_samples=0, temperature=1.0, "
             "clones=None, weighted=False, combine_with_logits=True, seed=None, silent=True)",
   returns="DataFrame[clone × phenotype] (rows sum to 1 unless weighted); "
           "n_samples>0 → stacked draws",
   writes="— (pure)", reads="uns[tcri_p_ct, tcri_local_scale, tcri_*_categories, "
          "tcri_metadata, tcri_{ct,cov}_array_for_cells, tcri_ct_to_*]; obsm[X_tcri_logits]",
   calls="—", upstream="register_model",
   invariants="posterior=True draws Dirichlet(local_scale·p_ct)+logit combine; "
              "posterior=False = point-estimate prior; FAILS LOUDLY on filtered-view length mismatch",
   edges="clones filter + reindex; weighted → mass-weighted, no renorm",
   mirror="—",
   source="MERGES joint_distribution_posterior (posterior=True) + joint_distribution (posterior=False)",
   notes="the single computational core under every metric")
fn(ns="pp", group="bookkeeping", name="group_singletons", status="keep",
   signature="(adata, *, clonotype_key='trb', groupby='patient', "
             "target_col='trb_unique', min_clone_size=10)",
   returns="None (writes obs)", writes="obs[target_col], obs[trb_candidate]",
   reads="obs[clonotype_key], obs[groupby]", calls="—", upstream="—",
   invariants="clones < min_clone_size → 'Singleton_<group>'", edges="—",
   mirror="—", source="keep; subsumes group_small_clones (hardcoded dup)",
   notes="canonical small-clone collapse")
fn(ns="pp", group="bookkeeping", name="clone_size", status="keep",
   signature="(adata, *, key_added='clone_size', return_counts=False)",
   returns="None | dict{clone: size}", writes="obs[clone_size]",
   reads="uns[tcri_metadata][clone_col] (was tcri_clone_key)", calls="—", upstream="register_model",
   invariants="size == cells per clone", edges="—", mirror="—",
   source="keep; retarget onto tcri_metadata (drop tcri_clone_key)", notes="")
fn(ns="pp", group="bookkeeping", name="filter_genes", status="rename + fix",
   signature="(adata, *, mt=True, rp=True, tcr=True, hsp=True, mtrn=True, ribo=True)",
   returns="adata (subset copy)", writes="— (returns new view)",
   reads="var_names", calls="—", upstream="—",
   invariants="each flag composes (AND), never resets the running mask",
   edges="HLA-* kept despite '-'/'.'", mirror="—",
   source="rename of remove_meaningless_genes + FIX flag-reset bug",
   notes="TCR-gene removal pre-embedding is methodologically in-scope; generic flags fixed")

# ============================ tl — metrics ==================================
fn(ns="tl", group="per-covariate", name="clonotypic_entropy", status="keep (+ single-phenotype mode)",
   signature="(adata, covariate, *, n_samples=0, temperature=1.0, clones=None, "
             "combine_with_logits=True, normalised=True, phenotype=None)",
   returns="Series[phenotype] (point) | ndarray[n_samples, P] (draws) | float (phenotype=)",
   writes="—", reads="uns[tcri_phenotype_categories]; (joint_distribution)",
   calls="pp.joint_distribution", upstream="register_model",
   invariants="H normalised by log2(n_clones); n_samples<1 → ValueError",
   edges="empty joint → NaN row", mirror="pl.clonotypic_entropy",
   source="keep; absorbs clonotypic_entropy_base via phenotype= arg",
   notes="H[P(c|phi,m)] — phenotype spread across clones")
fn(ns="tl", group="per-covariate", name="phenotypic_entropy", status="keep",
   signature="(adata, covariate, *, n_samples=0, temperature=1.0, clones=None, "
             "combine_with_logits=True, normalised=True)",
   returns="Series[clone] (point) | ndarray[n_samples, n_clones] (draws)",
   writes="—", reads="uns[tcri_metadata]; (joint_distribution)",
   calls="pp.joint_distribution", upstream="register_model",
   invariants="H normalised by log2(P); n_samples<1 → ValueError",
   edges="no clones at covariate → empty", mirror="pl.phenotypic_entropy",
   source="keep as-is", notes="H[P(phi|c,m)] — clone phenotypic plasticity")
fn(ns="tl", group="per-covariate", name="mutual_information", status="keep",
   signature="(adata, covariate, *, n_samples=0, temperature=1.0, clones=None, "
             "normalised=True, normalise_mode='average', posterior=True, "
             "combine_with_logits=True, verbose=True)",
   returns="float (point) | ndarray[n_samples] (draws)",
   writes="—", reads="(joint_distribution)", calls="pp.joint_distribution, _joint_to_mi",
   upstream="register_model",
   invariants="I = ΣΣ p·log2(p/(px·py)); normalised by mean/min marginal H",
   edges="posterior=False path implemented (no NotImplementedError)",
   mirror="pl.mutual_information",
   source="keep; wire the prior path that currently raises",
   notes="clone↔phenotype coupling")
fn(ns="tl", group="per-covariate", name="clonality", status="keep",
   signature="(adata)", returns="dict{phenotype: clonality∈[0,1]}",
   writes="—", reads="obs[tcri_phenotype], uns[tcri_metadata]", calls="—",
   upstream="register_model",
   invariants="1 - H(clone sizes)/log2(K); hard labels (no posterior)",
   edges="single clone → 1; nan→0", mirror="pl.clonality",
   source="keep; retarget onto tcri_metadata", notes="")
fn(ns="tl", group="between-covariate", name="flux", status="keep",
   signature="(adata, *, cov_from, cov_to, clones=None, distance_metric='l1', "
             "n_samples=0, temperature=1.0, weighted=False, posterior=True, "
             "combine_with_logits=True, seed=42)",
   returns="Series[clone] (point) | ndarray[n_samples, n_clones] (draws)",
   writes="—", reads="(joint_distribution ×2)", calls="pp.joint_distribution, _stats.distance",
   upstream="register_model",
   invariants="dist over common clones at both covariates",
   edges="no overlap → ValueError; metric ∈ {l1, dkl, callable}",
   mirror="pl.flux",
   source="keep; from_this/to_that → cov_from/cov_to; dkl via _stats registry",
   notes="phenotype-distribution shift per clone")
fn(ns="tl", group="between-covariate", name="delta_clonotypic_entropy", status="keep",
   signature="(adata, phenotype, *, cov_pre, cov_post, n_samples=1000, temperature=1.0, "
             "clones=None, weighted=False, normalised=True, posterior=True, "
             "combine_with_logits=True, seed=None)",
   returns="ndarray[n_samples] — H_post − H_pre",
   writes="—", reads="(clonotypic_entropy)", calls="tl.clonotypic_entropy",
   upstream="register_model",
   invariants="positive ⇒ entropy rose pre→post", edges="—",
   mirror="pl.ridge_delta_entropy (via delta_entropy_table)",
   source="keep; calls clonotypic_entropy(phenotype=)", notes="")
fn(ns="tl", group="tables", name="mi_compare", status="keep",
   signature="(adata, groupby, *, groups=None, treatment=None, n_samples=50, "
             "patient_col=None, clone_col=None, covariate_col=None, verbose=True)",
   returns="dict{samples, summary, pairs, params}",
   writes="—", reads="uns[tcri_metadata]; obs[groupby, patient_col]",
   calls="tl.mutual_information (per patient×covariate)", upstream="register_model",
   invariants="patient-level samples → group summary; pairs from groups",
   edges="missing group/cov skipped", mirror="pl.mi_compare",
   source="keep; uses shared _group_table loop", notes="patient-level MI comparison")
fn(ns="tl", group="tables", name="delta_entropy_table", status="keep",
   signature="(adata, *, cov_pre, cov_post, splitby='response', n_samples=1000, "
             "temperature=1.0, weighted=False, normalised=True, posterior=True, "
             "combine_with_logits=True, seed=42)",
   returns="DataFrame[phenotype × splitby] — delta_samples + summary stats",
   writes="—", reads="uns[tcri_metadata]; obs[splitby]",
   calls="tl.delta_clonotypic_entropy", upstream="register_model",
   invariants="keeps full delta vector per row", edges="—",
   mirror="pl.ridge_delta_entropy",
   source="keep; uses shared _group_table loop", notes="")
fn(ns="tl", group="tables", name="flux_table", status="keep",
   signature="(adata, *, cov_pre, cov_post, splitby='response', n_samples=0, "
             "temperature=1.0, weighted=False, posterior=True, "
             "combine_with_logits=True, distance_metric='l1', seed=42)",
   returns="DataFrame[clone × splitby] — flux_samples, flux_mean/sd, clone_size",
   writes="—", reads="uns[tcri_metadata]; obs[splitby]", calls="tl.flux",
   upstream="register_model", invariants="per-group clone scoping", edges="—",
   mirror="—", source="keep; uses shared _group_table loop", notes="")

# ============================ pl — plotting =================================
def plf(group, name, reads, viewtype, calls, status="keep", upstream="matching tl tool",
        source="keep", mirror="—"):
    fn(ns="pl", group=group, name=name, status=status,
       signature="(adata, *, ..., palette=None, figsize=..., ax=None, save=None)",
       returns="Axes | Figure | None", writes="—", reads=reads, calls=calls,
       upstream=upstream, invariants="reads computed state / calls tl; never owns model math",
       edges="—", mirror=mirror, source=source, notes=viewtype)

plf("metric", "mutual_information", "(via tl.mutual_information)", "box/strip across covariate ±splitby",
    "tl.mutual_information")
plf("metric", "clonotypic_entropy", "(via tl.clonotypic_entropy)", "per-phenotype box/dot ±covariate",
    "tl.clonotypic_entropy", status="rename", source="rename of clonotypic_entropy_by_phenotype")
plf("metric", "phenotypic_entropy", "(via tl.phenotypic_entropy)", "box/strip per covariate",
    "tl.phenotypic_entropy", status="fix", source="FIX broken tl call signature")
plf("metric", "clonality", "(via tl.clonality)", "stripplot per phenotype ±group",
    "tl.clonality, pl._metric_boxplot")
plf("metric", "flux", "(via tl.flux)", "box of flux distance by group",
    "tl.flux", status="fix", source="FIX broken key= passed to tl.flux")
plf("compare", "mi_compare", "uns/(via tl.mi_compare)", "patient MI box + AUROC/permutation stats",
    "tl.mi_compare, _stats.auc_and_label_permutation, _stats.bootstrap_auc")
plf("compare", "bayesian_mutual_information", "(via tl.mutual_information ×2)",
    "ΔMI KDE / posterior / bar across two covariates", "tl.mutual_information")
plf("compare", "ridge_delta_entropy", "DataFrame from tl.delta_entropy_table",
    "ridge plot of Δ-entropy posteriors", "—", status="fix",
    upstream="tl.delta_entropy_table", source="FIX undefined cm/st imports")
plf("distribution", "phenotypic_flux", "(via pp.joint_distribution)", "phenotype-flow sankey across covariates",
    "pp.joint_distribution, _build_sankey", source="keep; absorbs plot_pheno_sankey as private _build_sankey")
plf("distribution", "polar_plot", "(via pp.joint_distribution / tl.clonotypic_entropy)",
    "radar of phenotype distribution or entropy", "pp.joint_distribution, tl.clonotypic_entropy",
    status="fix", source="FIX undefined clonotypic_entropy ref + string phenotypes")
plf("umap", "clone_size_umap", "obs[clone_size], obsm[X_umap]", "UMAP colored by log clone size",
    "pp.clone_size")
plf("umap", "top_clone_umap", "obs[clone_col], obsm[X_umap]", "UMAP highlighting top-N clones", "—")
plf("umap", "phenotype_probabilities_umap", "obsm[X_tcri_probabilities]", "per-phenotype probability UMAP grid",
    "—", status="rename", source="rename of plot_phenotype_probabilities")
plf("diagnostic", "model_loss", "model.history_", "ELBO + dKL training curves", "—",
    status="move", upstream="model.train", source="moved from TCRIModel.plot_loss")
plf("diagnostic", "archetypes", "model.c2p_mat, model.centers", "archetype / clone-prior heatmaps", "—",
    status="move", upstream="model.train", source="moved from TCRIModel.plot_archetypes")
plf("diagnostic", "model_pgm", "— (static daft diagram)", "TCRI plate-diagram (PGM)", "—",
    status="move+merge", upstream="—",
    source="moved from ut.build_nested_tcri_pgm + ut.draw_tcri_pgm_nested (merged)")

# ============================ ut — session / IO =============================
fn(ns="ut", group="session", name="save_tcri_session", status="keep",
   signature="(model, adata, out_dir, *, save_adata=True, compression='gzip')",
   returns="dict{paths}", writes="run_dir/{model, pyro_params.pt, setup.json, adata.h5ad, meta.json}",
   reads="model.save, pyro store, adata", calls="write_adata_safely, _collect_setup_from_adata_or_model",
   upstream="train", invariants="adata written without tcri_manager (non-picklable)",
   edges="—", mirror="—", source="keep as-is", notes="")
fn(ns="ut", group="session", name="load_tcri_session", status="keep",
   signature="(run_dir, *, adata_path=None, map_location=None, layer=None)",
   returns="(model, adata)", writes="pyro param store (restored)",
   reads="run_dir artifacts", calls="TCRIModel.setup_anndata/.load, _pyro_load, "
   "_restore_category_order, _ensure_pyro_posterior_params, _disable_scvi_onload_train",
   upstream="save_tcri_session", invariants="category order restored; posterior params ensured",
   edges="missing pyro store → warn + uniform-prior fallback", mirror="—",
   source="keep as-is", notes="")
fn(ns="ut", group="session", name="write_adata_safely", status="keep",
   signature="(adata, path, *, compression='gzip')", returns="None",
   writes="path (h5ad without tcri_manager)", reads="adata", calls="_pop_nonserializables",
   upstream="—", invariants="strips non-serializable manager before write", edges="—",
   mirror="—", source="keep as-is", notes="")

# ---- shared primitives (grouped) -------------------------------------------
PRIMS = {
    "Console (tcri/_console.py) — replaces 3 duplicated copies": [
        ("_ok / _info / _warn / _fin", "every verbose tl/pp/pl/ml function"),
        ("_ascii_hist(samples)", "mutual_information, flux, delta_clonotypic_entropy (graph=)"),
        ("ANSI constants (RESET/BOLD/GRN/...)", "all of the above"),
    ],
    "Stats (tcri/_stats.py)": [
        ("_joint_to_mi(pxy, normalised, mode)", "tl.mutual_information"),
        ("distance(metric) -> f(p,q)  [l1 | dkl | callable]", "tl.flux, tl.flux_table"),
        ("auc_and_label_permutation(scores, labels)", "pl.mi_compare"),
        ("bootstrap_auc(scores, labels)", "pl.mi_compare"),
        ("_norm_entropy(p, base, n)", "clonotypic_entropy, phenotypic_entropy"),
    ],
    "Table builder (tcri/metrics/_tables.py)": [
        ("_group_table(adata, splitby, per_group_fn)", "mi_compare, delta_entropy_table, flux_table"),
    ],
    "Plot helpers (tcri/plotting/_base.py)": [
        ("_metric_boxplot(adata, fn, ...)", "clonality (+ any group×split metric box)"),
        ("_resolve_palette / tcri_colors", "every pl function"),
        ("_build_sankey / SankeyNode / _phenotype_mass_per_clone", "phenotypic_flux"),
    ],
    "Session internals (tcri/utils/_io.py)": [
        ("_ensure_dir, _to_jsonable, _pop_nonserializables", "save_tcri_session, write_adata_safely"),
        ("_collect_setup_from_adata_or_model, _restore_category_order", "save/load_tcri_session"),
        ("_pyro_load, _resolve_TCRIModel, _disable_scvi_onload_train, _ensure_pyro_posterior_params",
         "load_tcri_session"),
    ],
}

# ---- adata-state schema (the data contract; root of the DAG) ---------------
STATE = [
    ("uns[tcri_metadata]", "register_model", "{covariate,clone,phenotype,batch}_col",
     "≈ every tl/pp/pl function"),
    ("uns[tcri_p_ct]", "register_model", "learned clone×cov phenotype prior [CT,P]",
     "pp.joint_distribution"),
    ("uns[tcri_local_scale]", "register_model", "Dirichlet concentration scale",
     "pp.joint_distribution (posterior draw)"),
    ("uns[tcri_{covariate,clonotype,phenotype}_categories]", "register_model", "category orders",
     "joint_distribution, all metrics"),
    ("uns[tcri_ct_to_cov], uns[tcri_ct_to_c]", "register_model", "ct→cov / ct→clone maps",
     "pp.joint_distribution (posterior=False)"),
    ("uns[tcri_{ct,cov}_array_for_cells]", "register_model", "per-cell ct/cov indices",
     "pp.joint_distribution (cell selection)"),
    ("obsm[X_tcri]", "register_model", "latent posterior mean z", "pl.*_umap (if X_umap absent)"),
    ("obsm[X_tcri_logits]", "register_model", "classifier logits per cell",
     "pp.joint_distribution (combine_with_logits)"),
    ("obsm[X_tcri_probabilities]", "register_model", "softmax phenotype probs [N,P]",
     "pl.phenotype_probabilities_umap"),
    ("obs[tcri_phenotype]", "register_model", "hard phenotype label", "tl.clonality"),
    ("obs[clone_size]", "pp.clone_size", "cells per clone", "pl.clone_size_umap"),
    ("obs[trb_unique]", "pp.group_singletons", "collapsed clone id", "setup_anndata(clonotype_key)"),
    ("pyro param store", "ml.TCRIModel.train", "q_p_c_raw / q_p_ct_raw", "get_p_ct, save/load_session"),
]

DROPPED = [
    ("pp.classify_phenotypes", "deleted", "redundant cosine phenotype assignment; 0 callers"),
    ("pp.get_latent_embedding", "deleted", "trivial gaussian sampler; 0 callers"),
    ("pp.register_probability_columns", "deleted", "feeds only the dead probabilities()"),
    ("pp.gene_entropy", "deleted", "generic gene QC, out of scope; 0 callers"),
    ("pp.group_small_clones", "subsumed → group_singletons", "hardcoded inferior dup"),
    ("pp.register_phenotype_key / register_clonotype_key", "subsumed → register_model",
     "kill the tcri_*_key shadow convention"),
    ("pp._compute_logits_and_prior", "subsumed → ml.get_cell_phenotype_probs", "one prob path"),
    ("pp.joint_distribution_posterior", "subsumed → joint_distribution(posterior=True)", "unified engine"),
    ("tl.clonotypic_entropy_base", "subsumed → clonotypic_entropy(phenotype=)", "one entropy fn"),
    ("tl.clone_fraction", "deleted", "one-line value-counts; 0 callers"),
    ("tl.dkl", "subsumed → _stats.distance", "dead; flux had its own copy"),
    ("pl.compare_phenotypes", "deleted", "trivial crosstab heatmap; 0 callers"),
    ("pl.compare_joint_distribution", "deleted", "broken (undefined `model`)"),
    ("pl.probability_distribution", "deleted", "broken (self-recursion); covered by polar_plot"),
    ("pl.set_color_palette", "deleted", "buggy palette helper; 0 callers"),
    ("pl.plot_pheno_sankey", "subsumed → phenotypic_flux (_build_sankey)", "one public sankey"),
    ("pl.tcri_boxplot", "internalized → _metric_boxplot", "shared plot helper"),
    ("ut.probabilities", "deleted", "reads uns[joint_distribution] nothing ever writes"),
    ("ut.stars", "deleted", "imported but never called"),
    ("ut.build_nested_tcri_pgm / draw_tcri_pgm_nested", "moved+merged → pl.model_pgm",
     "describes the model, not IO"),
    ("ut.auc_and_label_permutation / bootstrap_auc", "internalized → _stats", "used only by pl.mi_compare"),
    ("SankeyNode.hex_to_rgb", "deleted", "unused method"),
    ("_ok/_info/_warn/_fin (×3), _ascii_hist (×2)", "subsumed → _console", "dedup"),
]

CONVENTIONS = (
    "<b>ml</b>: scvi/pyro model; the single source of phenotype probabilities "
    "(<code>get_cell_phenotype_probs</code>). "
    "<b>pp</b>: writes the canonical <code>tcri_*</code> state and owns the one "
    "<code>joint_distribution</code> engine (<code>posterior=</code> flag). "
    "<b>tl</b>: keyword-only; pure (reads state, returns arrays/frames, writes nothing); "
    "<code>n_samples=0</code> point estimate else posterior draws. "
    "<b>pl</b>: <code>ax/save/palette</code>; calls the matching <code>tl</code>/<code>pp</code> "
    "(never owns model math). "
    "<b>ut</b>: session save/load only. "
    "Every module declares <code>__all__</code>; shared logic lives in "
    "<code>_console</code>/<code>_stats</code>/<code>_base</code>, never copied."
)

# =============================================================================
# render
# =============================================================================
def mc(s):
    """Escape a value for use inside a Markdown table cell."""
    return str(s).replace("|", "\\|").replace("\n", " ")


def md():
    L = ["# TCRI — Target API Contract (`ml`/`pp`/`tl`/`pl`/`ut`)",
         "",
         "_The ideal post-refactor surface. Work toward this; do not drift. "
         "Generated from `build_tcri_contract.py` (single source of truth)._",
         "",
         "**Conventions.** " + CONVENTIONS.replace("<b>", "**").replace("</b>", "**")
         .replace("<code>", "`").replace("</code>", "`"),
         "",
         "## Namespaces", ""]
    for v, (c, label) in VIEWS.items():
        L.append(f"- **{label}** (`{v}`)")
    L += ["", "## Summary (target inventory)", "",
          "| ns | group | function | status | returns | writes |",
          "|---|---|---|---|---|---|"]
    for f in F:
        L.append(f"| `{f['ns']}` | {mc(f['group'])} | `{f['name']}` | {mc(f['status'])} | "
                 f"{mc(f['returns'])} | {mc(f['writes'])} |")
    L += ["", "## adata-state schema (the data contract — root of the DAG)", "",
          "| key | produced by | meaning | consumed by |", "|---|---|---|---|"]
    for k, prod, mean, cons in STATE:
        L.append(f"| `{k}` | `{prod}` | {mc(mean)} | {mc(cons)} |")
    L += ["", "## Detail cards", ""]
    for f in F:
        L.append(f"### `{f['ns']}.{f['name']}`  — _{f['group']}_  ·  {f['status']}")
        L.append("")
        sig = f["signature"] if f["signature"].startswith("(") else "(" + f["signature"] + ")"
        L.append(f"```python\n{f['ns']}.{f['name']}{sig}\n```")
        L.append(f"- **returns:** {f['returns']}")
        L.append(f"- **writes:** {f['writes']}")
        L.append(f"- **reads:** {f['reads']}")
        L.append(f"- **calls:** {f['calls']}")
        L.append(f"- **upstream:** {f['upstream']}")
        L.append(f"- **invariants:** {f['invariants']}")
        L.append(f"- **edge cases:** {f['edges']}")
        if f.get("mirror") and f["mirror"] != "—":
            L.append(f"- **plot mirror:** {f['mirror']}")
        L.append(f"- **provenance:** {f['source']}")
        if f.get("notes"):
            L.append(f"- **note:** {f['notes']}")
        L.append("")
    L += ["## Shared primitives", ""]
    for grp, items in PRIMS.items():
        L.append(f"**{grp}**")
        L.append("")
        for sig, used in items:
            L.append(f"- `{sig}` — {used}")
        L.append("")
    L += ["## Dropped / merged from the current code", "",
          "| current | disposition | why |", "|---|---|---|"]
    for cur, disp, why in DROPPED:
        L.append(f"| `{cur}` | {mc(disp)} | {mc(why)} |")
    L += ["", "_See `tcri_dependency_map.md` for the full call + producer/consumer graph._"]
    return "\n".join(L)


def esc(s):
    return _html.escape(str(s))


def render_html():
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}
    .wrap{max-width:1180px;margin:0 auto;padding:28px}
    h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:34px 0 12px;border-bottom:2px solid #eee;padding-bottom:6px}
    .sub{color:#555;margin:0 0 14px}
    .conv{background:#f1f5f9;border-left:4px solid #64748b;padding:10px 14px;border-radius:6px;margin:14px 0}
    .legend span{display:inline-block;margin:2px 8px 2px 0;padding:3px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:12px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0 20px}
    th,td{border:1px solid #e5e7eb;padding:6px 9px;text-align:left;vertical-align:top}
    th{background:#f3f4f6;position:sticky;top:0}
    code{background:#eef2ff;padding:1px 5px;border-radius:4px;font-size:92%}
    .badge{display:inline-block;padding:2px 9px;border-radius:11px;color:#fff;font-weight:600;font-size:11px;white-space:nowrap}
    .card{border:1px solid #e5e7eb;border-left:6px solid #999;border-radius:8px;padding:12px 16px;margin:12px 0;background:#fff}
    .card h3{margin:0 0 8px;font-size:15px}
    .card pre{background:#0f172a;color:#e2e8f0;padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:12px;white-space:pre-wrap}
    .card .row{margin:3px 0} .card .k{font-weight:700;color:#334155}
    .grp{font-weight:700;margin:12px 0 4px;color:#334155}
    .drop{color:#b91c1c} .merge{color:#b45309} .new{color:#15803d}
    ul{margin:4px 0 4px 18px}
    """
    def badge(v):
        c, label = VIEWS[v]
        return f'<span class="badge" style="background:{c}">{esc(label)}</span>'

    H = ['<!doctype html><meta charset="utf-8"><title>TCRI target API contract</title>',
         f'<style>{css}</style><div class="wrap">']
    H.append("<h1>TCRI — Target API Contract</h1>")
    H.append('<p class="sub">Ideal post-refactor surface · <code>ml</code>/<code>pp</code>/'
             '<code>tl</code>/<code>pl</code>/<code>ut</code> · work toward this, do not drift.</p>')
    H.append(f'<div class="conv"><b>Conventions.</b> {CONVENTIONS}</div>')
    H.append('<div class="legend">' + "".join(badge(v) for v in VIEWS) + "</div>")

    H.append("<h2>Summary (target inventory)</h2>")
    H.append("<table><tr><th>ns</th><th>group</th><th>function</th><th>status</th>"
             "<th>returns</th><th>writes</th></tr>")
    for f in F:
        H.append(f"<tr><td>{badge(f['ns'])}</td><td>{esc(f['group'])}</td>"
                 f"<td><code>{esc(f['name'])}</code></td><td>{esc(f['status'])}</td>"
                 f"<td>{esc(f['returns'])}</td><td>{esc(f['writes'])}</td></tr>")
    H.append("</table>")

    H.append("<h2>adata-state schema (the data contract — root of the DAG)</h2>")
    H.append("<table><tr><th>key</th><th>produced by</th><th>meaning</th><th>consumed by</th></tr>")
    for k, prod, mean, cons in STATE:
        H.append(f"<tr><td><code>{esc(k)}</code></td><td><code>{esc(prod)}</code></td>"
                 f"<td>{esc(mean)}</td><td>{esc(cons)}</td></tr>")
    H.append("</table>")

    H.append("<h2>Detail cards</h2>")
    for f in F:
        c = VIEWS[f["ns"]][0]
        H.append(f'<div class="card" style="border-left-color:{c}">')
        H.append(f'<h3>{badge(f["ns"])} &nbsp; <code>{esc(f["ns"])}.{esc(f["name"])}</code> '
                 f'<span style="color:#777;font-weight:400">· {esc(f["group"])} · {esc(f["status"])}</span></h3>')
        sig = f["signature"] if f["signature"].startswith("(") else "(" + f["signature"] + ")"
        H.append(f'<pre>{esc(f["ns"])}.{esc(f["name"])}{esc(sig)}</pre>')
        for k in ("returns", "writes", "reads", "calls", "upstream", "invariants", "edges"):
            label = {"edges": "edge cases"}.get(k, k)
            H.append(f'<div class="row"><span class="k">{label}:</span> {esc(f[k])}</div>')
        if f.get("mirror") and f["mirror"] != "—":
            H.append(f'<div class="row"><span class="k">plot mirror:</span> {esc(f["mirror"])}</div>')
        H.append(f'<div class="row"><span class="k">provenance:</span> {esc(f["source"])}</div>')
        if f.get("notes"):
            H.append(f'<div class="row"><span class="k">note:</span> {esc(f["notes"])}</div>')
        H.append("</div>")

    H.append("<h2>Shared primitives</h2>")
    for grp, items in PRIMS.items():
        H.append(f'<div class="grp">{esc(grp)}</div><ul>')
        for sig, used in items:
            H.append(f"<li><code>{esc(sig)}</code> — {esc(used)}</li>")
        H.append("</ul>")

    H.append("<h2>Dropped / merged from the current code</h2>")
    H.append("<table><tr><th>current</th><th>disposition</th><th>why</th></tr>")
    for cur, disp, why in DROPPED:
        cls = "drop" if disp.startswith("deleted") else ("merge" if ("subsumed" in disp or "internal" in disp or "moved" in disp) else "")
        H.append(f'<tr><td><code>{esc(cur)}</code></td><td class="{cls}">{esc(disp)}</td><td>{esc(why)}</td></tr>')
    H.append("</table>")
    H.append('<p class="sub">See <code>tcri_dependency_map.html</code> for the full call + '
             'producer/consumer graph.</p>')
    H.append("</div>")
    return "".join(H)


with open(OUT_MD, "w") as f:
    f.write(md())
with open(OUT_HTML, "w") as f:
    f.write(render_html())
print("wrote", OUT_MD)
print("wrote", OUT_HTML)
print("functions:", len(F), "| state keys:", len(STATE), "| dropped/merged:", len(DROPPED))
