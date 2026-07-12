# TCRI — Full Function Inventory & Consolidation Plan

_Generated from the inventory workflow (9 agents, 131 functions, completeness-verified) via `build_tcri_inventory.py`. Data: `tcri_inventory_data.json`. This is the working list we reduce the repo against._

## 0. Core definition

**Core = five things; everything else must justify itself against them.**
1. **Model (`ml`)** — `TCRIModel`: build / train / evaluate / register outputs onto an AnnData.
2. **Engine (`pp`)** — `joint_distribution`: Bayesian posterior sampling of the clone×phenotype distribution (the substrate every metric reads).
3. **Metrics (`tl`)** — `clonotypic_entropy`, `phenotypic_entropy`, `mutual_information` (+ `flux`, `delta_clonotypic_entropy`, and the tidy-table builders).
4. **Plotting (`pl`)** — plots that *directly* visualize those metrics and the joint distribution / flux.
5. **Utils (`ut`) + shared helpers** — session save/load; deduplicated console / stats / distance / color helpers.

## 1. Label counts (critic-corrected)

| label | count | disposition |
|---|---|---|
| core | 22 | keep (22 survive) |
| redundant | 4 | merge into core (5 groups) |
| plotting-beyond-core | 10 | move to examples / drop |
| model-construction | 31 | keep, split across model/_module,_priors,_classifier,_training |
| session-io | 12 | keep as utils/_session |
| helper | 35 | dedupe → _console/_stats/_distance/_base/_colors |
| dead-broken | 17 | 12 delete · 2 merge · 3 keep+fix |
| **total** | **131** | |

## 2. Grafiti reference layout (the target shape)

The reference layout (`../grafiti`) is a flat package with scanpy-style sub-packages aliased to short handles
(`model→ml`, `tools→tl`, `plotting→pl`, `preprocessing→pp`, `diagnostics→diag`, `datasets→ds`, `get.py→get`).
Six patterns tcri should copy:

1. **One file per topic, never a monolith.** `tools/_joint.py`, `tools/_motif.py`, … and `plotting/` mirrors them 1:1 by filename. (tcri's 1008-line `_metrics.py` and 1437-line `_plotting.py` are the anti-pattern.)
2. **Private cross-cutting sub-packages.** `_state/` (keys, resolve, storage, schemas) + `_compute/` (device-routed math) hold everything shared, so impl files stay thin. tcri analog: `_keys.py`, `_console.py`, `_stats.py`, `_distance.py` (and later a `_compute`).
3. **`__all__` at BOTH levels; NO `import *`.** Each impl module declares `__all__`; each `__init__` names every symbol explicitly and re-declares an aggregate `__all__` grouped by view. **This corrects the earlier `import *` instinct** — the mature pattern is explicit re-export, which keeps numpy/pandas/helpers *unexported*.
4. **`get.py` + `@tl_result` cache convention.** tl writes a versioned uns blob and returns a tidy result; `pl` functions are pure *cache renderers* (`load_result(adata, key)` → draw, never compute). *(tcri: adopt `_keys.py` now; defer the cache decorator — see open questions.)*
5. **`diagnostics/` returns DATA, not plots.** `gf.diag` runs read-only concordance/quality checks on the *finalized* model and returns a DataFrame ("did the model fit?"), deliberately outside the tl-writes / pl-reads loop. **This is exactly where PPCs + model-validation live.**
6. **Naming.** public package + private `_topic.py` impl modules; helper *packages* underscore-prefixed dirs, helper *files* underscore-prefixed, helper *functions* underscore-prefixed; tl↔pl twins share filename + function name.

## 3. Target tcri layout (grafiti-mirrored)

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

## 4. Full inventory — every function

Label is critic-corrected. Disposition folds in the five overlays (diagnostics reclass, keep+fix pl twins, n_samples convention).
### `tcri.ml` — model  (42 records)

| name | kind | label | disposition | purpose |
|---|---|---|---|---|
| `TCRIModel` | class | core | keep | High-level scvi model API: setup, build, train, and extract latent/phenotype/p_ct outputs. |
| `TCRIModel.__init__` | method | core | keep | Build clone->phenotype matrix, archetypes, clonotype-covariate index maps and class weights, then construct+prime TCRIModule. |
| `TCRIModel.get_cell_phenotype_probs` | method | core | keep | Per-cell phenotype probabilities by combining classifier logits with log p_ct prior (gate or additive), matching training. |
| `TCRIModel.get_latent_representation` | method | core | keep | Batched encode of an AnnData to a (n_cells, n_latent) numpy latent matrix. |
| `TCRIModel.get_p_ct` | method | core | keep | Return the learned clone-covariate x phenotype posterior p_ct as a numpy array. |
| `TCRIModel.setup_anndata` | method | core | keep | Register clonotype/phenotype/covariate/batch/count fields with scvi and stash the manager/layer on the AnnData. |
| `TCRIModel.train` | method | core | keep | Split data, build UnifiedTrainingPlan, and run TrainRunner with elbo_validation early stopping. |
| `TCRIModel.plot_archetypes` | method | plotting-beyond-core | → diagnostics | Heatmap the cluster-ordered clone-phenotype matrix and the archetype centroids. |
| `TCRIModel.plot_loss` | method | plotting-beyond-core | → diagnostics | Plot training/validation ELBO and prior-KL curves from self.history_. |
| `MixtureDirichlet` | class | model-construction | keep (internal) | Custom Pyro distribution: a mixture of Dirichlets over the phenotype simplex, used as the clonotype prior p_c. |
| `MixtureDirichlet.__call__` | method | model-construction | keep (internal) | Make the distribution callable as an alias for sample(). |
| `MixtureDirichlet.__init__` | method | model-construction | keep (internal) | Clamp concentrations, infer batch/B/K shapes, and init the TorchDistribution. |
| `MixtureDirichlet.log_prob` | method | model-construction | keep (internal) | Log-sum-exp of component Dirichlet log-probs weighted by (log) mixture weights. |
| `MixtureDirichlet.sample` | method | model-construction | keep (internal) | Sample a mixture component per batch element then draw from the selected Dirichlet. |
| `MixtureDirichlet.score_parts` | method | model-construction | keep (internal) | Return (log_prob, zero score-fn, zero entropy) so Pyro treats it as reparam-free. |
| `PhenotypeClassifier` | class | model-construction | keep (internal) | Temperature-scaled MLP head mapping latent z to phenotype logits. |
| `PhenotypeClassifier.__init__` | method | model-construction | keep (internal) | Build the stacked Linear/ReLU/Dropout MLP and store the softmax temperature. |
| `PhenotypeClassifier.forward` | method | model-construction | keep (internal) | Return MLP logits divided by temperature. |
| `TCRIModel.boost_phenotype_prior` | method | model-construction | keep (internal) | Multiply one phenotype's column in the clone prior (and optionally mixture centers) by a factor, renormalize, and overwrite module buffers. |
| `TCRIModule` | class | model-construction | keep (internal) | Pyro CVAE module with hierarchical clonotype->clonotype-covariate Dirichlet priors and a phenotype classifier. |
| `TCRIModule.__init__` | method | model-construction | keep (internal) | Construct encoder/decoder/classifier/VampPrior, px_r param, and register empty two-level buffers + class weights. |
| `TCRIModule._get_fn_args_from_batch` | method | model-construction | keep (internal) | Extract (x, batch_idx, log_library) tuple from a scvi batch dict for model()/guide(). |
| `TCRIModule.get_latent` | method | model-construction | keep (internal) | Encode a batch to the posterior-mean latent z_loc (collapses MC dim if present). |
| `TCRIModule.get_p_ct` | method | model-construction | keep (internal) | Read learned q_p_ct_raw from the Pyro param store and normalize (with guide-temperature) to the clone-covariate x phenotype posterior. |
| `TCRIModule.guide` | method | model-construction | keep (internal) | Variational guide: learnable Dirichlet params q(p_c), q(p_ct) and Normal q(z) from the encoder. |
| `TCRIModule.model` | method | model-construction | keep (internal) | Generative model: sample p_c (MixtureDirichlet), p_ct (Dirichlet), latent z (VampPrior), and ZINB gene obs. |
| `TCRIModule.prepare_two_level_params` | method | model-construction | keep (internal) | Normalize/temperature the clone-phenotype prior and register all two-level index buffers onto the module. |
| `TCRIModule.use_gate` | method | model-construction | keep (internal) | True when gate_prob is not None (selects convex-gate vs additive phenotype combination). |
| `UnifiedTrainingPlan` | class | model-construction | keep (internal) | Training plan adding KL warmup, diagnostics logging, and a validation_step emitting elbo_validation for early stopping. |
| `UnifiedTrainingPlan.__init__` | method | model-construction | keep (internal) | Choose TraceEnum_ELBO vs Trace_ELBO by module.use_enumeration and store optimizer/warmup config. |
| `UnifiedTrainingPlan.configure_optimizers` | method | model-construction | keep (internal) | Build an Adam optimizer over module parameters from optimizer_config. |
| `UnifiedTrainingPlan.loss` | method | model-construction | keep (internal) | Expose the configured ELBO loss object. |
| `UnifiedTrainingPlan.training_step` | method | model-construction | keep (internal) | Apply KL warmup, run the Pyro ELBO step, and log KL/entropy/confidence diagnostics. |
| `UnifiedTrainingPlan.validation_step` | method | model-construction | keep (internal) | Compute validation ELBO and prior-KL, logging elbo_validation for scvi early stopping. |
| `VampPrior` | class | model-construction | keep (internal) | VampPrior over latent z: a uniform mixture of encoder posteriors evaluated at learnable pseudo-inputs. |
| `VampPrior.__init__` | method | model-construction | keep (internal) | Register learnable pseudo-inputs as a Parameter and hold the shared encoder. |
| `VampPrior.get_mixture` | method | model-construction | keep (internal) | Encode pseudo-inputs and assemble a uniform MixtureSameFamily of Independent Normals as p(z). |
| `VampPrior.log_prob` | method | model-construction | keep (internal) | Log density of z under the VampPrior mixture. |
| `VampPrior.sample` | method | model-construction | keep (internal) | Draw samples from the VampPrior mixture. |
| `build_archetypes` | function | model-construction | keep (internal) | KMeans-cluster clone->phenotype rows into K normalized archetype centroids used as the Dirichlet-mixture prior concentration. |
| `TCRIModel.boost_phenotype_prior._ok` | inner-function | helper | extract → shared | Print a green check-marked status line. |
| `TCRIModel.use_gate` | method | helper | extract → shared | Public passthrough to module.use_gate. |

### `tcri.pp` — preprocessing  (20 records)

| name | kind | label | disposition | purpose |
|---|---|---|---|---|
| `joint_distribution_posterior` | function | core | keep | Draw one posterior Dirichlet sample of p_ct, combine with per-cell classifier logits, and aggregate to a clone x phenotype distribution … |
| `register_model` | function | core | keep | Register all TCRIModel outputs (priors, metadata, categories, per-cell ct/cov arrays, latent means, logits, log-posterior, phenotype … |
| `classify_phenotypes` | function | redundant | merge → register_model | Alternate per-cell phenotype assignment via cosine similarity of latent to per-phenotype archetypes, reweighted by posterior p_ct. |
| `joint_distribution` | function | redundant | merge → joint_distribution_posterior | Build a clone x phenotype distribution for a covariate from ct-level p_ct point estimates (n_samples=0) or Dirichlet draws (n_samples>0), … |
| `_ascii_hist` | function | helper | extract → shared | Render a numpy histogram of samples as an ASCII bar chart string. |
| `_compute_logits_and_prior` | function | helper | extract → shared | Run the model encoder+classifier over a data loader to extract per-cell classifier logits and the log-prior from get_p_ct. |
| `_fin` | function | helper | extract → shared | Print a magenta 'Done!' flourish unless quiet. |
| `_info` | function | helper | extract → shared | Print a dim key-value info line unless quiet. |
| `_ok` | function | helper | extract → shared | Print a green success line unless quiet. |
| `_warn` | function | helper | extract → shared | Print a yellow warning line unless quiet. |
| `clone_size` | function | helper | extract → shared | Compute per-clone cell counts from the registered clone key and write them per cell into obs. |
| `group_singletons` | function | helper | extract → shared | Collapse clones smaller than min_clone_size (per groupby) into 'Singleton_{group}' labels. |
| `group_singletons.collapse_singleton` | inner-function | helper | extract → shared | Map a row to 'Singleton_{group}' when its candidate clone count < min_clone_size, else keep candidate. |
| `register_clonotype_key` | function | helper | extract → shared | Register the clonotype obs column and its unique categories into uns. |
| `register_phenotype_key` | function | helper | extract → shared | Register the phenotype obs column and its unique categories into uns. |
| `gene_entropy` | function | dead-broken | delete | Compute per-gene Shannon entropy of expression-value counts (optionally per batch, aggregated) into var. |
| `get_latent_embedding` | function | dead-broken | delete | Draw Gaussian samples around the stored latent means with a scalar std. |
| `group_small_clones` | function | dead-broken | delete | Collapse clones with clone_size<4 into 'Singleton_{patient}', else '{trb}_{patient}', into obs['trb_unique']. |
| `register_probability_columns` | function | dead-broken | delete | Store a probability_columns list into uns. |
| `remove_meaningless_genes` | function | dead-broken | delete | Intend to filter out MT/RP/HSP/MTRN/TCR/RIK/GM/LINC/ambiguous genes and return a sliced copy. |

### `tcri.tl` — metrics  (21 records)

| name | kind | label | disposition | purpose |
|---|---|---|---|---|
| `_mi_from_joint` | function | core | keep | Compute (optionally normalised) mutual information from an already-normalised C×P joint table. |
| `clonotypic_entropy` | function | core | keep | Posterior mean (or per-draw matrix) normalised clonotypic entropy per phenotype at one covariate. |
| `delta_clonotypic_entropy` | function | core | keep | Monte-Carlo posterior samples of H_post−H_pre clonotypic entropy for one phenotype. |
| `delta_entropy_table` | function | core | keep | Tidy Δ-clonotypic-entropy table: one row per phenotype × splitby group with mean/sd/HDI/p and raw sample vector. |
| `flux` | function | core | keep | Per-clone phenotypic-distribution distance (l1/dkl/callable) between two covariates; point estimate or per-draw. |
| `flux_table` | function | core | keep | Tidy per-clone flux table: flux mean/sd + sample vector + clone size, per splitby group. |
| `mi_compare` | function | core | keep | Build a tidy per-patient MI samples+summary table across covariates and group pairs. |
| `mutual_information` | function | core | keep | Clone×phenotype mutual information at one covariate: point estimate (n_samples=0) or per-draw array. |
| `phenotypic_entropy` | function | core | keep | Posterior mean (or per-draw) normalised phenotypic entropy per clonotype at one covariate. |
| `clonality` | function | redundant | merge → clonotypic_entropy | Per-phenotype clonality 1 − H/log2K from observed hard clone-size counts. |
| `clonotypic_entropy_base` | function | redundant | merge → clonotypic_entropy | Single-draw normalised clonotypic entropy for ONE phenotype at ONE covariate. |
| `_ascii_hist` | function | helper | extract → shared | Build a text ASCII histogram of a sample vector for notebook/SSH display. |
| `_fin` | function | helper | extract → shared | Print a magenta 'Done!' flourish unless quiet. |
| `_info` | function | helper | extract → shared | Print a cyan key-value info line unless quiet. |
| `_ok` | function | helper | extract → shared | Print a green success/checkmark line unless quiet. |
| `_warn` | function | helper | extract → shared | Print a yellow warning line unless quiet. |
| `flux.dkl_func` | inner-function | helper | extract → shared | KL divergence kernel used when distance_metric=='dkl' inside flux. |
| `mutual_information._get_df` | inner-function | helper | extract → shared | Return one joint-distribution table for the MI computation (posterior draw). |
| `_ent` | function | dead-broken | delete | Normalise a vector and return its Shannon entropy in given base. |
| `clone_fraction` | function | dead-broken | delete | Per-group nested dict of each clone's frequency (count/total) within the group. |
| `dkl` | function | dead-broken | delete | KL divergence of p‖q via scipy.stats.entropy after clipping. |

### `tcri.pl` — plotting  (29 records)

| name | kind | label | disposition | purpose |
|---|---|---|---|---|
| `clonotypic_entropy_by_phenotype` | function | core | keep | Box-and-dot plot of clonotypic entropy per phenotype and covariate, with bootstrap/MWU significance brackets and per-patient dots. |
| `mi_compare` | function | core | keep | Plot per-patient TCRi normalized MI across covariate group pairs as boxplots with jittered points and AUROC/MWU/label-permutation stats. |
| `phenotypic_flux` | function | core | keep | Public flux Sankey across `order` values of `splitby`; thin convenience wrapper adding x-ticks and save. |
| `plot_pheno_sankey` | function | core | keep | Draw a Sankey of phenotype-distribution flow across ordered covariate values using per-clone outer-product flow geometry. |
| `clonality` | function | plotting-beyond-core | move→examples / drop | Boxplot of the clonality metric via tcri_boxplot. |
| `clone_size_umap` | function | plotting-beyond-core | move→examples / drop | UMAP scatter colored by log10 clone size. |
| `compare_phenotypes` | function | plotting-beyond-core | move→examples / drop | Heatmap of the row-normalized crosstab between two arbitrary obs columns. |
| `flux` | function | plotting-beyond-core | move→examples / drop | Boxplot of per-clone flux (flux_tl distance) between consecutive `order` values, grouped by `groupby` and colored by a `paint` obs category. |
| `plot_phenotype_probabilities` | function | plotting-beyond-core | move→examples / drop | UMAP panels colored by each per-cell phenotype probability. |
| `top_clone_umap` | function | plotting-beyond-core | move→examples / drop | UMAP scatter highlighting the top-N largest clones over a grey background. |
| `SankeyNode` | class | helper | extract → shared | Drawing primitive representing one rectangular sankey node and the ribbons flowing out of it, used to render the phenotypic-flux sankey. |
| `SankeyNode.__init__` | method | helper | extract → shared | Compute node bounding box (min/max x,y from center x, base y, width dx, height val) and build the mpatches.Rectangle patch. |
| `SankeyNode.plot` | method | helper | extract → shared | Render the node's rectangle patch onto the given matplotlib axis. |
| `SankeyNode.plot_node_connection` | method | helper | extract → shared | Draw the curved, color-interpolated ribbon (500 fill_between segments, sigmoid-shaped top/bottom edges) from this node to a destination … |
| `_fin` | function | helper | extract → shared | Print a magenta 'Done!' final flourish to stdout. |
| `_info` | function | helper | extract → shared | Print a cyan key-value info line to stdout. |
| `_ok` | function | helper | extract → shared | Print a green checkmark success line to stdout. |
| `_phenotype_mass_per_clone` | function | helper | extract → shared | Return {clone_id -> phenotype-mass vector} at one covariate by summing joint_distribution rows, optionally weighting each row by the … |
| `_warn` | function | helper | extract → shared | Print a yellow warning line to stdout. |
| `set_color_palette` | function | helper | extract → shared | Assign tcri_colors to the categories of each obs column and store them in uns['<col>_colors']; return category->color map. |
| `tcri_boxplot` | function | helper | extract → shared | Generic per-phenotype metric boxplot/stripplot engine that applies a metric `function` across groupby/splitby strata. |
| `SankeyNode.hex_to_rgb` | method | dead-broken | delete | Parse a #RRGGBB hex string into a normalized (r,g,b) float tuple in [0,1]. |
| `bayesian_mutual_information` | function | dead-broken | delete | 3-panel Bayesian Δ-MI (post-pre) analysis across splitby strata: Δ-MI KDEs, per-condition pre/post KDEs, and Δ bar summary with HDI/P(>0). |
| `compare_joint_distribution` | function | dead-broken | → diagnostics (PPC) | Side-by-side clustered heatmaps/dendrograms of model-inferred vs empirical joint distributions per covariate. |
| `mutual_information` | function | dead-broken | keep + FIX | Box/strip plot of clonotype<->phenotype mutual information per covariate and batch (intended core MI plot). |
| `phenotypic_entropy` | function | dead-broken | keep + FIX | Box/strip plot of phenotypic entropy per covariate and batch (intended core entropy plot). |
| `polar_plot` | function | dead-broken | delete | Radar/polar plot of per-phenotype distribution or entropy across split values. |
| `probability_distribution` | function | dead-broken | delete | Intended: barplots of the phenotype probability distribution per split value. |
| `ridge_delta_entropy` | function | dead-broken | keep + FIX | Ridge/joyplot of Δ-clonotypic-entropy posteriors per phenotype with significance brackets on the first two groups. |

### `tcri.ut` — utils  (19 records)

| name | kind | label | disposition | purpose |
|---|---|---|---|---|
| `build_nested_tcri_pgm` | function | plotting-beyond-core | → diagnostics | Construct a daft probabilistic-graphical-model diagram of the nested TCRI generative model (plates for batch/clonotype/ct/data, nodes … |
| `draw_tcri_pgm_nested` | function | plotting-beyond-core | → diagnostics | Render build_nested_tcri_pgm and save it to a hardcoded PDF, then plt.show(). |
| `_collect_setup_from_adata_or_model` | function | session-io | keep | Assemble the setup dict (phenotype/clone/covariate/batch cols, category lists, layer) from adata.uns['tcri_metadata']/categories and, if … |
| `_disable_scvi_onload_train` | function | session-io | keep | Context manager that monkey-patches scvi PyroBaseModuleClass.on_load to a no-op during model load, avoiding the one-step warmup train that … |
| `_disable_scvi_onload_train._noop` | inner-function | session-io | keep | Replacement on_load that swallows all args and just clears the Pyro param store (own params loaded afterward). |
| `_ensure_dir` | function | session-io | keep | os.makedirs(path, exist_ok=True) wrapper. |
| `_ensure_pyro_posterior_params` | function | session-io | keep | After load, guarantee the Pyro param 'q_p_ct_raw' exists; if missing, warn and re-init it to a uniform 1/P simplex so posterior metrics … |
| `_pop_nonserializables` | function | session-io | keep | Remove the non-picklable AnnDataManager (uns['tcri_manager']) before writing h5ad, returning a sidecar note. |
| `_pyro_load` | function | session-io | keep | torch.load a Pyro param-store state dict (weights_only=False for constraint objects) and set it into the global param store. |
| `_resolve_TCRIModel` | function | session-io | keep | Dynamically locate and import the TCRIModel class from common module paths or sibling files (editable installs). |
| `_restore_category_order` | function | session-io | keep | Re-impose saved categorical ordering on adata.obs phenotype/clone/covariate columns from the setup dict. |
| `load_tcri_session` | function | session-io | keep | Reconstruct a trained TCRIModel + AnnData from a saved run dir: read h5ad, restore setup/category order, re-run setup_anndata, load model … |
| `save_tcri_session` | function | session-io | keep | Persist a trained session: scvi model (weights+registry, no embedded adata), Pyro param store, setup.json, sanitized adata.h5ad, and … |
| `write_adata_safely` | function | session-io | keep | Write an AnnData to h5ad after stripping the non-serializable tcri_manager (not restored; rebuilt on load). |
| `_to_jsonable` | function | helper | extract → shared | Recursively coerce arbitrary values (numpy/torch/nested) into JSON-serializable primitives. |
| `auc_and_label_permutation` | function | helper | extract → shared | Compute observed ROC-AUC plus a two-sided permutation p-value (exact combinations if feasible, else Monte-Carlo). |
| `bootstrap_auc` | function | helper | extract → shared | Bootstrap 95% CI (2.5/97.5 quantiles) of ROC-AUC, resampling until both classes present. |
| `stars` | function | helper | extract → shared | Map a p-value to a significance-star string (****/***/**/*/ns). |
| `probabilities` | function | dead-broken | delete | Build a per-cell {barcode: {phenotype: prob}} dict from the probability columns and the joint_distribution index. |

## 5. Consolidation groups (redundant → core)

| into | members | rationale |
|---|---|---|
| `metrics.clonotypic_entropy` | metrics.clonotypic_entropy_base, metrics.clonality | clonotypic_entropy_base is the single-phenotype/single-draw special case (only extra is posterior=False + weighted) and its sole caller … |
| `preprocessing.joint_distribution_posterior` | preprocessing.joint_distribution | joint_distribution is the point-estimate/ct-level path producing the same covariate->clone x phenotype table WITHOUT per-cell logits. Unify into one engine … |
| `preprocessing.register_model` | preprocessing.classify_phenotypes | classify_phenotypes writes the same phenotype-probability + hard-label slots register_model produces, via a different archetype-cosine algorithm instead of … |
| `preprocessing.group_singletons` | preprocessing.group_small_clones | Both collapse sub-threshold clones into 'Singleton_{group}' labels written to obs['trb_unique']. group_small_clones is dead-broken/dataset-specific (0 … |
| `tcri/_distance.py (kl_divergence)` | metrics.dkl, metrics.flux.dkl_func | Module-level dkl is dead (0 callers) and flux's inner dkl_func reimplements the same KL kernel for the distance_metric=='dkl' branch. Fold both into one … |

## 6. Deletions

| function | reason |
|---|---|
| `metrics._ent` | 0 callers; would-be entropy helper never wired in — the entropy metrics inline their own clip/normalise/entropy. |
| `metrics.clone_fraction` | 0 callers; slow reimplementation of obs.groupby(groupby)[clone].value_counts(normalize=True); also uses legacy uns['tcri_clone_key']. |
| `preprocessing.remove_meaningless_genes` | 0 callers AND broken: the include_mtrn/include_hsp branches reassign `genes` from the FULL adata.var.index, silently discarding all prior filters. |
| `preprocessing.get_latent_embedding` | 0 callers; fabricates a spherical Gaussian with a constant posterior_scale (not the model's true latent variance), and default n_samples=0 yields an … |
| `preprocessing.register_probability_columns` | 0 callers; trivial setter whose uns['probability_columns'] key is never read anywhere in the package. |
| `preprocessing.gene_entropy` | 0 callers; per-GENE expression entropy unrelated to core clonotypic/phenotypic entropy; also drops the first count bin on an assumption. |
| `plotting.compare_joint_distribution` | **OVERLAY OVERRIDE → keep as diagnostics PPC** (was: dead-broken: references an undefined global `model` (model.adata_manager.registry) -> NameError at runtime.) |
| `plotting.probability_distribution` | dead-broken: infinite self-recursion (calls probability_distribution instead of utils.probabilities); also ax indexing fails when splitby=None. |
| `plotting.bayesian_mutual_information` | dead-broken (passes unsupported weighted= to tl.mutual_information -> TypeError) AND redundant with the functional core pl.mi_compare MI-comparison … |
| `plotting.polar_plot` | dead-broken: phenotypes defaults to a column-NAME string (iterated char-by-char) and the entropy branch calls unimported clonotypic_entropy -> … |
| `plotting._sankey.SankeyNode.hex_to_rgb` | 0 callers; plot_node_connection uses matplotlib mcolors.to_rgb instead; contract doc already marks it a deleted/unused method. |
| `utils.probabilities` | dead-broken: reads uns['joint_distribution'] which is never written anywhere in the package -> KeyError. |

## 7. Helper extraction (dedupe → shared modules)

| helper | → module | current copies |
|---|---|---|
| _ok, _info, _warn, _fin (ANSI console printers) | `tcri/_console.py` | metrics/_metrics.py:47-56; preprocessing/_preprocessing.py:50-59; … |
| _ascii_hist (numpy-histogram ASCII bar builder) | `tcri/_console.py` | metrics/_metrics.py:62 (the used copy); preprocessing/_preprocessing.py:62 (identical, … |
| stars, auc_and_label_permutation, bootstrap_auc (significance + AUROC stats) | `tcri/_stats.py` | utils/_utils.py (single copies today; relocate out of the utils monolith into a named … |
| kl_divergence + l1_distance + phenotype_distance dispatcher | `tcri/_distance.py` | metrics/_metrics.py:159 (dead module-level dkl) and metrics/_metrics.py flux.dkl_func … |
| _mi_from_joint (single-source MI kernel) | `tcri/metrics/_mutual_information.py (module-private)` | metrics/_metrics.py:_mi_from_joint (single copy; keep private, called by … |
| tcri_boxplot (generic per-phenotype box/strip engine) + _finish (scanpy show/save) | `tcri/plotting/_base.py` | plotting/_plotting.py:598 tcri_boxplot; _finish is new (formalize the fig/ax show/save … |
| SankeyNode drawing primitive + _phenotype_mass_per_clone data-prep | `tcri/plotting/_sankey.py` | plotting/_sankey.py (SankeyNode already; drop hex_to_rgb); … |
| set_color_palette + tcri_colors palette constants | `tcri/plotting/_colors.py` | plotting/_plotting.py set_color_palette (also fix: it writes uns on adata.copy() so … |
| register_phenotype_key, register_clonotype_key, _compute_logits_and_prior | `tcri/preprocessing/_register.py` | preprocessing/_preprocessing.py (relocate alongside register_model; migrate the legacy … |
| group_singletons (+ collapse_singleton inner), clone_size | `tcri/preprocessing/_clones.py` | preprocessing/_preprocessing.py:83 group_singletons; preprocessing/_preprocessing.py … |
| _to_jsonable (recursive JSON coercion of numpy/torch/nested) | `tcri/utils/_session.py` | utils/_utils.py:_to_jsonable (0 external callers today; wire it into save_tcri_session's … |

## 8. Plotting triage

**Core (keep):** `clonotypic_entropy_by_phenotype`, `mi_compare`, `phenotypic_flux`, `plot_pheno_sankey`, `mutual_information`, `phenotypic_entropy`, `ridge_delta_entropy`

**Beyond core (move→examples / drop):**

- compare_phenotypes (move->examples: generic categorical-crosstab heatmap, visualizes no core metric)
- top_clone_umap (move->examples: bespoke top-N-clone UMAP overlay, hardcoded title)
- clone_size_umap (move->examples: bespoke clone-size UMAP overlay; mutates adata.obs as a side effect)
- plot_phenotype_probabilities (move->examples: per-cell phenotype-probability UMAP panels)
- clonality (drop: only plots the merged-away redundant clonality metric)
- flux boxplot (drop: broken on default paint=None path — pcat used before assignment -> NameError — plus dataset-specific `paint` overlay; the Sankey is the flux visualization)
- plot_archetypes (drop or keep as optional TCRIModel model-diagnostic method: model prior heatmap, not a core-metric plot)
- plot_loss (drop or keep as optional TCRIModel model-diagnostic method: training ELBO/KL curves)
- build_nested_tcri_pgm (move->docs/examples: daft PGM diagram of the model architecture)
- draw_tcri_pgm_nested (drop: one-off PGM export with a hardcoded output PDF filename)

## 9. Rename / readability map

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

## 10. Metric conventions & scoping

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

## 11. Open questions (decide before executing)

_(see synthesis output)_

---
_Overlays applied on top of the workflow synthesis: (a) `diagnostics/` = PPCs + model validation (`compare_joint_distribution` reclassified from delete → diagnostics PPC seed; `plot_loss`/`plot_archetypes`/PGM relocated); (b) uniform `n_samples=0` point-estimate convention, drop `point_estimate=`; (c) prior-vs-posterior parked; (d) explicit `__all__` re-export (NOT `import *`) per grafiti; (e) full rename map._