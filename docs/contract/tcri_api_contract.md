# TCRI — Target API Contract (`ml`/`pp`/`tl`/`pl`/`ut`)

_The ideal post-refactor surface. Work toward this; do not drift. Generated from `build_tcri_contract.py` (single source of truth)._

**Conventions.** **ml**: scvi/pyro model; the single source of phenotype probabilities (`get_cell_phenotype_probs`). **pp**: writes the canonical `tcri_*` state and owns the one `joint_distribution` engine (`posterior=` flag). **tl**: keyword-only; pure (reads state, returns arrays/frames, writes nothing); `n_samples=0` point estimate else posterior draws. **pl**: `ax/save/palette`; calls the matching `tl`/`pp` (never owns model math). **ut**: session save/load only. Every module declares `__all__`; shared logic lives in `_console`/`_stats`/`_base`, never copied.

## Namespaces

- **Model (ml)** (`ml`)
- **Preprocess (pp)** (`pp`)
- **Metrics (tl)** (`tl`)
- **Plotting (pl)** (`pl`)
- **Session / IO (ut)** (`ut`)

## Summary (target inventory)

| ns | group | function | status | returns | writes |
|---|---|---|---|---|---|
| `ml` | model | `TCRIModel.setup_anndata` | keep | adata (registers scvi AnnDataManager) | uns[tcri_manager]; registry{clonotype/phenotype/covariate/batch_col} |
| `ml` | model | `TCRIModel.train` | keep | None (fits in place; populates history_) | pyro param store (q_p_c_raw, q_p_ct_raw); model weights |
| `ml` | model | `TCRIModel.get_latent_representation` | keep | ndarray[N, n_latent] — posterior-mean z | — |
| `ml` | model | `TCRIModel.get_cell_phenotype_probs` | keep — now the ONLY prob path | ndarray[N, P] — per-cell phenotype probabilities | — |
| `ml` | model | `TCRIModel.get_p_ct` | keep | ndarray[CT, P] — learned clone×covariate phenotype prior | — |
| `ml` | model | `TCRIModel.boost_phenotype_prior` | keep — advanced | None (mutates clone_phen_prior / mixture in place) | module.clone_phen_prior, module.mixture_concentration |
| `pp` | register | `register_model` | keep | adata (all tcri_* state written) | uns[tcri_metadata, tcri_p_ct, tcri_ct_to_cov, tcri_ct_to_c, tcri_local_scale, tcri_{covariate,clonotype,phenotype}_categories, tcri_{ct,cov}_array_for_cells]; obsm[X_tcri, X_tcri_logits, X_tcri_logposterior, X_tcri_probabilities]; obs[tcri_phenotype] |
| `pp` | engine | `joint_distribution` | merge — unified engine | DataFrame[clone × phenotype] (rows sum to 1 unless weighted); n_samples>0 → stacked draws | — (pure) |
| `pp` | bookkeeping | `group_singletons` | keep | None (writes obs) | obs[target_col], obs[trb_candidate] |
| `pp` | bookkeeping | `clone_size` | keep | None \| dict{clone: size} | obs[clone_size] |
| `pp` | bookkeeping | `filter_genes` | rename + fix | adata (subset copy) | — (returns new view) |
| `tl` | per-covariate | `clonotypic_entropy` | keep (+ single-phenotype mode) | Series[phenotype] (point) \| ndarray[n_samples, P] (draws) \| float (phenotype=) | — |
| `tl` | per-covariate | `phenotypic_entropy` | keep | Series[clone] (point) \| ndarray[n_samples, n_clones] (draws) | — |
| `tl` | per-covariate | `mutual_information` | keep | float (point) \| ndarray[n_samples] (draws) | — |
| `tl` | per-covariate | `clonality` | keep | dict{phenotype: clonality∈[0,1]} | — |
| `tl` | between-covariate | `flux` | keep | Series[clone] (point) \| ndarray[n_samples, n_clones] (draws) | — |
| `tl` | between-covariate | `delta_clonotypic_entropy` | keep | ndarray[n_samples] — H_post − H_pre | — |
| `tl` | tables | `mi_compare` | keep | dict{samples, summary, pairs, params} | — |
| `tl` | tables | `delta_entropy_table` | keep | DataFrame[phenotype × splitby] — delta_samples + summary stats | — |
| `tl` | tables | `flux_table` | keep | DataFrame[clone × splitby] — flux_samples, flux_mean/sd, clone_size | — |
| `pl` | metric | `mutual_information` | keep | Axes \| Figure \| None | — |
| `pl` | metric | `clonotypic_entropy` | rename | Axes \| Figure \| None | — |
| `pl` | metric | `phenotypic_entropy` | fix | Axes \| Figure \| None | — |
| `pl` | metric | `clonality` | keep | Axes \| Figure \| None | — |
| `pl` | metric | `flux` | fix | Axes \| Figure \| None | — |
| `pl` | compare | `mi_compare` | keep | Axes \| Figure \| None | — |
| `pl` | compare | `bayesian_mutual_information` | keep | Axes \| Figure \| None | — |
| `pl` | compare | `ridge_delta_entropy` | fix | Axes \| Figure \| None | — |
| `pl` | distribution | `phenotypic_flux` | keep | Axes \| Figure \| None | — |
| `pl` | distribution | `polar_plot` | fix | Axes \| Figure \| None | — |
| `pl` | umap | `clone_size_umap` | keep | Axes \| Figure \| None | — |
| `pl` | umap | `top_clone_umap` | keep | Axes \| Figure \| None | — |
| `pl` | umap | `phenotype_probabilities_umap` | rename | Axes \| Figure \| None | — |
| `pl` | diagnostic | `model_loss` | move | Axes \| Figure \| None | — |
| `pl` | diagnostic | `archetypes` | move | Axes \| Figure \| None | — |
| `pl` | diagnostic | `model_pgm` | move+merge | Axes \| Figure \| None | — |
| `ut` | session | `save_tcri_session` | keep | dict{paths} | run_dir/{model, pyro_params.pt, setup.json, adata.h5ad, meta.json} |
| `ut` | session | `load_tcri_session` | keep | (model, adata) | pyro param store (restored) |
| `ut` | session | `write_adata_safely` | keep | None | path (h5ad without tcri_manager) |

## adata-state schema (the data contract — root of the DAG)

| key | produced by | meaning | consumed by |
|---|---|---|---|
| `uns[tcri_metadata]` | `register_model` | {covariate,clone,phenotype,batch}_col | ≈ every tl/pp/pl function |
| `uns[tcri_p_ct]` | `register_model` | learned clone×cov phenotype prior [CT,P] | pp.joint_distribution |
| `uns[tcri_local_scale]` | `register_model` | Dirichlet concentration scale | pp.joint_distribution (posterior draw) |
| `uns[tcri_{covariate,clonotype,phenotype}_categories]` | `register_model` | category orders | joint_distribution, all metrics |
| `uns[tcri_ct_to_cov], uns[tcri_ct_to_c]` | `register_model` | ct→cov / ct→clone maps | pp.joint_distribution (posterior=False) |
| `uns[tcri_{ct,cov}_array_for_cells]` | `register_model` | per-cell ct/cov indices | pp.joint_distribution (cell selection) |
| `obsm[X_tcri]` | `register_model` | latent posterior mean z | pl.*_umap (if X_umap absent) |
| `obsm[X_tcri_logits]` | `register_model` | classifier logits per cell | pp.joint_distribution (combine_with_logits) |
| `obsm[X_tcri_probabilities]` | `register_model` | softmax phenotype probs [N,P] | pl.phenotype_probabilities_umap |
| `obs[tcri_phenotype]` | `register_model` | hard phenotype label | tl.clonality |
| `obs[clone_size]` | `pp.clone_size` | cells per clone | pl.clone_size_umap |
| `obs[trb_unique]` | `pp.group_singletons` | collapsed clone id | setup_anndata(clonotype_key) |
| `pyro param store` | `ml.TCRIModel.train` | q_p_c_raw / q_p_ct_raw | get_p_ct, save/load_session |

## Detail cards

### `ml.TCRIModel.setup_anndata`  — _model_  ·  keep

```python
ml.TCRIModel.setup_anndata(adata, *, layer=None, clonotype_key='unique_clone_id', phenotype_key='phenotype_col', covariate_key='timepoint', batch_key='patient')
```
- **returns:** adata (registers scvi AnnDataManager)
- **writes:** uns[tcri_manager]; registry{clonotype/phenotype/covariate/batch_col}
- **reads:** obs[clonotype/phenotype/covariate/batch_key]
- **calls:** —
- **upstream:** —
- **invariants:** all four obs columns must exist → ValueError otherwise
- **edge cases:** layer=None → counts from X
- **provenance:** keep as-is
- **note:** scvi classmethod entry point

### `ml.TCRIModel.train`  — _model_  ·  keep

```python
ml.TCRIModel.train(max_epochs=1000, batch_size=1000, lr=1e-3, reconstruction_loss_scale=1e-3, n_steps_kl_warmup=2000, **kw)
```
- **returns:** None (fits in place; populates history_)
- **writes:** pyro param store (q_p_c_raw, q_p_ct_raw); model weights
- **reads:** adata_manager
- **calls:** UnifiedTrainingPlan, TrainRunner, DataSplitter
- **upstream:** setup_anndata
- **invariants:** early-stops on elbo_validation
- **edge cases:** —
- **plot mirror:** pl.model_loss
- **provenance:** keep as-is

### `ml.TCRIModel.get_latent_representation`  — _model_  ·  keep

```python
ml.TCRIModel.get_latent_representation(adata=None, indices=None, batch_size=None)
```
- **returns:** ndarray[N, n_latent] — posterior-mean z
- **writes:** —
- **reads:** encoder
- **calls:** module.get_latent
- **upstream:** train
- **invariants:** deterministic (mean, no sampling)
- **edge cases:** —
- **provenance:** keep as-is

### `ml.TCRIModel.get_cell_phenotype_probs`  — _model_  ·  keep — now the ONLY prob path

```python
ml.TCRIModel.get_cell_phenotype_probs(adata=None, batch_size=256, eps=1e-8)
```
- **returns:** ndarray[N, P] — per-cell phenotype probabilities
- **writes:** —
- **reads:** encoder, classifier, get_p_ct(), module.ct_array
- **calls:** module.get_p_ct
- **upstream:** train
- **invariants:** softmax(gate·logits + (1-gate)·log prior) OR additive when no gate
- **edge cases:** use_gate toggles the two combination rules
- **provenance:** canonical; absorbs pp._compute_logits_and_prior + pp.classify_phenotypes
- **note:** register_model now calls THIS instead of reimplementing the softmax

### `ml.TCRIModel.get_p_ct`  — _model_  ·  keep

```python
ml.TCRIModel.get_p_ct()
```
- **returns:** ndarray[CT, P] — learned clone×covariate phenotype prior
- **writes:** —
- **reads:** pyro param store q_p_ct_raw
- **calls:** module.get_p_ct
- **upstream:** train
- **invariants:** rows sum to 1; NaN→uniform guard
- **edge cases:** —
- **provenance:** keep as-is

### `ml.TCRIModel.boost_phenotype_prior`  — _model_  ·  keep — advanced

```python
ml.TCRIModel.boost_phenotype_prior(phenotype_name, boost_factor=5.0, *, affect_mixture=True)
```
- **returns:** None (mutates clone_phen_prior / mixture in place)
- **writes:** module.clone_phen_prior, module.mixture_concentration
- **reads:** c2p_mat, centers
- **calls:** —
- **upstream:** (before) train
- **invariants:** rows renormalized to 1 after boost
- **edge cases:** unknown phenotype → ValueError
- **provenance:** keep; drop its inline _ok (use _console)
- **note:** niche manual-prior knob; kept but flagged advanced

### `pp.register_model`  — _register_  ·  keep

```python
pp.register_model(adata, model, *, latent_slot='X_tcri', batch_size=256, store_logits=True, store_logposterior=True, compute_umap=False, clonotype_key='trb_unique', ...)
```
- **returns:** adata (all tcri_* state written)
- **writes:** uns[tcri_metadata, tcri_p_ct, tcri_ct_to_cov, tcri_ct_to_c, tcri_local_scale, tcri_{covariate,clonotype,phenotype}_categories, tcri_{ct,cov}_array_for_cells]; obsm[X_tcri, X_tcri_logits, X_tcri_logposterior, X_tcri_probabilities]; obs[tcri_phenotype]
- **reads:** model.module.*, adata_manager.registry
- **calls:** model.get_latent_representation, model.get_cell_phenotype_probs, model.get_p_ct
- **upstream:** model.train
- **invariants:** per-cell arrays length == n_obs (guarded downstream)
- **edge cases:** compute_umap optional; prob slot only written if absent
- **provenance:** keep; folds in register_phenotype_key + register_clonotype_key (metadata only)
- **note:** THE bridge: model outputs → canonical adata state

### `pp.joint_distribution`  — _engine_  ·  merge — unified engine

```python
pp.joint_distribution(adata, covariate, *, posterior=True, n_samples=0, temperature=1.0, clones=None, weighted=False, combine_with_logits=True, seed=None, silent=True)
```
- **returns:** DataFrame[clone × phenotype] (rows sum to 1 unless weighted); n_samples>0 → stacked draws
- **writes:** — (pure)
- **reads:** uns[tcri_p_ct, tcri_local_scale, tcri_*_categories, tcri_metadata, tcri_{ct,cov}_array_for_cells, tcri_ct_to_*]; obsm[X_tcri_logits]
- **calls:** —
- **upstream:** register_model
- **invariants:** posterior=True draws Dirichlet(local_scale·p_ct)+logit combine; posterior=False = point-estimate prior; FAILS LOUDLY on filtered-view length mismatch
- **edge cases:** clones filter + reindex; weighted → mass-weighted, no renorm
- **provenance:** MERGES joint_distribution_posterior (posterior=True) + joint_distribution (posterior=False)
- **note:** the single computational core under every metric

### `pp.group_singletons`  — _bookkeeping_  ·  keep

```python
pp.group_singletons(adata, *, clonotype_key='trb', groupby='patient', target_col='trb_unique', min_clone_size=10)
```
- **returns:** None (writes obs)
- **writes:** obs[target_col], obs[trb_candidate]
- **reads:** obs[clonotype_key], obs[groupby]
- **calls:** —
- **upstream:** —
- **invariants:** clones < min_clone_size → 'Singleton_<group>'
- **edge cases:** —
- **provenance:** keep; subsumes group_small_clones (hardcoded dup)
- **note:** canonical small-clone collapse

### `pp.clone_size`  — _bookkeeping_  ·  keep

```python
pp.clone_size(adata, *, key_added='clone_size', return_counts=False)
```
- **returns:** None | dict{clone: size}
- **writes:** obs[clone_size]
- **reads:** uns[tcri_metadata][clone_col] (was tcri_clone_key)
- **calls:** —
- **upstream:** register_model
- **invariants:** size == cells per clone
- **edge cases:** —
- **provenance:** keep; retarget onto tcri_metadata (drop tcri_clone_key)

### `pp.filter_genes`  — _bookkeeping_  ·  rename + fix

```python
pp.filter_genes(adata, *, mt=True, rp=True, tcr=True, hsp=True, mtrn=True, ribo=True)
```
- **returns:** adata (subset copy)
- **writes:** — (returns new view)
- **reads:** var_names
- **calls:** —
- **upstream:** —
- **invariants:** each flag composes (AND), never resets the running mask
- **edge cases:** HLA-* kept despite '-'/'.'
- **provenance:** rename of remove_meaningless_genes + FIX flag-reset bug
- **note:** TCR-gene removal pre-embedding is methodologically in-scope; generic flags fixed

### `tl.clonotypic_entropy`  — _per-covariate_  ·  keep (+ single-phenotype mode)

```python
tl.clonotypic_entropy(adata, covariate, *, n_samples=0, temperature=1.0, clones=None, combine_with_logits=True, normalised=True, phenotype=None)
```
- **returns:** Series[phenotype] (point) | ndarray[n_samples, P] (draws) | float (phenotype=)
- **writes:** —
- **reads:** uns[tcri_phenotype_categories]; (joint_distribution)
- **calls:** pp.joint_distribution
- **upstream:** register_model
- **invariants:** H normalised by log2(n_clones); n_samples<1 → ValueError
- **edge cases:** empty joint → NaN row
- **plot mirror:** pl.clonotypic_entropy
- **provenance:** keep; absorbs clonotypic_entropy_base via phenotype= arg
- **note:** H[P(c|phi,m)] — phenotype spread across clones

### `tl.phenotypic_entropy`  — _per-covariate_  ·  keep

```python
tl.phenotypic_entropy(adata, covariate, *, n_samples=0, temperature=1.0, clones=None, combine_with_logits=True, normalised=True)
```
- **returns:** Series[clone] (point) | ndarray[n_samples, n_clones] (draws)
- **writes:** —
- **reads:** uns[tcri_metadata]; (joint_distribution)
- **calls:** pp.joint_distribution
- **upstream:** register_model
- **invariants:** H normalised by log2(P); n_samples<1 → ValueError
- **edge cases:** no clones at covariate → empty
- **plot mirror:** pl.phenotypic_entropy
- **provenance:** keep as-is
- **note:** H[P(phi|c,m)] — clone phenotypic plasticity

### `tl.mutual_information`  — _per-covariate_  ·  keep

```python
tl.mutual_information(adata, covariate, *, n_samples=0, temperature=1.0, clones=None, normalised=True, normalise_mode='average', posterior=True, combine_with_logits=True, verbose=True)
```
- **returns:** float (point) | ndarray[n_samples] (draws)
- **writes:** —
- **reads:** (joint_distribution)
- **calls:** pp.joint_distribution, _joint_to_mi
- **upstream:** register_model
- **invariants:** I = ΣΣ p·log2(p/(px·py)); normalised by mean/min marginal H
- **edge cases:** posterior=False path implemented (no NotImplementedError)
- **plot mirror:** pl.mutual_information
- **provenance:** keep; wire the prior path that currently raises
- **note:** clone↔phenotype coupling

### `tl.clonality`  — _per-covariate_  ·  keep

```python
tl.clonality(adata)
```
- **returns:** dict{phenotype: clonality∈[0,1]}
- **writes:** —
- **reads:** obs[tcri_phenotype], uns[tcri_metadata]
- **calls:** —
- **upstream:** register_model
- **invariants:** 1 - H(clone sizes)/log2(K); hard labels (no posterior)
- **edge cases:** single clone → 1; nan→0
- **plot mirror:** pl.clonality
- **provenance:** keep; retarget onto tcri_metadata

### `tl.flux`  — _between-covariate_  ·  keep

```python
tl.flux(adata, *, cov_from, cov_to, clones=None, distance_metric='l1', n_samples=0, temperature=1.0, weighted=False, posterior=True, combine_with_logits=True, seed=42)
```
- **returns:** Series[clone] (point) | ndarray[n_samples, n_clones] (draws)
- **writes:** —
- **reads:** (joint_distribution ×2)
- **calls:** pp.joint_distribution, _stats.distance
- **upstream:** register_model
- **invariants:** dist over common clones at both covariates
- **edge cases:** no overlap → ValueError; metric ∈ {l1, dkl, callable}
- **plot mirror:** pl.flux
- **provenance:** keep; from_this/to_that → cov_from/cov_to; dkl via _stats registry
- **note:** phenotype-distribution shift per clone

### `tl.delta_clonotypic_entropy`  — _between-covariate_  ·  keep

```python
tl.delta_clonotypic_entropy(adata, phenotype, *, cov_pre, cov_post, n_samples=1000, temperature=1.0, clones=None, weighted=False, normalised=True, posterior=True, combine_with_logits=True, seed=None)
```
- **returns:** ndarray[n_samples] — H_post − H_pre
- **writes:** —
- **reads:** (clonotypic_entropy)
- **calls:** tl.clonotypic_entropy
- **upstream:** register_model
- **invariants:** positive ⇒ entropy rose pre→post
- **edge cases:** —
- **plot mirror:** pl.ridge_delta_entropy (via delta_entropy_table)
- **provenance:** keep; calls clonotypic_entropy(phenotype=)

### `tl.mi_compare`  — _tables_  ·  keep

```python
tl.mi_compare(adata, groupby, *, groups=None, treatment=None, n_samples=50, patient_col=None, clone_col=None, covariate_col=None, verbose=True)
```
- **returns:** dict{samples, summary, pairs, params}
- **writes:** —
- **reads:** uns[tcri_metadata]; obs[groupby, patient_col]
- **calls:** tl.mutual_information (per patient×covariate)
- **upstream:** register_model
- **invariants:** patient-level samples → group summary; pairs from groups
- **edge cases:** missing group/cov skipped
- **plot mirror:** pl.mi_compare
- **provenance:** keep; uses shared _group_table loop
- **note:** patient-level MI comparison

### `tl.delta_entropy_table`  — _tables_  ·  keep

```python
tl.delta_entropy_table(adata, *, cov_pre, cov_post, splitby='response', n_samples=1000, temperature=1.0, weighted=False, normalised=True, posterior=True, combine_with_logits=True, seed=42)
```
- **returns:** DataFrame[phenotype × splitby] — delta_samples + summary stats
- **writes:** —
- **reads:** uns[tcri_metadata]; obs[splitby]
- **calls:** tl.delta_clonotypic_entropy
- **upstream:** register_model
- **invariants:** keeps full delta vector per row
- **edge cases:** —
- **plot mirror:** pl.ridge_delta_entropy
- **provenance:** keep; uses shared _group_table loop

### `tl.flux_table`  — _tables_  ·  keep

```python
tl.flux_table(adata, *, cov_pre, cov_post, splitby='response', n_samples=0, temperature=1.0, weighted=False, posterior=True, combine_with_logits=True, distance_metric='l1', seed=42)
```
- **returns:** DataFrame[clone × splitby] — flux_samples, flux_mean/sd, clone_size
- **writes:** —
- **reads:** uns[tcri_metadata]; obs[splitby]
- **calls:** tl.flux
- **upstream:** register_model
- **invariants:** per-group clone scoping
- **edge cases:** —
- **provenance:** keep; uses shared _group_table loop

### `pl.mutual_information`  — _metric_  ·  keep

```python
pl.mutual_information(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via tl.mutual_information)
- **calls:** tl.mutual_information
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep
- **note:** box/strip across covariate ±splitby

### `pl.clonotypic_entropy`  — _metric_  ·  rename

```python
pl.clonotypic_entropy(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via tl.clonotypic_entropy)
- **calls:** tl.clonotypic_entropy
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** rename of clonotypic_entropy_by_phenotype
- **note:** per-phenotype box/dot ±covariate

### `pl.phenotypic_entropy`  — _metric_  ·  fix

```python
pl.phenotypic_entropy(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via tl.phenotypic_entropy)
- **calls:** tl.phenotypic_entropy
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** FIX broken tl call signature
- **note:** box/strip per covariate

### `pl.clonality`  — _metric_  ·  keep

```python
pl.clonality(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via tl.clonality)
- **calls:** tl.clonality, pl._metric_boxplot
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep
- **note:** stripplot per phenotype ±group

### `pl.flux`  — _metric_  ·  fix

```python
pl.flux(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via tl.flux)
- **calls:** tl.flux
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** FIX broken key= passed to tl.flux
- **note:** box of flux distance by group

### `pl.mi_compare`  — _compare_  ·  keep

```python
pl.mi_compare(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** uns/(via tl.mi_compare)
- **calls:** tl.mi_compare, _stats.auc_and_label_permutation, _stats.bootstrap_auc
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep
- **note:** patient MI box + AUROC/permutation stats

### `pl.bayesian_mutual_information`  — _compare_  ·  keep

```python
pl.bayesian_mutual_information(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via tl.mutual_information ×2)
- **calls:** tl.mutual_information
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep
- **note:** ΔMI KDE / posterior / bar across two covariates

### `pl.ridge_delta_entropy`  — _compare_  ·  fix

```python
pl.ridge_delta_entropy(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** DataFrame from tl.delta_entropy_table
- **calls:** —
- **upstream:** tl.delta_entropy_table
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** FIX undefined cm/st imports
- **note:** ridge plot of Δ-entropy posteriors

### `pl.phenotypic_flux`  — _distribution_  ·  keep

```python
pl.phenotypic_flux(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via pp.joint_distribution)
- **calls:** pp.joint_distribution, _build_sankey
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep; absorbs plot_pheno_sankey as private _build_sankey
- **note:** phenotype-flow sankey across covariates

### `pl.polar_plot`  — _distribution_  ·  fix

```python
pl.polar_plot(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** (via pp.joint_distribution / tl.clonotypic_entropy)
- **calls:** pp.joint_distribution, tl.clonotypic_entropy
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** FIX undefined clonotypic_entropy ref + string phenotypes
- **note:** radar of phenotype distribution or entropy

### `pl.clone_size_umap`  — _umap_  ·  keep

```python
pl.clone_size_umap(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** obs[clone_size], obsm[X_umap]
- **calls:** pp.clone_size
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep
- **note:** UMAP colored by log clone size

### `pl.top_clone_umap`  — _umap_  ·  keep

```python
pl.top_clone_umap(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** obs[clone_col], obsm[X_umap]
- **calls:** —
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** keep
- **note:** UMAP highlighting top-N clones

### `pl.phenotype_probabilities_umap`  — _umap_  ·  rename

```python
pl.phenotype_probabilities_umap(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** obsm[X_tcri_probabilities]
- **calls:** —
- **upstream:** matching tl tool
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** rename of plot_phenotype_probabilities
- **note:** per-phenotype probability UMAP grid

### `pl.model_loss`  — _diagnostic_  ·  move

```python
pl.model_loss(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** model.history_
- **calls:** —
- **upstream:** model.train
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** moved from TCRIModel.plot_loss
- **note:** ELBO + dKL training curves

### `pl.archetypes`  — _diagnostic_  ·  move

```python
pl.archetypes(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** model.c2p_mat, model.centers
- **calls:** —
- **upstream:** model.train
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** moved from TCRIModel.plot_archetypes
- **note:** archetype / clone-prior heatmaps

### `pl.model_pgm`  — _diagnostic_  ·  move+merge

```python
pl.model_pgm(adata, *, ..., palette=None, figsize=..., ax=None, save=None)
```
- **returns:** Axes | Figure | None
- **writes:** —
- **reads:** — (static daft diagram)
- **calls:** —
- **upstream:** —
- **invariants:** reads computed state / calls tl; never owns model math
- **edge cases:** —
- **provenance:** moved from ut.build_nested_tcri_pgm + ut.draw_tcri_pgm_nested (merged)
- **note:** TCRI plate-diagram (PGM)

### `ut.save_tcri_session`  — _session_  ·  keep

```python
ut.save_tcri_session(model, adata, out_dir, *, save_adata=True, compression='gzip')
```
- **returns:** dict{paths}
- **writes:** run_dir/{model, pyro_params.pt, setup.json, adata.h5ad, meta.json}
- **reads:** model.save, pyro store, adata
- **calls:** write_adata_safely, _collect_setup_from_adata_or_model
- **upstream:** train
- **invariants:** adata written without tcri_manager (non-picklable)
- **edge cases:** —
- **provenance:** keep as-is

### `ut.load_tcri_session`  — _session_  ·  keep

```python
ut.load_tcri_session(run_dir, *, adata_path=None, map_location=None, layer=None)
```
- **returns:** (model, adata)
- **writes:** pyro param store (restored)
- **reads:** run_dir artifacts
- **calls:** TCRIModel.setup_anndata/.load, _pyro_load, _restore_category_order, _ensure_pyro_posterior_params, _disable_scvi_onload_train
- **upstream:** save_tcri_session
- **invariants:** category order restored; posterior params ensured
- **edge cases:** missing pyro store → warn + uniform-prior fallback
- **provenance:** keep as-is

### `ut.write_adata_safely`  — _session_  ·  keep

```python
ut.write_adata_safely(adata, path, *, compression='gzip')
```
- **returns:** None
- **writes:** path (h5ad without tcri_manager)
- **reads:** adata
- **calls:** _pop_nonserializables
- **upstream:** —
- **invariants:** strips non-serializable manager before write
- **edge cases:** —
- **provenance:** keep as-is

## Shared primitives

**Console (tcri/_console.py) — replaces 3 duplicated copies**

- `_ok / _info / _warn / _fin` — every verbose tl/pp/pl/ml function
- `_ascii_hist(samples)` — mutual_information, flux, delta_clonotypic_entropy (graph=)
- `ANSI constants (RESET/BOLD/GRN/...)` — all of the above

**Stats (tcri/_stats.py)**

- `_joint_to_mi(pxy, normalised, mode)` — tl.mutual_information
- `distance(metric) -> f(p,q)  [l1 | dkl | callable]` — tl.flux, tl.flux_table
- `auc_and_label_permutation(scores, labels)` — pl.mi_compare
- `bootstrap_auc(scores, labels)` — pl.mi_compare
- `_norm_entropy(p, base, n)` — clonotypic_entropy, phenotypic_entropy

**Table builder (tcri/metrics/_tables.py)**

- `_group_table(adata, splitby, per_group_fn)` — mi_compare, delta_entropy_table, flux_table

**Plot helpers (tcri/plotting/_base.py)**

- `_metric_boxplot(adata, fn, ...)` — clonality (+ any group×split metric box)
- `_resolve_palette / tcri_colors` — every pl function
- `_build_sankey / SankeyNode / _phenotype_mass_per_clone` — phenotypic_flux

**Session internals (tcri/utils/_io.py)**

- `_ensure_dir, _to_jsonable, _pop_nonserializables` — save_tcri_session, write_adata_safely
- `_collect_setup_from_adata_or_model, _restore_category_order` — save/load_tcri_session
- `_pyro_load, _resolve_TCRIModel, _disable_scvi_onload_train, _ensure_pyro_posterior_params` — load_tcri_session

## Dropped / merged from the current code

| current | disposition | why |
|---|---|---|
| `pp.classify_phenotypes` | deleted | redundant cosine phenotype assignment; 0 callers |
| `pp.get_latent_embedding` | deleted | trivial gaussian sampler; 0 callers |
| `pp.register_probability_columns` | deleted | feeds only the dead probabilities() |
| `pp.gene_entropy` | deleted | generic gene QC, out of scope; 0 callers |
| `pp.group_small_clones` | subsumed → group_singletons | hardcoded inferior dup |
| `pp.register_phenotype_key / register_clonotype_key` | subsumed → register_model | kill the tcri_*_key shadow convention |
| `pp._compute_logits_and_prior` | subsumed → ml.get_cell_phenotype_probs | one prob path |
| `pp.joint_distribution_posterior` | subsumed → joint_distribution(posterior=True) | unified engine |
| `tl.clonotypic_entropy_base` | subsumed → clonotypic_entropy(phenotype=) | one entropy fn |
| `tl.clone_fraction` | deleted | one-line value-counts; 0 callers |
| `tl.dkl` | subsumed → _stats.distance | dead; flux had its own copy |
| `pl.compare_phenotypes` | deleted | trivial crosstab heatmap; 0 callers |
| `pl.compare_joint_distribution` | deleted | broken (undefined `model`) |
| `pl.probability_distribution` | deleted | broken (self-recursion); covered by polar_plot |
| `pl.set_color_palette` | deleted | buggy palette helper; 0 callers |
| `pl.plot_pheno_sankey` | subsumed → phenotypic_flux (_build_sankey) | one public sankey |
| `pl.tcri_boxplot` | internalized → _metric_boxplot | shared plot helper |
| `ut.probabilities` | deleted | reads uns[joint_distribution] nothing ever writes |
| `ut.stars` | deleted | imported but never called |
| `ut.build_nested_tcri_pgm / draw_tcri_pgm_nested` | moved+merged → pl.model_pgm | describes the model, not IO |
| `ut.auc_and_label_permutation / bootstrap_auc` | internalized → _stats | used only by pl.mi_compare |
| `SankeyNode.hex_to_rgb` | deleted | unused method |
| `_ok/_info/_warn/_fin (×3), _ascii_hist (×2)` | subsumed → _console | dedup |

_See `tcri_dependency_map.md` for the full call + producer/consumer graph._