# Redo list — clean re-implementation after the notebook-census leak

Ordered work to redo, from the leak point onward. Each item is re-implemented **clean** (Hard Rules in
`REFACTOR_HISTORY.md`: no notebooks, no `examples/`, disposition = "is it core"). Check each against
`REFACTOR_HISTORY.md` §2 as it lands.

## What we did since the leak point (to re-implement)
1. **Corrected disposition map** (below) — the foundation the two docs got wrong. **Done in this file.**
2. **API + Function-Responsibility doc** — rebuild clean from the quarantined version: keep its (clean)
   math/stats spec, prior-vs-mean resolution, per-arg math, `covariate=None`=all semantics; **replace §11 and every
   disposition/`examples`/notebook reference** with the map below.
3. **Implementation Plan doc** — rebuild clean: keep the (clean) PR sequence, model→anndata streamline, GPU
   architecture, testing/CI; **delete Phase 0's "caller census," every "move to examples," and every "N notebook
   uses" justification**; notebook rewrite is a downstream chore, never a disposition driver.
4. **Fold the (clean) consistency findings** into the API doc: `random_state` naming; canonical arg order;
   `distance_metric` (not `metric`); `palette` (not `phenotype_colors`); `normalized` (not `normalize`);
   `order=`/`hue_order=` parity; `figsize` unify; keyword-only after `*`; `show=` on public `pl`; American spelling.
5. **Re-derive sufficiency from RESPONSIBILITY only** — a kept function gets an arg iff its own job needs it
   (`random_state` on sampling fns; `clones=` on per-clone metrics; `order=` for plot category ordering).
   **Drop every notebook-parity item** (`gene_entropy`, `probability_ternary`, `weighted=`, `minimum_clone_size=`,
   `base=`, `covariate_key=`, `pair_on=`-for-notebooks, etc.) unless responsibility independently requires it.
6. **Necessary/sufficient audit** of the rebuilt docs (consistency matrix over the surface; sufficiency vs responsibility).
7. **Final audit** of everything against `REFACTOR_HISTORY.md`.

---

## CORRECTED DISPOSITION MAP (authoritative; supersedes quarantined §11 / §3)

Decided by "is it core," never by callers. `examples/` does not appear. Allowed non-drop destinations: a kept
namespace, `diag/`, `docs/` (PGM only), or a private helper module.

### KEEP — core surface
- **ml:** `TCRIModel` = `setup_anndata`, `train`, `get_latent_representation`, `predict` (was `get_cell_phenotype_probs`),
  `get_p_ct`; private internals `TCRIModule` / priors / classifier / training-plan / `build_archetypes`.
- **pp:** `register_model`→`model.to_anndata` (streamlined), `group_singletons`, `clone_size`.
- **tl:** `joint_distribution` (+ top-level `tcri.joint_distribution`), `clonotypic_entropy`, `phenotypic_entropy`,
  `mutual_information`, `phenotypic_flux`, `compare_groups`.
- **pl:** `clonotypic_entropy`, `phenotypic_entropy`, `mutual_information`, `phenotypic_flux`; private
  `_sankey` (`SankeyNode` + `_phenotype_mass_per_clone`), `_metric_boxplot` (was `tcri_boxplot`), `_colors`/`resolve_palette`.
- **diag:** `joint_distribution_ppc` (fixed `compare_joint_distribution`), `phenotype_calibration`,
  `reconstruction_ppc`, `permutation_null`, `loss` (was `plot_loss`), `archetypes` (was `plot_archetypes`).
- **ut:** `save_tcri_session`, `load_tcri_session`.
- **shared:** `_keys`, `_console`, `_stats`, `_distance`.

### TRANSFORM (kept, relocated — NOT examples)
- `compare_joint_distribution` → `diag.joint_distribution_ppc` (fixed). `plot_loss`→`diag.loss`;
  `plot_archetypes`→`diag.archetypes`. `build_nested_tcri_pgm`/`draw_tcri_pgm_nested` → **`docs/`** figure script.
- `register_phenotype_key`/`register_clonotype_key`/`_compute_logits_and_prior` → folded into `to_anndata` (private).
- `dkl` → `_distance.kl_divergence`. `tcri_boxplot` → private `_metric_boxplot`.

### DROP — deleted (NOT moved to examples)
`clonality`, `probability_distribution`, `bayesian_mutual_information`, `polar_plot`, `compare_phenotypes`,
**`probability_ternary`** (was wrongly kept), **`gene_entropy`** (was wrongly →examples), **`top_clone_umap`**,
**`clone_size_umap`**, **`plot_phenotype_probabilities`** (all three were wrongly →examples), `mi_compare`,
`delta_clonotypic_entropy`, `delta_entropy_table`, `flux_table`, `clonotypic_entropy_base`, `ridge_delta_entropy`,
`classify_phenotypes`, `get_latent_embedding`, `group_small_clones`, `register_probability_columns`,
`remove_meaningless_genes`, `clone_fraction`, `probabilities`, `_ent`, `_ascii_hist` (+ all `graph=`/ASCII paths),
`SankeyNode.hex_to_rgb`, `write_adata_safely`, `_pop_nonserializables`, the plural `*_entropies` shims, the `*_tl`/
`centropy`/`pentropy` aliases.

> The 6 items in **bold** are the ones the leak wrongly kept-or-moved-to-examples. They are DROPPED.
