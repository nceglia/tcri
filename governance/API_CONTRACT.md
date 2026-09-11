# API contract

The public surface and every signature on it. The stub at the end of this file is parsed by
`tests/test_contract_conformance.py`, which asserts two things: the set of public callables
the package exposes equals the set the stub declares, in both directions, and every live
signature matches its declaration (parameter names, kinds, and which carry defaults). Nothing
else binds the interface. Adding a public function is a contract change by construction: the
test fails until it is declared here.

Namespaces: `tcri.ml` (`TCRIModel`), `tcri.pp`, `tcri.tl`, `tcri.pl`, `tcri.diag`,
`tcri.perturb`, `tcri.ut`, `tcri.datasets`. `tl`, `pl`, `pp`, `diag`, `perturb`, `ut` and
`datasets` are containers for namespacing only; `TCRIModel` is a real class. Private helpers
(`_state`, `_stats`, `_compute`, and every underscore name) are outside the contract.

## The scope principle

A comparison belongs in the API when producing it requires applying a metric at a level the
surface does not already expose. When it is arithmetic on values the package has already
computed, it belongs to the user. So the two metrics with an item axis (clone or phenotype)
have paired `delta_*` forms, because a within-draw difference needs the engine; mutual
information has no item axis, so its difference is a subtraction the caller performs. Always
know what a metric reduces to.

## Semantics that are not in a signature

**Parameter namespaces.** Pyro's parameter store is process-global, so the names a model
registers are the only thing separating it from another model in the same session. `name`
namespaces them: a model called `x` owns exactly the store keys under `x.`, and `""` (the
default) owns the unnamed layout every session saved before 0.12 uses. Two named models can be
fitted in one process without overwriting each other. `TCRIModel.name` reads it back.

**The substrate.** `TCRIModel.to_anndata()` writes the learned state under the keys in
`tcri._state.keys`: the fitted clone × covariate distributions `p_ct` (the posterior mean, no
temperature baked in), the index maps between cells, groups, clones and covariates, the
per-cell latent, logits and probabilities, the argmax label, and `local_scale`, `gate_prob` and
`classifier_temperature`. Every `tl` metric reads that substrate and takes no model; `diag`
functions that need the live networks take a model as well, and every `perturb` function takes
the model first. Session I/O (`ut`) restores the
model, the AnnData and the Pyro parameter store together; the store is process-global, so one
fitted model per process is the supported pattern.

**`n_samples`.** `0` is the deterministic plug-in at the posterior mean: no draw, reproducible
on repeat calls. `N > 0` draws `N` joints from the fitted Dirichlet posterior of `p_ct`, with
the guide's clamped concentration, seeded by `random_state` through a torch generator. One draw
serves every covariate, group and clone in a call (draw-once), so a self-delta is exactly zero.
The plug-in and the posterior mean are different estimators; see `METRICS_CONTRACT.md`.

**`temperature`.** Power-tempers `p_ct` once, `softmax(log(p_ct + 1e-8) / T)`, identically in
the mean and draw paths. `T = 1` is the identity. With `use_logits=True` the combined per-cell
logit is divided by `T` once; at `T = 1` the engine reproduces `predict()` exactly.

**`use_logits`.** `True` folds per-cell classifier logits into the group distribution with the
gate, exactly as `predict()` does, and aggregates per clone; `False` returns the clone ×
covariate table directly. Both use the posterior; neither touches the generative prior.

**`weighted`.** `False` (default) gives every clone equal mass; `True` weights by cell count.
Different estimands, both valid; the docstrings say which question each answers.

**`groupby` and `splitby`.** `groupby` restricts by cell and clone masks over the full object
(never by slicing the AnnData) and adds a group axis; clones are assumed disjoint across
groups. `splitby` produces the contrast between split levels in the `stats` slot: a two-sided
Mann–Whitney U on per-group values, uncorrected across contrasts, with the replicate unit
being the group, so items are averaged to one value per group before the test.

**Return shape.** Every `tl` returns `{table, result, stats}` and stores the same object under
`uns[key_added or "tcri_<metric>"]` with a `params` block recording every argument, defaults
included. `table` has one row per (covariate, group, item, draw) and is never reduced;
`result` reduces over draw only and carries `mean`, `sd`, `hdi_low`, `hdi_high` (a true
highest-density interval, NaN at `n_samples ≤ 1`); `stats` is `None` without `splitby` and
otherwise carries one row per (contrast, `quantity`). Every
frame survives an h5ad round trip: flat columns, no vectors in cells, per-draw values as rows
with a `draw` column. Read back with `tcri.get.result(adata, name)` and
`tcri.get.params(adata, name)`.

**References.** Every scored metric takes `null_model`, defaulting to `"auto"`: the reference is
computed and `result` gains `null_value` and `excess = value - null_value` beside every native
value column (`null_denom` has no excess; a denominator is not a reported quantity). `table`
carries them too, broadcast to every draw. `null_model=None` computes no reference and creates
no column. The reference is the caller's own call with two arguments changed — `fit` set to the
null and `null_model=None` — every other argument forwarded verbatim, so both sides are the same
functional at the same arguments. `joint_distribution` returns a matrix rather than a scored
quantity and takes no `null_model`.

**Substrate and fits.** One AnnData carries one main fit and any number of named ones, written
by `to_anndata(fit=...)`. Every per-fit key is prefixed through `fit_key`; the metadata and the
three category lists are shared. `fit=` on a substrate reader selects which fit the number is
computed ON; `fit=` on `tcri.get.*` selects which fit's result blob to read; a result computed on
a named fit is stored under the fit-suffixed key, so it never overwrites the main one. A bare
kind resolves to the fit that carries it.

**Plotting.** Each `pl` twin renders the result its `tl` twin stored; the covariate, groups,
splits and distance it draws are the ones `tl` used, read from `params`. `pl` functions take
no metric arguments. A mark shows one variance component: the coarsest unit that varies within
an x position (replicate over item over draw). Connecting lines are drawn only between points
sharing an identity across the compared levels.

**Diagnostics.** `reconstruction_ppc`, `loss` and `archetypes` take the model; the rest read the
stored substrate. `joint_distribution_ppc` and `phenotype_calibration` read PER-FIT keys and
therefore take `fit=`. `permutation_null` reads only the metadata and the category lists, which
are shared between every fit of an object, so it has no fit to select and takes none; it permutes
labels within each covariate on the empirical crosstab and draws no Dirichlet samples. It is a
model-free check on the data, not a reference for a model-based number — `tcri.null` is that.

**Perturbation.** `perturb.*` is a query on the fitted model with its parameters held fixed:
intervene on the expression matrix, read the phenotype call back through the per-cell rule
`predict()` uses (`use_gate=True`, the default) or through the head alone (`use_gate=False`).
The pass is deterministic at `n_samples=0`; no encoder sample is ever drawn. `knockout`
returns the `predict()` frame and stores nothing unless `key_added` names an `obsm` slot;
`gene_importance` returns and stores `{table, result, stats, shift}`, where `shift` is the
signed per-phenotype decomposition of each importance, and its `stats` contrast is per gene. It
takes `null_model` like every scored metric and no `fit`: it is a query on a model, so a
different fit is reached by handing it a different model, and `"auto"` rebuilds the phenotype
null from the parameter store and this object.
`adata` is required there because the result is stored into it. Definitions in
`METRICS_CONTRACT.md`. `pl.gene_importance` renders that cache: `kind="rank"` is the top
`n_top` genes under the mark rule, with each gene's own contrast starred above it when a
split was used; `kind="shift"` is the gene × phenotype `shift`, averaged over groups, on a
diverging scale centred on zero. Every twin takes `quantity`, selecting the value (with its
reference drawn behind it) or the excess (against a zero rule, with no interval);
`kind="shift"` accepts only the value, because the reference has no per-phenotype
decomposition stored.

## The stub

```python contract
from typing import Any, Optional
from anndata import AnnData
import pandas as pd


# ── model (ml) ───────────────────────────────────────────────────────────────
class TCRIModel:
    def __init__(
        self, adata: AnnData, n_latent: int = ..., n_hidden: int = ..., n_layers: int = ...,
        classifier_n_layers: int = ..., global_scale: float = ..., local_scale: float = ...,
        prior_temperature: float = ..., guide_temperature: float = ...,
        use_enumeration: bool = ..., patience_epochs: int = ..., classifier_hidden: int = ...,
        classifier_dropout: float = ..., n_pseudo_obs: int = ..., K: int = ...,
        gate_prob: Optional[float] = ..., kl_weight_max: float = ...,
        guide_init_scale: float = ..., classifier_temperature: float = ...,
        phenotype_kl_weight: float = ..., label_error_rate: Optional[float] = ...,
        seed: Optional[int] = ..., name: str = ..., permutation: Any = ..., **kwargs: Any,
    ) -> None: ...
    @classmethod
    def setup_anndata(
        cls, adata: AnnData, *, layer: Optional[str] = ...,
        clonotype_key: str = ..., phenotype_key: str = ...,
        covariate_key: str = ..., batch_key: str = ...,
        replicate: Optional[str] = ..., **kwargs: Any,
    ) -> None: ...
    def train(
        self, max_epochs: int = ..., batch_size: int = ..., lr: float = ...,
        reconstruction_loss_scale: float = ..., n_steps_kl_warmup: int = ..., **kwargs: Any,
    ) -> None: ...
    def get_latent_representation(
        self, adata: Optional[AnnData] = ..., indices: Any = ...,
        batch_size: Optional[int] = ...,
    ) -> Any: ...
    def predict(
        self, adata: Optional[AnnData] = ..., *, batch_size: int = ..., eps: float = ...,
    ) -> pd.DataFrame: ...
    def get_p_ct(self) -> Any: ...
    def to_anndata(
        self, adata: Optional[AnnData] = ..., *, batch_size: int = ..., compute_umap: bool = ...,
        fit: Optional[str] = ...,
    ) -> AnnData: ...


# ── preprocessing (pp) ───────────────────────────────────────────────────────
class pp:
    def group_singletons(
        adata: AnnData, *, clonotype_key: str = ..., groupby: str = ...,
        target_col: str = ..., min_clone_size: int = ...,
    ) -> None: ...
    def clone_size(adata: AnnData, *, key_added: str = ..., return_counts: bool = ...) -> Any: ...


# ── tools / metrics (tl) ─────────────────────────────────────────────────────
class tl:
    def joint_distribution(
        adata: AnnData, *, covariate: Optional[str] = ...,
        n_samples: int = ..., use_logits: bool = ..., weighted: bool = ...,
        clones: Any = ..., temperature: float = ..., random_state: Any = ...,
        device: Any = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def clonotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        n_clones_ref: Any = ..., random_state: Any = ..., device: Any = ...,
        null_model: Optional[str] = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def phenotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ..., random_state: Any = ...,
        device: Any = ..., null_model: Optional[str] = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def mutual_information(
        adata: AnnData, *, covariate: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        normalize_mode: str = ..., random_state: Any = ..., device: Any = ...,
        null_model: Optional[str] = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def phenotypic_flux(
        adata: AnnData, *, cov_from: str, cov_to: str, groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., distance_metric: str = ..., random_state: Any = ...,
        device: Any = ..., null_model: Optional[str] = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def delta_clonotypic_entropy(
        adata: AnnData, *, cov_from: str, cov_to: str, groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ...,
        n_clones_ref: Any = ..., random_state: Any = ..., device: Any = ...,
        null_model: Optional[str] = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...
    def delta_phenotypic_entropy(
        adata: AnnData, *, cov_from: str, cov_to: str, groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., weighted: bool = ..., normalized: bool = ..., random_state: Any = ...,
        device: Any = ..., null_model: Optional[str] = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...


# ── plotting (pl): twins render what tl stored ───────────────────────────────
class pl:
    def clonotypic_entropy(
        adata: AnnData, *, quantity: str = ..., key: Optional[str] = ..., order: Any = ...,
        hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_entropy(
        adata: AnnData, *, quantity: str = ..., key: Optional[str] = ..., order: Any = ...,
        hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def mutual_information(
        adata: AnnData, *, quantity: str = ..., key: Optional[str] = ..., order: Any = ...,
        hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_flux(
        adata: AnnData, *, quantity: str = ..., key: Optional[str] = ..., order: Any = ...,
        hue_order: Any = ...,
        palette: Any = ..., ax: Any = ..., figsize: Any = ..., save: Any = ...,
        show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def delta_clonotypic_entropy(
        adata: AnnData, *, kind: str = ..., quantity: str = ..., key: Optional[str] = ...,
        order: Any = ...,
        hue_order: Any = ..., palette: Any = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def delta_phenotypic_entropy(
        adata: AnnData, *, kind: str = ..., quantity: str = ..., key: Optional[str] = ...,
        order: Any = ...,
        hue_order: Any = ..., palette: Any = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def gene_importance(
        adata: AnnData, *, kind: str = ..., quantity: str = ..., n_top: int = ...,
        key: Optional[str] = ...,
        order: Any = ..., hue_order: Any = ..., palette: Any = ..., ax: Any = ...,
        figsize: Any = ..., save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def resolve_colors(
        adata: AnnData, cat_key: str, categories: Any = ..., *, palette: Any = ...,
        persist: bool = ...,
    ) -> dict: ...


# ── diagnostics (diag) ───────────────────────────────────────────────────────
class diag:
    def joint_distribution_ppc(
        adata: AnnData, *, covariate: Optional[str] = ..., distance_metric: str = ...,
        temperature: float = ..., clones: Any = ..., random_state: Any = ...,
        fit: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def phenotype_calibration(
        adata: AnnData, *, n_bins: int = ..., fit: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def reconstruction_ppc(
        model: Any, adata: Optional[AnnData] = ..., *, n_sims: int = ..., random_state: Any = ...,
    ) -> pd.DataFrame: ...
    def permutation_null(
        adata: AnnData, *, metric: str = ..., covariate: Optional[str] = ...,
        groupby: Optional[str] = ..., normalize_mode: str = ..., n_perm: int = ...,
        random_state: Any = ...,
    ) -> pd.DataFrame: ...
    def loss(model: Any, *, log_scale: bool = ..., ax: Any = ..., save: Any = ...) -> Any: ...
    def archetypes(model: Any, *, ax: Any = ..., save: Any = ...) -> Any: ...


# ── perturbation (perturb): queries on the fitted model, parameters fixed ────
class perturb:
    def knockout(
        model: Any, adata: Optional[AnnData] = ..., *, genes: Any, use_gate: bool = ...,
        batch_size: int = ..., key_added: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def gene_importance(
        model: Any, adata: AnnData, *, genes: Any = ..., covariate: Optional[str] = ...,
        groupby: Optional[str] = ..., splitby: Optional[str] = ..., n_samples: int = ...,
        use_gate: bool = ..., batch_size: int = ..., random_state: Any = ...,
        null_model: Any = ..., key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> dict: ...


# ── permutation references (null) ────────────────────────────────────────────
class null:
    def phenotype(
        model: Any, adata: AnnData, *, within: Any = ..., seed: Optional[int] = ...,
        key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> Any: ...
    def clonotype(
        model: Any, adata: AnnData, *, within: Any = ..., seed: Optional[int] = ...,
        key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> Any: ...
    def condition(
        model: Any, adata: AnnData, *, seed: Optional[int] = ...,
        key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> Any: ...
    def all(
        model: Any, adata: AnnData, *, kinds: Any = ..., within: Any = ...,
        seed: Optional[int] = ..., key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> dict: ...


# ── utils / session (ut) ─────────────────────────────────────────────────────
class ut:
    def save_tcri_session(
        model: Any, adata: AnnData, out_dir: str, *, save_adata: bool = ..., compression: str = ...,
    ) -> dict: ...
    def load_tcri_session(
        run_dir: str, *, adata_path: Optional[str] = ..., map_location: Any = ..., layer: Optional[str] = ...,
    ) -> Any: ...
    def auc_and_label_permutation(
        scores: Any, labels: Any, pos_label: Any = ..., n_perm: int = ..., seed: int = ...,
        max_exact: int = ...,
    ) -> tuple: ...
    def bootstrap_auc(
        scores: Any, labels: Any, pos_label: Any = ..., n_boot: int = ..., seed: int = ...,
    ) -> tuple: ...


# ── synthetic data (datasets) ────────────────────────────────────────────────
class datasets:
    def simulate_cohort(
        *, n_patients: int = ..., conditions: Any = ..., disease_fraction: float = ...,
        n_clones: Any = ..., n_phenotypes: int = ..., n_genes: int = ...,
        n_cells_per_sample: int = ..., clone_size_distribution: str = ...,
        clone_size_exponent: float = ..., disease_enrichment: float = ...,
        control_enrichment: float = ..., omega_concentration: float = ...,
        seed: int = ...,
    ) -> AnnData: ...
    def simulate_tcri(
        *, n_clones: int = ..., n_phenotypes: int = ..., n_genes: int = ...,
        n_cells: int = ..., n_covariates: int = ..., n_factors: int = ...,
        omega_concentration: float = ..., pi_concentration: float = ...,
        fuzziness: float = ..., label_error_rate: float = ..., seed: int = ...,
    ) -> AnnData: ...
    def simulate_from_fit_params(
        params: Any, *, n_cells: int = ..., temperature: float = ...,
        fuzziness: float = ..., label_error_rate: float = ..., seed: int = ...,
    ) -> AnnData: ...
    def temperature_scale(P: Any, T: Any, eps: float = ...) -> Any: ...
    def mi_from_joint_oracle(joint: Any) -> dict: ...
```
