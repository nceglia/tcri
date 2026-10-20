# API contract

The public surface and every signature on it. The stub at the end of this file is parsed by
`tests/test_contract_conformance.py`, which asserts three things: the set of public callables
the package exposes equals the set the stub declares, in both directions; every live
signature matches its declaration (parameter names, kinds, and which carry defaults); and every
declared parameter is read by the function that declares it. Nothing else binds the
interface. Adding a public function is a contract change by construction: the
test fails until it is declared here.

Namespaces: `tcri.ml` (`TCRIModel`), `tcri.pp`, `tcri.tl`, `tcri.pl`, `tcri.get`, `tcri.diag`,
`tcri.perturb`, `tcri.ut`, `tcri.datasets`. `tl`, `pl`, `get`, `pp`, `diag`, `perturb`, `ut` and
`datasets` are containers for namespacing only; `TCRIModel` is a real class. Private helpers
(`_state`, `_stats`, `_compute`, and every underscore name) are outside the contract.

## The scope principle

Every metric is computed at one covariate level or as a change between two (`cov_from`, `cov_to`),
per replicate and per item, and compared across splits with `splitby`: comparisons are one
mechanism on every metric, and none needs a function of its own. A change is taken within a
posterior draw, which needs the engine, so it is the metric's own call rather than a subtraction
of stored values. Always know what a metric reduces to.

## Semantics that are not in a signature

**Parameter namespaces.** Pyro's parameter store is process-global, so the names a model
registers are the only thing separating it from another model in the same session. `name`
namespaces them: a model called `x` owns exactly the store keys under `x.`, and its nulls the keys
under `x.null.`. A model constructed without `name=` gets a generated one (`tcri-` and 12 hex
digits), recorded with the model, so no two models share a key. `TCRIModel.name` reads it back.
Constructing a model under a name whose keys the store already holds raises; continuing a fit is
another `train()` on the same object. Each fit of a null gets its own namespace,
`x.null.<kind>[i].<token>`, recorded with the fit; refitting a reference of the same parent and
kind replaces every fit of the earlier one and removes their keys from the store.

**Loading.** `TCRIModel.load` and `load_tcri_session` restore the loaded model's guide
concentrations and its nulls' parameters from the saved store, take its networks from the saved
weights, and leave every other model's entries in the store as they are. Both restore the
arguments the model was trained with (`train_kwargs_`), which `tcri.null.*` replays, and the KL
ramp's position, so training a loaded model continues the ramp. A load that
finds the model's entries already in the store replaces them and warns. A load raises when the
file cannot supply a trained model's posteriors, and refuses a store that keeps them under bare
keys, without a model name.

**Registration.** Every cell `setup_anndata` registers must have a clonotype. NaN or None, an
empty or whitespace-only string, and the literal string `"nan"` in the `clonotype_key` column
are missing clonotypes: setup raises `ValueError` before registering anything, naming the
column and the number of cells, and the caller keeps only the cells that have one. The
constructor raises the same way for a cell whose registered clonotype, phenotype or covariate
is NaN or None, which covers `obs` edited after setup and every null, since a null is built on
its parent's registration without running setup again. No clone × covariate row is built for a
cell without a clonotype or a covariate level.

**Rare clones.** `pp.pool_rare_clones` runs on the source clonotype column, before setup. It
counts cells per (`groupby` group, clonotype), so a clonotype carried by two groups is two
clones. A clone is rare when it has fewer than `min_cells` cells (default 3) and all of them come
from one sample: `samples` names each cell's capture, and without it the rule is `min_cells`
alone. A clone seen in two or more samples is never rare. It writes `obs[key_added]` (default
`{clonotype_key}_pooled`), a categorical with sorted categories, in which each group's rare clones
are `pooled@{group}` and every other clone is `{clonotype}@{group}`, an id already ending in
`@{group}` unchanged; the source column is never modified. Each call records one step in
`uns["tcri_clonotype_derivations"]` with the columns it used, `min_cells`, the pool labels and the
numbers of clones and cells pooled; a later call with the same `key_added` replaces it. Missing
clonotypes, as defined above, and NaN or None in `groupby` or `samples` raise `ValueError`
naming the column and the number of cells; a group that keeps no clone warns.

**The substrate.** `TCRIModel.to_anndata()` writes the learned state under the keys in
`tcri._state.keys`: the fitted clone × covariate distributions `p_ct` (the posterior mean, no
temperature baked in), the index maps between cells, groups, clones and covariates, the
per-cell latent, logits and probabilities, the argmax label, and `local_scale`, `gate_prob`,
`classifier_temperature` and `label_error_rate` (NaN for a label-free fit). Every `tl` metric
reads that substrate and takes no model; `diag` functions that need the live networks take a
model as well, and every `perturb` function takes the model first. Session I/O (`ut`) restores
the model, the AnnData and the Pyro parameter store together; the store is process-global, so one
fitted model per process is the supported pattern.

**`n_samples`.** `0` is the deterministic plug-in at the posterior mean: no draw, reproducible
on repeat calls; `diag.joint_distribution_ppc` rejects it, since a single point estimate is not
a posterior predictive check. `N > 0` draws `N` joints from the fitted Dirichlet posterior of `p_ct`, with
the guide's clamped concentration, seeded by `random_state` through a torch generator. One draw
serves every covariate, group and clone in a call (draw-once), so a level against itself changes
by exactly zero.
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
being the group: the metrics test their replicate values, and gene importance each gene's
per-group values; the contrast refuses a frame holding more than one row per group and level.
`delta` is the second level's mean minus the first's. Every pair of split levels with a group has
its row, `comparison` `"split"`; an arm with no finite value has `n` 0 and NaN statistics. A
change call (`cov_from`, `cov_to`) also tests whether each replicate changed, in rows that come
first: `comparison` `"covariate"`, `level_a`/`level_b` the two covariate levels, a two-sided
Wilcoxon signed-rank test on the per-replicate change, one row per split level with `splitby` (the
split column holds the level) and one over every replicate without. Its arms are the change's
endpoints; the arms, `delta` and `n` are over the replicates the test used (both endpoints finite,
change not exactly 0), and fewer than two give NaN statistics. Flux's covariate row tests its
adjusted value against 0, its arms NaN; without a usable reference it has no test and `delta` is
the mean flux. With no replicate registered the covariate row is present with `n` of 1 or 0 and no
test. A contrast whose p is undefined has blank `stars`.

**Return shape.** Every `tl`, and `perturb.gene_importance`, stores its result under `uns[key_added
or "tcri_<metric>"]` with a `params` block recording every argument, defaults included, and returns
`None`; `inplace=False` stores nothing and returns the whole result. Nothing is read from a returned
object: `tcri.get` is the only reader of a stored result. A stateful `random_state` (a numpy or
torch generator) is turned into one integer seed before the call, and `params` records that
integer. Mutual information, the two entropies, flux and `perturb.gene_importance` store
`{replicate, item, stats}`. Each level holds `draws`, one row per (covariate, replicate[, item],
draw), never reduced, and `summary`, which reduces over draw only; a change labels its rows
`cov_from`/`cov_to` in place of `covariate` and carries `value_from`/`value_to` (mutual
information also `denom_from`/`denom_to`), and its summaries reduce the change and its endpoints
under one mask. The replicate level has one row per replicate, NaN when the replicate has no joint
at a level; the item level has one row per replicate and phenotype (clonotypic entropy) or clone
(phenotypic entropy; on a change and for flux, the persistent clones), and mutual information has
no item level. Gene importance labels every row with the gene and stores only the adjusted value
(**Perturbation**, below). `joint_distribution` stores `{summary, draws}`: the posterior-mean
clone × phenotype table and its draws, indexed by `[covariate,] clonotype[, draw]`, read with
`tcri.get.result` and `tcri.get.draws`. Every level's summary but gene importance's carries
`value`, `sd`, `hdi_low`, `hdi_high` (a true highest-density interval, NaN at `n_samples ≤ 1`);
`stats` is `None` when nothing is compared and otherwise carries one row per contrast, with
`quantity` naming the column contrasted. Every frame survives an h5ad round trip: flat columns, no vectors in
cells, per-draw values as rows with a `draw` column. Read back with `tcri.get.result(adata, name)`
(the whole result), `tcri.get.replicates`, `tcri.get.items`, `tcri.get.stats` and
`tcri.get.draws(adata, name, items=False)` (the frames of a result stored by level), and
`tcri.get.params(adata, name)`. A normalized replicate value whose normalizer is 0 raises
`ValueError` naming the metric, the replicate and the covariate level.

**References.** `tcri.null.phenotype`, `.clonotype`, `.covariate` and `.all` take `n_nulls`
(default 5): a reference is that many fits of the parent, `null.<kind>[0]` to
`null.<kind>[n_nulls-1]` (`null.<kind>.<key_added>[i]` when named), each with its own
permutation, namespace and records; the calls return the fitted models, a list per kind.
`tcri.null.all` therefore fits 3 x `n_nulls` nulls, and a metric's per-draw frames are
`n_nulls` times longer than with one. A null's permutation is drawn from the seed spawned by its
kind, `key_added` and index, so it does not depend on the fits already on the object. Every scored
metric takes `null_model`, defaulting to `"auto"`: the phenotype null at one covariate level, the
covariate null for flux and every change. The reference is computed and every summary gains
`null_value` and `adjusted = value - null_value` beside every native value column (`null_denom`
has no adjusted column; a denominator is not a reported quantity). Each level's `draws` carries
them per pair of draws, the fit's draw *d* paired with draw *d* of every fit of the reference,
`null` numbering those fits, and the summary summarizes all the pairs: `adjusted` is their mean,
with `adjusted_sd`, `adjusted_hdi_low` and `adjusted_hdi_high` beside it. At `n_samples=0` a
row has one pair per null: `adjusted_sd` is the sd between the nulls and the HDI is NaN. The adjusted value is the
quantity a metric reports: `stats` is computed on it, and `value` and `null_value` stay stored as
its inputs. `null_model` takes `"auto"` (every fit of the metric's default reference), a
reference's name (`"phenotype"`, `"null.phenotype"`: every fit), one fit's name
(`"null.phenotype[2]"`), one fitted model (that fit), or a list of fitted models (a reference of
those fits, such as the list `tcri.null.*` returned). `null_model=None` computes no reference and
creates no column, and `stats` is then on the value. The reference is the caller's own call on
each null — `fit` set to the null, `null_model=None`, and the draws seeded from their own seed
derived from `random_state` and the null's index — every other argument forwarded verbatim, so
both sides are the same functional at the same arguments. `fit=` selects one fit, so a
reference's name there raises, naming its fits. A covariate null can move a group only where one of
its counted clones has a (clonotype, batch) holding a level the call reads and another level; a
group it cannot move keeps its value, has NaN reference columns, is named in a warning and leaves
the tests, and building a covariate null that could move no cell raises before anything is fitted.
`joint_distribution` returns a matrix rather than a scored quantity and takes no `null_model`.

**Substrate and fits.** One AnnData carries one main fit and any number of named ones, written
by `to_anndata(fit=...)`. Every per-fit key is prefixed through `fit_key`; the metadata and the
three category lists are shared. `fit=` on a substrate reader selects which fit the number is
computed ON; `fit=` on `tcri.get.*` selects which fit's result blob to read; a result computed on
a named fit is stored under the fit-suffixed key, so it never overwrites the main one. A bare
kind resolves to the fit that carries it.

**Plotting.** Each `pl` twin renders the result its `tl` twin stored; the covariate, groups,
splits and distance it draws are the ones `tl` used, read from `params`. `pl` functions take
no metric arguments. A mark shows one variance component: the coarsest unit that varies within
an x position (replicate over item over draw). The quantity on the axis is `adjusted` when the
result carries one, against a zero rule, and in the metric twins `quantity="value"` draws the
value with its null behind it; the star a panel draws is always about the quantity on its axis.
Every twin takes `view=`, `"replicate"` (the default) drawing the replicate summary and `"item"`
the item summary. In the twins of mutual information, the two entropies and flux the replicate
view draws one mark per replicate, with the split on x and its test starred, and mutual
information has no items and raises for `"item"`; the views of `pl.gene_importance` are under
**Perturbation**. A change is drawn against a zero rule; flux, a distance that is never negative,
has its zero rule on the adjusted panel only. Connecting lines are drawn only between points
sharing an identity across the compared levels. Every twin takes
`legend=`, and `False` draws no legend, so a row of panels can carry one; a y label longer than
its axes is wrapped. In the metric twins the x axis is in the category order of the column on it
(`order=` overrides), so a level sits at the same position in every panel, and dots that are
replicates are colored by replicate unless a split is the hue; an item axis is never colored per
item.

**Diagnostics.** `reconstruction_ppc`, `loss` and `archetypes` take the model; the rest read the
stored substrate. `joint_distribution_ppc` and `phenotype_calibration` read PER-FIT keys and
therefore take `fit=`. `permutation_null` reads only the metadata, the category lists and the
derivation record, which are shared between every fit of an object, so it has no fit to select and
takes none; it permutes labels within each covariate on the empirical crosstab and draws no
Dirichlet samples. It is a model-free check on the data, not a reference for a model-based number
— `tcri.null` is that.

`joint_distribution_ppc` is a sampled-posterior predictive check of the labels. Each of
`n_samples` posterior draws of `p_ct` (the draws `tl.joint_distribution` makes for the same
`random_state`) rebuilds every cell's label likelihood, its classifier logits gated with its
group's draw and passed through the label-noise matrix of `label_error_rate`, and `n_sims`
simulated label sets are drawn cell by cell. A clone's `p` at a draw is the rank of its observed
distance among its replicate distances, ties broken at random, so under a correct fit each draw's
`p` is uniform across clones; `p` is never averaged over draws. Each fit is checked against its
own training data: a phenotype null against its permuted labels, a clonotype or covariate null
against its own groups. A label-free fit raises. It returns `draws` (one row per covariate, clone
and draw), `summary` (per clone, no `p`) and `calibration` (per covariate, doubling clone-size bin
and draw: the KS distance of `p` from uniform).

The four checks that compute (`joint_distribution_ppc`, `phenotype_calibration`,
`reconstruction_ppc`, `permutation_null`) return a dict of named DataFrames, one per level of
detail, and put no numbers in `attrs`: `phenotype_calibration` returns `bins` and a one-row
`summary` (`ece`, `n_cells`); `reconstruction_ppc` and `permutation_null` return `summary`. They
take `plot=True` and `save=None`. `plot` draws the check's figure from the values it just
computed, and the return value is the same whether or not it plots; `save` with `plot=False`
raises. Nothing is cached: a redraw reruns the check. The three with a panel count known before
the call also take `ax=None`, one Axes per panel; the PPC's panel count depends on the data, so it
always draws its own figure. `reconstruction_ppc` and `joint_distribution_ppc` both call their
count of simulated datasets `n_sims`.

**Perturbation.** `perturb.*` is a query on the fitted model with its parameters held fixed:
intervene on the expression matrix, read the phenotype call back through the per-cell rule
`predict()` uses (`use_gate=True`, the default) or through the head alone (`use_gate=False`).
The pass is deterministic at `n_samples=0`; no encoder sample is ever drawn. `knockout`
returns the `predict()` frame and stores nothing unless `key_added` names an `obsm` slot;
`gene_importance` stores `{replicate, item, stats}` and returns `None` (the whole result at
`inplace=False`). The gene is a label on every row: the replicate level is the importance, one
row per gene and group, and the item level its signed shift, one row per gene, group and
phenotype; a group with no cell at `covariate` has NaN rows. Only the adjusted value is stored:
each level's `draws` holds `adjusted` per pair (`null`, `draw`) and its `summary` holds
`adjusted`, `adjusted_sd` and the adjusted HDI; nothing is stored for the null runs. At
`n_samples > 0` draw `d` is paired with null `d mod n_nulls` and each null draws only its share;
at `n_samples = 0` the one value is paired with every null. `null_model=None` raises, and so does
`0 < n_samples < n_nulls`. Its `stats` contrast is per gene, on the adjusted importance. It
takes `null_model` like every scored metric and no `fit`: it is a query on a model, so a
different fit is reached by handing it a different model, and `"auto"` rebuilds the phenotype
null from the parameter store and this object.
`adata` is required there because the result is stored into it. Definitions in
`METRICS_CONTRACT.md`. `pl.gene_importance` renders that cache: `view="replicate"` is the top
`n_top` genes under the mark rule, with each gene's own contrast starred above it when a
split was used; `view="item"` is the gene × phenotype shift from the item summary, averaged over
groups, on a diverging scale centered on zero. It takes no `quantity`: both views draw the
adjusted value, the one stored, against zero, and both show the genes ranked by the adjusted
importance.

## The stub

```python contract
from typing import Any, Optional, Sequence
from anndata import AnnData
import numpy as np
import pandas as pd


# ── model (ml) ───────────────────────────────────────────────────────────────
class TCRIModel:
    def __init__(
        self, adata: AnnData, n_latent: int = ..., n_hidden: int = ..., n_layers: int = ...,
        classifier_n_layers: int = ..., global_scale: float = ..., local_scale: float = ...,
        prior_temperature: float = ..., guide_temperature: float = ...,
        classifier_hidden: int = ..., classifier_dropout: float = ...,
        n_pseudo_obs: int = ..., K: int = ...,
        gate_prob: Optional[float] = ..., max_kl_weight: float = ...,
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
    @classmethod
    def load(cls, dir_path: str, adata: Optional[AnnData] = ..., **kwargs: Any) -> Any: ...
    def train(
        self, max_epochs: int = ..., batch_size: int = ..., lr: float = ...,
        reconstruction_loss_scale: float = ..., n_epochs_kl_warmup: int = ...,
        num_particles: int = ..., early_stopping: bool = ..., early_stopping_patience: int = ...,
        early_stopping_min_delta: float = ..., **kwargs: Any,
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
        self, adata: Optional[AnnData] = ..., *, batch_size: int = ..., fit: Optional[str] = ...,
    ) -> AnnData: ...


# ── preprocessing (pp) ───────────────────────────────────────────────────────
class pp:
    def pool_rare_clones(
        adata: AnnData, *, clonotype_key: str, groupby: str, min_cells: int = ...,
        samples: Optional[str] = ..., key_added: Optional[str] = ...,
    ) -> None: ...
    def exclude_tcr_genes(
        adata: AnnData, *, keep: Sequence[str] = ..., gene_symbols: Optional[str] = ...,
        inplace: bool = ...,
    ) -> Optional[np.ndarray]: ...
    def clone_size(adata: AnnData, *, key_added: str = ..., return_counts: bool = ...) -> Any: ...
    def from_mudata(
        mdata: Any, *, clonotype_key: str, covariate_key: Optional[str] = ...,
        gex_mod: str = ..., airr_mod: str = ..., counts_layer: str = ...,
        drop_missing_clonotype: bool = ...,
    ) -> AnnData: ...
    def repertoire_summary(
        adata: AnnData, *, clonotype_key: str, groupby: Optional[str] = ...,
        covariate_key: Optional[str] = ..., phenotype_key: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def clone_persistence(
        adata: AnnData, *, clonotype_key: str, covariate_key: str,
        cov_from: Optional[str] = ..., cov_to: Optional[str] = ...,
        groupby: Optional[str] = ..., per_clone: bool = ...,
    ) -> pd.DataFrame: ...
    def clonotype_sharing(
        adata: AnnData, *, clonotype_key: str, groupby: str, per_clonotype: bool = ...,
    ) -> pd.DataFrame: ...


# ── tools / metrics (tl) ─────────────────────────────────────────────────────
class tl:
    def joint_distribution(
        adata: AnnData, *, covariate: Optional[str] = ...,
        n_samples: int = ..., use_logits: bool = ..., weighted: bool = ...,
        clones: Any = ..., exclude_pools: bool = ..., temperature: float = ...,
        random_state: Any = ..., device: Any = ..., fit: Optional[str] = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> Optional[dict]: ...
    def clonotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., cov_from: Optional[str] = ...,
        cov_to: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., exclude_pools: bool = ..., weighted: bool = ...,
        normalized: bool = ..., n_clones_ref: Any = ..., random_state: Any = ...,
        device: Any = ..., fit: Optional[str] = ..., null_model: Any = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> Optional[dict]: ...
    def phenotypic_entropy(
        adata: AnnData, *, covariate: Optional[str] = ..., cov_from: Optional[str] = ...,
        cov_to: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., exclude_pools: bool = ..., weighted: bool = ...,
        normalized: bool = ..., random_state: Any = ..., device: Any = ...,
        fit: Optional[str] = ..., null_model: Any = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> Optional[dict]: ...
    def mutual_information(
        adata: AnnData, *, covariate: Optional[str] = ..., cov_from: Optional[str] = ...,
        cov_to: Optional[str] = ..., groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., exclude_pools: bool = ..., weighted: bool = ...,
        normalized: bool = ..., normalize_mode: str = ..., random_state: Any = ...,
        device: Any = ..., fit: Optional[str] = ..., null_model: Any = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> Optional[dict]: ...
    def phenotypic_flux(
        adata: AnnData, *, cov_from: str, cov_to: str, groupby: Optional[str] = ...,
        splitby: Optional[str] = ..., n_samples: int = ..., temperature: float = ...,
        clones: Any = ..., exclude_pools: bool = ..., weighted: bool = ...,
        distance_metric: str = ..., random_state: Any = ..., device: Any = ...,
        fit: Optional[str] = ..., null_model: Any = ...,
        key_added: Optional[str] = ..., inplace: bool = ...,
    ) -> Optional[dict]: ...


# ── plotting (pl): twins render what tl stored ───────────────────────────────
class pl:
    def clonotypic_entropy(
        adata: AnnData, *, view: str = ..., quantity: str = ..., key: Optional[str] = ...,
        order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., legend: bool = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_entropy(
        adata: AnnData, *, view: str = ..., quantity: str = ..., key: Optional[str] = ...,
        order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., legend: bool = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def mutual_information(
        adata: AnnData, *, view: str = ..., quantity: str = ..., key: Optional[str] = ...,
        order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., legend: bool = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def phenotypic_flux(
        adata: AnnData, *, view: str = ..., quantity: str = ..., key: Optional[str] = ...,
        order: Any = ..., hue_order: Any = ...,
        palette: Any = ..., legend: bool = ..., ax: Any = ..., figsize: Any = ...,
        save: Any = ..., show: Any = ..., return_df: bool = ...,
    ) -> Any: ...
    def gene_importance(
        adata: AnnData, *, view: str = ..., n_top: int = ..., key: Optional[str] = ...,
        order: Any = ..., hue_order: Any = ..., palette: Any = ..., legend: bool = ...,
        ax: Any = ..., figsize: Any = ..., save: Any = ..., show: Any = ...,
        return_df: bool = ...,
    ) -> Any: ...
    def resolve_colors(
        adata: AnnData, cat_key: str, categories: Any = ..., *, palette: Any = ...,
        persist: bool = ...,
    ) -> dict: ...


# ── stored results (get) ──────────────────────────────────────────────────────
class get:
    def result(
        adata: AnnData, name: str, *, key: Optional[str] = ..., fit: Optional[str] = ...,
    ) -> Any: ...
    def params(
        adata: AnnData, name: str, *, key: Optional[str] = ..., fit: Optional[str] = ...,
    ) -> dict: ...
    def replicates(
        adata: AnnData, name: str, *, key: Optional[str] = ..., fit: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def items(
        adata: AnnData, name: str, *, key: Optional[str] = ..., fit: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def stats(
        adata: AnnData, name: str, *, key: Optional[str] = ..., fit: Optional[str] = ...,
    ) -> Optional[pd.DataFrame]: ...
    def draws(
        adata: AnnData, name: str, *, items: bool = ..., key: Optional[str] = ...,
        fit: Optional[str] = ...,
    ) -> pd.DataFrame: ...
    def fits(adata: AnnData) -> list: ...


# ── diagnostics (diag) ───────────────────────────────────────────────────────
class diag:
    def joint_distribution_ppc(
        adata: AnnData, *, covariate: Optional[str] = ..., distance_metric: Any = ...,
        n_samples: int = ..., n_sims: int = ..., clones: Any = ...,
        exclude_pools: bool = ..., random_state: Any = ..., fit: Optional[str] = ...,
        plot: bool = ..., save: Any = ...,
    ) -> dict: ...
    def phenotype_calibration(
        adata: AnnData, *, n_bins: int = ..., fit: Optional[str] = ..., plot: bool = ...,
        ax: Any = ..., save: Any = ...,
    ) -> dict: ...
    def reconstruction_ppc(
        model: Any, adata: Optional[AnnData] = ..., *, n_sims: int = ...,
        random_state: Any = ..., plot: bool = ..., ax: Any = ..., save: Any = ...,
    ) -> dict: ...
    def permutation_null(
        adata: AnnData, *, metric: str = ..., covariate: Optional[str] = ...,
        groupby: Optional[str] = ..., clones: Any = ..., exclude_pools: bool = ...,
        normalize_mode: str = ..., n_perm: int = ..., random_state: Any = ..., plot: bool = ...,
        ax: Any = ..., save: Any = ...,
    ) -> dict: ...
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
        groupby: Optional[str] = ..., splitby: Optional[str] = ..., exclude_pools: bool = ...,
        n_samples: int = ..., use_gate: bool = ..., batch_size: int = ...,
        random_state: Any = ..., null_model: Any = ..., key_added: Optional[str] = ...,
        inplace: bool = ...,
    ) -> Optional[dict]: ...


# ── permutation references (null) ────────────────────────────────────────────
class null:
    def phenotype(
        model: Any, adata: AnnData, *, n_nulls: int = ..., within: Any = ...,
        seed: Optional[int] = ..., key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> list: ...
    def clonotype(
        model: Any, adata: AnnData, *, n_nulls: int = ..., within: Any = ...,
        seed: Optional[int] = ..., key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> list: ...
    def covariate(
        model: Any, adata: AnnData, *, n_nulls: int = ..., seed: Optional[int] = ...,
        key_added: Optional[str] = ..., **train_kwargs: Any,
    ) -> list: ...
    def all(
        model: Any, adata: AnnData, *, n_nulls: int = ..., kinds: Any = ..., within: Any = ...,
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


# ── synthetic data (datasets) ────────────────────────────────────────────────
class datasets:
    def simulate_cohort(
        *, n_patients: int = ..., timepoints: Any = ..., disease_fraction: float = ...,
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
