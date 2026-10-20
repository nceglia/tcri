# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # tcri — an end-to-end walkthrough
#
# One pass through the package on a synthetic cohort: **8 patients**, sampled **pre** and
# **post** treatment, split into a **disease** and a **control** arm.
#
# The order below is the order you would actually work in — build, fit, *check the fit*,
# then measure, then plot. The diagnostics come before the metrics on purpose: none of the
# numbers in section 4 mean anything if section 3 looks wrong.

# %%
# %matplotlib inline
import logging
import warnings

# before importing anything else: tqdm warns about ipywidgets on import, which is noise here
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro

import tcri
from tcri._state import keys as K
from tcri.datasets import simulate_cohort
from tcri.model import TCRIModel

plt.rcParams["figure.dpi"] = 110
SEED = 0

# %% [markdown]
# ## 1. Build a cohort
#
# `simulate_cohort` gives the shape most analyses have: patients as replicates, an ordered
# timepoint axis *within* each patient, and a `disease_status` label *between* them.
#
# Two properties matter downstream:
#
# - **Clones are paired across timepoints.** Each patient is simulated once, fixing its
#   clone→phenotype structure, and every timepoint is drawn from that one population. Without
#   this, `phenotypic_flux` and every change from pre to post would have nothing to measure.
# - **Clone sizes are heavy-tailed** — a power law, as real repertoires are: a few large
#   expanded clones over a long tail of singletons.
#
# Only the clone→phenotype *concentration* changes between timepoints. Responders' clones
# commit; control clones barely move. Nothing is relabeled, so each cell's phenotype still
# matches the expression it was generated with.

# %%
adata = simulate_cohort(
    n_patients=16,                       # 8 per arm -- 4 per arm cannot reach p<0.05
    timepoints=("pre", "post"),
    disease_fraction=0.5,
    n_clones=(14, 24),                   # ragged, as real cohorts are
    n_phenotypes=4,
    n_genes=40,
    n_cells_per_sample=260,
    clone_size_distribution="powerlaw",  # "uniform" for a flat repertoire
    clone_size_exponent=2.0,
    disease_enrichment=12.0,
    control_enrichment=1.1,
    seed=SEED,
)
adata

# %%
print(adata.obs.groupby(["disease_status", "timepoint"], observed=True).size().to_frame("cells").T)

sizes = (adata.obs.query("timepoint == 'pre'")
         .groupby("patient", observed=True)["clone_id"].value_counts())
sizes = sizes[sizes > 0]
top = sizes.groupby(level=0).apply(lambda s: s.iloc[0] / s.sum())
print(f"\nclones per patient : {sorted(adata.uns['tcri_truth']['per_sample']['n_clones'].unique())}")
print(f"largest clone holds: {top.min():.0%}–{top.max():.0%} of a patient's cells")

# %%
# the clone size distribution, which is what "heavy-tailed" means concretely
fig, ax = plt.subplots(figsize=(5.5, 4))
for patient, s in sizes.groupby(level=0):
    v = np.sort(s.to_numpy())[::-1]
    ax.plot(np.arange(1, len(v) + 1), v, marker="o", ms=3, lw=1, alpha=0.7, label=patient)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("clone rank"); ax.set_ylabel("cells")
ax.set_title("clone size vs rank (a straight line here is a power law)")
ax.legend(fontsize=7, ncol=2, frameon=False)
plt.show()

# %% [markdown]
# ### Preprocessing
#
# `pool_rare_clones` pools each patient's rare clones into one `pooled@{patient}` label: a
# clone is rare when it has fewer than `min_cells` cells and all of them come from one sample.
# On a heavy-tailed repertoire most clones are tiny, and a clone seen through a cell or two says
# little about its own phenotype distribution while still counting as a row in every entropy
# normalizer. This notebook keeps every clone and only shows what the call would do.
#
# (`tcri.pp` also has `clone_size`, but it reads the registry that `to_anndata` writes, so it
# runs *after* fitting rather than here — it appears in section 2.)

# %%
preview = adata.copy()
tcri.pp.pool_rare_clones(preview, clonotype_key="clone_id", groupby="patient")
print(f"clones: {adata.obs['clone_id'].nunique()} -> "
      f"{preview.obs['clone_id_pooled'].nunique()} if clones with fewer than 3 cells were pooled")

# %% [markdown]
# ## 2. Register and fit
#
# `setup_anndata` names the columns the model needs. Two of them are easy to confuse and mean
# entirely different things, even when they are the same column:
#
# - **`batch_key`** is a *modeling* choice — one-hot encoded into every hidden layer of the
#   encoder and decoder, so the network can absorb batch structure.
# - **`replicate`** is a *statistical* choice — the independent unit. Registering it once lets
#   `groupby` be left implicit on every metric, and the effective value still lands in each
#   result's provenance rather than a `None` placeholder.

# %%
pyro.clear_param_store()
TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="timepoint",   # the axis a change is taken across
    batch_key="patient",         # one-hot into the networks
    replicate="patient",         # the independent unit for statistics
)

model = TCRIModel(adata, n_latent=10, n_hidden=64, n_layers=2,
                  classifier_n_layers=1, classifier_hidden=64, K=4, seed=SEED)

with contextlib.redirect_stdout(io.StringIO()):
    model.train(max_epochs=150, batch_size=256, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    model.to_anndata(adata)
    # The permutation references, fitted once. Every scored metric below is read against one:
    # the same model, the same knobs, the same seed, on one permuted label vector. Without
    # them a metric raises rather than reporting a bare number.
    tcri.null.all(model, adata, enable_progress_bar=False, enable_model_summary=False)

# what actually happened, not what was requested -- early stopping can end a run short
record = model.training_record_
pd.Series({k: v for k, v in record.items()
           if not hasattr(v, "__len__") or isinstance(v, str)}).to_frame("value")

# %% [markdown]
# `to_anndata` wrote everything the metrics read. Two model outputs are useful directly:
# `predict` gives per-cell phenotype probabilities, and `get_latent_representation` gives the
# latent embedding you would hand to a neighbors graph or a UMAP.

# %%
# `clone_size` reads uns[METADATA]['clone_col'], which to_anndata just wrote
tcri.pp.clone_size(adata, key_added="clone_size")
print(adata.obs[["clone_size"]].describe().round(1).T.to_string())

probs = model.predict(adata)
latent = model.get_latent_representation(adata)
print(f"predict : {probs.shape}  columns = {list(probs.columns)}")
print(f"latent  : {latent.shape}")
print(f"\nagreement between argmax(predict) and the observed label: "
      f"{(probs.to_numpy().argmax(1) == adata.obs['phenotype'].cat.codes.to_numpy()).mean():.1%}")
probs.head().round(3)

# %% [markdown]
# ## 3. Diagnostics — before trusting any metric
#
# `tcri.diag` asks whether the *fit* is worth interpreting. Every function returns a
# DataFrame, so the numbers are inspectable and not only drawn.

# %%
# did training converge, and did the VampPrior components separate?
tcri.diag.loss(model, log_scale=True)
plt.show()
tcri.diag.archetypes(model)
plt.show()

# %% [markdown]
# **`joint_distribution_ppc`** — does the model's clone×phenotype table resemble the observed
# crosstab? Per-clone L1 distance, where 0 is perfect and 2 is maximal.

# %%
ppc = tcri.diag.joint_distribution_ppc(adata, distance_metric="l1")
print(f"median per-clone L1: {ppc['distance'].median():.3f}")
ppc.head()

# %% [markdown]
# **`phenotype_calibration`** — when the classifier says 0.8, is it right 80% of the time?
# `ECE` is the expected calibration error; lower is better.

# %%
calib = tcri.diag.phenotype_calibration(adata, n_bins=8)
print(f"ECE = {calib.attrs['ECE']:.3f}")

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, c="0.5", label="perfect")
ax.plot(calib["mean_pred"], calib["emp_freq"], marker="o", color="#AE81FF", label="observed")
ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical frequency")
ax.set_title("phenotype calibration"); ax.legend(frameon=False)
plt.show()

# %% [markdown]
# **`reconstruction_ppc`** — does the decoder generate counts like the real ones?
# **`permutation_null`** — how large is the metric under shuffled labels? This is the
# reference any MI should be read against, and it is *model-free*: it scores the empirical
# crosstab and draws no posterior samples.

# %%
recon = tcri.diag.reconstruction_ppc(model, adata, n_sims=25, random_state=SEED)
print("reconstruction PPC:")
print(recon.round(3).to_string(index=False))

null = tcri.diag.permutation_null(adata, metric="mutual_information", covariate="post",
                                  n_perm=200, random_state=SEED)
print("\npermutation null — `observed` is the real labels, `null_mean` is shuffled:")
print(null[["covariate", "observed", "null_mean", "null_sd", "z", "p"]]
      .round(4).to_string(index=False))

# %% [markdown]
# ## 4. `tl` — compute once, cached with provenance
#
# Every `tl` call stores its result in `uns` and returns `None`; `tcri.get` reads it back.
# Mutual information, the two entropies and `phenotypic_flux` store it **by level**:
#
# | frame | one row per | read with |
# |---|---|---|
# | `replicate.summary` | (covariate, replicate) | `tcri.get.replicates` |
# | `replicate.draws` | (covariate, replicate, draw) | `tcri.get.draws` |
# | `item.summary` | (covariate, replicate, item) | `tcri.get.items` |
# | `item.draws` | (covariate, replicate, item, draw) | `tcri.get.draws(..., items=True)` |
# | `stats` | test | `tcri.get.stats` |
#
# A summary is built *from* its draws, so the two cannot drift. Mutual information has one
# value per joint and no item level. A change between two timepoints (section 6) labels its
# rows `cov_from`/`cov_to` in place of `covariate`. `stats` holds a split row for each pair of
# `splitby` levels, and a change adds its covariate rows; its **n counts replicates, never
# items** — 18 clones from 4 patients give n=4, not n=18. `joint_distribution` stores `table`
# and `result`, read with `tcri.get.result`.

# %%
tcri.tl.mutual_information(
    adata,
    covariate="post",
    groupby="patient",     # could be omitted -- `replicate` is registered
    splitby="disease_status",    # -> produces `stats`
    n_samples=100,         # posterior draws; 0 = plug-in point estimate
    normalize_mode="min",  # "average" is the manuscript's eq 6
    weighted=False,        # one vote per CLONE; True = one vote per CELL
    random_state=SEED,
)
print("levels:", sorted(tcri.get.result(adata, "mutual_information")))
mi_post = tcri.get.replicates(adata, "mutual_information")
mi_post[["patient", "disease_status", "value", "sd", "hdi_low", "hdi_high"]].round(4)

# %%
# the contrast. `n_a`/`n_b` count PATIENTS -- that is what makes pseudoreplication impossible
mi_stats = tcri.get.stats(adata, "mutual_information")
mi_stats[["level_a", "level_b", "n_a", "n_b", "mean_a", "mean_b", "delta", "p", "stars"]].round(4)

# %% [markdown]
# **`n_clones_ref` matters here.** This cohort has ragged clone counts by design. A patient's
# clonotypic entropy, the replicate value H(C|P), is normalized by `log2(C)`, where `C` is the
# number of clones that patient has at the covariate, so two patients with different `C` are on
# different scales. Passing `n_clones_ref` normalizes every patient by `log2(n_clones_ref)`
# instead. The table reads the replicate summary both ways, one row per patient.

# %%
free = tcri.tl.clonotypic_entropy(adata, covariate="post", groupby="patient",
                                  random_state=SEED, inplace=False)["replicate"]["summary"]
pinned = tcri.tl.clonotypic_entropy(adata, covariate="post", groupby="patient",
                                    n_clones_ref=int(adata.obs.groupby("patient", observed=True)
                                                     ["clone_id"].nunique().max()),
                                    random_state=SEED, inplace=False)["replicate"]["summary"]
comparison = (free.set_index("patient")["value"].rename("own log2(C)")
              .to_frame()
              .join(pinned.set_index("patient")["value"].rename("pinned")))
comparison["difference"] = comparison["pinned"] - comparison["own log2(C)"]
comparison.round(4).head(8)

# %%
# the other metrics, plus the joint they all reduce
tcri.tl.joint_distribution(adata, covariate="post", n_samples=20, random_state=SEED)
jd = tcri.get.result(adata, "joint_distribution")
print(f"joint: {jd['result'].shape} (clone x phenotype), rows sum to 1: "
      f"{np.allclose(jd['result'].sum(axis=1), 1.0)}")

tcri.tl.clonotypic_entropy(adata, covariate="post", groupby="patient", splitby="disease_status",
                           n_samples=60, random_state=SEED)
tcri.tl.phenotypic_entropy(adata, covariate="post", groupby="patient", splitby="disease_status",
                           n_samples=60, random_state=SEED)
tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post", groupby="patient",
                        splitby="disease_status", distance_metric="kl",
                        n_samples=60, random_state=SEED)

# a second call REPLACES the cached result (the scanpy convention). `key_added` keeps both.
tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", splitby="disease_status",
                           n_samples=100, random_state=SEED, key_added="mi_pre")

pre_vs_post = pd.concat([
    tcri.get.replicates(adata, "mutual_information", key="mi_pre").assign(timepoint="pre"),
    mi_post.assign(timepoint="post"),
])
pre_vs_post.groupby(["disease_status", "timepoint"], observed=True)["value"].mean().unstack("timepoint").round(4)

# %% [markdown]
# ### Reading results back
#
# `tcri.get` is how anything reads a cached result without knowing the storage format — `pl`
# uses it, and so should your own code. With `inplace=False` a call stores nothing and returns
# the whole result instead, the same frames `tcri.get` reads. `params` carries every argument
# the tool ran with, including defaults that were never passed, so a figure from six months ago
# can still say what produced it.

# %%
fresh = tcri.tl.mutual_information(adata, covariate="post", groupby="patient",
                                   splitby="disease_status", n_samples=100,
                                   random_state=SEED, inplace=False)
assert fresh["replicate"]["summary"].equals(mi_post)
params = tcri.get.params(adata, "mutual_information")
print({k: params[k] for k in ("covariate", "groupby", "splitby", "n_samples",
                              "normalize_mode", "weighted")})

# the unreduced substrate: one row per patient, null fit and draw
tcri.get.draws(adata, "mutual_information").head().round(4)

# %% [markdown]
# ## 5. `pl` — twins that render the cache
#
# A `pl` twin takes **no metric arguments**. It reads what `tl` stored, so the covariate,
# groupby, splitby and n_samples it draws are the ones actually used — a figure cannot
# disagree with the frame in your hand. Run the `tl` twin first.
#
# The mark follows one rule: **a mark shows one variance component.** Within an x position
# the sample is the coarsest unit that varies there (replicate > item > draw). So the dots are
# patients wherever a p-value over patients sits above them, and draws are never pooled across
# replicates.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
tcri.pl.mutual_information(adata, ax=axes[0, 0])
axes[0, 0].set_title("mutual_information — post\ndots = patients, bracket from `stats`", fontsize=10)
tcri.pl.clonotypic_entropy(adata, view="item", ax=axes[0, 1])
axes[0, 1].set_title("clonotypic_entropy — one value per PHENOTYPE", fontsize=10)
tcri.pl.phenotypic_entropy(adata, view="item", ax=axes[1, 0])
axes[1, 0].set_title("phenotypic_entropy — one value per CLONE", fontsize=10)
tcri.pl.phenotypic_flux(adata, ax=axes[1, 1])
axes[1, 1].set_title("phenotypic_flux — how far clones moved pre → post", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Colors are a property of the level
#
# `resolve_colors` caches under scanpy's `uns["<key>_colors"]`, so a level keeps its color in
# every later figure — and `sc.pl.umap(color="disease_status")` matches too.

# %%
palette = tcri.pl.resolve_colors(adata, "disease_status", palette={"disease": "#2E8B57", "control": "#C1440E"})
print("disease_status palette:", palette)
print("stored in uns   :", adata.uns[K.colors("disease_status")])

# %% [markdown]
# ## 6. Changes between timepoints — what moved, per patient and per clone
#
# Pass `cov_from`/`cov_to` in place of `covariate` and mutual information or either entropy
# reports the **change** `cov_to − cov_from`. Both timepoints come from one shared posterior
# sample and the subtraction happens **within each draw**, so the reported interval is the
# interval of the *difference*. HDIs do not subtract — you cannot recover it from the
# endpoints' intervals, which is why the call takes both timepoints rather than leaving the
# subtraction to you.
#
# Each patient's change is the difference of its two per-patient values, each side counting
# every clone the patient has at that timepoint. The endpoints stay beside it, as
# `value_from`/`value_to` and `adjusted_from`/`adjusted_to`; mutual information also keeps
# `denom_from`/`denom_to`, each side's own normalizer.
#
# On a change the default reference is the **covariate null**, which shuffles the timepoint
# within each clone and batch, so `adjusted` is the change minus the null's change. The
# adjusted endpoints are read against that null, so they are not the adjusted values of a
# single-timepoint call, which are read against the phenotype null.
#
# `stats` on a change holds two kinds of row, told apart by `comparison`:
#
# - `"covariate"` — did this arm change? A Wilcoxon signed-rank test on each patient's change,
#   one row per `splitby` level (one over every patient without `splitby`), with
#   `level_a`/`level_b` = `cov_from`/`cov_to`.
# - `"split"` — did the change differ between arms? A Mann–Whitney test between the arms on
#   the per-patient changes.

# %%
# a change: cov_from/cov_to in place of covariate. key_added keeps section 4's post-only MI
tcri.tl.mutual_information(adata, cov_from="pre", cov_to="post", splitby="disease_status",
                           n_samples=100, random_state=SEED, key_added="mi_change")
mi_change = tcri.get.replicates(adata, "mutual_information", key="mi_change")
mi_change[["patient", "disease_status", "value_from", "value_to", "value",
           "adjusted", "adjusted_hdi_low", "adjusted_hdi_high"]].round(4)

# %%
# the covariate rows (did each arm change), then the split row (did the arms differ).
# `n_a`/`n_b` count PATIENTS here too
STATS = ["comparison", "disease_status", "level_a", "level_b", "quantity",
         "n_a", "n_b", "mean_a", "mean_b", "delta", "p", "stars"]
tcri.get.stats(adata, "mutual_information", key="mi_change")[STATS].round(4)

# %% [markdown]
# **Per item, a change covers the persistent clones**, the clones a patient has at both
# timepoints. Phenotypic entropy has one item row per persistent clone; clonotypic entropy has
# one per phenotype, recomputed on the persistent clones so both sides share one clone set and
# one normalizer. Mutual information has no items. `pp.clone_persistence` counts the persistent
# clones of each patient.
#
# `phenotypic_flux` is a change by construction and is stored the same way: per persistent
# clone, the distance its phenotype distribution moved (the item rows), and per patient those
# distances weighted by each clone's share of the `cov_from` joint (the replicate rows). Its covariate
# row tests the adjusted flux against 0, since flux is never negative.
#
# A change warns when some patients have no clone at a timepoint, and when no patient has a
# clone at both, whose usual cause is a covariate that is constant within each patient.

# %%
tcri.tl.phenotypic_entropy(adata, cov_from="pre", cov_to="post", splitby="disease_status",
                           n_samples=60, random_state=SEED)

# the change is taken within each draw, so it is exactly the difference of its endpoints there
pe_draws = tcri.get.draws(adata, "phenotypic_entropy")
print("value == value_to - value_from on every draw:",
      np.allclose(pe_draws["value"], pe_draws["value_to"] - pe_draws["value_from"],
                  equal_nan=True))


def item_rows(name):
    n = tcri.get.items(adata, name).groupby("patient", observed=True).size()
    return pd.Series(n.to_numpy(), index=n.index.astype(str))


# the item rows of the entropy's change and of flux are the persistent clones
persistence = tcri.pp.clone_persistence(adata, clonotype_key="clone_id",
                                        covariate_key="timepoint", cov_from="pre",
                                        cov_to="post", groupby="patient")
persistence = persistence.set_index("patient")[["clones_from_only", "clones_to_only",
                                                "persistent_clones"]]
persistence["entropy items"] = item_rows("phenotypic_entropy")
persistence["flux items"] = item_rows("phenotypic_flux")
persistence.head(8)

# %%
# flux, computed in section 4, read the same way: one row per patient
flux = tcri.get.replicates(adata, "phenotypic_flux")
flux[["patient", "disease_status", "value", "null_value", "adjusted",
      "adjusted_hdi_low", "adjusted_hdi_high"]].round(4)

# %%
# the same two kinds of row on the entropy's change and on flux
pd.concat({name: tcri.get.stats(adata, name)[STATS]
           for name in ("phenotypic_entropy", "phenotypic_flux")}).round(4)

# %%
# a change panel draws each patient's change against a zero line; view="item" draws each
# persistent clone's
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
tcri.pl.mutual_information(adata, key="mi_change", ax=axes[0, 0])
axes[0, 0].set_title("mutual_information — pre → post\none dot per patient", fontsize=10)
tcri.pl.phenotypic_entropy(adata, ax=axes[0, 1])
axes[0, 1].set_title("phenotypic_entropy — pre → post\none dot per patient", fontsize=10)
tcri.pl.phenotypic_entropy(adata, view="item", ax=axes[1, 0])
axes[1, 0].set_title("phenotypic_entropy, view='item'\none dot per persistent clone", fontsize=10)
tcri.pl.phenotypic_flux(adata, view="item", ax=axes[1, 1])
axes[1, 1].set_title("phenotypic_flux, view='item'\none dot per persistent clone", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Saving and resuming
#
# `ut.save_tcri_session` writes the fitted model, its pyro parameter store and the AnnData
# together, so a later session picks up exactly where this one stopped — including every
# cached `tl` result, since those live in `uns`.

# %%
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    manifest = tcri.ut.save_tcri_session(model, adata, f"{tmp}/run")
    print("wrote:", sorted(manifest))

    reloaded_model, reloaded = tcri.ut.load_tcri_session(f"{tmp}/run")
    cached = tcri.get.replicates(reloaded, "mutual_information")
    print(f"\nreloaded {reloaded.n_obs} cells")
    print(f"cached MI survived the round trip: "
          f"{np.allclose(cached['value'], mi_post['value'])}")
    print(f"provenance survived: {tcri.get.params(reloaded, 'mutual_information')['covariate']!r}")
