# %% [markdown]
# # tcri end-to-end — a multi-patient, multi-condition cohort
#
# One pass through the whole package: **8 patients**, each sampled **pre** and **post**
# treatment, split into **responders (R)** and **non-responders (NR)**.
#
# The two arms are constructed to differ, so the figures have something to show:
#
# * **R patients** start diffuse and end sharply coupled — clones commit to phenotypes.
# * **NR patients** start diffuse and stay that way.
#
# Run as a script (`python examples/end_to_end_workflow.py`) or open as a notebook — the
# `# %%` markers are jupytext cells. Figures land in `examples/end_to_end_figures/`.

# %%
import contextlib
import io
import logging
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyro

matplotlib.use("Agg")          # drop this line in a notebook
import matplotlib.pyplot as plt

import tcri
from tcri import _keys as K
from tcri.datasets import simulate_cohort
from tcri.model import TCRIModel

warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.INFO)

FIGS = Path(__file__).parent / "end_to_end_figures"
FIGS.mkdir(exist_ok=True)
SEED = 0

# %% [markdown]
# ## 1. Build the cohort — one line
#
# `simulate_cohort` produces the shape most analyses have: patients as replicates, an ordered
# condition axis *within* each patient, and a response label *between* them.
#
# Two properties it gives you that matter downstream:
#
# * **Clones are paired across conditions.** Each patient is simulated once, fixing its
#   clone→phenotype structure, and every condition is drawn from that one population. (Doing
#   it the obvious way — simulating each condition separately and matching clone names —
#   yields ids that line up over unrelated structure, so `phenotypic_flux` and the `delta_*`
#   metrics would have nothing to measure.)
# * **Clone sizes are heavy-tailed**, as real repertoires are: a power law, a few large
#   expanded clones over a long tail of singletons. `simulate_tcri`'s own clone abundance is a
#   symmetric Dirichlet, which is not.
#
# Only the clone→phenotype *concentration* changes between conditions — responders' clones
# commit, non-responders' barely move. Nothing is relabelled, so a cell's phenotype still
# matches the expression it was generated with.

# %%
adata = simulate_cohort(
    n_patients=8,
    conditions=("pre", "post"),
    responder_fraction=0.5,
    n_clones=(14, 24),                  # ragged, as real cohorts are
    n_phenotypes=4,
    n_genes=40,
    n_cells_per_sample=260,
    clone_size_distribution="powerlaw", # "uniform" for a flat repertoire
    clone_size_exponent=2.0,
    responder_enrichment=12.0,
    nonresponder_enrichment=1.1,
    seed=SEED,
)
print(f"{adata.n_obs} cells x {adata.n_vars} genes")
print(adata.obs.groupby(["response", "condition"], observed=True).size().to_frame("cells").T)

sizes = (adata.obs.query("condition == 'pre'")
         .groupby("patient", observed=True)["clone_id"].value_counts())
sizes = sizes[sizes > 0]
top = sizes.groupby(level=0).apply(lambda s: s.iloc[0] / s.sum())
print(f"\nclones per patient: {sorted(adata.uns['tcri_truth']['per_sample']['n_clones'].unique())}")
print(f"largest clone holds {top.min():.0%}-{top.max():.0%} of a patient's cells")

# %% [markdown]
# ## 2. Register and train
#
# `setup_anndata` names the four columns the model needs, plus two that are easy to confuse:
#
# * **`batch_key`** — a *modelling* choice. It is one-hot encoded into every hidden layer of
#   the encoder and decoder, so the model can absorb batch structure.
# * **`replicate`** — a *statistical* choice: the independent unit. Registering it once means
#   `groupby` can be left implicit on every metric, and the effective value still lands in
#   each result's provenance rather than a `None` placeholder.
#
# They are often the same column (here, `patient`) and mean entirely different things.

# %%
pyro.clear_param_store()
TCRIModel.setup_anndata(
    adata,
    layer="counts",
    clonotype_key="clone_id",
    phenotype_key="phenotype",
    covariate_key="condition",     # the axis a delta is taken across
    batch_key="patient",           # one-hot into the networks
    replicate="patient",           # the independent unit for statistics
)

model = TCRIModel(adata, n_latent=10, n_hidden=64, n_layers=2,
                  classifier_n_layers=1, classifier_hidden=64, K=4, seed=SEED)

with contextlib.redirect_stdout(io.StringIO()):
    model.train(max_epochs=150, batch_size=256, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    model.to_anndata(adata)

record = getattr(model, "training_record_", None)
if record:
    print({k: v for k, v in record.items() if not hasattr(v, "__len__") or isinstance(v, str)})

# %% [markdown]
# ## 3. Diagnostics — before trusting any metric
#
# `tcri.diag` asks whether the *fit* is worth interpreting. Every function returns a
# DataFrame, so the numbers are inspectable rather than only drawn.
#
# * **`loss`** / **`archetypes`** — did training converge, and did the VampPrior components
#   separate?
# * **`joint_distribution_ppc`** — does the model's clone×phenotype table resemble the
#   observed crosstab?
# * **`phenotype_calibration`** — when the classifier says 0.8, is it right 80% of the time?
# * **`reconstruction_ppc`** — does the decoder generate counts like the real ones?
# * **`permutation_null`** — how large is the metric under shuffled labels? This is the
#   reference any MI should be read against, and it is *model-free*: it scores the empirical
#   crosstab and draws no posterior samples.

# %%
# `loss` and `archetypes` are two-panel figures that build their own axes — use their `save=`
# rather than handing them an `ax` (a single ax cannot hold two panels).
tcri.diag.loss(model, log_scale=True, save=FIGS / "1a_training_loss.png")
tcri.diag.archetypes(model, save=FIGS / "1b_archetypes.png")

ppc = tcri.diag.joint_distribution_ppc(adata, distance_metric="l1")
calib = tcri.diag.phenotype_calibration(adata, n_bins=8)
recon = tcri.diag.reconstruction_ppc(model, adata, n_sims=25, random_state=SEED)

print(f"joint PPC        : median per-clone L1 = {ppc['distance'].median():.3f}  "
      f"(0 = perfect, 2 = maximal)")
print(f"calibration      : ECE = {calib.attrs.get('ECE', float('nan')):.3f}")
print(f"reconstruction   : {recon.shape[0]} rows, cols {list(recon.columns)[:4]}")

null = tcri.diag.permutation_null(adata, metric="mutual_information", covariate="post",
                                  n_perm=200, random_state=SEED)
print("\npermutation null (one row per covariate level):")
print(null[["covariate", "observed", "null_mean", "null_sd", "z", "p"]]
      .round(4).to_string(index=False))
# `observed` is the empirical MI on the real labels; `null_mean` is what shuffling gives.

# %% [markdown]
# ## 4. `tl` — compute once, cached with provenance
#
# Every `tl` returns **the same three slots** and stores that object in `uns`:
#
# | slot | one row per | reduced over |
# |---|---|---|
# | `table` | (covariate, group, item, draw) | nothing — the substrate |
# | `result` | (covariate, group, item) | `draw` only |
# | `stats` | (split_a, split_b) | items → groups, then contrast |
#
# `result` is built *from* `table`, so they cannot drift. `stats` appears only with
# `splitby`, and its **n counts replicates, never items** — 18 clones from 4 patients give
# n=4.

# %%
mi_post = tcri.tl.mutual_information(
    adata, covariate="post",
    groupby="patient",          # could be omitted: `replicate` is registered
    splitby="response",         # -> produces `stats`
    n_samples=100,              # posterior draws; 0 = plug-in point estimate
    normalize_mode="min",       # "average" reproduces the note's eq 6
    weighted=False,             # one vote per CLONE; True = one vote per CELL
    random_state=SEED,
)
print("slots:", sorted(mi_post))
print("\nresult (one row per patient):")
print(mi_post["result"][["patient", "response", "value", "sd", "hdi_low", "hdi_high"]]
      .round(4).to_string(index=False))
print("\nstats (the contrast — note n counts PATIENTS):")
print(mi_post["stats"][["level_a", "level_b", "n_a", "n_b", "mean_a", "mean_b",
                        "delta", "p", "stars"]].round(4).to_string(index=False))

# %%
# the other three, plus the joint everything reduces
jd = tcri.tl.joint_distribution(adata, covariate="post", n_samples=20, random_state=SEED)
print(f"joint: result {jd['result'].shape} (clone x phenotype), rows sum to 1: "
      f"{np.allclose(jd['result'].sum(axis=1), 1.0)}")

tcri.tl.clonotypic_entropy(adata, covariate="post", groupby="patient", splitby="response",
                           n_samples=60, random_state=SEED)
tcri.tl.phenotypic_entropy(adata, covariate="post", groupby="patient", splitby="response",
                           n_samples=60, random_state=SEED)
tcri.tl.phenotypic_flux(adata, cov_from="pre", cov_to="post", groupby="patient",
                        splitby="response", distance_metric="kl",
                        n_samples=60, random_state=SEED)

# A second call REPLACES the cached result (the scanpy convention). `key_added` is how you
# keep both side by side instead -- here, the same metric at the other condition.
tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", splitby="response",
                           n_samples=100, random_state=SEED, key_added="mi_pre")

pre_vs_post = pd.concat([
    tcri.get.result(adata, "mutual_information", key="mi_pre")["result"].assign(condition="pre"),
    mi_post["result"].assign(condition="post"),
])
print("\nboth conditions, from two cached results:")
print(pre_vs_post.groupby(["response", "condition"], observed=True)["value"]
      .mean().unstack("condition").round(4).to_string())
# ...and `pl` renders either one: pass the same `key` you stored it under.

# %% [markdown]
# ### Reading results back
#
# `tcri.get` is how anything reads a cached result without knowing the blob format — `pl`
# uses it, and so should your own code. `params` carries every argument the tool ran with,
# including defaults never passed, so a figure from six months ago can still say what made it.

# %%
assert tcri.get.result(adata, "mutual_information")["result"].equals(mi_post["result"])
params = tcri.get.params(adata, "mutual_information")
print({k: params[k] for k in ("covariate", "groupby", "splitby", "n_samples",
                              "normalize_mode", "weighted")})
print("\nthe unreduced substrate, one row per draw:")
print(tcri.get.table(adata, "mutual_information", which="table").head(3).round(4)
      .to_string(index=False))

# %% [markdown]
# ## 5. `pl` — twins that render the cache
#
# A `pl` twin takes **no metric arguments**. It reads what `tl` stored, so the covariate,
# groupby, splitby and n_samples it draws are the ones actually used — a figure cannot
# disagree with the frame in your hand. Run the `tl` twin first.
#
# The mark follows one rule: **a mark shows one variance component.** Within an x position
# the sample is the coarsest unit that varies there (replicate > item > draw). So dots are
# patients wherever a p-value over patients sits above them, and draws are only ever pooled
# within a single replicate.

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
tcri.pl.mutual_information(adata, ax=axes[0, 0])
axes[0, 0].set_title("mutual_information — post\ndots = patients, bracket from `stats`", fontsize=10)
tcri.pl.clonotypic_entropy(adata, ax=axes[0, 1])
axes[0, 1].set_title("clonotypic_entropy — one value per PHENOTYPE", fontsize=10)
tcri.pl.phenotypic_entropy(adata, ax=axes[1, 0])
axes[1, 0].set_title("phenotypic_entropy — one value per CLONE", fontsize=10)
tcri.pl.phenotypic_flux(adata, ax=axes[1, 1])
axes[1, 1].set_title("phenotypic_flux — how far clones moved pre -> post", fontsize=10)
# `key=` renders a non-default result -- the pre-treatment MI stored above:
#     tcri.pl.mutual_information(adata, key="mi_pre")
fig.suptitle("The four metric twins", fontsize=13, y=0.99)
fig.tight_layout()
fig.savefig(FIGS / "2_metrics.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ### Colours are a property of the level
#
# `resolve_colors` caches under scanpy's `uns["<key>_colors"]`, so a level keeps its colour
# across every later figure — and `sc.pl.umap(color="response")` matches too.

# %%
palette = tcri.pl.resolve_colors(adata, "response", palette={"R": "#2E8B57", "NR": "#C1440E"})
print("response palette:", palette)
print("stored in uns  :", adata.uns[K.colors("response")])

# %% [markdown]
# ## 6. The paired entropies — what changed, per clone
#
# `delta_*` takes `cov_from` / `cov_to` and subtracts **within a posterior draw**, so the
# reported interval is the interval of the *difference*. HDIs do not subtract: you cannot
# recover it from the endpoints' intervals, which is why this is a function rather than
# arithmetic you do yourself.
#
# Support is the **intersection** — clones present at both timepoints, within each patient.
# A delta needs both endpoints, and the drop is warned about because it moves `n`.
#
# There is deliberately **no `delta_mutual_information`**: MI has no item axis, so it is
# already the repertoire-level number and its "delta" is a subtraction of two cached scalars.
# That one belongs to you, not the package.

# %%
d_phen = tcri.tl.delta_phenotypic_entropy(
    adata, cov_from="pre", cov_to="post", groupby="patient", splitby="response",
    n_samples=60, random_state=SEED,
)
d_clon = tcri.tl.delta_clonotypic_entropy(
    adata, cov_from="pre", cov_to="post", groupby="patient", splitby="response",
    n_samples=60, random_state=SEED,
)
per_arm = (d_phen["result"].groupby("response", observed=True)["value"]
           .agg(["mean", "count"]).round(4))
print("delta phenotypic entropy (post - pre), per arm:")
print(per_arm.to_string())
print("\ncontrast:")
print(d_phen["stats"][["level_a", "level_b", "n_a", "n_b", "delta", "p", "stars"]]
      .round(4).to_string(index=False))

# %%
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
tcri.pl.delta_phenotypic_entropy(adata, kind="delta", ax=axes[0, 0])
axes[0, 0].set_title("delta_phenotypic_entropy — the change\nzero rule; dots = patients",
                     fontsize=10)
tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints", ax=axes[0, 1])
axes[0, 1].set_title("kind='endpoints' — a line per patient\n(a clonotype persists: matched)",
                     fontsize=10)
tcri.pl.delta_clonotypic_entropy(adata, kind="delta", ax=axes[1, 0])
axes[1, 0].set_title("delta_clonotypic_entropy — per phenotype", fontsize=10)
tcri.pl.delta_clonotypic_entropy(adata, kind="endpoints", ax=axes[1, 1])
axes[1, 1].set_title("no lines — a phenotype is a bin,\nnot a barcode", fontsize=10)
fig.suptitle("The paired entropies (pre -> post)", fontsize=13, y=0.99)
fig.tight_layout()
fig.savefig(FIGS / "3_deltas.png", dpi=130, bbox_inches="tight")

# %%
print(f"figures written to {FIGS}")
for f in sorted(FIGS.glob("*.png")):
    print(f"  {f.name}")
