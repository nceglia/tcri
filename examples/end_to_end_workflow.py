# %% [markdown]
# # tcri end-to-end — a multi-patient, two-timepoint cohort
#
# One pass through the whole package on synthetic data with a known answer:
# **8 patients**, each sampled **pre** and **post** treatment, split into
# **responders (R)** and **non-responders (NR)**.
#
# The cohort is built so the truth is known in advance:
#
# * **R patients** start diffuse and end sharply coupled — clones commit to phenotypes.
# * **NR patients** start diffuse and stay that way.
#
# So clone↔phenotype mutual information should *rise* pre→post in R and barely move in NR,
# and the paired entropies should show R clones becoming less plastic. Nothing below is
# tuned to make that come out; if the package is right, it falls out.
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
from tcri.datasets import simulate_tcri
from tcri.model import TCRIModel

warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.INFO)

FIGS = Path(__file__).parent / "end_to_end_figures"
FIGS.mkdir(exist_ok=True)
SEED = 0

# %% [markdown]
# ## 1. Build the cohort
#
# The truth has to be *in* the data, and getting this right took a correction worth stating:
# simulating each timepoint independently and giving the two runs matching clone names does
# **not** produce paired clones. The ids line up, but the underlying clone→phenotype
# relationship is unrelated between the runs, so there is no "clone *c* became more
# committed" to find — only two unrelated draws.
#
# Instead, each patient is simulated **once** (fixing that patient's clone→phenotype
# structure), and the two timepoints are drawn from that one population:
#
# * **pre** — an unbiased sample.
# * **post** — a sample *enriched* for each clone's dominant phenotype. Strongly for
#   responders, barely for non-responders.
#
# So the same clones are observed twice, expression stays consistent with its phenotype label
# (nothing is relabelled), and only the clone→phenotype *concentration* moves. Clone ids are
# suffixed per patient (`clone_3@P02`) so none spans two patients — the metric `groupby`
# restricts by clone id, and `tcri` raises rather than let one patient absorb another's cells.

# %%
#: (patient, arm, enrichment). Higher enrichment = post-treatment clones commit harder.
COHORT = [
    ("P01", "R",  14.0), ("P02", "R",  10.0), ("P03", "R", 18.0), ("P04", "R",  12.0),
    ("P05", "NR",  1.2), ("P06", "NR",  1.0), ("P07", "NR", 1.3), ("P08", "NR",  1.1),
]
N_CLONES, N_PHENOTYPES, N_GENES, N_CELLS = 18, 4, 40, 260


def build_cohort():
    rng = np.random.default_rng(SEED)
    blocks = []
    for i, (patient, arm, enrichment) in enumerate(COHORT):
        # ONE simulation per patient: this fixes that patient's clone->phenotype structure
        pool = simulate_tcri(
            n_clones=N_CLONES, n_phenotypes=N_PHENOTYPES, n_genes=N_GENES,
            n_cells=3 * N_CELLS, n_covariates=1, omega_concentration=0.9, seed=SEED + i,
        )
        obs = pool.obs
        modal = obs.groupby("clone_id", observed=True)["phenotype"].agg(
            lambda x: x.value_counts().idxmax())
        is_modal = (obs["clone_id"].map(modal).astype(str) == obs["phenotype"].astype(str))

        # pre: unbiased. post: oversample each clone's dominant phenotype, so the SAME clones
        # concentrate. Nothing is relabelled, so expression still matches its phenotype.
        take = {}
        take["pre"] = rng.choice(pool.n_obs, size=N_CELLS, replace=False)
        w = np.where(is_modal.to_numpy(), enrichment, 1.0)
        take["post"] = rng.choice(pool.n_obs, size=N_CELLS, replace=False, p=w / w.sum())

        for timepoint, idx in take.items():
            block = pool[idx].copy()
            block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
            block.obs["timepoint"] = timepoint
            block.obs["patient"] = patient
            block.obs["response"] = arm
            block.obs_names = [f"{patient}_{timepoint}_{k}" for k in range(block.n_obs)]
            blocks.append(block)

    adata = ad.concat(blocks, join="outer", label=None)
    for col in ("clone_id", "phenotype", "timepoint", "patient", "response"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.layers["counts"] = adata.X.copy()
    return adata


adata = build_cohort()
print(f"{adata.n_obs} cells x {adata.n_vars} genes")
print(adata.obs.groupby(["response", "timepoint"], observed=True).size().to_frame("cells").T)

# The truth, straight off the labels — no model involved. Computed PER PATIENT and then
# averaged, because that is how `tl.mutual_information(groupby="patient")` computes it.
# Pooling an arm's patients first would mix clones across patients and inflate the MI, and
# comparing that against a per-patient estimate would be a unit mismatch, not a benchmark.
from tcri.datasets import mi_from_joint_oracle

rows = []
for (patient, arm, tp), g in adata.obs.groupby(["patient", "response", "timepoint"],
                                               observed=True):
    ct = pd.crosstab(g["clone_id"], g["phenotype"]).to_numpy(float)
    rows.append({"patient": patient, "response": arm, "timepoint": tp,
                 "nmi": mi_from_joint_oracle(ct)["nmi_min"]})
truth_per_patient = pd.DataFrame(rows)
truth = (truth_per_patient.groupby(["response", "timepoint"], observed=True)["nmi"].mean()
         .unstack("timepoint"))
truth["change"] = truth["post"] - truth["pre"]
print("\nground truth from the labels alone (per patient, then averaged):")
print(truth.round(4).to_string())

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
    covariate_key="timepoint",     # the axis a delta is taken across
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
null_mean = float(null["null_mean"].iloc[0])

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

# `key_added` keeps a second result side by side rather than replacing the first
tcri.tl.mutual_information(adata, covariate="pre", groupby="patient", splitby="response",
                           n_samples=100, random_state=SEED, key_added="mi_pre")

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

# %% [markdown]
# ## 7. Truth vs estimate
#
# The cohort was built with responders sharpening and non-responders flat. Nothing above was
# tuned to produce that, so this is a fair, if small, benchmark.

# %%
mi_pre = tcri.get.result(adata, "mutual_information", key="mi_pre")["result"]
est = (pd.concat([mi_pre.assign(timepoint="pre"),
                  mi_post["result"].assign(timepoint="post")])
       .groupby(["response", "timepoint"], observed=True)["value"].mean()
       .unstack("timepoint"))
est["change"] = est["post"] - est["pre"]

comparison = truth.join(est, lsuffix=" (truth)", rsuffix=" (model)")
print(comparison[["pre (truth)", "pre (model)", "post (truth)", "post (model)",
                  "change (truth)", "change (model)"]].round(4).to_string())

r_t, r_m = float(truth.loc["R", "change"]), float(est.loc["R", "change"])
nr_t, nr_m = float(truth.loc["NR", "change"]), float(est.loc["NR", "change"])
print(f"\ndirection: R moves more than NR in the truth ({r_t:+.3f} vs {nr_t:+.3f}) "
      f"and in the estimate ({r_m:+.3f} vs {nr_m:+.3f}) -> "
      f"{'recovered' if (r_m > nr_m) == (r_t > nr_t) else 'NOT recovered'}")
print(f"magnitude: the estimate captures {100 * r_m / r_t:.0f}% of the true R change.")

# %% [markdown]
# **Read that second number carefully.** On this fixture the *direction* is recovered and the
# *magnitude* is heavily attenuated. Both are worth knowing, and the attenuation is not a
# defect to be explained away — it has at least four candidate sources, and this example is
# far too small to separate them:
#
# 1. **Structural shrinkage.** In the model (Note 1 eq 2), a clone's phenotype distribution at
#    each covariate, `ϕ_(c,m)`, is drawn around a *single* clone-level `ω_c` shared across
#    covariates: `ϕ_m | ω ~ Dir(β·ω)`. The hierarchy actively pulls a clone's pre and post
#    toward each other, with `β` (`local_scale`) setting how hard. A within-clone change is
#    therefore shrunk by construction, and how much is a property of the fitted `β`.
# 2. **Estimator.** At `n_samples>0` tcri reports `E_s[NMI(J_s)]`, which is not the NMI of the
#    posterior mean — an open question in the metrics contract
#    (`OPEN_QUESTIONS['posterior_summary_of_a_nonlinear_metric']`).
# 3. **Fit.** 150 epochs on 4160 cells, with a calibration ECE of ~0.18 — the classifier is
#    not sharp, and a soft phenotype assignment blurs the joint the metric reads.
# 4. **Ceiling.** The empirical NMI here is computed on hard labels; the model's is computed on
#    a posterior over soft assignments, and the two are not the same quantity.
#
# The honest summary for this run: **tcri orders the arms correctly and separates both from
# the permutation null by a wide margin, while under-reporting the size of the change.** If
# you need calibrated effect sizes rather than rankings, that gap is the thing to characterise
# first — on a real dataset, at a real training budget, with `β` swept.

# %%
print(f"figures written to {FIGS}")
for f in sorted(FIGS.glob("*.png")):
    print(f"  {f.name}")
