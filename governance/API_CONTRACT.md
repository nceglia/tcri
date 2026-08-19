# TCRI — API & Function Responsibilities (Final)

*The definitive, publishable specification for the refactored, grafiti-mirrored `tcri` package (Door A: standalone scverse package). It folds the API-surface draft and the math/stats draft into a single artifact and applies every fix from the plan-correctness, math/stats, prior-vs-mean, missing-links, and GPU/optimization audits **inline** — no known-wrong math survives below. For each function you get: exact final signature and module home; responsibility; the precise math/stats it performs; how every argument dictates that math; and the exact return shape per argument combination.*

---

## The scope principle — what the package computes, and what it hands over

> **A comparison belongs in the API when producing it requires applying a metric at a level the
> public surface does not already expose. When the comparison is arithmetic on values the
> package has already computed, it belongs to the user.**

This is the test for whether a proposed function should exist. It governs §7 and §8 and takes
precedence over "it would be convenient."

**The two sides.**

*Inside — the package owes the computation.* A per-clone or per-phenotype quantity across two
covariate levels — `H(φ|c)` at `pre` versus at `post` for the **same** clone — requires the
metric evaluated inside the engine's draw loop, with clone identity aligned across two covariate
blocks that hold different row sets. The raw material is reachable
(`joint_distribution(covariate=None, n_samples=S)["table"]` is clone × phenotype at every level
for every draw, off one shared draw stack), but turning it into that quantity means the caller
**reimplementing the metric**: the support handling, the normalizer, the NaN and zero-mass
conventions. That is re-deriving a frozen definition in notebook code, which the metrics contract
exists to prevent — a delta computed by different code is not a delta of the same metric. So the
package must provide it.

*Outside — the package owes the material, not the answer.* A repertoire-level comparison —
`mutual_information(patient 1, pre)` versus `mutual_information(patient 1, post)` — is a
subtraction of two numbers already computed, already cached, already at the granularity the
question asks about. Nothing below the public surface is needed. And everything that makes such a
comparison *interesting* — how to handle repertoire imbalance between the levels, whether to pin a
normalizer, which replicates are comparable, which test — is a study-specific analysis decision.
Pre-empting it in the API imposes one answer on every user.

**The consequence for the payload.** `table` in the cache is not an implementation detail of
`result`. It **is** the population-level interface: the per-draw, per-item substrate, with
provenance, that a user builds their own replicate-level comparison on. `result` and `stats` are
the two reductions the package commits to; anything else is the user's to reduce.

**The test in practice.** A metric with an item axis (clone or phenotype) is defined per item, so
a cross-covariate comparison of it needs engine access — the API provides it. A metric with no
item axis is already the repertoire-level number, so a cross-covariate comparison of it is
subtraction — the user performs it. This is why `phenotypic_flux` and the entropy deltas are
functions while a "Δ mutual information" is not: MI has no item axis, so it would be a subtraction
dressed as a metric.

Corollary, and the reason this is worth writing down: **always know what a metric reduces to.**
It should be a clonotype or a phenotype wherever possible. If a proposed function cannot name its
unit of reduction, that is the signal to re-examine whether it belongs on this side of the line.

---

## 0. Conventions, notation, and resolved decisions

### 0.1 Layout principle

Mirrors `grafiti`: one file per topic (never a monolith), private cross-cutting helper modules, explicit `__all__` re-export, **never `import *`**. Five view directories map to short handles exactly as grafiti does — `model→ml`, `tools→tl`, `preprocessing→pp`, `plotting→pl`, `diagnostics→diag` — plus `utils→ut` (tcri-specific session I/O) and a new private `_compute/` numeric+device seam. The `tl` view lives in `tcri/tools/` (grafiti `tools↔tl`), superseding the earlier working name `metrics/`.

### 0.2 Notation

| Symbol | Meaning |
|---|---|
| $P$ | number of phenotypes (columns of every joint) |
| $C$ | number of clonotypes (rows of a joint at one covariate) |
| $M$ | number of covariate values (e.g. timepoints) |
| $\mathrm{ct}$ | a $(\text{clonotype},\text{covariate})$ group; `ct_count` = number of them |
| $p_{ct}\in\Delta^{P}$ | learned per-`ct` phenotype distribution; `uns[K.P_CT]`, shape `(ct_count, P)` |
| $m$ | $=\text{normalize}(q\_p\_ct\_raw)=\mathbb{E}_q[p_{ct}]=$ `uns[K.P_CT]` |
| $\ell_i\in\mathbb{R}^P$ | per-cell classifier logits, `obsm[K.X_LOGITS]` (already scaled by classifier temperature) |
| $s$ | local scale, `uns[K.LOCAL_SCALE]` (Dirichlet total concentration; controls draw spread only) |
| $g$ | `gate_prob`, `uns[K.GATE_PROB]` (scalar $\in[0,1]$, or `None` → additive) |
| $\tau_{\text{cls}}$ | classifier temperature, `uns[K.CLASSIFIER_TEMPERATURE]` |
| $T$ | analysis-time `temperature` argument |
| $\varepsilon$ | numerical floor; values stated in situ |

Index maps: `uns[K.CT_TO_COV]`, `uns[K.CT_TO_C]`, `uns[K.CT_ARRAY]` (per-cell ct index), `uns[K.COV_ARRAY]` (per-cell covariate index).

### 0.3 The one substrate every metric reads

Training learns a variational Dirichlet posterior per `ct` row:

$$q(p_{ct})=\mathrm{Dirichlet}(\alpha),\qquad \alpha=\mathrm{clamp}(s\cdot m,\ \min=10^{-3}),\quad m=\text{normalize}(q\_p\_ct\_raw)\in\Delta^{P}.$$

Where the clamp is **inactive** (the common case), $\alpha=s\cdot m$, so $\sum_k\alpha_k=s$ and the mean is closed-form:

$$\mathbb{E}_q[p_{ct}]=\frac{\alpha}{\sum_k\alpha_k}=m=\texttt{get\_p\_ct()}=\texttt{uns[K.P\_CT]}.$$

The local scale $s$ **cancels in the mean** and matters only for the spread of draws.

> **Persisted-substrate decision (fixes the double-tempering bug).** `uns[K.P_CT]` stores the **raw** normalized posterior mean $m=\text{normalize}(q\_p\_ct\_raw)$ with **no** guide-temperature and **no** analysis-temperature baked in. `TCRIModel.get_p_ct()` returns exactly this at its default `guide_temperature=1.0`. The analysis-time `temperature` argument (§0.9) is therefore the **single** tempering knob; it is never composed on top of a pre-tempered vector.

### 0.4 RESOLVED — the point estimate is the closed-form posterior MEAN (prior vs mean vs MAP vs draw)

This is the audit's required decision. `n_samples=0` returns the **closed-form variational posterior mean** $\mathbb{E}_q[p_{ct}]=m=\texttt{uns[K.P\_CT]}=\texttt{get\_p\_ct()}$ (temperature-adjusted per §0.9). It is read directly and **never sampled**.

Options analyzed and their disposition:

| Option | Verdict | Reason |
|---|---|---|
| **(a) Closed-form posterior mean** $\mathbb{E}_q[p_{ct}]=m$ | **ADOPTED** | Exact, closed-form, deterministic, reproducible; already stored in `uns[K.P_CT]` (zero sampling cost). It is the Bayes point estimator under squared-error loss and lies in the simplex interior. Invariant to $s$ (which only sets spread). |
| (b) Generative prior `clone_phen_prior` / archetype `mixture_concentration` | **REJECTED** | It is guide **initialization** / generative anchor, not what training learned; built from argmax hard labels (leakage/circularity); indexed at clone/archetype level, not the `ct` level metrics need. No metric may read it. |
| (c) MAP / posterior mode $(\alpha-1)/(\sum\alpha-P)$ | **REJECTED** | $\alpha_k=s\,m_k$ is routinely $<1$ (small $s$, $m_k<1$), so the mode sits on the simplex boundary / is undefined — unstable, discontinuous in $s$. |
| (d) Mean of many Dirichlet draws | **REJECTED** | Converges to (a) only up to Monte-Carlo noise; a strictly Rao-Blackwell-dominated, non-reproducible estimator of a quantity available in closed form. |

**Fate of the `posterior=` argument.** The *only* real difference between today's two engines was never prior-vs-posterior — both already used the posterior mean of $p_{ct}$; neither ever touched the generative prior. The real axis is **whether per-cell classifier logits $\ell_i$ are folded in**. Therefore:

- **`posterior=` is DELETED from all four metrics and from `phenotypic_flux`.** They always use the learned posterior (mean at `n_samples=0`, draws at `n_samples>0`) and, given an `adata`, compute the joint with logits folded in.
- **On the engine `joint_distribution`, the flag survives but is REDEFINED and RENAMED to `use_logits`** (alias `cell_informed=`), replacing both `posterior=` and the old `combine_with_logits=`. It is a **classifier-mixing switch**, not a prior/posterior switch: `use_logits=True` folds per-cell logits into $\log(\text{base})$ exactly like `predict()`; `use_logits=False` returns the `ct`-level table directly. **Both branches use the posterior mean/draws of $p_{ct}$; neither ever touches the generative prior.** The dead `mutual_information(posterior=False) → NotImplementedError` branch and the "prior path" are removed, not implemented.

### 0.5 Uniform sampling convention (`n_samples`)

| `n_samples` | Operation |
|---|---|
| `0` | **Deterministic point estimate.** Use the posterior mean $m$ (temperature-adjusted); **no Dirichlet draw**; bit-reproducible on repeat calls. Fixes today's latent bug where `mutual_information`/`flux` at `n_samples=0` returned *one random draw*. |
| `N>0` | **$N$ i.i.d. posterior draws** from the **exact guide posterior** $p_{ct}^{(s)}\sim\mathrm{Dirichlet}\big(\mathrm{clamp}(s\cdot\tilde p_{ct},\ \min=10^{-3})\big)$, seeded (§0.11). Adds a sample axis; posterior mean/HDI of the functional fall out. |

The old `point_estimate=` argument is **deleted**; `n_samples` is the only point-vs-draws knob.

> **Default `n_samples = 250`** on the engine, all four metrics, and the sampling `diag` functions — every default call **samples** and reports mean + interval (the honest posterior). `n_samples=0` (the deterministic point estimate) is **opt-in** for speed. (HDI is stabler at `n_samples ≳ 500` near boundaries — tunable.) Signatures ship this default; the `n_samples=0` shown in per-function code blocks below denotes the point-estimate *identity*, not the default.

> **Clamp fix (blocking).** Draws use the guide's clamped concentration `clamp(local_scale·m̃, 1e-3)` — **not** the bare `local_scale·m̃` or `local_scale·m̃+1e-8` variants that appear in today's three inconsistent engines and summarize a distribution the model never learned. See §0.10 for the induced (documented, intentional) mean discrepancy on committed clones.

### 0.6 Estimator honesty — plug-in vs posterior-mean (Jensen gap)

Entropy, MI, and KL/L1 flux are **nonlinear** functionals of $p$, so $\text{metric}(\mathbb{E}_q[p])\neq\mathbb{E}_q[\text{metric}(p)]$; the difference is a **Jensen gap**, *not* Monte-Carlo noise. Consequently the two numbers below are **different estimators by design** and must be documented and tested as such:

- **`n_samples=0`** computes the **plug-in-at-posterior-mean** estimator $\text{metric}(m̃)$.
- The **`mean` summary column of `n_samples>0`** estimates the **posterior mean of the functional** $\mathbb{E}_q[\text{metric}(p)]$.

Directions of the gap (Shannon entropy concave; L1/KL flux convex):

| Metric | Relationship |
|---|---|
| clonotypic / phenotypic entropy | plug-in $\ge$ posterior-mean (over-estimates) |
| phenotypic flux (L1, KL) | plug-in $\le$ posterior-mean (under-estimates); a clone with no real shift reads exactly $0$ at `n_samples=0` but strictly $>0$ in the `n_samples>0` mean |
| mutual information $=H(\phi)-H(\phi\mid c)$ | gaps partly cancel; **sign indeterminate**, magnitudes differ |

Additionally, because draws use the **clamped** concentration while the `n_samples=0` base is the **unclamped** mean $m$, the two also differ on **committed clones** (where $s\,m_k<10^{-3}$) by a second, clamp-induced term. **No conformance test may assert `n_samples=0 == mean(n_samples>0)`.** Docstrings label the two estimators distinctly.

### 0.7 Uniform return-shape rule

**One shape, for every metric and every combination of axes — see §7.10.** `{table, result,
stats}`, returned *and* stored under `uns[key_added or "tcri_<metric>"]` with a `params` block.

The table that used to live here gave four different return types keyed on `groupby` × `n_samples`
(scalar / draw-array+dict / tidy frame / tidy frame + summary), so a caller had to branch on their
own arguments to read their own result. That is gone.

> **`p_gt` is not a column of `result`.** Entropy, MI, L1 and KL flux are all $\ge 0$, so
> `P(draw>0) ≈ 1` and the quantity is vacuous on a single metric. It is not restored for the
> **delta** metrics either, even though there zero *is* a meaningful reference: `p_gt` is a
> posterior direction probability that reads as a frequentist p-value, its resolution is capped at
> `1/n_samples` (so `1.0` means "no draw crossed zero", not certainty), and per-item it invites
> filtering on the survivors. `hdi_low`/`hdi_high` already answer the direction question in the
> Bayesian idiom — an interval excluding zero — and the graded version is one line off the cached
> `table` for a caller who wants it, which is the scope principle applied.
>
> This keeps exactly one representative per inferential frame: `hdi_*` in `result` (Bayesian,
> within a replicate, over draws) and `p`/`stars` in `stats` (frequentist, between replicates,
> over groups). They answer different questions and neither can be dropped without dropping a
> question — see §7.10.

> **HDI fix.** The interval columns are a **true highest-density interval** (`hdi_low/hdi_high`), i.e. the narrowest interval containing `hdi_prob` mass — **not** the equal-tailed `np.percentile(x,[2.5,97.5])` mislabeled "HDI" in today's code. For the bounded, right-skewed entropy/flux posteriors (mass piled against the boundary for committed clones) the equal-tailed interval is materially wrong. HDIs from few hundred draws near a boundary are documented as unstable.

> **Draw-coherence rule (correctness).** For `n_samples>0`, **all clones within one sample share the same $p_{ct}$ draw** (one coherent joint per sample). Metrics iterate the `sample_id` level and compute the full-joint metric per draw, then summarize — never independent per-clone draws.

### 0.8 The `weighted` axis — KEPT as a dial (default `False`)

`weighted` is **retained** on the engine and all four metrics, **default `False`**: each clonotype is one unit on the simplex (a **repertoire-level / per-clonotype** statistic). `weighted=True` recovers the **cell-weighted** statistic (large clones dominate — a cell-level statistic). The two answer different biological questions (repertoire structure vs clonal expansion), can reorder samples, and are both valid — the choice is the user's, stated in each docstring. The refactor **fixes the current weight-lookup bug** (a `ct`-indexed `Counter` keyed with clone indices) and unifies the two engines so weighting is applied consistently (removing the inconsistency where `joint_distribution` normalized to sum 1 while `joint_distribution_posterior` returned un-normalized counts). `min`/uncertainty-coefficient normalization (§7.4) is robust to either mode (`MI ≤ min(H_c, H_p)` holds regardless).

> **Behavior-change note (changelog).** The default is now `weighted=False`, flipping `pl.mutual_information`'s displayed MI from the old cell-weighted default to per-clonotype. `weighted=True` restores the old behavior.

### 0.9 Temperature — single knob, one consistent placement

$T$ power-tempers the base **once**, identically in the mean and draw paths:

$$\tilde p_{ct}=\mathrm{softmax}\!\Big(\tfrac1T\log(m+\varepsilon)\Big)=\frac{m^{1/T}}{\sum_\phi m_\phi^{1/T}},\qquad \varepsilon=10^{-8}.$$

$T=1$ is the identity (renormalization only); $T<1$ sharpens, $T>1$ flattens. This fixes today's split where the two engines tempered at different stages.

- For `use_logits=True`, the combined per-cell logit is divided by $T$ **once**: $P(\phi\mid i)=\mathrm{softmax}\big(\text{combine}(\ell_i,\log\tilde b)/T\big)$. At **$T=1$** this reproduces `predict()` **bit-for-bit** (classifier temperature $\tau_{\text{cls}}$ is already baked into $\ell_i$; no second division). $T\neq1$ is an analysis-time temper that intentionally diverges from `predict()`; documented.
- For `n_samples>0`, draws are centered on the **re-tempered** $\tilde p_{ct}$; docstrings state that $T\neq1$ makes the sampled distribution a re-tempered object, not the raw learned posterior.

### 0.10 Reproducibility / seeding

`random_state` (`int | numpy.Generator | torch.Generator | None`) is added to `joint_distribution`, all four metrics, `compare_groups`'s bootstrap, and the sampling `diag` functions. It seeds a **`torch.Generator`** (and, on the GPU path, the CUDA RNG) because draws are `torch` Dirichlet — fixing the standing no-op where `seed` only touched `np.random`. `n_samples=0` is deterministic regardless of `random_state`.

---

## 1. Package tree

```
tcri/
  __init__.py               # explicit re-export + sys.modules aliases (tl/pp/pl/ml/ut/diag); top-level joint_distribution; NO import *
  _keys.py                  # single source of every uns/obsm/obs key string (constants only)
  _console.py               # leveled, silenceable logging over scanpy.logging (no raw ANSI, no _ascii_hist)
  _stats/                   # statistics, private
    _core.py                #   stars, AUROC+permutation, bootstrap, MWU, prob_direction, hdi
    _compare.py             #   compare_groups (INTERNAL; reached only via a metric's splitby)
  _compute/                 # NEW private numeric+device seam (grafiti-mirrored)
    _xp.py                  #   resolve_device, torch_device, asnumpy (torch-first, cupy optional later, CPU default)
    _joint.py               #   _joint_draws(p_ct, ct_to_cov, ct_to_c, ct_array, cov_array, *, ...) -> (blocks, n_draws)
    _distance.py            #   kl_divergence, l1_distance, js_divergence, phenotype_distance dispatcher
    _tables.py              #   metric_table / build_result / build_stats / collapse_to_replicates
                            #   (the plumbing every tools/ metric reduces through)
  model/                    # ml
    _model.py               #   TCRIModel
    _module.py              #   TCRIModule (pyro model/guide, get_latent, get_p_ct)
    _priors.py              #   MixtureDirichlet, VampPrior
    _classifier.py          #   PhenotypeClassifier
    _training.py            #   UnifiedTrainingPlan, build_archetypes
  preprocessing/            # pp
    _register.py            #   registration writers behind TCRIModel.to_anndata (all private)
    _clones.py              #   group_singletons, clone_size
  tools/                    # tl (mirrors pl by filename)
    _joint.py               #   joint_distribution (THE ENGINE)
    _entropy.py             #   clonotypic_entropy, phenotypic_entropy
    _mutual_information.py   #   mutual_information (+ private _mi_from_joint)
    _flux.py                #   phenotypic_flux
  plotting/                 # pl
    _base.py                #   _metric_boxplot, _finish
    _colors.py              #   tcri_colors, NA_COLOR, resolve_colors
    _entropy.py             #   clonotypic_entropy, phenotypic_entropy
    _mutual_information.py   #   mutual_information
    _flux.py                #   phenotypic_flux (sankey)
    _sankey.py              #   SankeyNode, _phenotype_mass_per_clone (private)
  diagnostics/              # diag (NEW)
    _ppc.py                 #   joint_distribution_ppc, phenotype_calibration, reconstruction_ppc, permutation_null
    _training.py            #   loss, archetypes
  utils/                    # ut
    _session.py             #   save_tcri_session, load_tcri_session (+ private helpers)
```

**Dropped entirely** (deleted — not relocated anywhere): `top_clone_umap`, `clone_size_umap`, `plot_phenotype_probabilities`, `gene_entropy`, `polar_plot`, `probability_ternary`. Out of the package to `docs/` (a figure script only): the model PGM (`build_nested_tcri_pgm` / `draw_tcri_pgm_nested`).

---

## 2. Top-level `__init__.py` and the `__all__` story

`__all__` is declared at **both** levels (grafiti pattern). Every impl module declares its own `__all__`; every view `__init__` imports symbols by name and re-declares an aggregate `__all__`. The root imports the six view packages, aliases them into `sys.modules`, and re-exports `joint_distribution` for prominence. **No `import *` anywhere** — numpy/pandas/torch and every `_helper` stay unexported; GPU libs are never imported at module top (§4.3).

```python
# tcri/__init__.py
from importlib.metadata import PackageNotFoundError, version as _version
try:
    __version__ = _version("tcri")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import model as ml
from . import diagnostics as diag
from . import utils as ut
from .tools import joint_distribution          # tcri.joint_distribution

import sys
sys.modules.update({f"{__name__}.{m}": globals()[m]
                    for m in ("tl", "pp", "pl", "ml", "diag", "ut")})

__all__ = ["tl", "pp", "pl", "ml", "diag", "ut", "joint_distribution", "__version__"]
```

| View `__init__` | `__all__` |
|---|---|
| `tools/__init__.py` (`tl`) | `joint_distribution`, `clonotypic_entropy`, `phenotypic_entropy`, `mutual_information`, `phenotypic_flux`, `compare_groups` |
| `preprocessing/__init__.py` (`pp`) | `group_singletons`, `clone_size` |
| `plotting/__init__.py` (`pl`) | `clonotypic_entropy`, `phenotypic_entropy`, `mutual_information`, `phenotypic_flux`, `resolve_colors`, `tcri_colors`, `NA_COLOR` |
| `model/__init__.py` (`ml`) | `TCRIModel` |
| `diagnostics/__init__.py` (`diag`) | `joint_distribution_ppc`, `phenotype_calibration`, `reconstruction_ppc`, `permutation_null`, `loss`, `archetypes` |
| `utils/__init__.py` (`ut`) | `save_tcri_session`, `load_tcri_session` |

Private modules (`_keys`, `_console`, `_stats`, `_distance`, `_compute/*`) and every private symbol (`_mi_from_joint`, `_metric_boxplot`, `_finish`, `SankeyNode`, `_phenotype_mass_per_clone`, all `_register.py` writers) are **absent** from every `__all__`.

---

## 3. Shared private helper modules

### 3.1 `tcri/_keys.py` — canonical AnnData key registry (constants only)

Import as `from tcri import _keys as K`; no key literal lives anywhere else.

| Constant | Value | Slot | Meaning |
|---|---|---|---|
| `K.METADATA` | `"tcri_metadata"` | `uns` | dict: `covariate_col`, `clone_col`, `phenotype_col`, `batch_col` (single scheme; replaces dual `tcri_clone_key`/`tcri_phenotype_key`) |
| `K.PHENOTYPE_CATEGORIES` | `"tcri_phenotype_categories"` | `uns` | ordered phenotype categories |
| `K.CLONOTYPE_CATEGORIES` | `"tcri_clonotype_categories"` | `uns` | ordered clonotype categories |
| `K.COVARIATE_CATEGORIES` | `"tcri_covariate_categories"` | `uns` | ordered covariate categories |
| `K.P_CT` | `"tcri_p_ct"` | `uns` | `(ct_count, P)` **raw** posterior mean $m=\mathbb{E}_q[p_{ct}]$ (no temper baked in) |
| `K.CT_TO_COV` | `"tcri_ct_to_cov"` | `uns` | `(ct_count,)` ct→covariate index |
| `K.CT_TO_C` | `"tcri_ct_to_c"` | `uns` | `(ct_count,)` ct→clonotype index |
| `K.CT_ARRAY` | `"tcri_ct_array_for_cells"` | `uns` | `(n_obs,)` per-cell ct index |
| `K.COV_ARRAY` | `"tcri_cov_array_for_cells"` | `uns` | `(n_obs,)` per-cell covariate index |
| `K.LOCAL_SCALE` | `"tcri_local_scale"` | `uns` | scalar Dirichlet total concentration $s$ (draws only) |
| **`K.GATE_PROB`** | `"tcri_gate_prob"` | `uns` | **NEW** — scalar gate $g\in[0,1]$ or `None`; required for gate-aware `use_logits=True` parity with `predict()` |
| **`K.CLASSIFIER_TEMPERATURE`** | `"tcri_classifier_temperature"` | `uns` | **NEW** — $\tau_{\text{cls}}$; persisted for reproducibility/auditing (already baked into stored logits) |
| `K.X_LATENT` | `"X_tcri"` | `obsm` | `(n_obs, n_latent)` encoder posterior-mean latent |
| **`K.X_LOGITS`** | `"X_tcri_logits"` | `obsm` | `(n_obs, P)` classifier logits — **part of the canonical write-set** (§5.1); the `use_logits=True` engine path requires it |
| `K.X_PROBABILITIES` | `"X_tcri_probabilities"` | `obsm` | `(n_obs, P)` per-cell phenotype probabilities (`predict()`) |
| `K.PHENOTYPE_LABEL` | `"tcri_phenotype"` | `obs` | per-cell argmax hard label |
| `K.CLONE_SIZE` | `"clone_size"` | `obs` | per-cell clone cell-count |
| `K.OBS_INDICES` | `"indices"` | `obs` | per-cell integer index — **registration glue** written by `setup_anndata` (§5.1); not analysis output |

> The AnnDataManager is **no longer** stashed in `uns` (`tcri_manager` retired) — deleting the `write_adata_safely` / `_pop_nonserializables` hack. The stash lived in `setup_anndata`, so it is removed there (§5.1).

### 3.2 `tcri/_console.py` — leveled, silenceable logging (private)

Thin wrappers over `scanpy.logging`; respects scanpy verbosity. Raw ANSI prints and `_ascii_hist` (and every `graph=`/ASCII-histogram code path) are **deleted**.

| Signature | Responsibility |
|---|---|
| `info(msg, *, deep=None)` | `scanpy.logging.info`; silenced by scanpy verbosity. |
| `warning(msg)` | `scanpy.logging.warning`. |
| `success(msg)` | `scanpy.logging.hint`. |
| `done(msg="done")` | terminal completion line. |

### 3.3 `tcri/_stats/` — significance + posterior-comparison statistics (private)

| Signature | Responsibility / math |
|---|---|
| `stars(p)` | $p<10^{-4}\to$`****`; $<10^{-3}\to$`***`; $<10^{-2}\to$`**`; $<0.05\to$`*`; else `ns`. |
| `mann_whitney(a, b, *, alternative="two-sided")` | Mann–Whitney $U=\sum_{i,j}\mathbb1[a_i>b_j]+\tfrac12\mathbb1[a_i=b_j]$; two-sided $p$ from the rank-sum null (`scipy.stats.mannwhitneyu`). Returns `(U, p)`. |
| `prob_direction(delta)` | **Signed-contrast only.** Given a paired difference-draw vector $\Delta$: $p_{gt}=\frac1N\sum_s\mathbb1[\Delta^{(s)}>0]$, $p_{lt}=\frac1N\sum_s\mathbb1[\Delta^{(s)}<0]$. Returns `(p_gt, p_lt)`. |
| `hdi(samples, *, hdi_prob=0.94)` | **True** highest-density interval: over sorted samples, the **narrowest** window containing $\lceil hdi\_prob\cdot N\rceil$ points. Returns `(low, high)`. Documented unstable near a bounded posterior's boundary for small $N$. |
| `summarize(samples, *, hdi_prob=0.94)` | Reducer for a **raw metric** draw vector → `{mean, sd, hdi_low, hdi_high}`. **No `p_gt`** (vacuous for $\ge0$ metrics, §0.7). |
| `auc_and_label_permutation(scores, labels, *, pos_label=None, n_perm=200_000, seed=42, max_exact=200_000)` | Observed ROC-AUC + two-sided permutation $p$: exact enumeration when $\binom{n}{k}\le$`max_exact`, else Monte-Carlo; $p_{\text{perm}}=\text{mean}(|\mathrm{AUC}_{\text{perm}}-0.5|\ge|\mathrm{AUC}_{\text{obs}}-0.5|)$. Returns `(auc, p, perm_stats, mode)`. |
| `bootstrap_auc(scores, labels, *, pos_label=None, n_boot=5000, seed=42)` | Resample cells with replacement (reject draws missing a class), recompute AUROC, return the 2.5/97.5 quantiles. Returns `np.array([lo, hi])`. |

### 3.4 `tcri/_compute/_distance.py` — phenotype-distribution distances (private)

Dedupes the old module-level `dkl` and `flux.dkl_func`; **one base (bits, $\log_2$) and one $\varepsilon=10^{-12}$ library-wide**, matching entropy/MI.

| Signature | Responsibility / math |
|---|---|
| `l1_distance(p, q)` | $\sum_i|p_i-q_i|$; symmetric; range $[0,2]$ on the simplex. Defensively renormalizes inputs. Returns float. |
| `kl_divergence(p, q, *, base=2.0, eps=1e-12)` | $\mathrm{KL}(p\Vert q)=\sum_i p_i\log_2\frac{p_i}{q_i}$; clip to $[\varepsilon,1]$ then renormalize. **Asymmetric**, range $[0,\infty)$. **Single base fixed to $\log_2$ / single $\varepsilon$** (repairs the natural-log-vs-$\log_2$ and $10^{-10}$-vs-$10^{-15}$ divergence between the two dead copies). Returns float. |
| `js_divergence(p, q, *, base=2.0, eps=1e-12)` | **NEW** Jensen–Shannon $=\tfrac12\mathrm{KL}(p\Vert r)+\tfrac12\mathrm{KL}(q\Vert r)$, $r=\tfrac12(p+q)$; **symmetric, bounded $[0,1]$ bit** — the recommended symmetric shift measure. Returns float. |
| `phenotype_distance(p, q, *, metric="l1")` | Dispatcher: `"l1"`→`l1_distance`; `"kl"`/`"dkl"`→`kl_divergence` (directional, unbounded, bits); `"jsd"`→`js_divergence`; a callable `f(p,q)`; else `ValueError`. |

---

## 4. `tcri/_compute/` — numeric + device seam (NEW, private)

The engine's numeric core is written **once** as a batched, device-routable function so the acceleration is additive and reversible, and so the four metrics share one joint-draw stack.

### 4.1 `_xp.py` — the device seam (grafiti reference, copied 1:1)

| Signature | Responsibility |
|---|---|
| `resolve_device(device)` | `None`/`"cpu"`→`"cpu"`; `"mps"`→`"cpu"`; `"cuda"`/`"gpu"`/`"auto"`→GPU **iff** the backend imports AND a device is present (`getDeviceCount()>0`), else CPU. Explicit `"cuda"` warns on fallback; `"auto"`/`"gpu"` silent; unknown warns. |
| `torch_device(device)` | Return the resolved `torch.device` (`resolve_device` maps the ladder to cpu/cuda). torch-first (already a hard dep → zero new deps); cupy optional later. GPU libs imported **lazily inside** the function. |
| `asnumpy(x)` | Host-boundary shim: `cupy.asnumpy(x)` / `x.cpu().numpy()` / `np.asarray(x)`. Every accelerated function returns a plain numpy array. |

### 4.2 `_joint.py` / `_reduce.py` — the batched core

- **`_joint_draws(p_ct, ct_to_cov, ct_to_c, ct_array, cov_array, *, local_scale, n_samples, temperature, use_logits, covariate_idx, logits, gate_prob, weighted, random_state, device) -> (blocks, n_draws)`** — the adata-unpacking lives in the `tools/_joint` wrapper; this core takes **decomposed uns arrays** and returns a **list of per-covariate `(cov_idx, clone_idx, J[S, n_rows, P])` blocks** plus the draw count (`covariate=None` stacks variable-length per-covariate blocks). Draws all `n_samples` Dirichlet samples over **all ct rows in one batched kernel** from `clamp(s·m̃, 1e-3)` (the shared-draw invariant), then slices per covariate; softmaxes the (optionally gated) per-cell combination batched on the leading axis; reduces per clone with a **scatter-add** (`torch.index_add_`) instead of a per-draw `pandas.groupby` — the dominant win. `float64` accumulators for CPU/GPU parity; `asnumpy` at the boundary. **Phase-6 (with the GPU path / `_reduce` wiring):** on-device per-row-sum $\approx1$ validation + chunking over cells/draws (§7.4 guardrails 5/7/8).
- **`_reduce.py`** — batched `entropy`, `mutual_information`, `distance` as `xlogx`/outer-product reductions over the whole stack (no per-draw scipy call, no per-clone `.loc`), plus the `summarize`/`hdi` reduction over the sample axis.

### 4.3 GPU guardrails (replicated uniformly from grafiti)

Lazy GPU imports (never at module top — `import tcri` never touches a GPU lib; the old module-top `import umap` is moved inside its function); GPU deps never in `install_requires` (CPU path always fully functional); permissive device ladder with device-count verification; `asnumpy`/`output_type="numpy"` at every boundary; `try/except` degrade-to-CPU reporting which backend ran; `float64` where parity must hold; validate-before-compute on **per-row** invariants; chunked reductions to bound host+device memory on the large `[n_samples, n_cells, P]` tensor. `n_samples=0` performs **zero** draws (closed-form read of `uns[K.P_CT]`). cuML UMAP for the one-off latent embedding sits behind a `_use_gpu(device)` gate with the `umap-learn` CPU fallback; layouts differ (both valid), documented, not claimed bit-identical.

---

## 5. `tcri.ml` — model (`model/`)

### 5.1 `model/_model.py`

**`class TCRIModel(BaseModelClass)`** — register → build → train → extract → write.

| Method (signature) | Responsibility |
|---|---|
| `@classmethod setup_anndata(cls, adata, *, layer=None, clonotype_key="unique_clone_id", phenotype_key="phenotype_col", covariate_key="timepoint", batch_key="patient", **kwargs)` | **Registration only** — register clonotype/phenotype/covariate/batch/count fields with scvi and store the layer. **Writes `obs["indices"]=range(n)` and registers it** (`CategoricalObsField`) — this is registration glue that `training_step`/`validation_step` consume via `batch["indices"]`; it is **not** analysis output. Invariant is **"no analysis/label `obs` mutation"** (labels/probabilities are written only by `to_anndata`). **Removes the `uns["tcri_manager"]` stash** (was here), deleting the need for `write_adata_safely`. |
| `__init__(self, adata, *, n_latent=128, n_hidden=128, n_layers=3, classifier_n_layers=3, global_scale=5.0, local_scale=3.0, prior_temperature=1.0, guide_temperature=1.0, use_enumeration=False, patience_epochs=300, classifier_hidden=128, classifier_dropout=0.1, n_pseudo_obs=10, K=10, phenotype_weights=None, gate_prob=None, kl_weight_max=1.0, guide_init_scale=10.0, classifier_temperature=1.0, **kwargs)` | Build the empirical clone→phenotype prior + KMeans archetypes + clonotype/covariate index maps + class weights, then construct/prime `TCRIModule`. Note `gate_prob=None` default ⇒ ungated model; the gate-parity guarantee is only exercised when a gate is trained. |
| `train(self, *, max_epochs=1000, batch_size=1000, lr=1e-3, reconstruction_loss_scale=1e-3, n_steps_kl_warmup=2000, **kwargs)` | 0.9/0.1 split, `UnifiedTrainingPlan`, `TrainRunner` with `elbo_validation` early stopping. |
| `get_latent_representation(self, adata=None, *, indices=None, batch_size=None) -> np.ndarray` | Batched encode to the `(n_cells, n_latent)` posterior-mean latent. |
| `predict(self, adata=None, *, batch_size=256, eps=1e-8) -> pd.DataFrame` | **(renamed from `get_cell_phenotype_probs`)** Per-cell phenotype-probability `DataFrame` (index = `adata.obs_names`, columns = phenotypes). Combines classifier logits with $\log p_{ct}$ (gate or additive), matching training (scvi/CellAssign idiom). **Reference the `use_logits=True` joint must reproduce at $T=1$** (§0.9, §7.1). Uses an **order-preserving loader** (shuffle=False / sequential sampler) and the registered `indices` field so ct-lookup and barcode labels cannot drift. |
| `get_p_ct(self, *, guide_temperature=1.0) -> np.ndarray` | Return the learned `(ct_count, P)` posterior mean $m=\text{normalize}(q\_p\_ct\_raw)$. At the default `guide_temperature=1.0` this equals `uns[K.P_CT]` exactly. |
| `to_anndata(self, adata=None, *, batch_size=256, compute_umap=False) -> AnnData` | **(replaces the heavy `register_model`; signature matches the frozen `_contract.pyi` as of PR4 — the canonical key names come from `_keys`, NOT per-call arguments)** Thin writer of the **canonical minimum**: metadata + categories (from registry); `X_tcri` latent; **`obsm[K.X_LOGITS]` per-cell logits** (restored — the `use_logits=True` engine path hard-requires them); `predict()` probs + argmax hard labels; `p_ct` (+ `ct_to_cov`, `ct_to_c`, per-cell ct/cov arrays); **`local_scale`**, **`gate_prob`**, **`classifier_temperature`**. No manager stash; no other writes. |

> Relocated off the model: `plot_archetypes`→`diag.archetypes`; `plot_loss`→`diag.loss`. `boost_phenotype_prior`, `use_gate` remain internal.

### 5.2 `model/_module.py`

**`class TCRIModule(PyroBaseModuleClass)`** — Pyro CVAE with hierarchical clonotype→(clonotype×covariate) Dirichlet priors and a phenotype classifier. *(internal)*

| Member (signature) | Responsibility |
|---|---|
| `__init__(self, n_input, n_latent, P, n_batch, *, global_scale=10.0, local_scale=5.0, prior_temperature=1.0, guide_temperature=1.0, gate_prob=0.5, mixture_concentration=None, n_pseudo_obs=10, use_enumeration=False, classifier_hidden=128, classifier_dropout=0.1, classifier_n_layers=3, n_hidden=128, n_layers=3, class_weights=None, kl_weight_max=1.0, guide_init_scale=10.0, classifier_temperature=1.0)` | Construct encoder/decoder/classifier/VampPrior, `px_r`; register empty two-level buffers + class weights (`mixture_concentration` required). |
| `prepare_two_level_params(self, clone_phen_prior_mat, ct_to_c, ct_to_cov, ct_array_for_cells, cov_array_for_cells, *, eps=1e-6)` | Normalize/temperature the clone-phenotype prior; register two-level index buffers. |
| `model(self, x, batch_idx, log_library, ...)` | Generative: sample $p_c$ (MixtureDirichlet), $p_{ct}$ (Dirichlet centered at $p_c$), latent $z$ (VampPrior), ZINB gene obs. |
| `guide(self, x, batch_idx, log_library, ...)` | Guide: learnable Dirichlet params $q(p_c)$, $q(p_{ct})$ with **`clamp(min=1e-3)`** on the concentration (the floor the draw path must reproduce, §0.5); Normal $q(z)$ from the encoder. |
| `get_latent(self, tensor_dict) -> torch.Tensor` | Encode a batch to posterior-mean latent $z_{\text{loc}}$. |
| `get_p_ct(self) -> torch.Tensor` | Read `q_p_ct_raw` from the (process-global) param store; return the row-normalized `(ct_count, P)` posterior mean. |
| `use_gate(self) -> bool` (property) | `True` when `gate_prob is not None`. |
| `@staticmethod _get_fn_args_from_batch(tensor_dict) -> tuple` | Extract `(x, batch_idx, log_library)` from a scvi batch dict. |

> **Param-store caveat (documented).** `get_p_ct` reads the **process-global** Pyro param store (`q_p_ct_raw`); loading two sessions in one process clobbers it. `to_anndata` and every `diag` PPC must be called immediately after the intended model's params are set; `load_tcri_session` sets the store before any `get_p_ct`/`to_anndata` call. Single-model-per-process otherwise.

### 5.3 `model/_priors.py`

**`class MixtureDirichlet(dist.TorchDistribution)`** *(internal)* — clonotype prior $p_c$. Members: `__init__(self, mixture_weights, concentration, validate_args=None)`; `sample`; `log_prob` (log-sum-exp of component Dirichlet log-probs); `score_parts` (returns `(log_prob, 0, 0)` → reparam-free); `__call__` (alias for `sample`).

**`class VampPrior(torch.nn.Module)`** *(internal)* — VampPrior over $z$. Members: `__init__(self, pseudo_inputs, encoder)`; `get_mixture` (uniform `MixtureSameFamily` of `Independent` Normals); `log_prob(self, z)`; `sample`.

### 5.4 `model/_classifier.py`

**`class PhenotypeClassifier(nn.Module)`** *(internal)* — `__init__(self, n_latent, classifier_hidden, P, *, num_layers=3, dropout_rate=0.1, temperature=1.0)`; `forward(self, x)` returns MLP logits divided by `temperature` ($\tau_{\text{cls}}$, baked into the stored logits).

### 5.5 `model/_training.py`

**`class UnifiedTrainingPlan(PyroTrainingPlan)`** *(internal)* — `__init__(self, module, *, n_steps_kl_warmup=1000, reconstruction_loss_scale=1e-2, num_particles=5, optimizer_config=None, class_weights=None, **kwargs)`; `loss` (property); `configure_optimizers`; `training_step`; `validation_step` (logs `elbo_validation`).

**Module function:** `build_archetypes(c2p_mat, *, K=10) -> tuple[np.ndarray, np.ndarray]` — KMeans-cluster clone→phenotype rows into `K` normalized archetype centroids. **Returns `(centers, labels)`** — labels are retained so `diag.archetypes` can reproduce the cluster-ordered heatmap. `K` default is **10** (aligned to `TCRIModel`, repairing the former `K=4` default mismatch).

---

## 6. `tcri.pp` — preprocessing (`preprocessing/`)

### 6.1 `preprocessing/_clones.py` — public

| Signature | Responsibility |
|---|---|
| `group_singletons(adata, *, clonotype_key="trb", groupby="patient", target_col="trb_unique", min_clone_size=10) -> AnnData` | Collapse clones smaller than `min_clone_size` (per `groupby`) into `"Singleton_{group}"` labels in `target_col`. **Ordering invariant (documented + enforced):** any clone relabeling must run **before** `setup_anndata`/`train`, else the learned clonotype categories and `p_ct`'s `ct_to_c` map desync from `obs`; `setup_anndata` refuses registration if a later relabel is detected. |
| `clone_size(adata, *, key_added="clone_size", return_counts=False)` | Per-clone cell counts, written per cell into `obs[key_added]`. **Reads `uns[K.METADATA]["clone_col"]`** (migrated off the retired `tcri_clone_key` in the same change that stops writing it). |

**Private inner:** `group_singletons.collapse_singleton(row)`.

### 6.2 `preprocessing/_register.py` — private (the `to_anndata` writers)

Called only by `TCRIModel.to_anndata`; folds in the old `register_phenotype_key` / `register_clonotype_key` / `_compute_logits_and_prior`.

| Signature | Responsibility |
|---|---|
| `_write_metadata(adata, model)` | `uns[K.METADATA]` (single scheme) + the three category lists from the registry. |
| `_register_clonotype_key(adata, clonotype_key, *, order=None)` | Register the clonotype `obs` column + ordered categories. |
| `_register_phenotype_key(adata, phenotype_key, *, order=None)` | Register the phenotype `obs` column + ordered categories. |
| `_write_latent(adata, model, *, latent_key="X_tcri", batch_size=256)` | Encoder posterior-mean latent → `obsm`. |
| `_write_logits(adata, model, *, logits_key="X_tcri_logits", batch_size=256)` | **Per-cell classifier logits → `obsm[K.X_LOGITS]`** (canonical; required by the default engine path). |
| `_write_predictions(adata, model, *, predictions_key="X_tcri_probabilities", label_key="tcri_phenotype", batch_size=256)` | `predict()` probs → `obsm`; argmax hard labels → `obs`. |
| `_write_p_ct(adata, model)` | `p_ct`, `ct_to_cov`, `ct_to_c`, per-cell ct/cov arrays, `local_scale`, **`gate_prob`**, **`classifier_temperature`** → `uns`. |
| `_compute_logits_and_prior(model, adata, *, batch_size=256, eps=1e-8) -> tuple[np.ndarray, np.ndarray]` | Run encoder+classifier to extract per-cell logits and $\log p_{ct}$ from `get_p_ct()`. |

---

## 7. `tcri.tl` — tools / metrics (`tools/`)

### 7.1 Engine — `tools/_joint.py`

```python
joint_distribution(
    adata, *, covariate=None, n_samples=0, use_logits=True, weighted=False,
    clones=None, temperature=1.0, random_state=None, device=None,
    key_added=None, inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```
Re-exported top-level as `tcri.joint_distribution`. Unifies today's `joint_distribution` + `joint_distribution_posterior`.

**(a) Responsibility.** Produce the clone×phenotype distribution (the substrate every metric consumes) at a covariate value from the learned variational posterior of $p_{ct}$ — a deterministic posterior-mean table or posterior draws. Provenance in `df.attrs["params"]` (and, for cache-friendliness, in a flat `_provenance` sidecar column, §7.7).

**(b) Math.** Select `ct` rows for covariate $m$ via `ct_to_cov`; each clone maps to exactly one `ct`, so rows index by clonotype. Temperature-temper the base **once** (§0.9): $\tilde p_{ct}=\mathrm{softmax}(\tfrac1T\log(m+10^{-8}))$.

*Base per point/draw:*
- `n_samples=0`: base $b=\tilde p_{ct}$ (posterior mean; deterministic).
- `n_samples=N`: bases $b^{(s)}\sim\mathrm{Dirichlet}\big(\mathrm{clamp}(s\cdot\tilde p_{ct},\ \min=10^{-3})\big)$, $s=$`local_scale`, seeded (§0.10). One coherent joint per sample (§0.7).

*`use_logits=False` (ct-level table):* row $c=b$. For `n_samples=0` this is exactly $\tilde p_{ct}$; at $T=1$ it equals `uns[K.P_CT]` restricted to the covariate — the clean closed-form identity used in tests.

*`use_logits=True` (fold per-cell logits; matches `predict()`):* per cell $i$ with clonotype $c(i)$, group $\mathrm{ct}(i)$, base $b_{\mathrm{ct}(i)}$:
$$P(\phi\mid i)=\mathrm{softmax}\!\Big(\tfrac1T\,\mathrm{combine}\big(\ell_i,\ \log(b_{\mathrm{ct}(i)}+\varepsilon)\big)\Big),\quad
\mathrm{combine}=\begin{cases}g\,\ell_i+(1-g)\log b & g=\texttt{gate\_prob}\neq\text{None}\\ \ell_i+\log b & \text{additive}\end{cases}$$
then $J[c,\phi]=\sum_{i\in c}P(\phi\mid i)$, row-normalize. At $T=1$, `n_samples=0`, this equals `predict()` aggregated per clone — **bit-for-bit**, gate-aware (fixes the standing disagreement where metrics used one Dirichlet *draw* and *never* applied the gate).

*`covariate=None`:* compute the joint for **all** covariate values from a **single shared draw** per sample (the draw-once invariant, §7.8), stacking a covariate axis.

*`groupby=g`:* restrict by cell/clone masks into the **full-space** `uns` arrays (never by slicing the AnnData — see the guard note below), computing per group value and stacking a group axis.

**(c) Arguments → math.**

| Argument | Effect |
|---|---|
| `covariate` | Selects `ct` rows via `ct_to_cov`. `None` → all covariates in one shared-draw pass. |
| `groupby` | Separate joint per group value (adds a group axis), implemented by **restriction over full adata**, not slicing. Requires the cell-informed path or a clone-constant key (see semantics note). |
| `n_samples` | `0` → posterior-mean table (deterministic); `N` → $N$ clamped-Dirichlet draws. Only place `local_scale` enters. |
| `use_logits` | `True` → fold logits with $\log b$ (gate-aware), aggregate per clone, row-normalize; `False` → `ct`-level $\tilde p_{ct}$ rows. Neither is the generative prior. |
| `clones` | Filters rows to the listed clonotypes; with `use_logits=True` also restricts aggregated cells; final reindex to the exact list (absent clones → dropped, **not** all-zero rows — see §7.2 fix). |
| `temperature` | $T$ tempers the base once (§0.9). $T=1$ identity; at $T=1$, `use_logits=True` reproduces `predict()`. |
| `random_state` | Seeds the torch (and CUDA) Dirichlet generator for `n_samples>0`. Ignored at `n_samples=0`. |
| `device` | Routes the numeric core through `_compute/_xp` (CPU / torch-CUDA / cupy); result is always host numpy. |

**(d) Return shape** — `pandas.DataFrame`, columns = phenotype categories.

| `covariate` | `groupby` | `n_samples` | Index / axes |
|---|---|---|---|
| set | unset | `0` | rows = clonotype id; `(C_m, P)` |
| set | unset | `N>0` | MultiIndex (clonotype, `sample_id`); `(C_m·N, P)` |
| set | set | `0` | MultiIndex (group, clonotype) |
| set | set | `N>0` | MultiIndex (group, clonotype, `sample_id`) |
| `None` | — | — | adds a leading covariate level to any of the above |

> **groupby ↔ alignment guard (blocking fix).** `joint_distribution_posterior` hard-raises if per-cell `uns[...array_for_cells]` lengths $\neq$ `n_obs`. Passing a **sliced** AnnData (today's `tcri_boxplot` pattern) trips this. groupby is therefore implemented by **positional cell/clone masks into the full-space `uns` arrays** + `clones=`, never by handing a slice to the engine. `_metric_boxplot` (§8.5) is rewritten off the slice-and-call pattern.

> **groupby ↔ covariate semantics.** `p_ct` is indexed by `ct=(clonotype, covariate)` only. A `groupby` key that is **not** functionally determined by clonotype-at-fixed-covariate (e.g. a tissue cross-cutting one clone) is unrepresentable in the `use_logits=False` table and is only well-defined on the cell-informed `use_logits=True` path. The engine **requires the cell-informed path for such keys**, or requires the key be clone-nested / constant within a clone×covariate; it errors/warns on `use_logits=False` + a non-clone-determined groupby. The whole per-group scheme assumes **clones are disjoint across groups** (a TCR clone never spans two patients) — stated explicitly.

`__all__ = ["joint_distribution"]`

### 7.2 `tools/_entropy.py` — `clonotypic_entropy`

```python
clonotypic_entropy(
    adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    n_clones_ref=None, random_state=None, device=None, key_added=None,
    inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```

**(a) Responsibility.** For each phenotype $\phi$ (at covariate $m$), the normalized Shannon entropy of the distribution over clonotypes carrying that phenotype, $H[P(c\mid\phi,m)]$ — spread of a phenotype across clones. **Repertoire-level (uniform-clonotype prior; §0.8).**

**(b) Math.** From joint $J$, take column $\phi$ over the **supported** clones only (absent/all-zero clones excluded — see fix), normalize, take entropy in bits:
$$v_c=\frac{J[c,\phi]}{\sum_{c'\in\text{supp}}J[c',\phi]},\qquad H_\phi=-\sum_{c\in\text{supp}} v_c\log_2 v_c.$$
If `normalized`: divide by $\log_2 C_{\text{den}}$ where $C_{\text{den}}$ = number of **supported** clones (default), or `n_clones_ref` if given (a fixed reference for cross-group comparability). No division when $C_{\text{den}}\le1$. Base fixed to 2.

- **Estimator (§0.6):** `n_samples=0` = plug-in $H_\phi(m̃)$; `n_samples>0` `mean` = $\mathbb{E}_q[H_\phi]$ (plug-in $\ge$ posterior-mean for entropy). Documented as distinct.
- **Fixes:** deterministic `n_samples=0` (no forced draw); `normalized` exposed (was hard-normalized); base fixed to 2; **absent/zero-support clones excluded before normalizing** (no $\varepsilon$-clip fabricating uniform mass or inflating $C$).

**(c) Arguments → math.**

| Argument | Effect |
|---|---|
| `adata` | Compute $J$ internally via §7.1 (`use_logits=True`; `covariate` required). AnnData only — the `adata_or_jd` union and its precomputed-joint fast path are deleted (§7.9). |
| `covariate` / `groupby` | Condition $m$; per-group entropy → tidy rows (group × phenotype). |
| `n_samples` | `0` → plug-in per phenotype; `N` → per-draw + summary. |
| `temperature` | Tempers $J$ before the column is read. |
| `clones` | Restricts the clone set → changes support and the default $\log_2 C_{\text{den}}$. |
| `normalized` / `n_clones_ref` | `True` → divide by $\log_2 C_{\text{den}}$ (range $[0,1]$); `n_clones_ref` fixes the denominator for comparability; `False` → raw bits. |
| `random_state` / `device` | Seeding / backend routing for `n_samples>0`. |

**(d) Return shape.** `n_samples=0`, no `groupby` → `Series` over phenotypes; `n_samples>0`, no `groupby` → per-phenotype `mean, sd, hdi_low, hdi_high`; `groupby` → tidy DataFrame row per (group, phenotype) [+ summary]. **Absent phenotype → `NaN`, not 0.**

> **Comparability note.** Because the default denominator is group-specific, normalized clonotypic entropy is **within-group** unless `n_clones_ref` (a common denominator) is supplied. The `pl` twin defaults cross-group plots to a common `n_clones_ref`.

### 7.3 `tools/_entropy.py` — `phenotypic_entropy`

```python
phenotypic_entropy(
    adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    random_state=None, device=None, key_added=None, inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```

**(a) Responsibility.** For each clonotype $c$, the normalized Shannon entropy of its phenotype distribution $H[P(\phi\mid c,m)]$ — plasticity vs commitment.

**(b) Math.** Row $c$ = $P(\phi\mid c)$; normalize over phenotypes, entropy in bits, divide by $\log_2 P$ if `normalized` and $P>1$:
$$p_\phi=\frac{J[c,\phi]}{\sum_{\phi'}J[c,\phi']},\quad H_c=-\sum_\phi p_\phi\log_2 p_\phi.$$
Estimator convention as §0.6 (plug-in at `n_samples=0`). **Critical bug fix:** a clone present in `obs` but with **zero posterior mass** returns **`NaN`** and is excluded — it is **not** reindexed to zeros, $\varepsilon$-clipped, and normalized to uniform → spurious $H=1.0$. Base fixed to 2; `normalized` exposed.

**(c) Arguments → math.** As §7.2, except the normalizer is $\log_2 P$ (depends on $P$, not clone count), so `clones` does not change the divisor; $P\le1\Rightarrow$ divisor 1.

**(d) Return shape.** `n_samples=0`, no `groupby` → `Series` over clonotypes; `n_samples>0` → per-clone `mean, sd, hdi_low, hdi_high`; `groupby` → tidy DataFrame row per (group, clone) [+ summary].

### 7.4 `tools/_mutual_information.py` — `mutual_information` (+ kernel)

```python
mutual_information(
    adata, *, covariate=None, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    normalize_mode='min', random_state=None, device=None, key_added=None,
    inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```

**(a) Responsibility.** $I(c;\phi\mid m)$ in bits — strength of clone–phenotype coupling. Optionally normalized to $[0,1]$.

**(b) Math — kernel `_mi_from_joint(pxy, *, normalized, mode="min") -> float`.** Normalize the whole joint: $\text{pxy}=J/\sum J$; marginals $p_x=\sum_\phi\text{pxy}$, $p_y=\sum_c\text{pxy}$.
$$I=\sum_{c,\phi}\text{pxy}\,\log_2\frac{\text{pxy}+\varepsilon}{p_x p_y^\top+\varepsilon}\ \text{(bits)},\quad \varepsilon=10^{-15}.$$
With $H_c=-\sum p_x\log_2 p_x$, $H_p=-\sum p_y\log_2 p_y$:
$$I_{\text{norm}}=\frac{I}{D},\quad D=\begin{cases}\min(H_c,H_p) & \texttt{mode="min"}\ \text{(coefficient of constraint; default)}\\[2pt]\tfrac12(H_c+H_p) & \texttt{mode="average"}\end{cases}$$
returning 0 if $D\le0$.

> **Default `normalize_mode="min"` (blocking fix).** Under the uniform-clonotype prior (§0.8), each row sums to 1 and the table is divided by its sum, so $p_x=P(c)=1/C$ **exactly** and $H_c=\log_2 C$ is **structural and maximal**. `"average"` denom $=\tfrac12(\log_2 C+H_p)$ throttles normalized MI by $\sim1/\log_2 C$ and shrinks its ceiling as $C$ grows — non-comparable across groups/covariates with different $C$, breaking the groupby-comparison workflow. `"min"` gives $I/H_p$ (reaches 1 when clone determines phenotype, $C$-independent) and is the default. Docstring states $H_c=\log_2 C$ is not a meaningful normalizer here.

**Fixes:** `n_samples=0` = deterministic plug-in $I(m̃)$ (was one random draw); `posterior=False → NotImplementedError` deleted (§0.4). Estimator honesty per §0.6 (MI Jensen-gap sign indeterminate).

**(c) Arguments → math.** As the shared table; additionally `normalize_mode` selects $D$. `clones` restricts rows; `normalized` toggles $I$ vs $I/D$.

**(d) Return shape.** The same for every metric and every combination of axes — see §7.10.
The old shape (scalar `float` / array+dict / tidy DataFrame, depending on `n_samples` and
`groupby`) meant the caller had to branch on their own arguments to read their own result.

`__all__ = ["mutual_information"]`

### 7.5 `tools/_flux.py` — `phenotypic_flux` (renamed from `flux`)

```python
phenotypic_flux(
    adata, *, cov_from, cov_to, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, distance_metric='kl',
    random_state=None, device=None, key_added=None, inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```

**(a) Responsibility.** Per-clonotype distance between a clone's phenotype distribution at `cov_from` vs `cov_to`, over the clone intersection.

**(b) Math.** $J_{\text{from}}$, $J_{\text{to}}$ (rows $P(\phi\mid c)$); over common clones $c$, $p=J_{\text{from}}[c]$, $q=J_{\text{to}}[c]$:
$$d_c=\text{phenotype\_distance}(p,q,\ \text{metric}=\texttt{distance\_metric}),$$
dispatched through `_distance` (§3.4): `"l1"` (default, bounded $[0,2]$), `"kl"`/`"dkl"` (directional, unbounded, **bits**), `"jsd"` (symmetric, bounded $[0,1]$ bit), or callable.

**Fixes carried by the rewrite:** (1) the `posterior=False` dead branch is gone (no more `TypeError` from unsupported `silent=`/`combine_with_logits=` kwargs); (2) **reproducibility** — draws use a seeded **torch** generator (old `seed=` only touched NumPy → no-op); (3) **unit consistency** — KL is $\log_2$/bits, single $\varepsilon$; (4) the `flux_table` column-misalignment (`clones_g.index(cl)` vs `common`-ordered columns) is removed by returning a tidy per-(group,clone) frame keyed by clone id. **Estimator honesty (§0.6):** `n_samples=0` = plug-in $d_c(m̃)$ (convex → under-estimates $\mathbb{E}_q[d_c]$; a clone with no real shift reads exactly 0 at `n_samples=0` but $>0$ in the `n_samples>0` mean).

**(c) Arguments → math.**

| Argument | Effect |
|---|---|
| `cov_from`, `cov_to` | The two conditions compared (was `from_this`/`to_that`). |
| `groupby` | Per-group flux → tidy rows (group × clone) with a `clone_size` column (replaces `flux_table`), via full-space restriction. |
| `n_samples` | `0` → deterministic per-clone plug-in `Series`; `N` → $N$ redrawn distance vectors + summary. |
| `temperature` | Tempers both joints identically before differencing. |
| `clones` | Restricts both sides; distances over the intersection. |
| `distance_metric` | `"l1"` / `"kl"` / `"jsd"` / callable. |
| `random_state` / `device` | Seeding / backend. |

**(d) Return shape.** `n_samples=0`, no `groupby` → `Series` over common clones; `n_samples>0` → per-clone `mean, sd, hdi_low, hdi_high`; `groupby` → tidy DataFrame row per (group, clone) + `clone_size` [+ summary].

`__all__ = ["phenotypic_flux"]`

### 7.6 `tcri/_stats/_compare.py` — `compare_groups` (INTERNAL: the one contrast implementation)

```python
compare_groups(
    df, *,
    value,                    # column holding the per-unit metric value or draw vector
    by,                       # grouping column (e.g. "response")
    reference=None,           # baseline level; None → all pairwise
    paired=False,             # True → paired posterior-draw contrast (uses prob_direction)
    hdi_prob=0.94,
    alternative="two-sided",
) -> pandas.DataFrame
```

**Responsibility.** The **public** replacement for the deleted `mi_compare` / `delta_entropy_table` / `flux_table`: turn a tidy `groupby` result (per-unit point estimates, e.g. per patient) or paired posterior-draw vectors into group contrasts. This closes the audit gap where "`groupby` + `_stats` subsumes `*_compare`/`*_delta`" was non-functional because `_stats` is private.

**Math.** For each contrast (`reference` vs other, or all pairs):
- **Unpaired point estimates:** Mann–Whitney $U$ + two-sided $p$ (`_stats.mann_whitney`), group means, and $\Delta=\text{mean}_B-\text{mean}_A$.
- **Paired posterior draws** (`paired=True`, one draw vector per group per unit, aligned by `sample_id`): the signed difference $\Delta^{(s)}=\text{metric}_B^{(s)}-\text{metric}_A^{(s)}$, then `mean(Δ)`, `hdi(Δ)`, and **`p_gt`/`p_lt` via `prob_direction`** — the **only** place a direction probability is emitted (§0.7).

**Return.** Tidy DataFrame, one row per contrast: `group_a, group_b, mean_a, mean_b, delta, U, p, p_gt, hdi_low, hdi_high, stars`.

**NOT PUBLIC.** It was `tl.compare_groups` — a second function you had to remember to call, on
the right frame, having picked the replicate unit yourself; picking the row-level frame gave
`p=0.040` with a star off 15 clones from 2 patients. `splitby` now produces the contrast as
part of the metric (the `stats` slot), and `build_stats` calls this after collapsing items to
replicates, so that choice is no longer available to get wrong. One implementation, one caller.

**The `paired=True` branch has no producer** — it wants a frame whose cells are draw *vectors*,
and no `tl` emits that shape. Kept rather than deleted because `table` makes a paired
replicate-level contrast reachable, and which estimand it should use is an open question.

`tools/__init__.py __all__ = ["joint_distribution", "clonotypic_entropy", "phenotypic_entropy",
"mutual_information", "phenotypic_flux", "delta_clonotypic_entropy",
"delta_phenotypic_entropy"]`

### 7.6a `tools/_delta.py` — the paired entropies

```python
delta_clonotypic_entropy(
    adata, *, cov_from, cov_to, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    n_clones_ref=None, random_state=None, device=None, key_added=None,
    inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```

```python
delta_phenotypic_entropy(
    adata, *, cov_from, cov_to, groupby=None, splitby=None, n_samples=0,
    temperature=1.0, clones=None, weighted=False, normalized=True,
    random_state=None, device=None, key_added=None, inplace=True
) -> dict   # {table, result, stats}; also stored in uns (§7.10)
```

**(a) Responsibility.** `value(cov_to) − value(cov_from)` per item, taken **within a posterior
draw**. Only the two metrics with an item axis have a paired form — `mutual_information` has
none, so a "Δ MI" would be a subtraction of two cached scalars and belongs to the caller (the
scope principle). Positive means it increased.

**(b) Support.** The **intersection** of clones present at both levels, within each replicate,
as `phenotypic_flux` already requires. It does two jobs depending on where the item axis sits:
for `delta_phenotypic_entropy` it decides which rows exist (the pairing itself — the same
clonotype at two timepoints); for `delta_clonotypic_entropy` it constrains the clone set summed
over *inside* H(c\|φ), making `log2(C)` identical on both sides so the normalizer cancels.
Without it a repertoire contracting 150 → 90 clones reports **+0.078** normalized entropy
having not redistributed at all. The drop moves `n`, so it warns.

**(c) Seeding.** Both sides come from **one shared sample** — the engine draws over every `ct`
row then selects a covariate's block, so the same seed realises the same underlying draw. A
self-delta is therefore exactly `0`. `phenotypic_flux` learned this: unpinned, its self-flux
read 0.209 at `n_samples=16`, which was the noise floor reported as a result.

**(d) Return.** `{table, result, stats}` as §7.10, plus `value_from` / `value_to` carried
through so the paired endpoints view is renderable from this result alone — and therefore
matched by construction. **No `p_gt`**: see §0.7.

`__all__ = ["delta_clonotypic_entropy", "delta_phenotypic_entropy"]`

### 7.7 h5ad-serializable return shapes

Every `tl` return frame must survive an h5ad round-trip: **flat columns only, no object-dtype
"samples" columns** holding numpy vectors (`AnnData.write` drops them), and no tuples anywhere in
the payload (h5py cannot write them — a MultiIndex `.to_numpy()` yields tuples, which is why
`_encode_index` flattens one array per level under a `__tcri_multiindex__` tag). Per-draw values
live as **rows carrying an explicit `draw` column**, never as vectors inside cells.

Provenance is a **sibling key in the stored blob** — `{table, result, stats, params, version}` —
written by `@tl_result` and read back with `tcri.get.params()`; `tcri.get.result()` strips
`params`/`version` so the return mirrors the tool's natural return exactly.

> **This section previously described a design that never shipped**, and is corrected here rather
> than being allowed to keep reading as current. It called `@tl_result` "deferred" after it had
> landed, and specified three mechanisms that do not exist: a `_provenance` JSON-string column, a
> `df.attrs["params"]` durable copy, and a cache key hashed over the argument list. The `attrs`
> hand-roll was in fact **removed** during implementation — `tcri/tools/_joint.py:220` records why
> (attrs does not survive most pandas operations), so the contract was mandating the exact carrier
> the code had rejected. There is no cache-key hashing at all: `key_added` names the slot. A
> forward-compat clause written for a future that arrives differently does not expire on its own.

### 7.8 Draw-once efficiency invariant

For `n_samples>0`, the engine draws the `p_ct` table **once per sample** and **reuses that draw across all covariates, groups, and clones**; groups are formed by cell/clone masking, not re-drawing. `covariate=None`, the flux sankey's pairwise series, and per-patient analyses all consume one shared draw stack. `diag.permutation_null` does **not** — it is model-free, scoring the empirical clone×phenotype crosstab, and draws no Dirichlet samples at all. A test/counter asserts the number of Dirichlet draws equals `n_samples`, independent of `#groups` and `#covariates`.

### 7.9 The precomputed-joint fast path — REMOVED

Three metrics used to accept a bare joint DataFrame in place of an AnnData. It was declared
in the first contract freeze, implemented because it was declared, and had exactly one caller
in the repo. Its root cause was `joint_distribution` returning a naked table; now that every
tool stores its result, there is nothing to hand back in. Gone with it: the `adata_or_jd`
union type, the `is_precomputed_joint` branch in three metrics, and the
`reject_stacked_covariate_joint` guard that existed only to police that branch.

`tl.*` takes an AnnData. That is the whole rule.

### 7.10 Return shape — one shape, stored once

Every `tl` returns `{table, result, stats}` **and** writes the same object to
`uns[key_added or "tcri_<metric>"]` with a `params` provenance block recording every argument
it ran with, including untouched defaults.

| slot | one row per | reduced over |
|---|---|---|
| `table` | (covariate, group, item, draw) | nothing — the substrate |
| `result` | (covariate, group, item) | `draw` only |
| `stats` | (split_a, split_b) pair | items → groups, then contrast |

`result` is built *from* `table`, so the two cannot drift. `stats` is `None` without
`splitby`; its replicate unit is the **group**, so item rows are averaged to one value per
group before the contrast and 15 clones from 2 patients give n=2.

Two uncertainty families, named apart on purpose: `hdi_*` on `result` is the within-group
posterior interval over draws; `ci_*`/`sd_*`/`n_*` on `stats` is the between-replicate spread
of each arm.

Read a cached result back with `tcri.get.result(adata, "<metric>")` and its provenance with
`tcri.get.params(adata, "<metric>")`.

---

## 8. `tcri.pl` — plotting (`plotting/`)

Twins mirror `tl` by filename and function name. Each renderer accepts its `tl` twin's metric arguments (computing the joint internally when needed) plus rendering args. Cross-group comparison is driven by **`groupby`** (dots = aggregation unit) and **`splitby`** (box hue = comparison cohort) — **both axes are retained by design** because most figures carry two categorical axes simultaneously (e.g. dots = patient, boxes = response, x = phenotype). Statistics come from `_stats` / `compare_groups`.

### 8.1 `plotting/_entropy.py`

```python
clonotypic_entropy(
    adata, *, key=None, order=None, hue_order=None, palette=None, ax=None,
    figsize=(8, 4), save=None, show=None, return_df=False
) -> matplotlib.axes.Axes | pandas.DataFrame  # renders the cache (§7.10)
```
**(renamed from `clonotypic_entropy_by_phenotype`)** Box-and-dot plot of clonotypic entropy per phenotype across covariate values, per-group dots, significance brackets. Cross-group plots default to a common `n_clones_ref` for comparability (§7.2).

```python
phenotypic_entropy(
    adata, *, key=None, order=None, hue_order=None, palette=None, ax=None,
    figsize=(8, 4), save=None, show=None, return_df=False
) -> matplotlib.axes.Axes | pandas.DataFrame  # renders the cache (§7.10)
```
**[FIXED]** Box/strip plot of phenotypic entropy per covariate/group.

`__all__ = ["clonotypic_entropy", "phenotypic_entropy"]`

### 8.2 `plotting/_mutual_information.py`

```python
mutual_information(
    adata, *, key=None, order=None, hue_order=None, palette=None, ax=None,
    figsize=(8, 4), save=None, show=None, return_df=False
) -> matplotlib.axes.Axes | pandas.DataFrame  # renders the cache (§7.10)
```
Renders the cached `tl.mutual_information`. The axes come from that call's `params`, not from
arguments here: with `groupby` it boxes one MI per group, with `splitby` it boxes by split and
brackets the contrast from `stats`. `weighted`, `normalize_mode`, `n_samples` and the rest are
`tl` arguments and appear nowhere on this signature — a plot that could compute is a plot that
can disagree with the frame in your hand.

`__all__ = ["mutual_information"]`

### 8.3 `plotting/_flux.py`

```python
phenotypic_flux(
    adata, *, key=None, order=None, hue_order=None, palette=None, ax=None,
    figsize=(8, 4), save=None, show=None, return_df=False
) -> matplotlib.axes.Axes | pandas.DataFrame  # renders the cache (§7.10)
```
Renders the cached `tl.phenotypic_flux` — per-clone flux from `cov_from` to `cov_to`, boxed.
The endpoints AND the distance metric come from the cached `params`, which is what makes the
axis label agree with the data: `tl` defaulted `distance_metric="kl"` and `pl` to `"l1"`, so
the two could describe different quantities.

`order` here is the plotting order of the x categories, not a covariate sequence. The
phenotype-flow Sankey over an ordered covariate series is a deferred enhancement with its own
issue; `_sankey.py` (§8.4) is its unused drawing primitives.

`__all__ = ["phenotypic_flux"]`

### 8.4 `plotting/_delta.py` — the paired entropy twins

```python
delta_clonotypic_entropy(
    adata, *, kind='delta', key=None, order=None, hue_order=None,
    palette=None, ax=None, figsize=(8, 4), save=None, show=None,
    return_df=False
) -> matplotlib.axes.Axes | pandas.DataFrame  # renders the cache (§7.10)
```

```python
delta_phenotypic_entropy(
    adata, *, kind='delta', key=None, order=None, hue_order=None,
    palette=None, ax=None, figsize=(8, 4), save=None, show=None,
    return_df=False
) -> matplotlib.axes.Axes | pandas.DataFrame  # renders the cache (§7.10)
```

Two views of one cached result, selected by `kind`:

| `kind` | shows | marks |
|---|---|---|
| `"delta"` (default) | the change | the mark rule (§8.5) + a zero rule |
| `"endpoints"` | `cov_from` and `cov_to` side by side | one dot per replicate per level |

`"endpoints"` lives here rather than on `pl.phenotypic_entropy` because the endpoints are in
the *delta's* payload, computed over the intersected clone set. The same figure drawn from two
separate single-covariate results would use a **different clone set on each side**, and the
values differ substantially — rendering it only from the delta result makes the unmatched
version unreachable rather than merely discouraged.

**Connecting lines only on `delta_phenotypic_entropy`.** A line asserts the two points are the
same entity observed twice. A clonotype persists — a biological barcode. A phenotype is a bin:
the same category measured twice, its occupants changed. Both can show their endpoints; only
one may join them.

**Dot area is the matched clone count**, with a size legend. It varies per replicate (one
patient may match 240 of 300, another 30 of 400), which is why it is encoded per point rather
than stated once in a title. A replicate's two endpoint dots are the same size by
construction — the matched set is the same on both sides — and a difference would mean the
intersection did not hold.

### 8.4b `plotting/_sankey.py` — private drawing primitives

**`class SankeyNode`** *(internal)* — `__init__(self, x, y, val, *, dx=0.2, color=None, **kwargs)`; `plot(self, ax)`; `plot_node_connection(self, destination_node, ax, **kwargs)` (curved, color-interpolated ribbon). `_phenotype_mass_per_clone(adata, covariate, clones, normalize) -> dict[str, np.ndarray]` — `{clone → phenotype-mass vector}` at one covariate. `SankeyNode.hex_to_rgb` is **deleted** (0 callers; ribbons use `mcolors.to_rgb`).

### 8.5 `plotting/_base.py` — private plotting engine

| Signature | Responsibility |
|---|---|
| `render_metric(adata, name, *, ylabel, item_col=None, item_as_x=False, key=None, ...)` | The shared cache renderer every twin delegates to. Reads `tcri.get.result`/`params`, picks the x axis from the cached `groupby`/`splitby`, applies the mark rule, routes colours through `resolve_colors`, and brackets `stats` when x IS the split. Takes **no** metric arguments and never calls `tl`. |
| `_sample_unit(frame, table, *, x, groupby, item_col)` | **The mark rule.** Returns the coarsest unit that varies within an x position — `replicate` > `item` > `draw`, or `None`. One variance component per mark. |
| `_boxstrip(...)` | Box + strip. Reached when replicates or items are the sample; items are collapsed to replicates first via `collapse_to_replicates`, the same function `build_stats` uses. |
| `_violins(...)` | Violin over the draw distribution, read from `table`. Reached only when draws are the coarsest varying unit, so a violin can never span replicates. |
| `_points(...)` | The floor: one point per x with the HDI as an error bar, when nothing varies. A point rather than a bar because a bar's area encodes a magnitude from zero these metrics do not have. |
| `_annotate_contrasts(ax, stats, levels)` | Bracket + stars per contrast, drawn only where `stats` has a row for that exact pair of x levels. Bracket artists carry `BRACKET_LABEL` so the connector guard can tell them from a matched-identity line. |
| `_finish(fig, ax, *, save=None, show=None)` | scanpy-style show/save/return finalizer. |

**The mark rule (§8.5).** A mark shows ONE variance component; within an x position the sample
is the coarsest unit that varies there. Pooling draws across replicates would render 6 patients
× 100 draws as 600 samples — the pseudoreplication `build_stats` collapses away, drawn as a
picture. Measured before the fix, on `phenotypic_entropy(groupby="patient", splitby="response")`:
the box and strip described **47 clones** while the p-value bracketed above them described
**6 patients**.

**Connecting lines** are drawn only between points sharing an identity across the compared
levels — never from adjacency. A line asserts the two points are the same entity observed
twice, which is a claim only matched data supports.

### 8.6 `plotting/_colors.py`

| Symbol | Responsibility |
|---|---|
| `tcri_colors` (`list[str]`) | Canonical categorical hex palette. |
| `resolve_colors(adata, cat_key, categories=None, *, palette=None, persist=True) -> dict` | Resolve `{category: hex}`, cached under `uns["<cat_key>_colors"]` (scanpy's convention, so `sc.pl.umap` matches). Priority: explicit `palette` (dict / list / cmap name) → an existing stored assignment of matching length → `tcri_colors`, cycled. A partial dict fills its gaps from the cycle. |

`__all__ = ["tcri_colors", "NA_COLOR", "resolve_colors"]`

---

## 9. `tcri.diag` — diagnostics (`diagnostics/`) — NEW

Read-only checks on the finalized model. PPCs return `DataFrame`s; the two relocated model plots render figures. **`model` is required exactly where the live decoder/param store is needed, optional where `adata` suffices** (stated per function).

### 9.1 `diagnostics/_ppc.py`

| Signature | Responsibility / math |
|---|---|
| `joint_distribution_ppc(adata, *, covariate=None, distance_metric="l1", temperature=1.0) -> pandas.DataFrame` | **(fixed `compare_joint_distribution`)** Model vs empirical per-clone phenotype frequencies. $P_{\text{model}}(\phi\mid c,m)=\texttt{joint\_distribution}(adata, covariate=m)[c]$; $P_{\text{emp}}(\phi\mid c,m)=\frac{\#\{i\in c,m:\text{pheno}_i=\phi\}}{\#\{i\in c,m\}}$; per-clone $\delta_c=\text{L1}$ or $\text{KL}(P_{\text{emp}}\Vert P_{\text{model}})$, plus per-covariate aggregate. **Model-free (adata only).** **Bug fix:** reads `clonotype_col`/`phenotype_col` from `uns[K.METADATA]` instead of the undefined global `model` (repairs the `NameError`). |
| `phenotype_calibration(adata, *, n_bins=10) -> pandas.DataFrame` | Reliability of `predict()` probabilities: bin cells by predicted max-prob; per bin compare mean predicted prob to empirical accuracy; $\text{ECE}=\sum_b\frac{n_b}{N}|\text{acc}_b-\text{conf}_b|$. **adata only.** Returns `(bin, mean_pred, emp_freq, count)` + scalar `ECE`. |
| `reconstruction_ppc(model, adata=None, *, n_sims=100, random_state=0) -> pandas.DataFrame` | ZINB reconstruction PPC: simulate from the fitted decoder ($\mu,\theta,\pi_{\text{dropout}}$), compare library size / per-gene dropout / mean–variance vs observed. **`model` REQUIRED** (live decoder lives on the module, not in `adata`). Returns statistic × {observed, simulated, discrepancy}. |
| `permutation_null(adata, *, metric="mutual_information", covariate=None, groupby=None, normalize_mode="min", n_perm=1000, random_state=None) -> pandas.DataFrame` | Permute phenotype labels within each covariate $R$ times, recompute the metric on the **empirical** crosstab to form a null; $p=\frac{1+\#\{\text{null}\ge\text{obs}\}}{1+R}$, $z=\frac{\text{obs}-\overline{\text{null}}}{\text{sd(null)}}$. **adata only, model-free — no Dirichlet draws.** `normalize_mode` must match the statistic this is a null for. `groupby` strata match the metric surface (one row per covariate x group, permuting within the stratum; same clone-disjointness requirement). Returns per stratum: `observed, null_mean, null_sd, z, p`. |

`__all__ = ["joint_distribution_ppc", "phenotype_calibration", "reconstruction_ppc", "permutation_null"]`

### 9.2 `diagnostics/_training.py`

| Signature | Responsibility |
|---|---|
| `loss(model, *, log_scale=False, ax=None, save=None)` | **(relocated `plot_loss`)** Plot training/validation ELBO and prior-KL from `model.history_`. |
| `archetypes(model, *, ax=None, save=None)` | **(relocated `plot_archetypes`)** Cluster-ordered clone-phenotype heatmap + archetype centroids, ordered by the `labels` from `build_archetypes` (retained, §5.5). |

`diagnostics/__init__.py __all__ = ["joint_distribution_ppc", "phenotype_calibration", "reconstruction_ppc", "permutation_null", "loss", "archetypes"]`

> The model PGM (`build_nested_tcri_pgm` / `draw_tcri_pgm_nested`) is moved **out of the package** to `docs/`.

---

## 10. `tcri.ut` — utilities (`utils/`)

### 10.1 `utils/_session.py` — public

| Signature | Responsibility |
|---|---|
| `save_tcri_session(model, adata, out_dir, *, save_adata=True, compression="gzip") -> dict` | Persist a trained session: scvi model (weights + registry, no embedded adata), Pyro param store, `setup.json`, the h5ad. |
| `load_tcri_session(run_dir, *, adata_path=None, map_location=None, layer=None) -> (TCRIModel, AnnData)` | Reconstruct `TCRIModel` + `AnnData`: read h5ad, restore setup/category order, re-run `setup_anndata`, load model + Pyro params. **Sets the global Pyro store before any `get_p_ct`/`to_anndata` call** (param-store caveat, §5.2). |

### 10.2 `utils/_session.py` — private helpers (not re-exported)

`_to_jsonable(x)`; `_collect_setup_from_adata_or_model(adata, model)`; `_restore_category_order(adata, setup)`; `_resolve_TCRIModel()`; `_disable_scvi_onload_train()`; `_ensure_pyro_posterior_params(model, adata)` (guarantees `q_p_ct_raw`; if missing, warn + re-init to uniform $1/P$); `_pyro_load(path, *, map_location=None)`; `_ensure_dir(path)`.

> **Removed from utils:** `write_adata_safely`, `_pop_nonserializables` (manager stash retired at `setup_anndata`); `probabilities` (dead: read a never-written `uns` key — **and its module-top import in `_plotting.py` is removed in the same PR**, §11); `build_nested_tcri_pgm`/`draw_tcri_pgm_nested` (→ `docs/`); `stars`/`auc_and_label_permutation`/`bootstrap_auc` (→ `_stats.py`).

---

## 11. Surface deltas (removed / renamed / moved)

**Disposition rule.** Every function is kept or dropped by ONE test — *is it core?* (the model, the joint engine, the four metrics + their plots, session I/O, PPC diagnostics, shared helpers). Non-core = **dropped (deleted)**; nothing is relocated to `examples/`, and the disposable notebooks are never consulted for disposition. Deletion PRs grep import-sites as well as call-sites so a top-level import (e.g. `utils.probabilities` at `_plotting.py:18`) is removed in the same PR as its symbol.

- **Deleted (not core — dropped, never moved to examples):** `clonality` (tl + pl), `probability_ternary`, `gene_entropy`, `top_clone_umap`, `clone_size_umap`, `plot_phenotype_probabilities`, `polar_plot`, `compare_phenotypes`, `clonotypic_entropy_base`, `delta_clonotypic_entropy`, `delta_entropy_table`, `mi_compare` (tl + pl), `flux_table`, `bayesian_mutual_information`, `probability_distribution`, `classify_phenotypes`, `get_latent_embedding`, `group_small_clones`, `register_probability_columns`, `remove_meaningless_genes`, `clone_fraction`, `_ent`, `ridge_delta_entropy`, `dkl` (→ `_distance.kl_divergence`), `probabilities` (**and its `_plotting.py` import**), `SankeyNode.hex_to_rgb`, `_ascii_hist` (+ all `graph=`/ASCII paths), `write_adata_safely`, `_pop_nonserializables`, and the retired `uns` keys `tcri_manager`, `tcri_clone_key`, `tcri_phenotype_key`, the `X_tcri_phenotypes` obsm slot.
- **Renamed:** `flux`→`tl.phenotypic_flux`; `get_cell_phenotype_probs`→`TCRIModel.predict`; `register_model`→`TCRIModel.to_anndata`; `clonotypic_entropy_by_phenotype`→`pl.clonotypic_entropy`; `tcri_boxplot`→`_base._metric_boxplot`; `set_color_palette`→`resolve_palette`→`resolve_colors`; params `from_this`/`to_that`→`cov_from`/`cov_to`; engine `posterior=`→`use_logits=` (alias `cell_informed=`); `point_estimate=`→removed (use `n_samples`). `weighted=` is **retained** (default `False`; §0.8), not removed.
- **Made private (folded in):** `register_clonotype_key` / `register_phenotype_key` / `_compute_logits_and_prior` → `preprocessing/_register` internals, folded into `TCRIModel.to_anndata`.
- **Relocated (kept, NOT to examples):** `compare_joint_distribution`→`diag.joint_distribution_ppc`, `plot_loss`→`diag.loss`, `plot_archetypes`→`diag.archetypes`; `build_nested_tcri_pgm` / `draw_tcri_pgm_nested` → **`docs/`** (out of the package, a figure script only).
- **Subsumed by `groupby` + `compare_groups` (removed, with migration recipe):** the plural batch wrappers `clonotypic_entropies` / `phenotypic_entropies`, `pl.phenotypic_entropy_delta`, and every `*_compare` / `*_delta` / `*_table` variant.

---

## 12. Appendix — current → target math/stats deltas (what changed and why)

| # | Site | Current | Target | Rationale |
|---|---|---|---|---|
| 1 | engine `n_samples=0` | `joint_distribution_posterior` always draws 1 Dirichlet sample | closed-form posterior **mean** $m$, no draw | reproducible, Rao-Blackwell (§0.4) |
| 2 | `mutual_information`/`flux` `n_samples=0` | returns one random draw | deterministic plug-in point estimate | latent bug (§0.5) |
| 3 | `posterior=` semantics | conflates draw-vs-mean **and** logit-folding; MI `posterior=False` raises | axis renamed `use_logits`, means *fold per-cell logits* only; both branches use the posterior, never the generative prior | §0.4 |
| 4 | metric ↔ model agreement | metrics use a Dirichlet **draw** and **never** apply the gate | `use_logits=True` at $T=1$ uses the same gate-aware, mean-prior rule as `predict()` (needs persisted `X_tcri_logits`, `gate_prob`, `classifier_temperature`) | removes silent disagreement (§0.9, §5.1) |
| 5 | **plug-in vs posterior-mean** | drafts equate `n_samples=0` with `mean(n_samples>0)` | documented as **different estimators** (Jensen gap: entropy plug-in $\ge$ mean; flux plug-in $\le$ mean; MI indeterminate); **no equality test** | §0.6 |
| 6 | **`p_gt` summary** | attached to every `n_samples>0` metric | **removed** from single-metric summaries; emitted only by `compare_groups` on a signed $\Delta$ | metrics are $\ge0$ ⇒ $P(>0)\approx1$ (§0.7) |
| 7 | **posterior draw concentration** | three inconsistent variants: `clamp(s·m,1e-3)` (guide) vs `s·p_ct` vs `s·p_ct+1e-8` | draw from the **exact guide** `Dirichlet(clamp(s·m̃, 1e-3))` | HDIs must summarize the learned posterior (§0.5) |
| 8 | **MI `normalize_mode` default** | `"average"` ⇒ denom $\tfrac12(\log_2C+H_p)$, $C$-dependent | **`"min"`** ⇒ $I/H_p$, $C$-independent; document $H_c=\log_2C$ structural | cross-group comparability (§7.4) |
| 9 | clonotypic-entropy denominator | $\log_2$ of raw reindexed row count (inflated by absent clones) | $\log_2$ of **supported** clones; optional fixed `n_clones_ref` | comparability (§7.2) |
| 10 | phenotypic-entropy zero clone | zero-mass clone → uniform → $H=1.0$ | zero-support clone → **`NaN`/excluded** | §7.3 |
| 11 | flux `seed` | seeds NumPy only; torch draws unaffected | seed a **torch (+CUDA) Generator**; `random_state` on engine/metrics | reproducibility (§0.10) |
| 12 | KL base/$\varepsilon$ | natural log in flux; $\log_2$ elsewhere; mixed $\varepsilon$ | one base ($\log_2$/bits), one $\varepsilon=10^{-12}$; add bounded symmetric `jsd` | unit consistency (§3.4) |
| 13 | "HDI" | equal-tailed percentiles labeled HDI | **true** highest-density interval `hdi_low/hdi_high` | correct for skewed bounded posteriors (§0.7) |
| 14 | temperature | applied at different stages in the two engines; double-tempered with `guide_temperature` | **single** power-temper of the base; `uns[K.P_CT]` stores the **raw** mean; $T=1$ reproduces `predict()` | §0.9 |
| 15 | joint-distribution PPC | references undefined global `model` → `NameError` | reads cols from `uns[K.METADATA]` | §9.1 |
| 16 | groupby via slicing | `function(adata[mask])` trips the full-space alignment guard | full-space cell/clone **restriction**; `_metric_boxplot` rewritten | §7.1, §8.5 |
| 17 | `local_scale` fallback | `uns.get("tcri_local_scale", 1.0)` silently corrupts draw variance if unwritten | `to_anndata` always writes `K.LOCAL_SCALE`; engine **raises** (no `1.0` default) when missing at `n_samples>0` | draw-variance integrity (§5.1) |
| 18 | partial posterior | intervals silently read as full predictive uncertainty | documented: `n_samples>0` captures **`p_ct` uncertainty only** (classifier logits fixed at their posterior-mean encoding) | §0.6 |

---

*Source of truth cross-checked against `tcri/model/_model.py`, `tcri/model/_module.py`, `tcri/preprocessing/_preprocessing.py`, `tcri/metrics/_metrics.py`, `tcri/plotting/_plotting.py`, `tcri/plotting/_sankey.py`, `tcri/utils/_utils.py`, and the grafiti reference at `/Users/ceglian/Codebase/GitHub/grafiti/grafiti`. Intended document home: `/Users/ceglian/Codebase/GitHub/tcri/governance/API_CONTRACT.md`.*