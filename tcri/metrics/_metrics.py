# Standard library imports
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc
from typing import Optional, List, Union

# Third-party imports
from scipy.stats import entropy
from scipy.spatial import distance
import gseapy as gp
import numpy as np, pandas as pd, torch, umap
from tqdm.auto import tqdm
from scvi import REGISTRY_KEYS
import itertools
# Local imports
from ..preprocessing._preprocessing import joint_distribution,joint_distribution_posterior

warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from typing import Optional, List
from tqdm.auto import tqdm   # nice progress bar in notebooks
# ╭──────────────────────────────────────────────────────────────╮
# │  Δ-E N T R O P Y   T A B L E   B U I L D E R  (v2)           │
# ╰──────────────────────────────────────────────────────────────╯
import numpy as np, pandas as pd
from typing import Optional, List
from tqdm.auto import tqdm


# ------------ simple ANSI helpers ------------ #
RESET  = "\x1b[0m"
BOLD   = "\x1b[1m"
DIM    = "\x1b[2m"
GREEN  = "\x1b[32m"
CYAN   = "\x1b[36m"
MAGENT = "\x1b[35m"

# ╭─ colour / pretty-print helpers ─────────────────────────────────────────╮
RESET  = "\x1b[0m";  BOLD  = "\x1b[1m";  DIM  = "\x1b[2m"
GRN = "\x1b[32m";  CYN = "\x1b[36m";  MAG = "\x1b[35m";  YLW = "\x1b[33m"; RED = "\x1b[31m"

def _ok(msg:str, quiet=False):    # success mark
    if not quiet: print(f"{GRN}✅ {msg}{RESET}")

def _info(key:str, txt:str, quiet=False):       # key-value info line
    if not quiet: print(f"   {CYN}🎯 {key:<22}{DIM}{txt}{RESET}")

def _warn(msg:str, quiet=False):   # warning line
    if not quiet: print(f"{YLW}⚠️  {msg}{RESET}")

def _fin(quiet=False):             # final flourish
    if not quiet: print(f"{MAG}✨  Done!{RESET}")
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭─ tiny ASCII histogram (handy in notebooks/SSH) ──────────────────────────╮
def _ascii_hist(samples, bins=25, width=40) -> str:
    hist, edges = np.histogram(samples, bins=bins)
    top = hist.max()
    lines=[]
    for h,e0,e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "█"*int(width*h/top) if top else ""
        lines.append(f"{e0:7.3f}-{e1:7.3f} | {bar}")
    return "\n".join(lines)
# ╰─────────────────


# ╭─ MI helper (single source of truth) ─────────────────────────────────────╮
def _mi_from_joint(pxy: np.ndarray, normalised: bool, mode: str="average") -> float:
    """
    Mutual information (optionally normalised) from an *already normalised*
    joint table pxy (shape C×P).
    """
    eps = 1e-15
    px  = pxy.sum(1, keepdims=True)
    py  = pxy.sum(0, keepdims=True)
    mi  = np.sum(pxy * np.log2((pxy+eps) / (px @ py + eps)))

    if not normalised:
        return mi
    h_c = -np.sum(px * np.log2(px+eps))
    h_p = -np.sum(py * np.log2(py+eps))
    denom = 0.5*(h_c+h_p) if mode == "average" else min(h_c, h_p)
    return mi/denom if denom > 0 else 0.0
# ╰──────────────────────────────────────────────────────────────────────────╯


def mi_compare(adata, groupby, groups=None, treatment=None, n_samples=50,
               patient_col=None, clone_col=None, covariate_col=None,
               verbose=True, **mi_kwargs):
    meta = adata.uns["tcri_metadata"]
    patient_col = patient_col or meta["batch_col"]
    clone_col = clone_col or meta["clone_col"]
    covariate_col = covariate_col or meta["covariate_col"]

    covariates = treatment
    if covariates is None:
        covariates = adata.obs[covariate_col].cat.categories.tolist()
    elif isinstance(covariates, str):
        covariates = [covariates]

    # resolve groups into list of 2-tuples
    unique_groups = adata.obs[groupby].dropna().unique().tolist()
    if groups is None:
        pairs = list(itertools.combinations(sorted(unique_groups), 2))
    elif isinstance(groups[0], (list, tuple)):
        pairs = [tuple(g) for g in groups]
    else:
        pairs = list(itertools.combinations(groups, 2))

    # compute patient-level MI samples
    keep_groups = set(g for pair in pairs for g in pair)
    records = []
    patients = adata.obs[patient_col].unique()
    for p in tqdm(patients, disable=not verbose, desc="MI per patient"):
        pmask = adata.obs[patient_col] == p
        group_val = adata.obs.loc[pmask, groupby].iloc[0]
        if group_val not in keep_groups:
            continue
        clones = adata.obs.loc[pmask, clone_col].unique().tolist()
        for cov in covariates:
            try:
                samples = mutual_information(
                    adata, cov, n_samples=n_samples,
                    clones=clones, verbose=False, **mi_kwargs
                )
            except Exception as e:
                if verbose:
                    print(f"  Skip {p}/{cov}: {e}")
                continue
            for s in np.atleast_1d(samples):
                records.append({"patient": p, "group": group_val,
                                "covariate": cov, "MI": float(s)})

    result = pd.DataFrame(records)
    summary = (
        result.groupby(["patient", "group", "covariate"])["MI"]
        .agg(mean="mean", median="median",
             lo=lambda x: np.quantile(x, 0.025),
             hi=lambda x: np.quantile(x, 0.975),
             sd="std", n="size")
        .reset_index()
    )

    return {
        "samples": result,
        "summary": summary,
        "pairs": pairs,
        "params": {"groupby": groupby, "covariates": covariates,
                   "n_samples": n_samples, "patient_col": patient_col},
    }


def dkl(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return entropy(p, q) 

import numpy as np, pandas as pd
from   scipy.stats import entropy        # Shannon entropy
from   typing import Optional, List, Union
import numpy as np, pandas as pd
from   typing import Optional, List, Union
from   scipy.stats import entropy                         # Shannon H

# ---------- ANSI helpers  -------------------------------------------
RESET="\x1b[0m"; BOLD="\x1b[1m"; DIM="\x1b[2m"
GRN="\x1b[32m"; CYN="\x1b[36m"; MAG="\x1b[35m"; YLW="\x1b[33m"

# small helper ----------------------------------------------
def _ent(p, base=2):
    p   = np.asarray(p, dtype=float)
    eps = 1e-15
    p   = p.clip(eps) / p.sum()
    return entropy(p, base=base)

def clonotypic_entropy_base(
    adata,
    covariate_label: str,
    phenotype: str,
    *,
    base: int = 2,
    normalised: bool = True,
    temperature: float = 1.0,
    clones: Optional[List[str]] = None,
    weighted: bool = False,
    posterior: bool = True,
    combine_with_logits: bool = True,
) -> float:
    if posterior:
        jd = joint_distribution_posterior(
                adata,
                covariate_label     = covariate_label,
                temperature         = temperature,
                clones              = clones,
                weighted            = weighted,
                combine_with_logits = combine_with_logits,
                silent              = True)
    else:
        jd = joint_distribution(
                adata,
                covariate_label = covariate_label,
                temperature     = temperature,
                n_samples       = 0,
                clones          = clones,
                weighted        = weighted)

    if jd is None or jd.empty or phenotype not in jd.columns:
        return 0.0

    vec = jd[phenotype].to_numpy(dtype=float)
    eps = 1e-15
    vec = np.clip(vec, eps, None)
    vec = vec / vec.sum()                           # ensure ∑=1

    H = entropy(vec, base=base)
    if normalised and len(vec) > 1:
        H /= np.log(len(vec)) / np.log(base) #        # max-entropy normalisation
    return H

def clonotypic_entropy(
    adata,
    covariate: str,
    phenotype: str,
    *,
    temperature: float = 1.0,
    n_samples: int = 0,           # 0 → single value
    clones: Optional[List[str]] = None,
    weighted: bool = False,
    normalised: bool = True,
    base: int = 2,
    posterior: bool = True,
    combine_with_logits: bool = True,
    verbose: bool = True,
    graph: bool = False,          # ASCII bar plot of posterior
) -> Union[float, np.ndarray]:

    if verbose:
        print(f"{BOLD}{MAG}🧮  Clonotypic entropy – {phenotype} @ '{covariate}'{RESET}")
        _info("posterior",      posterior)
        _info("weighted",       weighted)
        _info("# samples",      n_samples)

    # ---- single deterministic draw ---------------------------------
    if n_samples == 0:
        H = clonotypic_entropy_base(
                adata, covariate, phenotype,
                base               = base,
                normalised         = normalised,
                temperature        = temperature,
                clones             = clones,
                weighted           = weighted,
                posterior          = posterior,
                combine_with_logits= combine_with_logits)
        if verbose:
            _ok(f"H = {H:.4f}")
        return H

    # ---- Monte-Carlo sampling --------------------------------------
    H_samp = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        H_samp[i] = clonotypic_entropy_base(
            adata, covariate, phenotype,
            base               = base,
            normalised         = normalised,
            temperature        = temperature,
            clones             = clones,
            weighted           = weighted,
            posterior          = posterior,
            combine_with_logits= combine_with_logits)

    if verbose:
        _ok("sampling complete")
        _info("mean ± sd",  f"{H_samp.mean():.4f} ± {H_samp.std():.4f}")
        lo,hi = np.percentile(H_samp,[2.5,97.5])
        _info("95 % CI",    f"[{lo:.4f}, {hi:.4f}]")
        if graph:
            print(f"{DIM}\nASCII histogram:\n{_ascii_hist(H_samp)}{RESET}")

    return H_samp

def delta_clonotypic_entropy(
    adata,
    phenotype: str,
    *,
    cov_pre:  str = "Pre-treatment",
    cov_post: str = "Post-treatment",
    n_samples: int = 1_000,
    temperature: float = 1.0,
    clones: Optional[List[str]] = None,
    weighted: bool = False,
    normalised: bool = True,
    base: int = 2,
    posterior: bool = True,
    combine_with_logits: bool = True,
    verbose: bool = True,
    graph: bool = False,        # ASCII plot of Δ posterior
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample Δ-entropy =  H_post – H_pre  for one phenotype.

    Returns
    -------
    delta_samples : ndarray  (shape = (n_samples,))
        Positive values ⇒ entropy increased from pre → post.
    """
    if seed is not None:
        np.random.seed(seed)

    if verbose:
        print(f"{BOLD}{MAG}Δ-Entropy  {phenotype}: "
              f"'{cov_pre}' ⟶ '{cov_post}'{RESET}")
        _info("# samples", n_samples)
        _info("posterior", posterior)
        _info("weighted",  weighted)

    Δ = np.empty(n_samples, dtype=float)

    # --- Monte Carlo loop -----------------------------------------
    for i in range(n_samples):
        H_pre  = clonotypic_entropy_base(
                    adata, cov_pre,  phenotype,
                    base=base, normalised=normalised,
                    temperature=temperature,
                    clones=clones, weighted=weighted,
                    posterior=posterior,
                    combine_with_logits=combine_with_logits)

        H_post = clonotypic_entropy_base(
                    adata, cov_post, phenotype,
                    base=base, normalised=normalised,
                    temperature=temperature,
                    clones=clones, weighted=weighted,
                    posterior=posterior,
                    combine_with_logits=combine_with_logits)
        Δ[i] = H_post - H_pre

    if verbose:
        _ok("sampling complete")
        _info("mean ± sd", f"{Δ.mean():.4f} ± {Δ.std():.4f}")
        lo,hi = np.percentile(Δ,[2.5,97.5])
        _info("95 % CI",  f"[{lo:.4f}, {hi:.4f}]")
        if graph:
            print(f"{DIM}\nASCII histogram of Δ:\n{_ascii_hist(Δ)}{RESET}")

    return Δ


def delta_entropy_table(
    adata,
    *,
    cov_pre : str = "Pre-treatment",
    cov_post: str = "Post-treatment",
    splitby : str = "response",
    n_samples : int = 1_000,
    temperature: float = 1.0,
    weighted   : bool = False,
    normalised : bool = True,
    base       : int  = 2,
    posterior  : bool = True,
    combine_with_logits : bool = True,
    seed : Optional[int] = 42,
    show_progress: bool  = True
) -> pd.DataFrame:
    """
    Build a tidy Δ-clonotypic-entropy table (post – pre).

    Each row ⇢ one phenotype × one `splitby` group.
    The `delta_samples` column keeps the full NumPy vector so you can
    re-plot KDEs or run further stats without re-sampling.
    """
    # reproducibility
    if seed is not None:
        np.random.seed(seed)

    meta       = adata.uns["tcri_metadata"]
    clone_col  = meta["clone_col"]
    phen_col   = meta["phenotype_col"]

    groups     = sorted(adata.obs[splitby].dropna().unique().tolist())
    phenotypes = adata.obs[phen_col].astype("category").cat.categories.tolist()

    records = []
    iterator = tqdm(groups, desc="Δ-entropy groups") if show_progress else groups

    for g in iterator:
        # — restrict ONLY the clone list, keep full AnnData for index integrity
        mask_g   = adata.obs[splitby] == g
        clones_g = adata.obs.loc[mask_g, clone_col].unique().tolist()

        for ph in phenotypes:
            delta = delta_clonotypic_entropy(
                        adata, ph,
                        cov_pre         = cov_pre,
                        cov_post        = cov_post,
                        n_samples       = n_samples,
                        temperature     = temperature,
                        clones          = clones_g,      # ⬅ scoped clones
                        weighted        = weighted,
                        normalised      = normalised,
                        base            = base,
                        posterior       = posterior,
                        combine_with_logits = combine_with_logits,
                        verbose         = False
                    )

            d_mean  = delta.mean()
            d_sd    = delta.std()
            hdi_lo, hdi_hi = np.percentile(delta, [2.5, 97.5])
            p_gt    = (delta > 0).mean()
            p_lt    = (delta < 0).mean()

            records.append(dict(
                **{splitby: g, "phenotype": ph},
                delta_samples = delta,
                delta_mean    = d_mean,
                delta_sd      = d_sd,
                hdi_low       = hdi_lo,
                hdi_high      = hdi_hi,
                p_greater     = p_gt,
                p_less        = p_lt
            ))

    return pd.DataFrame.from_records(records)


def phenotypic_entropy(
    adata,
    covariate              : str,
    *,
    clonotypes             : Optional[Union[str,List[str]]] = None,
    base                   : int   = 2,
    normalised             : bool  = True,
    temperature            : float = 1.0,
    n_samples              : int   = 0,
    weighted               : bool  = False,
    posterior              : bool  = True,
    combine_with_logits    : bool  = True,
    verbose                : bool  = True,
    graph                  : bool  = False,
    seed                   : Optional[int] = 42
) -> Union[float, np.ndarray]:

    if seed is not None:
        np.random.seed(seed)

    # ── header ────────────────────────────────────────────────── #
    if verbose:
        print(f"{BOLD}{MAG}📈  Phenotypic-entropy at '{covariate}'{RESET}")
        _info("posterior",   posterior,   False)
        _info("n_samples",   n_samples,   False)
        _info("normalised",  normalised,  False)

    # ── pull meta info ----------------------------------------- #
    meta           = adata.uns["tcri_metadata"]
    clone_col      = meta["clone_col"]
    covariate_col  = meta["covariate_col"]

    # determine clonotype list
    if clonotypes is None:
        clones_list = (
            adata.obs.loc[adata.obs[covariate_col] == covariate, clone_col]
            .unique()
            .tolist()
        )
    elif isinstance(clonotypes, str):
        clones_list = [clonotypes]
    else:
        clones_list = clonotypes

    if len(clones_list) == 0:
        _warn("no clonotypes found – returning 0", quiet=not verbose)
        return 0.0

    log_base = np.log(base)

    # ── ONE posterior/prior draw → mean entropy across clones ── #
    def _one_sample() -> float:
        if posterior:
            jd = joint_distribution_posterior(
                    adata,
                    covariate_label     = covariate,
                    temperature         = temperature,
                    clones              = clones_list,
                    weighted            = weighted,
                    combine_with_logits = combine_with_logits,
                    silent              = True)
        else:
            jd = joint_distribution(
                    adata,
                    covariate_label = covariate,
                    temperature     = temperature,
                    n_samples       = 0,
                    clones          = clones_list,
                    weighted        = weighted)

        if jd is None or jd.empty:
            return 0.0
        ent_vals = []
        for cl in clones_list:
            if cl not in jd.index:
                continue
            p   = jd.loc[cl].to_numpy(dtype=float)
            eps = 1e-15
            p   = p.clip(eps)
            h   = entropy(p, base=base)
            if normalised:
                h /= np.log(len(p)) / log_base
            ent_vals.append(h)
        return float(np.mean(ent_vals)) if ent_vals else 0.0

    # ── no-sampling branch ------------------------------------ #
    if n_samples == 0:
        val = _one_sample()
        _ok("entropy computed", quiet=not verbose)
        if verbose:
            _info("value", f"{val:.4f}")
            _fin()
        return val

    # ── Monte-Carlo sampling ---------------------------------- #
    samples = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        samples[i] = _one_sample()

    if verbose:
        _ok(f"generated {n_samples:,} samples")
        _info("mean ± sd", f"{samples.mean():.4f} ± {samples.std():.4f}")
        _info("95% CI",
              f"[{np.percentile(samples,2.5):.4f}, "
              f"{np.percentile(samples,97.5):.4f}]")
        if graph:
            print(f"{DIM}\nASCII histogram:\n{_ascii_hist(samples)}{RESET}")
        _fin()

    return samples

def clonotypic_entropies(adata, covariate, normalized=True, base=2, temperature=1., decimals=5, n_samples=0):
    unique_phenotypes = adata.uns["tcri_phenotype_categories"]
    phenotype_entropies = dict()
    for phenotype in unique_phenotypes:
        cent = clonotypic_entropy(adata, covariate, phenotype, temperature=temperature)
        phenotype_entropies[phenotype] = np.round(cent,decimals=decimals)
    return phenotype_entropies

def clonality(adata):
    phenotypes = adata.obs[adata.uns["tcri_phenotype_key"]].tolist()
    unique_phenotypes = np.unique(phenotypes)
    entropys = dict()
    df = adata.obs.copy()
    for phenotype in unique_phenotypes:
        pheno_df = df[df[adata.uns['tcri_phenotype_key']]==phenotype]
        clonotypes = pheno_df[adata.uns["tcri_clone_key"]].tolist()
        unique_clonotypes = np.unique(clonotypes)
        nums = []
        for tcr in unique_clonotypes:
            tcrs = pheno_df[pheno_df[adata.uns["tcri_clone_key"]]==tcr]
            nums.append(len(tcrs))
        clonality = 1 - entropy(np.array(nums),base=2) / np.log2(len(nums))
        entropys[phenotype] = np.nan_to_num(clonality)
    return entropys

def clone_fraction(adata, groupby):
    frequencies = dict()
    for group in set(adata.obs[groupby]):
        frequencies[group] = dict()
        sdata = adata[adata.obs[groupby] == group]
        total_cells = len(sdata.obs.index.tolist())
        clones = sdata.obs[sdata.uns["tcri_clone_key"]].tolist()
        for c in set(clones):
            frequencies[group][c] = clones.count(c) / total_cells
    return frequencies


def mutual_information(
    adata,
    covariate: str,
    *,
    temperature: float = 1.0,
    n_samples: int = 0,
    clones: Optional[List[str]] = None,
    weighted: bool = False,
    normalised: bool = True,
    normalise_mode: str = "average",
    posterior: bool = True,
    combine_with_logits: bool = True,
    verbose: bool = True,
    graph: bool = False,
) -> Union[float, np.ndarray]:
    """
    MI between clonotype and phenotype at one covariate value.

    posterior=True  → Dirichlet draw of p_ct (+ optional logits)  
    posterior=False → prior-only (uses your original joint_distribution).
    """

    if verbose:
        print(f"{BOLD}{MAGENT}📊  MI for '{covariate}'{RESET}")
        _info("posterior",  posterior)
        _info("weighted",   weighted)
        _info("n_samples",  n_samples)

    # helper to obtain *one* joint table
    def _get_df(silent_flag: bool):
        if posterior:
            return joint_distribution_posterior(
                adata,
                covariate_label     = covariate,
                temperature         = temperature,
                clones              = clones,
                weighted            = weighted,
                combine_with_logits = combine_with_logits,
                silent              = silent_flag,
            )
        else:
            # fall back to prior-only version (not shown here)
            raise NotImplementedError("prior-only joint_distribution not included")

    # ---------- single draw ----------------------------------------
    if n_samples == 0:
        df  = _get_df(silent_flag=not verbose)
        pxy = df.to_numpy().astype(float)
        pxy /= pxy.sum()
        mi  = _mi_from_joint(pxy, normalised, normalise_mode)
        _ok("computed MI",           quiet=not verbose)
        _info("value", f"{mi:.4f}",  quiet=not verbose)
        return mi

    # ---------- multiple draws -------------------------------------
    mi_samples = np.empty(n_samples, dtype=float)
    iter_bar   = tqdm(range(n_samples), disable=not verbose, leave=False,
                      desc=f"sampling MI [{covariate}]")
    for i in iter_bar:
        df = _get_df(silent_flag=True)
        pxy = df.to_numpy().astype(float)
        pxy /= pxy.sum()
        mi_samples[i] = _mi_from_joint(pxy, normalised, normalise_mode)

    _ok("computed MI for all samples", quiet=not verbose)
    _info("mean ± sd",
          f"{mi_samples.mean():.4f} ± {mi_samples.std():.4f}",
          quiet=not verbose)
    _info("95% CI",
          f"[{np.percentile(mi_samples,2.5):.4f}, "
          f"{np.percentile(mi_samples,97.5):.4f}]",
          quiet=not verbose)

    if graph and n_samples > 1 and verbose:
        print(f"{DIM}\nASCII histogram of MI posterior:\n"
              f"{_ascii_hist(mi_samples)}{RESET}")

    return mi_samples



# ────────────────────────────────────────────────────────────────
def flux_table(
    adata,
    *,
    cov_pre:  str = "Pre-treatment",
    cov_post: str = "Post-treatment",
    splitby:  str = "response",
    n_samples: int = 0,
    temperature: float = 1.0,
    weighted: bool = False,
    posterior: bool = True,
    combine_with_logits: bool = True,
    distance_metric: str = "l1",
    seed: Optional[int] = 42,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Build a tidy table with per-clone flux distances and clone size.

    Nested progress bars:
        • outer: groups in `splitby`
        • inner: clones within that group
    """

    if seed is not None:
        np.random.seed(seed)

    meta        = adata.uns["tcri_metadata"]
    clone_col   = meta["clone_col"]

    groups   = sorted(adata.obs[splitby].dropna().unique().tolist())
    records  = []

    outer_it = groups if not show_progress else tqdm(groups, desc="groups")

    for g in outer_it:
        # -------- subset AnnData to *this* group -----------------
        mask_g   = adata.obs[splitby] == g
        adata_g  = adata[mask_g]
        clones_g = adata_g.obs[clone_col].unique().tolist()

        # -------- clone sizes (within this group) ----------------
        c_sizes = adata_g.obs[clone_col].value_counts().to_dict()

        # -------- flux distances  (vector, index = clone_id) -----
        dist = flux(
            adata,
            from_this       = cov_pre,
            to_that         = cov_post,
            clones          = clones_g,
            temperature     = temperature,
            distance_metric = distance_metric,
            n_samples       = n_samples,
            weighted        = weighted,
            posterior       = posterior,
            combine_with_logits = combine_with_logits,
            graph           = False            # keep helper quiet
        )

        # nested bar over clones ----------------------------------
        inner_it = clones_g if not show_progress else tqdm(
            clones_g, desc=f"{g}: clones", leave=False)

        for cl in inner_it:
            if n_samples == 0:
                val   = float(dist.get(cl, np.nan))
                sd    = 0.0
                vec   = np.array([val])
            else:
                # `dist` is (n_samples, n_clones); retrieve column
                idx   = clones_g.index(cl)
                vec   = dist[:, idx]
                val   = float(vec.mean())
                sd    = float(vec.std(ddof=1))

            records.append(dict(
                **{splitby: g, "clone_id": cl},
                flux_samples = vec,
                flux_mean    = val,
                flux_sd      = sd,
                clone_size   = c_sizes.get(cl, 0)
            ))

    return pd.DataFrame.from_records(records)


# ────────────────────────────────────────────────────────────────
def flux(
    adata,
    *,
    from_this           : str,
    to_that             : str,
    clones              : Optional[Union[str, List[str]]] = None,
    temperature         : float = 1.0,
    distance_metric     : Union[str, callable] = "l1",      # MODIFIED: Type hint updated
    n_samples           : int   = 0,         # posterior draws
    weighted            : bool  = False,
    posterior           : bool  = True,
    combine_with_logits : bool  = True,
    graph               : bool  = False,     # ASCII histogram
    seed                : Optional[int] = 42
) -> Union[pd.Series, np.ndarray]:
    """
    Flux distance D(p_clone^from , p_clone^to)  per clone.

    Returns
    -------
    • `pd.Series` (index = clone_id)   if `n_samples == 0`
    • `np.ndarray` shape = (n_samples, n_clones) otherwise
      (rows correspond to posterior draws)
    """
    if seed is not None:
        np.random.seed(seed)

    # ---------- which clones? ----------------------------------
    clone_col = adata.uns["tcri_metadata"]["clone_col"]
    if clones is None:
        clones = adata.obs[clone_col].unique().tolist()
    elif isinstance(clones, str):
        clones = [clones]

    # ---------- get joint tables -------------------------------
    get = joint_distribution_posterior if posterior else joint_distribution

    jd_from = get(
        adata,
        covariate_label     = from_this,
        temperature         = temperature,
        clones              = clones,
        weighted            = weighted,
        combine_with_logits = combine_with_logits if posterior else None,
        silent              = True
    )
    jd_to   = get(
        adata,
        covariate_label     = to_that,
        temperature         = temperature,
        clones              = clones,
        weighted            = weighted,
        combine_with_logits = combine_with_logits if posterior else None,
        silent              = True
    )

    if jd_from.empty or jd_to.empty:
        raise ValueError("No overlap between requested clones and data.")

    common = jd_from.index.intersection(jd_to.index)

    # --- MINIMAL CHANGE BLOCK 1 ---
    # Define _dkl once if needed
    _dkl = None
    if isinstance(distance_metric, str) and distance_metric.lower() == "dkl":
        eps = 1e-15
        def dkl_func(p, q):
            p = p.clip(eps)/p.sum(); q = q.clip(eps)/q.sum()
            return float(np.sum(p*np.log(p/q)))
        _dkl = dkl_func

    if isinstance(distance_metric, collections.abc.Callable):
        dist = pd.Series(
            {cl: distance_metric(jd_from.loc[cl], jd_to.loc[cl]) for cl in common}
        )
    elif isinstance(distance_metric, str) and distance_metric.lower() == "l1":
        dist = (jd_from.loc[common] - jd_to.loc[common]).abs().sum(axis=1)
    elif _dkl is not None:
        dist = pd.Series(
            {cl: _dkl(jd_from.loc[cl], jd_to.loc[cl]) for cl in common}
        )
    else:
        raise ValueError("distance_metric must be 'l1', 'dkl', or a callable function.")
    # --- END OF CHANGE ---

    # ---------- single / multi-sample handling -----------------
    if n_samples == 0:
        return dist

    # posterior draws (one extra call per sample)
    samples = np.empty((n_samples, len(common)), dtype=float)
    for i in range(n_samples):
        jd_from_s = get(adata, covariate_label = from_this, temperature=temperature,
                        clones=clones, weighted=weighted,
                        combine_with_logits=combine_with_logits if posterior else None,
                        silent=True)
        jd_to_s   = get(adata, covariate_label = to_that,  temperature=temperature,
                        clones=clones, weighted=weighted,
                        combine_with_logits=combine_with_logits if posterior else None,
                        silent=True)
        
        # --- MINIMAL CHANGE BLOCK 2 ---
        if isinstance(distance_metric, collections.abc.Callable):
            samples[i] = [distance_metric(jd_from_s.loc[c], jd_to_s.loc[c]) for c in common]
        elif isinstance(distance_metric, str) and distance_metric.lower() == "l1":
            samples[i] = (jd_from_s.loc[common] - jd_to_s.loc[common]).abs().sum(axis=1)
        else: # Handles the 'dkl' case using the function defined above
            samples[i] = [ _dkl(jd_from_s.loc[c], jd_to_s.loc[c]) for c in common ]
        # --- END OF CHANGE ---

    if graph:
        print(_ascii_hist(samples.ravel()))

    return samples