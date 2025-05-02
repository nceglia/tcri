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

# # ---------- ASCII histogram (quick visual check) --------------------
# def _ascii_hist(v, bins=25, width=40):
#     h, edges = np.histogram(v, bins=bins)
#     top = h.max()
#     out=[]
#     for c,e0,e1 in zip(h, edges[:-1], edges[1:]):
#         bar = "█"*int(width*c/top) if top else ""
#         out.append(f"{e0:7.3f}–{e1:7.3f} | {bar}")
#     return "\n".join(out)

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
    """Shannon entropy of a 1-D probability vector (with tiny ϵ-guard)."""
    p   = np.asarray(p, dtype=float)
    eps = 1e-15
    p   = p.clip(eps) / p.sum()          # re-normalise + avoid log(0)
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
    """
    One-shot Shannon entropy **H(C)** over clonotypes inside a phenotype
    at the specified covariate level.

    It relies on the *joint-distribution* helpers already in your code‐
    base.  These return (clone × phenotype) tables:

        • posterior=True  → draws p_ct (and optionally adds logits)  
        • posterior=False → uses the prior estimate only
    """
    # ----------- fetch (clone × phenotype) probability table --------
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

    # jd is (clone × phenotype) – select the column of interest
    vec = jd[phenotype].to_numpy(dtype=float)
    eps = 1e-15
    vec = np.clip(vec, eps, None)
    vec = vec / vec.sum()                           # ensure ∑=1

    H = entropy(vec, base=base)
    if normalised and len(vec) > 1:
        H /= np.log(len(vec)) / np.log(base)        # max-entropy normalisation
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

def phenotypic_entropy(adata, covariate, clonotype, base=2, normalized=True, temperature=1.0, n_samples=0, clones=None):
    logf = lambda x : np.log(x) / np.log(base)
    jd = joint_distribution(adata, covariate, temperature=temperature, n_samples=n_samples, clones=clones).T
    res = jd.loc[clonotype].to_numpy()
    pent = entropy(res, base=base)
    if normalized:
        pent = pent / logf(len(res))
    return pent

def phenotypic_entropies(adata, covariate, base=2, normalized=True, temperature=1., n_samples=0):
    tcr_sequences = adata.obs[adata.uns["tcri_clone_key"]].tolist()
    unique_tcrs = np.unique(tcr_sequences)
    tcr_to_entropy_dict = {
        tcr: phenotypic_entropy(
            adata, 
            covariate, 
            tcr, 
            base=base, 
            normalized=normalized, 
            temperature=temperature,
            n_samples=n_samples
        ) 
        for tcr in unique_tcrs
    }
    return tcr_to_entropy_dict

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

def flux(adata, from_this, to_that, clones=None, temperature=1.0, distance_metric="l1", n_samples=0, weighted=False):
    distances = dict()
    jd_this = joint_distribution(adata, from_this, temperature=temperature, n_samples=n_samples, clones=clones,weighted=weighted)
    jd_that = joint_distribution(adata, to_that, temperature=temperature, n_samples=n_samples, clones=clones,weighted=weighted)
    common_indices = jd_this.index.intersection(jd_that.index)
    if distance_metric == "l1":
        distances = (jd_this.loc[common_indices] - jd_that.loc[common_indices]).abs().sum(axis=1)
    else:
        distances = pd.Series(
            {idx: dkl(jd_this.loc[idx], jd_that.loc[idx]) for idx in common_indices}
        )
    return distances

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